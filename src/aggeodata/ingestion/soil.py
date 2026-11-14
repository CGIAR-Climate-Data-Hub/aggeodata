"""
aggeodata.ingestion.soil
=========================

Data Ingestion Layer — Soil.

Rule: This module ONLY downloads files to disk.

Sources covered:

* :class:`SoilGridsDownloader` — SoilGrids 2.0 from one of two backends:

  - ``"hf"`` (default): the 1 km SoilGrids v2 Zarr datacube on Hugging Face
    (``aaguilar90/soilgrids-v2-1km``).  All 12 properties, 5 depths, one
    EPSG:4326 grid; a bounding box is one indexed read per variable.
  - ``"isric"``: ISRIC's own services — 250 m layers via the WCS API (VRT
    fallback) and 1 km hydraulic properties from Google Cloud Storage.

Both backends write the same files: one int16 GeoTIFF per variable and depth,
``{var}_{depth}cm_mean.tif``, in ISRIC's mapped units (e.g. clay in g/kg), so
:class:`~aggeodata.transform.soil_cube.SoilDataCubeBuilder` reads either.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Migrated from: ag_cube_cm.ingestion.soil
# ---------------------------------------------------------------------------

__all__ = ["SoilGridsDownloader", "HF_SOILGRIDS_URL"]


import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Google Cloud Storage root for 1-km aggregated hydraulic variables.
GOOGLESTORAGE: str = (
    "https://storage.googleapis.com/isric-share-soilgrids/pre-release/"
)
#: ISRIC HTTPS root for 1-km aggregated hydraulic variables.
AGGREGATESTORAGE_1000M: str = (
    "https://files.isric.org/soilgrids/latest/data_aggregated/1000m/"
)

#: ISRIC HTTPS root for the 250 m layers as Cloud-Optimised VRTs — the
#: fallback when the WCS server (maps.isric.org) times out.
VRTSTORAGE_250M: str = "https://files.isric.org/soilgrids/latest/data/"

#: Variables available via the SoilGrids WCS/coverage API (250 m native res).
_WCS_VARIABLES: frozenset[str] = frozenset(
    {"bdod", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt", "soc", "cec"}
)
#: Variables available via the 1-km Google Storage bucket.
_STORAGE_1KM_VARIABLES: frozenset[str] = frozenset({"wv0010", "wv0033", "wv1500"})

#: SoilGrids v2 at 1 km as a consolidated Zarr v3 datacube on Hugging Face.
#: int16 ``(depth, lat, lon)`` arrays on a 1/120 degree EPSG:4326 grid, stored
#: in ISRIC's mapped units; chunks hold all five depths of a 480 x 480 tile.
HF_SOILGRIDS_URL: str = (
    "https://huggingface.co/datasets/aaguilar90/soilgrids-v2-1km/"
    "resolve/main/soilgrids-v2-1km.zarr"
)
#: Variables in the Hugging Face store.
_HF_VARIABLES: frozenset[str] = _WCS_VARIABLES | _STORAGE_1KM_VARIABLES
_NODATA: int = -32768


# ---------------------------------------------------------------------------
# Low-level rasterio helper
# ---------------------------------------------------------------------------


def _download_using_rasterio(url: str, extent: list[float], output_path: str) -> None:
    """Stream a remote Cloud-Optimised GeoTIFF, clip it, and save locally.

    Parameters
    ----------
    url : str
        Remote GeoTIFF/COG URL.
    extent : list[float]
        Bounding box ``[xmin, ymin, xmax, ymax]`` in the source CRS
        (ESRI:54052 for SoilGrids WCS, EPSG:4326 for 1-km bucket).
    output_path : str
        Local file path to write the clipped GeoTIFF.
    """
    x1, y1, x2, y2 = extent
    with rasterio.open(url) as src:
        # Reproject the EPSG:4326 bbox to the source CRS when the source is
        # projected (Homolosine metres).  Passing degree coordinates to a
        # projected transform produces a garbage window and 1×1 output.
        # Densify the edges: in Homolosine x depends on latitude, so the two
        # corners alone do not bound the box.
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
        if src.crs is not None and not src.crs.is_geographic:
            from rasterio.warp import transform_bounds
            rx1, ry1, rx2, ry2 = transform_bounds(
                "EPSG:4326", src.crs, x1, y1, x2, y2, densify_pts=21
            )

        window = from_bounds(rx1, ry1, rx2, ry2, src.transform)
        transform = src.window_transform(window)
        # Derive pixel dimensions directly from the window (works for any CRS /
        # resolution) rather than the old hardcoded ÷250 formula.
        height = max(1, round(window.height))
        width  = max(1, round(window.width))

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "tiled": True,
                "compress": "deflate",
                "dtype": "int16",
                "nodata": -32768,
                "height": height,
                "width": width,
                "transform": transform,
            }
        )
        tags = src.tags()

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.update_tags(**tags)
            dst.write(src.read(window=window, out_shape=(src.count, height, width)))

    logger.debug("Saved %s", output_path)


# ---------------------------------------------------------------------------
# Main downloader class
# ---------------------------------------------------------------------------


class SoilGridsDownloader:
    """Download SoilGrids raster layers to local GeoTIFF files.

    Backends (the *backend* parameter):

    * ``"hf"`` (default) — SoilGrids v2 at 1 km from the Hugging Face Zarr
      datacube (:data:`HF_SOILGRIDS_URL`).  Every variable on one EPSG:4326
      grid; each variable is a single read covering all requested depths.
      No credentials.
    * ``"isric"`` — ISRIC's services: physical/chemical variables (clay,
      sand, bdod, …) at 250 m via the WCS API (VRT fallback), hydraulic
      variables (wv0010, wv0033, wv1500) at 1 km from Google Cloud Storage.

    If the Hugging Face store cannot be read, the affected files are fetched
    from ISRIC instead, with a warning — those layers then come at 250 m.

    Parameters
    ----------
    soil_layers : list[str]
        SoilGrids variable names to download.
    depths : list[str]
        Depth intervals in ``"lo-hi"`` format (e.g. ``["0-5", "5-15"]``).
    output_folder : str
        Root folder where downloaded GeoTIFFs are saved.  Existing files are
        reused, whichever backend wrote them — use a new folder when switching
        backends.
    backend : str
        ``"hf"`` (default) or ``"isric"``.
    zarr_url : str
        Store URL for ``backend="hf"``.  Defaults to :data:`HF_SOILGRIDS_URL`.

    Examples
    --------
    >>> dl = SoilGridsDownloader(
    ...     soil_layers=["clay", "sand", "wv1500"],
    ...     depths=["0-5", "5-15", "15-30"],
    ...     output_folder="data/raw/soil_mwi",
    ... )
    >>> dl.download(boundaries=[-90.5, 13.0, -88.5, 15.5])
    """

    def __init__(
        self,
        soil_layers: list[str],
        depths: list[str],
        output_folder: str,
        backend: Literal["hf", "isric"] = "hf",
        zarr_url: str = HF_SOILGRIDS_URL,
    ) -> None:
        if backend not in {"hf", "isric"}:
            raise ValueError(f"backend must be 'hf' or 'isric', got '{backend}'")
        self.soil_layers = soil_layers
        self.depths = depths
        self.output_folder = output_folder
        self.backend = backend
        self.zarr_url = zarr_url
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, boundaries: list[float]) -> list[str]:
        """Download all configured variables and depths for the given extent.

        Parameters
        ----------
        boundaries : list[float]
            ``[xmin, ymin, xmax, ymax]`` bounding box in **EPSG:4326**.

        Returns
        -------
        list[str]
            Paths to all files that were successfully written.
        """
        written: list[str] = []
        pending: list[tuple[str, str]] = []
        for var in self.soil_layers:
            for depth in self.depths:
                out_path = self._expected_output_path(var, depth)
                if os.path.exists(out_path):
                    logger.info("Already exists, skipping: %s", out_path)
                    written.append(out_path)
                else:
                    pending.append((var, depth))

        if pending and self.backend == "hf":
            try:
                done = self._download_from_zarr(pending, boundaries)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Hugging Face SoilGrids store failed (%s); falling back to ISRIC", exc
                )
                done = []
            written.extend(done)
            pending = [
                (v, d) for v, d in pending if self._expected_output_path(v, d) not in done
            ]
            if pending:
                logger.warning(
                    "Fetching %d layer(s) from ISRIC instead: %s. Physical/chemical "
                    "variables from ISRIC are 250 m, not 1 km.",
                    len(pending), ", ".join(f"{v} {d}" for v, d in pending),
                )

        if pending:
            written.extend(self._download_from_isric(pending, boundaries))
        return written

    def _download_from_isric(
        self, pending: list[tuple[str, str]], boundaries: list[float]
    ) -> list[str]:
        """Fetch each variable + depth from ISRIC's WCS / VRT / 1-km storage."""
        x1, y1, x2, y2 = boundaries
        written: list[str] = []

        for var, depth in pending:
            out_path = self._expected_output_path(var, depth)
            try:
                if var in _WCS_VARIABLES:
                    try:
                        self._get_from_soilgrid_package(
                            var, depth, [x1, y1, x2, y2], out_path
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "WCS failed for %s @ %s (%s); falling back to VRT",
                            var, depth, exc,
                        )
                        self._get_from_soilgrid_vrt(
                            var, depth, [x1, y1, x2, y2], out_path
                        )
                elif var in _STORAGE_1KM_VARIABLES:
                    self._get_from_soilgrid_1000aggregate(
                        var, depth, [x1, y1, x2, y2], self.output_folder
                    )
                else:
                    logger.warning("Unknown SoilGrids variable: %s", var)
                    continue

                logger.info("Created: %s", out_path)
                written.append(out_path)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to download %s @ %s: %s", var, depth, exc
                )

        return written

    # ------------------------------------------------------------------
    # Internal download methods
    # ------------------------------------------------------------------

    def _expected_output_path(self, var: str, depth: str) -> str:
        """Return the canonical output file path for a variable + depth."""
        return os.path.join(self.output_folder, f"{var}_{depth}cm_mean.tif")

    def _download_from_zarr(
        self, pending: list[tuple[str, str]], boundaries: list[float]
    ) -> list[str]:
        """Slice the Hugging Face 1-km store and write one GeoTIFF per layer.

        The store's stored int16 values are kept (``mask_and_scale=False``), so
        files hold ISRIC's mapped units and match what the ISRIC backend writes.
        One variable is loaded at a time — every chunk carries all five depths,
        so a variable's depths cost no extra requests.

        Parameters
        ----------
        pending : list[tuple[str, str]]
            ``(variable, depth)`` pairs still to write.
        boundaries : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.

        Returns
        -------
        list[str]
            Paths written.  Pairs that failed are left for the caller.

        Raises
        ------
        ValueError
            If the extent does not intersect the store.
        """
        import rioxarray  # noqa: F401

        from .zarr_backend import open_remote_zarr

        ds = open_remote_zarr(self.zarr_url, mask_and_scale=False)

        # Depth labels ("0-5") -> the store's layer-midpoint coordinate.
        bounds = ds["depth_bnds"].values
        depth_coord = {
            f"{int(lo)}-{int(hi)}": float(mid)
            for (lo, hi), mid in zip(bounds, ds["depth"].values, strict=True)
        }

        # Pad by half a cell so every cell the box touches is kept, as the
        # rasterio window read of the ISRIC backend does.
        x1, y1, x2, y2 = boundaries
        half = abs(float(ds["lon"].values[1] - ds["lon"].values[0])) / 2
        box = ds.sel(lat=slice(y2 + half, y1 - half), lon=slice(x1 - half, x2 + half))
        if box.sizes["lat"] == 0 or box.sizes["lon"] == 0:
            raise ValueError(
                f"Extent {boundaries} does not intersect the SoilGrids store "
                f"(lat {float(ds.lat.min()):.1f}..{float(ds.lat.max()):.1f}, "
                f"lon {float(ds.lon.min()):.1f}..{float(ds.lon.max()):.1f})."
            )

        by_var: dict[str, list[str]] = {}
        for var, depth in pending:
            by_var.setdefault(var, []).append(depth)

        written: list[str] = []
        for var, depths in by_var.items():
            if var not in _HF_VARIABLES or var not in ds.data_vars:
                logger.warning("Unknown SoilGrids variable: %s", var)
                continue
            unknown = [d for d in depths if d not in depth_coord]
            if unknown:
                logger.error(
                    "Depths %s are not SoilGrids layers (%s)", unknown, sorted(depth_coord)
                )
            depths = [d for d in depths if d in depth_coord]
            if not depths:
                continue

            try:
                logger.info("Zarr: reading %s @ %s", var, ", ".join(depths))
                arr = box[var].sel(depth=[depth_coord[d] for d in depths]).load()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to read %s from the Zarr store: %s", var, exc)
                continue

            tags = {
                "source": self.zarr_url,
                "aggeodata_backend": "hf",
                "long_name": str(arr.attrs.get("long_name", var)),
                "units": str(arr.attrs.get("mapped_units", "")),
                "conversion_factor": str(arr.attrs.get("conversion_factor", "")),
            }
            for depth in depths:
                out_path = self._expected_output_path(var, depth)
                layer = (
                    arr.sel(depth=depth_coord[depth])
                    .drop_vars(["depth", "depth_bnds", "spatial_ref"], errors="ignore")
                    .astype("int16")
                )
                # Scale/offset attrs would be written into the GeoTIFF band;
                # the files keep plain mapped-unit integers.
                layer.attrs = {}
                layer.encoding = {}
                layer = (
                    layer.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                    .rio.write_crs("EPSG:4326")
                    .rio.write_nodata(_NODATA)
                )
                try:
                    layer.rio.to_raster(
                        out_path, driver="GTiff", tiled=True, compress="deflate", tags=tags
                    )
                except Exception as exc:  # noqa: BLE001
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    logger.error("Failed to write %s @ %s: %s", var, depth, exc)
                    continue
                logger.info("Created: %s", out_path)
                written.append(out_path)

        return written

    @staticmethod
    def _get_from_soilgrid_1000aggregate(
        var: str,
        depth: str,
        extent: list[float],
        output_folder: str,
        source: str = "google_storage",
    ) -> None:
        """Download a 1-km SoilGrids variable from Google Storage / ISRIC.

        Parameters
        ----------
        var : str
            SoilGrids variable name (e.g. ``"wv1500"``).
        depth : str
            Depth string (e.g. ``"0-5"``).
        extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        output_folder : str
            Local folder to save the GeoTIFF.
        source : str
            ``"google_storage"`` (default) or ``"isric"``.
        """
        if source == "google_storage":
            file_name = f"{var}_{depth}cm_mean.tif"
            url = GOOGLESTORAGE + f"{var}/{file_name}"
        else:
            file_name = f"{var}_{depth}cm_mean_1000.tif"
            url = AGGREGATESTORAGE_1000M + f"{var}/{file_name}"

        logger.info("Downloading 1-km layer: %s", url)
        output_path = os.path.join(output_folder, f"{var}_{depth}cm_mean.tif")
        _download_using_rasterio(url, extent, output_path)

    @staticmethod
    def _get_from_soilgrid_vrt(
        var: str,
        depth: str,
        extent: list[float],
        output_path: str,
    ) -> None:
        """Download a 250 m SoilGrids variable by streaming ISRIC's VRT.

        Parameters
        ----------
        var : str
            SoilGrids variable (e.g. ``"clay"``).
        depth : str
            Depth string (e.g. ``"5-15"``).
        extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        output_path : str
            Full output file path for the GeoTIFF.
        """
        url = VRTSTORAGE_250M + f"{var}/{var}_{depth}cm_mean.vrt"
        logger.info("VRT request: %s -> %s", url, output_path)
        # Partially written files would be skipped as "already exists" next run
        try:
            _download_using_rasterio(url, extent, output_path)
        except Exception:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise

    @staticmethod
    def _get_from_soilgrid_package(
        var: str,
        depth: str,
        extent: list[float],
        output_path: str,
    ) -> None:
        """Download a SoilGrids variable via the WCS API (pyproj + soilgrids).

        Parameters
        ----------
        var : str
            SoilGrids variable (e.g. ``"clay"``).
        depth : str
            Depth string (e.g. ``"5-15"``).
        extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        output_path : str
            Full output file path for the GeoTIFF.
        """
        try:
            from pyproj import Transformer
            from soilgrids import SoilGrids
        except ImportError as exc:
            raise ImportError(
                "pyproj and soilgrids are required for WCS downloads. "
                "Install with: pip install pyproj soilgrids"
            ) from exc

        x1, y1, x2, y2 = extent

        if -180.0 < x1 < 180.0:
            proj_igh = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
            tfm = Transformer.from_crs("EPSG:4326", proj_igh, always_xy=True)
            x1, y1 = tfm.transform(float(x1), float(y1))
            x2, y2 = tfm.transform(float(x2), float(y2))

        logger.info("WCS request: %s @ %s -> %s", var, depth, output_path)
        sg = SoilGrids()
        sg.get_coverage_data(
            service_id=var,
            coverage_id=f"{var}_{depth}cm_mean",
            west=x1,
            south=y1,
            east=x2,
            north=y2,
            crs="urn:ogc:def:crs:EPSG::152160",
            output=output_path,
        )
