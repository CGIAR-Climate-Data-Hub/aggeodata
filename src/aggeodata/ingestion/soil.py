"""
aggeodata.ingestion.soil
=========================

Data Ingestion Layer — Soil.

Rule: This module ONLY downloads files to disk.

Sources covered:

* :class:`SoilGridsDownloader` — ISRIC SoilGrids 250 m (WCS API) and 1 km
  hydraulic properties (Google Cloud Storage).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Migrated from: ag_cube_cm.ingestion.soil
# ---------------------------------------------------------------------------

__all__ = ["SoilGridsDownloader"]


import logging
import os
from pathlib import Path

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

#: Variables available via the SoilGrids WCS/coverage API (250 m native res).
_WCS_VARIABLES: frozenset[str] = frozenset(
    {"bdod", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt", "soc", "cec"}
)
#: Variables available via the 1-km Google Storage bucket.
_STORAGE_1KM_VARIABLES: frozenset[str] = frozenset({"wv0010", "wv0033", "wv1500"})


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
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
        if src.crs is not None and not src.crs.is_geographic:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            rx1, ry1 = tr.transform(x1, y1)
            rx2, ry2 = tr.transform(x2, y2)

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

    Variable dispatch:

    * Physical/chemical variables (clay, sand, bdod, …) → SoilGrids WCS API
      via the ``soilgrids`` Python package (requires Homolosine coordinates).
    * Hydraulic variables (wv0010, wv0033, wv1500) → 1-km Google Storage
      bucket via ``rasterio`` streaming.

    Parameters
    ----------
    soil_layers : list[str]
        SoilGrids variable names to download.
    depths : list[str]
        Depth intervals in ``"lo-hi"`` format (e.g. ``["0-5", "5-15"]``).
    output_folder : str
        Root folder where downloaded GeoTIFFs are saved.

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
    ) -> None:
        self.soil_layers = soil_layers
        self.depths = depths
        self.output_folder = output_folder
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
        x1, y1, x2, y2 = boundaries
        written: list[str] = []

        for var in self.soil_layers:
            for depth in self.depths:
                out_path = self._expected_output_path(var, depth)
                if os.path.exists(out_path):
                    logger.info("Already exists, skipping: %s", out_path)
                    written.append(out_path)
                    continue

                try:
                    if var in _WCS_VARIABLES:
                        self._get_from_soilgrid_package(
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
