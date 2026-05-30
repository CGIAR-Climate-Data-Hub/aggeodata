"""
aggeodata.ingestion.nasa_power
================================

NASA POWER daily data downloader.

Two backends are available:

* **S3 Zarr** (default, ``NASAPowerS3Downloader``) — reads directly from the
  public NASA POWER Zarr store on Amazon S3.  No rate limits; fast spatial/
  temporal slicing; no API key required.  Requires ``s3fs`` and ``zarr``.

* **REST API** (``NASAPowerDownloader``) — uses the NASA POWER LARC regional
  API (10° × 10° tile limit, slower, subject to rate limits).

Rule: this module ONLY downloads files to disk.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import date
from pathlib import Path

import requests
import xarray as xr

from .cf_names import rename_cf_vars

logger = logging.getLogger(__name__)

_POWER_REGIONAL_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
_S3_ZARR_PATH = "nasa-power/merra2/temporal/power_merra2_daily_temporal_lst.zarr"
_MAX_DEGREE = 10.0
_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tile_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float, max_size: float = _MAX_DEGREE
) -> list[tuple[float, float, float, float]]:
    """Split a large bounding box into tiles of at most max_size°."""
    tiles = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            tiles.append((
                round(x, 6), round(y, 6),
                round(min(x + max_size, xmax), 6),
                round(min(y + max_size, ymax), 6),
            ))
            y += max_size
        x += max_size
    return tiles


def _download_tile(
    xmin: float, ymin: float, xmax: float, ymax: float,
    start: str, end: str,
    parameters: list[str],
    community: str,
) -> xr.Dataset | None:
    params = {
        "parameters": ",".join(parameters),
        "community": community,
        "longitude-min": xmin, "longitude-max": xmax,
        "latitude-min": ymin, "latitude-max": ymax,
        "start": start, "end": end,
        "format": "netcdf",
        "user": "aggeodata",
        "header": "true",
        "time-standard": "UTC",
    }
    resp = requests.get(_POWER_REGIONAL_URL, params=params, timeout=_TIMEOUT, stream=True)
    if resp.status_code == 404:
        logger.warning("NASA POWER: 404 for tile [%.2f,%.2f,%.2f,%.2f]", xmin, ymin, xmax, ymax)
        return None
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=65536):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        return xr.open_dataset(tmp_path, engine="netcdf4").load()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _yyyymmdd(date_str: str) -> str:
    return date_str.replace("-", "")


def _yearly_chunks(start: str, end: str) -> list[tuple[str, str]]:
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    chunks: list[tuple[str, str]] = []
    cur = d0
    while cur <= d1:
        chunk_end = date(cur.year, 12, 31)
        if chunk_end > d1:
            chunk_end = d1
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = date(cur.year + 1, 1, 1)
    return chunks


def _normalize_power_dataset(ds: xr.Dataset, parameters: list[str]) -> xr.Dataset:
    rename: dict[str, str] = {}
    for d in ds.dims:
        if d.lower() in ("lon", "longitude"):
            rename[d] = "lon"
        elif d.lower() in ("lat", "latitude"):
            rename[d] = "lat"
        elif d.lower() in ("time", "date"):
            rename[d] = "time"
    if rename:
        ds = ds.rename(rename)
    for var in ds.data_vars:
        if var in parameters:
            ds[var] = ds[var].where(ds[var] > -990)
    return ds


# ---------------------------------------------------------------------------
# REST API backend
# ---------------------------------------------------------------------------

class NASAPowerDownloader:
    """Download daily regional climate data from the NASA POWER REST API.

    Parameters
    ----------
    parameters : list[str]
        NASA POWER parameter codes (community AG).  Common codes:
        ``ALLSKY_SFC_SW_DWN``, ``T2M``, ``T2M_MAX``, ``T2M_MIN``,
        ``RH2M``, ``WS2M``, ``PRECTOTCORR``.
    community : str
        NASA POWER community.  ``"AG"`` (default), ``"RE"``, or ``"SB"``.

    Examples
    --------
    >>> dl = NASAPowerDownloader(parameters=["T2M_MAX", "T2M_MIN", "RH2M"])
    >>> path = dl.download(
    ...     extent=[-90.5, 13.0, -88.5, 15.5],
    ...     starting_date="2015-01-01",
    ...     ending_date="2015-12-31",
    ...     output_folder="data/raw/nasa_power",
    ... )
    """

    _DEFAULT_PARAMS = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN"]

    def __init__(
        self,
        parameters: list[str] | None = None,
        community: str = "AG",
    ) -> None:
        self.parameters = parameters or self._DEFAULT_PARAMS
        self.community = community

    def download(
        self,
        extent: list[float],
        starting_date: str,
        ending_date: str,
        output_folder: str,
        force: bool = False,
    ) -> str:
        """Download NASA POWER data for the given extent and date range.

        Parameters
        ----------
        extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        starting_date : str
            ISO 8601 start ``"YYYY-MM-DD"``.
        ending_date : str
            ISO 8601 end ``"YYYY-MM-DD"``.
        output_folder : str
            Folder where the output NetCDF will be saved.
        force : bool
            Re-download even if a cached file exists.  Default: False.

        Returns
        -------
        str
            Path to the saved NetCDF file.
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        fname = f"nasa_power_{starting_date}_{ending_date}.nc"
        out_nc = os.path.join(output_folder, fname)

        if os.path.exists(out_nc) and not force:
            logger.info("NASA POWER: using cached file %s", out_nc)
            return out_nc

        xmin, ymin, xmax, ymax = extent
        tiles = _tile_bbox(xmin, ymin, xmax, ymax)
        chunks = _yearly_chunks(starting_date, ending_date)

        all_datasets: list[xr.Dataset] = []
        for chunk_start, chunk_end in chunks:
            start = _yyyymmdd(chunk_start)
            end = _yyyymmdd(chunk_end)
            for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                logger.info("NASA POWER: tile %d/%d  %s→%s", i + 1, len(tiles), chunk_start, chunk_end)
                ds_tile = _download_tile(tx1, ty1, tx2, ty2, start, end, self.parameters, self.community)
                if ds_tile is not None:
                    all_datasets.append(_normalize_power_dataset(ds_tile, self.parameters))

        if not all_datasets:
            raise RuntimeError("NASA POWER: no data downloaded for any tile/chunk.")

        merged = (
            xr.combine_by_coords(all_datasets, combine_attrs="override")
            if len(all_datasets) > 1
            else all_datasets[0]
        )
        merged = rename_cf_vars(merged)
        encoding = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
        merged.to_netcdf(out_nc, encoding=encoding, engine="netcdf4")
        logger.info("NASA POWER saved -> %s", out_nc)
        return out_nc


# ---------------------------------------------------------------------------
# S3 Zarr backend (preferred — no rate limits)
# ---------------------------------------------------------------------------

class NASAPowerS3Downloader:
    """Download NASA POWER data directly from the public S3 Zarr store.

    Faster than the REST API: no tiling, no rate limits, direct spatial/
    temporal slicing over the full 1981–present daily global archive.
    Requires ``s3fs`` and ``zarr``.

    Parameters
    ----------
    parameters : list[str] | None
        NASA POWER variable codes.  Default: T2M_MAX, T2M_MIN, RH2M,
        WS2M, ALLSKY_SFC_SW_DWN.

    Examples
    --------
    >>> dl = NASAPowerS3Downloader(parameters=["T2M_MAX", "RH2M"])
    >>> path = dl.download(
    ...     extent=[-90.5, 13.0, -88.5, 15.5],
    ...     starting_date="2015-01-01",
    ...     ending_date="2015-12-31",
    ...     output_folder="data/raw/nasa_power",
    ... )
    """

    # ALLSKY_SFC_SW_DWN (solar radiation) is NOT in this MERRA-2 temporal Zarr
    # store — use the REST API backend (NASAPowerDownloader) for radiation vars.
    _DEFAULT_PARAMS = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M"]

    def __init__(self, parameters: list[str] | None = None) -> None:
        self.parameters = parameters or self._DEFAULT_PARAMS

    def download(
        self,
        extent: list[float],
        starting_date: str,
        ending_date: str,
        output_folder: str,
        force: bool = False,
    ) -> str:
        """Slice the S3 Zarr store to extent/dates and save to NetCDF.

        Parameters
        ----------
        extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        starting_date : str
            ISO 8601 start ``"YYYY-MM-DD"``.
        ending_date : str
            ISO 8601 end ``"YYYY-MM-DD"``.
        output_folder : str
            Folder where the output NetCDF will be saved.
        force : bool
            Re-download even if a cached file exists.  Default: False.

        Returns
        -------
        str
            Path to the saved NetCDF.
        """
        try:
            import s3fs
        except ImportError as exc:
            raise ImportError("s3fs is required: pip install s3fs") from exc

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        fname = f"nasa_power_{starting_date}_{ending_date}.nc"
        out_nc = os.path.join(output_folder, fname)

        if os.path.exists(out_nc) and not force:
            logger.info("NASA POWER S3: using cached file %s", out_nc)
            return out_nc

        xmin, ymin, xmax, ymax = extent
        logger.info("NASA POWER S3: opening Zarr store")

        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(_S3_ZARR_PATH, s3=fs)
        ds_full = xr.open_zarr(store, consolidated=True)

        missing = [v for v in self.parameters if v not in ds_full.data_vars]
        if missing:
            raise ValueError(f"Variables not in S3 Zarr store: {missing}. Available: {sorted(ds_full.data_vars)}")

        ds = (
            ds_full[self.parameters]
            .sel(time=slice(starting_date, ending_date))
            .sel(lat=slice(ymin, ymax), lon=slice(xmin, xmax))
        )
        for var in ds.data_vars:
            ds[var] = ds[var].where(ds[var] > -990)

        ds = ds.compute()
        ds = rename_cf_vars(ds)
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(out_nc, encoding=encoding, engine="netcdf4")
        logger.info("NASA POWER S3 saved -> %s", out_nc)
        return out_nc
