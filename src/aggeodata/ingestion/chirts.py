"""
aggeodata.ingestion.chirts
===========================

CHIRTS-daily temperature downloader.

Downloads Tmax and/or Tmin GeoTIFFs from the UCSB Climate Hazards Center
servers, clips to an AOI, and saves each day as a NetCDF file.

Two data products are supported via the *source* parameter:

* ``"era5"`` (default) — CHIRTS-ERA5 (experimental reanalysis blend):
  ``https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/``
* ``"chirts"`` — Original CHIRTS-daily v1.0 COG/TIF products:
  ``https://data.chc.ucsb.edu/products/CHIRTS-daily/``

Two transports are supported via the *backend* parameter:

* ``"ucsb"`` (default) — one GeoTIFF request per day from the URLs above.
* ``"hf"`` — the CHIRTS-ERA5 daily Zarr datacube mirrored on Hugging Face
  (:data:`~aggeodata.ingestion.zarr_backend.HF_CHIRTS_URL`).  A bounding box plus a
  date range is a single indexed read, which is far faster for long series.
  This mirror holds the ``"era5"`` product only.

Both backends write the same per-day NetCDF layout
(``<output_folder>/<variable>/<year>/chirts_<variable>_YYYYMMDD.nc``), so
downstream stages do not care which one was used.

Rule: this module ONLY downloads files to disk.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

from .cf_names import to_cf_name
from .files_manager import create_yearly_query
from .gis_functions import from_xyxy_2polygon, numpy_to_xarray
from .zarr_backend import HF_CHIRTS_URL, download_daily_from_zarr

logger = logging.getLogger(__name__)

_URL_ERA5 = (
    "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/{variable}/tifs/daily/{year}/"
    "CHIRTS-ERA5.daily_{VarTitle}.{date}.tif"
)
_URL_COG = (
    "https://data.chc.ucsb.edu/products/CHIRTS-daily/cogs/{Variable}/{year}/"
    "CHIRTSdaily.v1.0.{Variable}.{date}.cog"
)
_URL_TIF = (
    "https://data.chc.ucsb.edu/products/CHIRTS-daily/tifs/global_cogs_v1.0/{Variable}/{year}/"
    "CHIRTSdaily.v1.0.{Variable}.{date}.tif"
)


_VAR_TITLE = {"tmax": "Tmax", "tmin": "Tmin"}

#: CHIRTS variable -> variable name inside the Hugging Face Zarr store.
_HF_VAR = {"tmax": "Tmax", "tmin": "Tmin"}


class CHIRTSDownloader:
    """Download CHIRTS Tmax/Tmin from UCSB servers or the Hugging Face mirror.

    Parameters
    ----------
    variables : list[str]
        One or both of ``['tmax', 'tmin']``.  Default: both.
    source : str
        ``"era5"`` *(default)* — CHIRTS-ERA5 experimental reanalysis blend.
        ``"chirts"`` — Original CHIRTS-daily v1.0 COG product.
    backend : str
        ``"ucsb"`` *(default)* — one GeoTIFF request per day from UCSB.
        ``"hf"`` — slice the CHIRTS-ERA5 daily Zarr datacube on Hugging Face.
        The mirror carries the ``"era5"`` product only; combining
        ``backend="hf"`` with ``source="chirts"`` raises ``ValueError``.
    hf_url : str | None
        Override the default Hugging Face Zarr URL — only used when
        ``backend="hf"``.

    Examples
    --------
    >>> dl = CHIRTSDownloader(variables=["tmax"], source="era5")
    >>> paths = dl.download(
    ...     extent=[-90.5, 13.0, -89.5, 14.5],
    ...     starting_date="2015-01-01",
    ...     ending_date="2015-12-31",
    ...     output_folder="data/raw/chirts",
    ... )

    The same request against the Hugging Face datacube:

    >>> dl = CHIRTSDownloader(variables=["tmax"], backend="hf")
    >>> paths = dl.download(
    ...     extent=[-90.5, 13.0, -89.5, 14.5],
    ...     starting_date="2015-01-01",
    ...     ending_date="2015-12-31",
    ...     output_folder="data/raw/chirts",
    ...     ncores=4,
    ... )
    """

    URL_HF: str = HF_CHIRTS_URL

    def __init__(
        self,
        variables: list[str] | None = None,
        source: Literal["era5", "chirts"] = "era5",
        backend: Literal["ucsb", "hf"] = "ucsb",
        hf_url: str | None = None,
    ) -> None:
        self.variables = [v.lower() for v in (variables or ["tmax", "tmin"])]
        self.source = source
        _valid = {"tmax", "tmin"}
        if not set(self.variables).issubset(_valid):
            raise ValueError(f"CHIRTS variables must be in {_valid}, got {self.variables}")
        if source not in {"era5", "chirts"}:
            raise ValueError(f"source must be 'era5' or 'chirts', got '{source}'")
        if backend not in {"ucsb", "hf"}:
            raise ValueError(f"backend must be 'ucsb' or 'hf', got '{backend}'")
        if backend == "hf" and source != "era5":
            raise ValueError(
                "The Hugging Face CHIRTS mirror holds the CHIRTS-ERA5 product only. "
                "Use source='era5' with backend='hf', or backend='ucsb' for "
                "the original CHIRTS-daily v1.0."
            )
        self.backend = backend
        self.hf_url = hf_url or self.URL_HF

    def download(
        self,
        extent: list[float],
        starting_date: str,
        ending_date: str,
        output_folder: str,
        ncores: int = 1,
        polite_delay: float = 0.5,
    ) -> dict[str, dict[str, str]]:
        """Download CHIRTS data for the given extent and date range.

        Parameters
        ----------
        extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        starting_date : str
            ISO 8601 start ``"YYYY-MM-DD"``.
        ending_date : str
            ISO 8601 end ``"YYYY-MM-DD"``.
        output_folder : str
            Root folder; ``{output_folder}/{variable}/{year}/`` sub-folders
            are created automatically.
        ncores : int
            With ``backend="ucsb"``: concurrent day-downloads, hard-capped at 1.
            With ``backend="hf"``: number of time-batches fetched concurrently
            (no cap — the Hugging Face CDN handles parallel range reads).
            Default: 1.
        polite_delay : float
            Per-worker sleep in seconds after each request.  Default: 0.5.
            Ignored by the Hugging Face backend.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{variable: {year: year_folder_path}}``
        """
        from tqdm import tqdm

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        yearly_dates = create_yearly_query(starting_date, ending_date)
        workers = min(ncores, 1)  # UCSB now limits to 1 concurrent connection

        jobs: list[tuple[str, str, str, str]] = []
        year_folders: dict[str, dict[str, str]] = {v: {} for v in self.variables}

        for var in self.variables:
            for year, monthly_dates in yearly_dates.items():
                year_folder = os.path.join(output_folder, var, year)
                Path(year_folder).mkdir(parents=True, exist_ok=True)
                year_folders[var][year] = year_folder
                for month, days in monthly_dates.items():
                    for day in days:
                        jobs.append((var, year, month, day))

        if self.backend == "hf":
            self._download_from_hf(extent, starting_date, ending_date, output_folder, ncores)
            return year_folders

        pbar = tqdm(total=len(jobs), desc="CHIRTS", unit="day")

        def _run(job: tuple[str, str, str, str]) -> None:
            var, year, month, day = job
            try:
                self._download_one_day(var, year, month, day, output_folder, extent)
            except Exception as exc:  
                logger.warning("CHIRTS %s %s-%s-%s failed: %s", var, year, month, day, exc)
            finally:
                pbar.update(1)
            if polite_delay > 0:
                time.sleep(polite_delay)

        if workers > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_run, jobs))
        else:
            for job in jobs:
                _run(job)

        pbar.close()
        return year_folders

    # ------------------------------------------------------------------
    # Hugging Face Zarr backend
    # ------------------------------------------------------------------

    @staticmethod
    def _output_path(output_folder: str, variable: str, date_str: str) -> str:
        """Return the per-day NetCDF path for ``variable`` on ``date_str``.

        ``date_str`` is ``"YYYYMMDD"``; ``variable`` is the native CHIRTS name
        (``"tmax"`` / ``"tmin"``), which also names the sub-folder.
        """
        return os.path.join(
            output_folder, variable, date_str[:4], f"chirts_{variable}_{date_str}.nc"
        )

    def _download_from_hf(
        self,
        extent: list[float],
        starting_date: str,
        ending_date: str,
        output_folder: str,
        ncores: int,
    ) -> None:
        """Slice the Hugging Face Zarr store into per-day NetCDF files."""
        # {store variable: CF variable} for the export, plus the CF -> native
        # reverse map the per-day path builder needs for the sub-folder name.
        store_to_cf = {_HF_VAR[v]: to_cf_name(v) for v in self.variables}
        cf_to_native = {to_cf_name(v): v for v in self.variables}

        download_daily_from_zarr(
            url=self.hf_url,
            variables=store_to_cf,
            extent=extent,
            starting_date=starting_date,
            ending_date=ending_date,
            path_for=lambda cf_var, date_str: self._output_path(
                output_folder, cf_to_native[cf_var], date_str
            ),
            ncores=ncores,
            desc="CHIRTS (HF)",
        )

    # ------------------------------------------------------------------
    # UCSB GeoTIFF backend
    # ------------------------------------------------------------------

    def _build_urls(self, variable: str, var_title: str, year: str, date_str: str) -> list[str]:
        if self.source == "era5":
            return [_URL_ERA5.format(variable=variable, VarTitle=var_title, year=year, date=date_str)]
        return [
            _URL_COG.format(Variable=var_title, year=year, date=date_str),
            _URL_TIF.format(Variable=var_title, year=year, date=date_str),
        ]

    def _download_one_day(
        self,
        variable: str,
        year: str,
        month: str,
        day: str,
        output_folder: str,
        extent: list[float],
    ) -> None:
        var_title = _VAR_TITLE[variable]
        date_str = f"{year}.{month}.{day}"
        out_nc = self._output_path(output_folder, variable, f"{year}{month}{day}")
        if os.path.exists(out_nc):
            return

        urls = self._build_urls(variable, var_title, year, date_str)
        last_exc: Exception | None = None

        for url in urls:
            try:
                with rasterio.open(url) as src:
                    aoi_geom = gpd.GeoSeries([from_xyxy_2polygon(*extent)])
                    masked, transform = rio_mask(dataset=src, shapes=aoi_geom, crop=True)
                    if masked.shape[0] > 1:
                        masked = np.expand_dims(masked[-1], axis=0)
                    xrm = numpy_to_xarray(masked, transform, crs=str(src.crs), var_name=to_cf_name(variable))
                    xrm.to_netcdf(out_nc)
                    logger.debug("CHIRTS saved: %s", out_nc)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.debug("CHIRTS URL failed (%s): %s", url, exc)

        logger.warning("CHIRTS could not download %s %s: %s", variable, date_str, last_exc)
