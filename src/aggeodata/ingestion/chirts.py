"""
aggeodata.ingestion.chirts
===========================

CHIRTS-daily temperature downloader.

Downloads Tmax and/or Tmin GeoTIFFs from the UCSB Climate Hazards Center
servers, clips to an AOI, and saves each day as a NetCDF file.

Two data sources are supported via the *source* parameter:

* ``"era5"`` (default) — CHIRTS-ERA5 (experimental reanalysis blend):
  ``https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/``
* ``"chirts"`` — Original CHIRTS-daily v1.0 COG/TIF products:
  ``https://data.chc.ucsb.edu/products/CHIRTS-daily/``

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


class CHIRTSDownloader:
    """Download CHIRTS Tmax/Tmin from UCSB servers.

    Parameters
    ----------
    variables : list[str]
        One or both of ``['tmax', 'tmin']``.  Default: both.
    source : str
        ``"era5"`` *(default)* — CHIRTS-ERA5 experimental reanalysis blend.
        ``"chirts"`` — Original CHIRTS-daily v1.0 COG product.

    Examples
    --------
    >>> dl = CHIRTSDownloader(variables=["tmax"], source="era5")
    >>> paths = dl.download(
    ...     extent=[-90.5, 13.0, -89.5, 14.5],
    ...     starting_date="2015-01-01",
    ...     ending_date="2015-12-31",
    ...     output_folder="data/raw/chirts",
    ... )
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        source: Literal["era5", "chirts"] = "era5",
    ) -> None:
        self.variables = [v.lower() for v in (variables or ["tmax", "tmin"])]
        self.source = source
        _valid = {"tmax", "tmin"}
        if not set(self.variables).issubset(_valid):
            raise ValueError(f"CHIRTS variables must be in {_valid}, got {self.variables}")
        if source not in {"era5", "chirts"}:
            raise ValueError(f"source must be 'era5' or 'chirts', got '{source}'")

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
            Concurrent day-downloads (hard-capped at 1).  Default: 1.
        polite_delay : float
            Per-worker sleep in seconds after each request.  Default: 0.5.

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
        out_nc = os.path.join(
            output_folder, variable, year,
            f"chirts_{variable}_{year}{month}{day}.nc",
        )
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
