"""
aggeodata.ingestion.chirps
===========================

CHIRPS daily precipitation downloader.

Downloads Cloud Optimized GeoTIFFs directly from the UCSB data warehouse,
clips to an AOI, and saves each day as a local NetCDF file.

Rule: this module ONLY downloads files to disk.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

from .files_manager import create_yearly_query
from .gis_functions import from_xyxy_2polygon, numpy_to_xarray

logger = logging.getLogger(__name__)


class CHIRPSDownloader:
    """Download CHIRPS daily precipitation from the UCSB COG endpoints.

    Parameters
    ----------
    frequency : str
        ``'daily'`` (default) or ``'monthly'``.
    sp_resolution : str
        Spatial resolution code — ``'05'`` for 0.05°.  Default ``'05'``.
    version : str
        ``'3.0'`` (default) or ``'2.0'``.

    Examples
    --------
    >>> dl = CHIRPSDownloader()
    >>> paths = dl.download(
    ...     extent=[-90.5, 13.0, -89.5, 14.5],
    ...     starting_date="2018-01-01",
    ...     ending_date="2018-12-31",
    ...     output_folder="data/raw/chirps",
    ... )
    """

    URL_V2: str = (
        "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
        "global_{freq}/cogs/p{res}/{year}/chirps-v2.0.{date}.cog"
    )
    URL_V3: str = (
        "https://data.chc.ucsb.edu/products/CHIRP-v3.0/"
        "{freq}/global/tifs/{year}/chirp-v3.0.{date}.tif"
    )

    def __init__(
        self,
        frequency: str = "daily",
        sp_resolution: str = "05",
        version: str = "3.0",
    ) -> None:
        self.frequency = frequency
        self.resolution = sp_resolution
        self.version = version

    def download(
        self,
        extent: list[float],
        starting_date: str,
        ending_date: str,
        output_folder: str,
        ncores: int = 2,
        polite_delay: float = 0.15,
    ) -> dict[str, str]:
        """Download CHIRPS data for the given extent and date range.

        Parameters
        ----------
        extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        starting_date : str
            ISO 8601 start date ``"YYYY-MM-DD"``.
        ending_date : str
            ISO 8601 end date ``"YYYY-MM-DD"``.
        output_folder : str
            Root folder; one sub-folder per year is created automatically.
        ncores : int
            Maximum concurrent day-downloads.  Hard-capped at 3 internally
            to avoid HTTP 403 from the UCSB server.  Default: 4.
        polite_delay : float
            Per-worker sleep in seconds after each completed request.  Default: 0.1.

        Returns
        -------
        dict[str, str]
            ``{year_str: year_folder_path}``
        """
        from tqdm import tqdm

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        yearly_dates = create_yearly_query(starting_date, ending_date)

        year_folders: dict[str, str] = {}
        jobs: list[tuple[str, str, str]] = []
        for year, monthly_dates in yearly_dates.items():
            year_folder = os.path.join(output_folder, year)
            Path(year_folder).mkdir(parents=True, exist_ok=True)
            year_folders[year] = year_folder
            for month, days in monthly_dates.items():
                for day in days:
                    jobs.append((year, month, day))

        workers = min(ncores, 2)  # avoid ban
        pbar = tqdm(total=len(jobs), desc="CHIRPS", unit="day")

        def _run(job: tuple[str, str, str]) -> None:
            year, month, day = job
            try:
                self._download_one_day(year, month, day, output_folder, extent)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CHIRPS failed %s-%s-%s: %s", year, month, day, exc)
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

    def _build_url(self, year: str, date_str: str, version: str | None = None) -> str:
        v = version or self.version
        if v == "3.0" and int(year) <= 2000:
            v = "2.0"
        if v == "2.0":
            return self.URL_V2.format(
                freq=self.frequency, res=self.resolution, year=year, date=date_str
            )
        return self.URL_V3.format(freq=self.frequency, year=year, date=date_str)

    def _download_one_day(
        self,
        year: str,
        month: str,
        day: str,
        output_folder: str,
        extent: list[float],
    ) -> None:
        date_str = f"{year}.{month}.{day}"
        out_nc = os.path.join(output_folder, year, f"chirps_precipitation_{year}{month}{day}.nc")
        if os.path.exists(out_nc):
            return

        primary_url = self._build_url(year, date_str)
        fallback_url = self._build_url(year, date_str, version="2.0") if self.version != "2.0" else None
        urls = [primary_url] + ([fallback_url] if fallback_url else [])

        last_exc: Exception | None = None
        for url in urls:
            try:
                with rasterio.Env(GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_MAX_RETRY=2, GDAL_HTTP_RETRY_DELAY=3):
                    with rasterio.open(url) as src:
                        aoi_geom = gpd.GeoSeries([from_xyxy_2polygon(*extent)])
                        masked, transform = rio_mask(dataset=src, shapes=aoi_geom, crop=True)
                        if masked.shape[0] > 1:
                            masked = np.expand_dims(masked[-1], axis=0)
                        xrm = numpy_to_xarray(masked, transform, crs=str(src.crs), var_name="pr")
                        xrm.to_netcdf(out_nc)
                        logger.debug("CHIRPS saved: %s", out_nc)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.debug("CHIRPS URL failed (%s): %s", url, exc)

        logger.warning("CHIRPS could not download %s: %s", date_str, last_exc)
