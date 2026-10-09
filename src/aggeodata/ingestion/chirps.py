"""
aggeodata.ingestion.chirps
===========================

CHIRPS daily precipitation downloader.

Two backends are supported via the *backend* parameter:

* ``"ucsb"`` (default) — Cloud Optimized GeoTIFFs from the UCSB data warehouse,
  one HTTP request per day.
* ``"hf"`` — the CHIRPS v3 daily Zarr datacube mirrored on Hugging Face
  (:data:`~aggeodata.ingestion.hf_zarr.HF_CHIRPS_URL`).  A bounding box plus a
  date range is a single indexed read, which is far faster for long series.

Both backends write the same per-day NetCDF layout
(``<output_folder>/<year>/chirps_precipitation_YYYYMMDD.nc``), so downstream
stages do not care which one was used.

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

from .files_manager import create_yearly_query
from .gis_functions import from_xyxy_2polygon, numpy_to_xarray
from .hf_zarr import HF_CHIRPS_URL, download_daily_from_hf

logger = logging.getLogger(__name__)


class CHIRPSDownloader:
    """Download CHIRPS daily precipitation from UCSB or the Hugging Face mirror.

    Parameters
    ----------
    frequency : str
        ``'daily'`` (default) or ``'monthly'``.  Only ``'daily'`` is available
        from the Hugging Face backend.
    sp_resolution : str
        Spatial resolution code — ``'05'`` for 0.05°.  Default ``'05'``.
    version : str
        ``'3.0'`` (default) or ``'2.0'``.  Ignored by the Hugging Face backend,
        which mirrors v3.0 only.
    backend : str
        ``"ucsb"`` *(default)* — one COG/TIF request per day from UCSB.
        ``"hf"`` — slice the CHIRPS v3 daily Zarr datacube on Hugging Face.
    hf_url : str | None
        Override the default Hugging Face Zarr URL — only used when
        ``backend="hf"``.

    Examples
    --------
    >>> dl = CHIRPSDownloader()
    >>> paths = dl.download(
    ...     extent=[-90.5, 13.0, -89.5, 14.5],
    ...     starting_date="2018-01-01",
    ...     ending_date="2018-12-31",
    ...     output_folder="data/raw/chirps",
    ... )

    The same request against the Hugging Face datacube:

    >>> dl = CHIRPSDownloader(backend="hf")
    >>> paths = dl.download(
    ...     extent=[-90.5, 13.0, -89.5, 14.5],
    ...     starting_date="2018-01-01",
    ...     ending_date="2018-12-31",
    ...     output_folder="data/raw/chirps",
    ...     ncores=4,
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

    URL_HF: str = HF_CHIRPS_URL

    #: Precipitation variable name inside the Hugging Face Zarr store.
    HF_VARIABLE: str = "precip"

    def __init__(
        self,
        frequency: str = "daily",
        sp_resolution: str = "05",
        version: str = "3.0",
        backend: Literal["ucsb", "hf"] = "ucsb",
        hf_url: str | None = None,
    ) -> None:
        if backend not in {"ucsb", "hf"}:
            raise ValueError(f"backend must be 'ucsb' or 'hf', got '{backend}'")
        if backend == "hf" and frequency != "daily":
            raise ValueError(
                "The Hugging Face CHIRPS store is daily only; "
                f"use backend='ucsb' for frequency='{frequency}'."
            )
        self.frequency = frequency
        self.resolution = sp_resolution
        self.version = version
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
            With ``backend="ucsb"``: maximum concurrent day-downloads, hard-capped
            at 1 internally to avoid HTTP 403 from the UCSB server.
            With ``backend="hf"``: number of time-batches fetched concurrently
            (no cap — the Hugging Face CDN handles parallel range reads).
            Default: 1.
        polite_delay : float
            Per-worker sleep in seconds after each completed request.  Default: 0.5.
            Ignored by the Hugging Face backend.

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

        if self.backend == "hf":
            self._download_from_hf(extent, starting_date, ending_date, output_folder, ncores)
            return year_folders

        workers = min(ncores, 1)  # UCSB now limits to 1 concurrent connection
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

    # ------------------------------------------------------------------
    # Hugging Face Zarr backend
    # ------------------------------------------------------------------

    @staticmethod
    def _output_path(output_folder: str, date_str: str) -> str:
        """Return the per-day NetCDF path for ``date_str`` (``"YYYYMMDD"``)."""
        return os.path.join(
            output_folder, date_str[:4], f"chirps_precipitation_{date_str}.nc"
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
        download_daily_from_hf(
            url=self.hf_url,
            variables={self.HF_VARIABLE: "pr"},
            extent=extent,
            starting_date=starting_date,
            ending_date=ending_date,
            path_for=lambda _var, date_str: self._output_path(output_folder, date_str),
            ncores=ncores,
            desc="CHIRPS (HF)",
        )

    # ------------------------------------------------------------------
    # UCSB COG/TIF backend
    # ------------------------------------------------------------------

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
        out_nc = self._output_path(output_folder, f"{year}{month}{day}")
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
