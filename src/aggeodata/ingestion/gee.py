"""
aggeodata.ingestion.gee
========================

Google Earth Engine ImageCollection downloader.

Three CHC/AgERA5 collections are pre-configured; additional collections can be
supported by appending an entry to ``_DATASET_CONFIGS``.

Pre-configured collections
--------------------------
* ``"UCSB-CHG/CHIRPS/DAILY"``   — CHIRPS precipitation (band ``precipitation``)
* ``"UCSB-CHG/CHIRTS/DAILY"``   — CHIRTS Tmax / Tmin (bands ``Tmax``, ``Tmin``)
* ``"projects/climate-engine-pro/assets/ce-ag-era5-v2/daily"``
  — Climate Engine AgERA5 v2 (multiple bands; temperature bands in Kelvin)

Rule: this module ONLY downloads files to disk.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import rasterio

from .files_manager import create_yearly_query
from .gis_functions import numpy_to_xarray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset catalogue
# ---------------------------------------------------------------------------

_DATASET_CONFIGS: dict[str, dict] = {
    "UCSB-CHG/CHIRPS/DAILY": {
        "cf_bands": {"pr": "precipitation"},
        "scale": 5566,       # ~0.05° at equator in metres
        "kelvin_vars": frozenset(),
    },
    "UCSB-CHG/CHIRTS/DAILY": {
        "cf_bands": {"tasmax": "Tmax", "tasmin": "Tmin"},
        "scale": 5566,
        "kelvin_vars": frozenset(),
        # GEE public asset covers 1983-01-01 → 2016-12-31 only.
        # For dates after 2016 use projects/climate-engine-pro/assets/ce-ag-era5-v2/daily.
    },
    "projects/climate-engine-pro/assets/ce-ag-era5-v2/daily": {
        "cf_bands": {
            "pr":     "Precipitation_Flux",
            "tasmax": "Temperature_Air_2m_Max_24h",
            "tasmin": "Temperature_Air_2m_Min_24h",
            "tas":    "Temperature_Air_2m_Mean_24h",
            "tdps":   "Dew_Point_Temperature_2m_Mean_24h",
            "rsds":   "Solar_Radiation_Flux",
            "vp":     "Vapour_Pressure_Mean_24h",
            "etr":    "ReferenceET_PenmanMonteith_FAO56",
        },
        "scale": 11132,      # ~0.1° at equator in metres
        "kelvin_vars": frozenset({"tasmax", "tasmin", "tas", "tdps"}),
    },
}


class GEEDownloader:
    """Download daily data from a Google Earth Engine ImageCollection.

    Parameters
    ----------
    dataset_id : str
        GEE ImageCollection ID.  Pre-configured options:

        * ``"UCSB-CHG/CHIRPS/DAILY"``
        * ``"UCSB-CHG/CHIRTS/DAILY"``
        * ``"projects/climate-engine-pro/assets/ce-ag-era5-v2/daily"``
    variables : list[str] | None
        CF variable names to download.  Must be keys in the dataset's
        ``cf_bands`` mapping.  Defaults to all available variables.
    project : str | None
        GEE cloud project for ``ee.Initialize(project=...)``.
        Required for accounts that use the new project-based API.

    Examples
    --------
    >>> dl = GEEDownloader("UCSB-CHG/CHIRPS/DAILY")
    >>> paths = dl.download(
    ...     extent=[-90.5, 13.0, -89.5, 14.5],
    ...     starting_date="2020-01-01",
    ...     ending_date="2020-12-31",
    ...     output_folder="data/raw/gee",
    ... )
    """

    def __init__(
        self,
        dataset_id: str,
        variables: list[str] | None = None,
        project: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.project = project

        cfg = _DATASET_CONFIGS.get(dataset_id)
        if cfg is None:
            raise ValueError(
                f"Unknown dataset_id '{dataset_id}'. "
                f"Known datasets: {sorted(_DATASET_CONFIGS)}. "
                "To add a custom dataset, append an entry to "
                "aggeodata.ingestion.gee._DATASET_CONFIGS."
            )
        self._cfg = cfg

        available = set(cfg["cf_bands"])
        requested = set(variables or available)
        unknown = requested - available
        if unknown:
            raise ValueError(
                f"Variables {sorted(unknown)} are not available in '{dataset_id}'. "
                f"Available CF variables: {sorted(available)}"
            )
        self.variables: list[str] = sorted(requested)
        self._ee = self._init_ee()

    # ------------------------------------------------------------------

    def _init_ee(self):
        try:
            import ee
        except ImportError as exc:
            raise ImportError(
                "earthengine-api is required for GEEDownloader. "
                "Install with: pip install earthengine-api"
            ) from exc
        try:
            if self.project:
                ee.Initialize(project=self.project)
            else:
                ee.Initialize()
        except Exception:
            ee.Authenticate()
            if self.project:
                ee.Initialize(project=self.project)
            else:
                ee.Initialize()
        return ee

    # ------------------------------------------------------------------

    def download(
        self,
        extent: list[float],
        starting_date: str,
        ending_date: str,
        output_folder: str,
        ncores: int = 1,
        polite_delay: float = 0.0,
    ) -> dict[str, dict[str, str]]:
        """Download GEE data for the given extent and date range.

        Parameters
        ----------
        extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        starting_date : str
            ISO 8601 start date ``"YYYY-MM-DD"``.
        ending_date : str
            ISO 8601 end date ``"YYYY-MM-DD"``.
        output_folder : str
            Root folder; ``{output_folder}/{variable}/{year}/`` sub-folders
            are created automatically.
        ncores : int
            Concurrent day-downloads.  Default: 1.
        polite_delay : float
            Sleep in seconds between day requests.  Default: 0.0.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{cf_variable: {year: year_folder_path}}``
        """
        from tqdm import tqdm

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        yearly_dates = create_yearly_query(starting_date, ending_date)

        year_folders: dict[str, dict[str, str]] = {v: {} for v in self.variables}
        jobs: list[tuple[str, str, str]] = []

        for var in self.variables:
            for year, monthly_dates in yearly_dates.items():
                year_folder = os.path.join(output_folder, var, year)
                Path(year_folder).mkdir(parents=True, exist_ok=True)
                year_folders[var][year] = year_folder
                for month, days in monthly_dates.items():
                    for day in days:
                        jobs.append((year, month, day))

        # Deduplicate: all variables share the same image per day
        unique_days = list(dict.fromkeys(jobs))

        pbar = tqdm(total=len(unique_days), desc="GEE", unit="day")

        def _run(job: tuple[str, str, str]) -> None:
            year, month, day = job
            try:
                self._download_one_day(year, month, day, output_folder, extent)
            except Exception as exc:
                logger.warning("GEE failed %s-%s-%s: %s", year, month, day, exc)
            finally:
                pbar.update(1)
            if polite_delay > 0:
                time.sleep(polite_delay)

        if ncores > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as pool:
                list(pool.map(_run, unique_days))
        else:
            for job in unique_days:
                _run(job)

        pbar.close()
        return year_folders

    # ------------------------------------------------------------------

    def _download_one_day(
        self,
        year: str,
        month: str,
        day: str,
        output_folder: str,
        extent: list[float],
    ) -> None:
        ee = self._ee
        date_str = f"{year}-{month}-{day}"
        next_str = (
            datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        xmin, ymin, xmax, ymax = extent
        region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

        ic = ee.ImageCollection(self.dataset_id).filterDate(date_str, next_str)
        if ic.size().getInfo() == 0:
            logger.warning(
                "No image in '%s' for %s — skipping. "
                "Check the collection's temporal coverage "
                "(UCSB-CHG/CHIRTS/DAILY ends 2016-12-31; "
                "use projects/climate-engine-pro/assets/ce-ag-era5-v2/daily for later dates).",
                self.dataset_id, date_str,
            )
            return
        img = ic.first()

        scale = self._cfg["scale"]
        kelvin_vars: frozenset[str] = self._cfg["kelvin_vars"]
        cf_bands: dict[str, str] = self._cfg["cf_bands"]

        for cf_var in self.variables:
            out_nc = os.path.join(
                output_folder, cf_var, year,
                f"gee_{cf_var}_{year}{month}{day}.nc",
            )
            if os.path.exists(out_nc):
                continue

            url = img.select(cf_bands[cf_var]).getDownloadURL({
                "format": "GEO_TIFF",
                "region": region,
                "scale": scale,
                "crs": "EPSG:4326",
            })

            resp = requests.get(url, timeout=300)
            resp.raise_for_status()

            with rasterio.open(BytesIO(resp.content)) as src:
                data = src.read().astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                transform = src.transform
                crs = str(src.crs)

            # Mask GEE fill values before any unit conversion
            data[data < -9000] = np.nan

            if cf_var in kelvin_vars:
                data -= 273.15

            xrm = numpy_to_xarray(data, transform, crs=crs, var_name=cf_var)
            xrm.to_netcdf(out_nc)
            logger.debug("GEE saved: %s", out_nc)
