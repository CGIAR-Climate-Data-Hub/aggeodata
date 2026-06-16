"""
aggeodata.ingestion.agera5
===========================

AgERA5 agrometeorological indicator downloader via the Copernicus CDS API.

Rule: this module ONLY downloads files to disk.
"""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray
from tqdm import tqdm

from .files_manager import (
    create_yearly_query,
    days_range_asstring,
    find_date_instring,
    months_range_asstring,
    split_date,
    uncompress_zip_path,
)
from .cf_names import AGERA5_CF_MAP
from .gis_functions import read_raster_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variable map — config key → CDS API request fragment
# ---------------------------------------------------------------------------

AGERA5_VARIABLE_MAP: dict[str, dict] = {
    "precipitation": {
        "variable": "precipitation_flux"
    },
    "wind_speed": {
        "variable": "10m_wind_speed",
        "statistic": ["24_hour_mean"],
    },
    "vapour_pressure": {
        "variable": "vapour_pressure",
        "statistic": ["24_hour_mean"],
    },
    "vapour_pressure_defficit": {
        "variable": "vapour_pressure_deficit_at_maximum_temperature",
        "statistic": ["24_hour_mean"],
    },
    "relative_humidity_max": {
        "variable": "2m_relative_humidity_derived",
        "statistic": ["24_hour_maximum"],
    },
    "relative_humidity_min": {
        "variable": "2m_relative_humidity_derived",
        "statistic": ["24_hour_minimum"],
    },
    "relative_humidity_06": {"variable": "2m_relative_humidity", "time": ["06_00"]},
    "relative_humidity_09": {"variable": "2m_relative_humidity", "time": ["09_00"]},
    "relative_humidity_12": {"variable": "2m_relative_humidity", "time": ["12_00"]},
    "relative_humidity_15": {"variable": "2m_relative_humidity", "time": ["15_00"]},
    "relative_humidity_18": {"variable": "2m_relative_humidity", "time": ["18_00"]},
    "reference_evapotranspiration": {"variable": "reference_evapotranspiration"},
    "solar_radiation": {"variable": "solar_radiation_flux"},
    "dew_point_temperature": {
        "variable": "2m_dewpoint_temperature",
        "statistic": ["24_hour_mean"],
    },
    "temperature_tmax": {
        "variable": "2m_temperature",
        "statistic": ["24_hour_maximum"],
    },
    "temperature_tmin": {
        "variable": "2m_temperature",
        "statistic": ["24_hour_minimum"],
    },
}

AGERA5_SHORT_NAMES: dict[str, str] = {
    "precipitation": "precipitation",
    "solar_radiation": "srad",
    "temperature_tmax": "tmax",
    "temperature_tmin": "tmin",
    "vapour_pressure": "vp",
    "vapour_pressure_defficit": "vpd",
    "dew_point_temperature": "dpt",
    "wind_speed": "ws",
    "reference_evapotranspiration": "etr",
    "relative_humidity_06": "rh06",
    "relative_humidity_09": "rh09",
    "relative_humidity_12": "rh12",
    "relative_humidity_15": "rh15",
    "relative_humidity_18": "rh18",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _transform_dates_for_query(
    year: int,
    init_day: int | None = None,
    end_day: int | None = None,
    init_month: int | None = None,
    end_month: int | None = None,
) -> dict[str, list[str]]:
    return {
        "year": [str(year)],
        "month": months_range_asstring(init_month or 1, end_month or 12),
        "day": days_range_asstring(init_day or 1, end_day or 31),
    }


def _process_file(
    year_path_folder: str,
    filename: str,
    date: str,
    xdim_name: str,
    ydim_name: str,
    depthdim_name: str,
    cf_name: str | None = None,
) -> xarray.Dataset:
    """Read a single AgERA5 NetCDF file and assign a time coordinate."""
    dateasdatetime = datetime.strptime(date, "%Y%m%d")
    filepath = os.path.join(year_path_folder, filename)
    xrdata = read_raster_data(filepath, ydim_name=ydim_name, xdim_name=xdim_name)
    varname = list(xrdata.data_vars.keys())[0]
    two_var = xrdata[varname].values[0] if xrdata[varname].values.ndim == 3 else xrdata[varname].values
    out_varname = cf_name or varname
    xrdata = xarray.Dataset(
        data_vars={out_varname: ([ydim_name, xdim_name], two_var)},
        coords={xdim_name: xrdata[xdim_name].values, ydim_name: xrdata[ydim_name].values},
    )
    xrdata = xrdata.expand_dims(dim={depthdim_name: 1}, axis=0)
    xrdata[depthdim_name] = [dateasdatetime]
    return xrdata


def read_annual_data(
    path: str,
    year: str,
    xdim_name: str = "longitude",
    ydim_name: str = "latitude",
    depthdim_name: str = "time",
    crs: str = "EPSG:4326",
    cf_name: str | None = None,
) -> xarray.Dataset:
    """Read all per-day NetCDF files for one year and concatenate along time."""
    import rioxarray  # noqa: F401

    year_path_folder = uncompress_zip_path(path, year)
    times = [
        [fn, find_date_instring(fn, pattern=year)]
        for fn in os.listdir(year_path_folder)
        if fn.endswith(".nc")
    ]
    list_xrdata = [
        _process_file(os.path.join(path, year), fn, date, ydim_name, xdim_name, depthdim_name, cf_name)
        for fn, date in times
    ]
    annual_data = xarray.concat(list_xrdata, dim=depthdim_name)

    tmp = list_xrdata[0].copy().rio.write_crs(crs)
    spatial_ref = tmp.rio.write_transform(tmp.rio.transform()).spatial_ref
    annual_data = annual_data.assign(crs=spatial_ref)

    if "spatial_ref" in list(annual_data.coords.keys()):
        return annual_data.drop_vars("spatial_ref")
    return annual_data


# ---------------------------------------------------------------------------
# AgEra5Downloader
# ---------------------------------------------------------------------------

class AgEra5Downloader:
    """Download AgERA5 agrometeorological data from the Copernicus CDS API.

    Parameters
    ----------
    product : str
        CDS dataset identifier. Default: ``"sis-agrometeorological-indicators"``.
    version : str
        AgERA5 product version — ``"2_0"`` (default) or ``"1_1"``.
    max_attempts : int
        Maximum retry attempts per year.  Default: 3.

    Examples
    --------
    >>> dl = AgEra5Downloader()
    >>> paths = dl.download(
    ...     variable="2m_temperature",
    ...     statistic=["24_hour_maximum"],
    ...     starting_date="2010-01-01",
    ...     ending_date="2010-12-31",
    ...     output_folder="data/raw/tmax",
    ...     aoi_extent=[-90.5, 13.0, -88.5, 15.5],
    ... )
    """

    PRODUCT: str = "sis-agrometeorological-indicators"

    def __init__(
        self,
        product: str = "sis-agrometeorological-indicators",
        version: str = "2_0",
        max_attempts: int = 3,
    ) -> None:
        self.product = product
        self.version = version
        self.max_attempts = max_attempts

    def download(
        self,
        variable: str | list[str],
        starting_date: str,
        ending_date: str,
        output_folder: str,
        aoi_extent: list[float],
        statistic: list[str] | None = None,
        time: list[str] | None = None,
        ncores: int = 4,
    ) -> dict[str, str]:
        """Download one AgERA5 variable for a date range.

        Parameters
        ----------
        variable : str | list[str]
            CDS variable name(s), e.g. ``"2m_temperature"``.
        starting_date : str
            ISO 8601 start ``"YYYY-MM-DD"``.
        ending_date : str
            ISO 8601 end ``"YYYY-MM-DD"``.
        output_folder : str
            Root directory; one ``.zip`` per year is written here.
        aoi_extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        statistic : list[str] | None
            Statistic filter, e.g. ``["24_hour_maximum"]``.
        time : list[str] | None
            Hour filter for relative-humidity (e.g. ``["12_00"]``).
        ncores : int
            Parallel threads.  Default: 4.

        Returns
        -------
        dict[str, str]
            ``{year_str: zip_file_path}``
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        init_year, init_month, init_day = split_date(starting_date)
        end_year, end_month, end_day = split_date(ending_date)
        years = list(range(init_year, end_year + 1))

        # CDS expects [N, W, S, E]
        cds_area = [aoi_extent[3], aoi_extent[0], aoi_extent[1], aoi_extent[2]]
        base_query: dict[str, Any] = {
            "version": self.version,
            "area": cds_area,
            "variable": variable if isinstance(variable, list) else [variable],
        }
        if statistic:
            base_query["statistic"] = statistic
        if time:
            base_query["time"] = time

        if ncores > 0:
            return self._parallel_download(
                years, base_query, output_folder,
                init_year, end_year, init_month, end_month, init_day, end_day, ncores,
            )
        return self._sequential_download(
            years, base_query, output_folder,
            init_year, end_year, init_month, end_month, init_day, end_day,
        )

    @staticmethod
    def stack_annual_to_netcdf(
        raw_folder: str,
        init_year: int,
        end_year: int,
        output_folder: str | None = None,
        remove_source: bool = True,
        config_key: str | None = None,
    ) -> None:
        """Stack per-day NetCDF files for each year into a single annual NetCDF.

        Parameters
        ----------
        config_key : str | None
            Key from ``AGERA5_VARIABLE_MAP`` (e.g. ``"solar_radiation"``) used to
            resolve the CF output variable name.  If omitted the native CDS name
            is kept as-is.
        """
        cf_name = AGERA5_CF_MAP.get(config_key) if config_key else None
        out = output_folder or raw_folder
        for year in range(init_year, end_year + 1):
            try:
                ds = read_annual_data(raw_folder, str(year), cf_name=cf_name)
                nc_path = os.path.join(out, f"{year}.nc")
                ds.to_netcdf(nc_path)
                logger.info("Stacked year %d -> %s", year, nc_path)
                if remove_source:
                    yr_dir = os.path.join(raw_folder, str(year))
                    yr_zip = yr_dir + ".zip"
                    if os.path.isdir(yr_dir):
                        shutil.rmtree(yr_dir)
                    if os.path.isfile(yr_zip):
                        os.remove(yr_zip)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not stack year %d: %s", year, exc)

    def _download_one_year(
        self,
        year: int,
        base_query: dict[str, Any],
        output_folder: str,
        init_year: int,
        end_year: int,
        init_month: int,
        end_month: int,
        init_day: int,
        end_day: int,
    ) -> str | None:
        try:
            import cdsapi
        except ImportError as exc:
            raise ImportError("cdsapi is required: pip install cdsapi") from exc

        try:
            if year == init_year:
                dates = _transform_dates_for_query(year, init_day=init_day, end_day=31, init_month=init_month, end_month=12)
            elif year == end_year:
                dates = _transform_dates_for_query(year, init_day=1, end_day=end_day, init_month=1, end_month=end_month)
            else:
                dates = _transform_dates_for_query(year)

            query = copy.deepcopy(base_query)
            query.update(dates)
            zip_path = os.path.join(output_folder, f"{year}.zip")
            year_dir = os.path.join(output_folder, str(year))

            if os.path.exists(zip_path) or os.path.isdir(year_dir):
                logger.info("AgERA5 year=%d already exists — skipping", year)
                return zip_path if os.path.exists(zip_path) else year_dir

            client = cdsapi.Client()
            client.retrieve(self.product, query, zip_path)
            logger.info("AgERA5 downloaded year=%d -> %s", year, zip_path)
            return zip_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("AgERA5 year %d failed: %s", year, exc)
            return None

    def _sequential_download(
        self, years, base_query, output_folder,
        init_year, end_year, init_month, end_month, init_day, end_day,
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        for year in years:
            for attempt in range(1, self.max_attempts + 1):
                path = self._download_one_year(
                    year, base_query, output_folder,
                    init_year, end_year, init_month, end_month, init_day, end_day,
                )
                if path:
                    results[str(year)] = path
                    break
        return results

    def _parallel_download(
        self, years, base_query, output_folder,
        init_year, end_year, init_month, end_month, init_day, end_day, ncores,
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        tasks_to_retry = {y: 1 for y in years}
        while tasks_to_retry:
            this_round = tasks_to_retry.copy()
            tasks_to_retry.clear()
            with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as pool:
                future_map = {
                    pool.submit(
                        self._download_one_year, y, base_query, output_folder,
                        init_year, end_year, init_month, end_month, init_day, end_day,
                    ): y
                    for y in this_round
                }
                for future in concurrent.futures.as_completed(future_map):
                    year = future_map[future]
                    attempt = this_round[year]
                    try:
                        path = future.result()
                        if path:
                            results[str(year)] = path
                        else:
                            raise RuntimeError("download returned None")
                    except Exception as exc:  # noqa: BLE001
                        if attempt < self.max_attempts:
                            tasks_to_retry[year] = attempt + 1
        return results
