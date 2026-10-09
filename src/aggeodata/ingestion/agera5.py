"""
aggeodata.ingestion.agera5
===========================

AgERA5 agrometeorological indicator downloader.

Two backends are supported via the *backend* parameter:

* ``"cds"`` (default) — the Copernicus CDS API.  One queued request per year,
  returning a zip of per-day NetCDFs.
* ``"zarr"`` — the ECMWF ARCO datastore, where AgERA5 is published as a
  consolidated Zarr datacube.  A bounding box plus a date range is a single
  indexed read, with no request queue, which is far faster for long series.

Both backends authenticate with the same ``~/.cdsapirc`` key and write the same
per-day NetCDF layout, so downstream stages do not care which one was used.

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
from typing import Any, Literal

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
from .zarr_backend import (
    AGERA5_ARCO_GEO_URL,
    AGERA5_ARCO_TIME_URL,
    cds_storage_options,
    download_daily_from_zarr,
)

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

# ---------------------------------------------------------------------------
# Variable map — config key → variable name in the ECMWF ARCO Zarr store
#
# The ARCO store publishes 28 AgERA5 variables; only the ones aggeodata already
# exposes through AGERA5_VARIABLE_MAP are mapped here.  The remaining ARCO
# variables (cloud cover, snow thickness, precipitation duration fractions,
# day/night-time temperature means, ...) are intentionally left out — add an
# entry here *and* to AGERA5_VARIABLE_MAP to support one.
# ---------------------------------------------------------------------------

AGERA5_ARCO_VARIABLES: dict[str, str] = {
    "precipitation":                "Precipitation_Flux",
    "solar_radiation":              "Solar_Radiation_Flux",
    "temperature_tmax":             "Temperature_Air_2m_Max_24h",
    "temperature_tmin":             "Temperature_Air_2m_Min_24h",
    "wind_speed":                   "Wind_Speed_10m_Mean_24h",
    "dew_point_temperature":        "Dew_Point_Temperature_2m_Mean_24h",
    "vapour_pressure":              "Vapour_Pressure_Mean_24h",
    "vapour_pressure_defficit":     "Vapour_Pressure_Deficit_at_Maximum_Temperature",
    "reference_evapotranspiration": "ReferenceET_PenmanMonteith_FAO56",
    "relative_humidity_max":        "Derived_Relative_Humidity_2m_Max_24h",
    "relative_humidity_min":        "Derived_Relative_Humidity_2m_Min_24h",
    "relative_humidity_06":         "Relative_Humidity_2m_06h",
    "relative_humidity_09":         "Relative_Humidity_2m_09h",
    "relative_humidity_12":         "Relative_Humidity_2m_12h",
    "relative_humidity_15":         "Relative_Humidity_2m_15h",
    "relative_humidity_18":         "Relative_Humidity_2m_18h",
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
    """Download AgERA5 agrometeorological data from the CDS API or ECMWF ARCO.

    Parameters
    ----------
    product : str
        CDS dataset identifier. Default: ``"sis-agrometeorological-indicators"``.
    version : str
        AgERA5 product version — ``"2_0"`` (default) or ``"1_1"``.  Only used by
        the CDS backend; the ARCO store publishes a single current version.
    max_attempts : int
        Maximum retry attempts per year.  Default: 3.  CDS backend only.
    backend : str
        ``"cds"`` *(default)* — queued CDS API request, one zip per year.
        ``"zarr"`` — slice the ECMWF ARCO Zarr datacube.
    chunking : str
        Which ARCO store to read when ``backend="zarr"``.  ``"geo"`` *(default)*
        is chunked ``(2112 time, 16 lat, 32 lon)`` and suits many days over a
        modest AOI; ``"time"`` is chunked ``(1, 1024, 1024)`` and suits a few
        days over a very large AOI.
    zarr_url : str | None
        Override the ARCO store URL — only used when ``backend="zarr"``.

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

    The same request against the ARCO datacube — pass the aggeodata config key
    rather than the CDS variable/statistic pair:

    >>> dl = AgEra5Downloader(backend="zarr")
    >>> paths = dl.download(
    ...     variable="2m_temperature",
    ...     statistic=["24_hour_maximum"],
    ...     starting_date="2010-01-01",
    ...     ending_date="2010-12-31",
    ...     output_folder="data/raw/tmax",
    ...     aoi_extent=[-90.5, 13.0, -88.5, 15.5],
    ...     config_key="temperature_tmax",
    ...     ncores=4,
    ... )
    """

    PRODUCT: str = "sis-agrometeorological-indicators"

    def __init__(
        self,
        product: str = "sis-agrometeorological-indicators",
        version: str = "2_0",
        max_attempts: int = 3,
        backend: Literal["cds", "zarr"] = "cds",
        chunking: Literal["geo", "time"] = "geo",
        zarr_url: str | None = None,
    ) -> None:
        if backend not in {"cds", "zarr"}:
            raise ValueError(f"backend must be 'cds' or 'zarr', got '{backend}'")
        if chunking not in {"geo", "time"}:
            raise ValueError(f"chunking must be 'geo' or 'time', got '{chunking}'")
        self.product = product
        self.version = version
        self.max_attempts = max_attempts
        self.backend = backend
        self.chunking = chunking
        self.zarr_url = zarr_url or (
            AGERA5_ARCO_GEO_URL if chunking == "geo" else AGERA5_ARCO_TIME_URL
        )

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
        config_key: str | None = None,
    ) -> dict[str, str]:
        """Download one AgERA5 variable for a date range.

        Parameters
        ----------
        variable : str | list[str]
            CDS variable name(s), e.g. ``"2m_temperature"``.  Used by the CDS
            backend; the Zarr backend resolves its variable from *config_key*
            (falling back to matching *variable* + *statistic* + *time*
            against :data:`AGERA5_VARIABLE_MAP`).
        starting_date : str
            ISO 8601 start ``"YYYY-MM-DD"``.
        ending_date : str
            ISO 8601 end ``"YYYY-MM-DD"``.
        output_folder : str
            Root directory.  CDS backend: one ``.zip`` per year.  Zarr backend:
            ``<year>/agera5_<short_name>_YYYYMMDD.nc`` per day.
        aoi_extent : list[float]
            ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
        statistic : list[str] | None
            Statistic filter, e.g. ``["24_hour_maximum"]``.
        time : list[str] | None
            Hour filter for relative-humidity (e.g. ``["12_00"]``).
        ncores : int
            Parallel threads (CDS: concurrent years; Zarr: concurrent
            time-batches).  Default: 4.
        config_key : str | None
            Key from :data:`AGERA5_VARIABLE_MAP` (e.g. ``"temperature_tmax"``).
            Lets the Zarr backend pick the right ARCO variable without
            re-deriving it from the CDS request fragment.

        Returns
        -------
        dict[str, str]
            ``{year_str: zip_file_path}`` (CDS) or ``{year_str: year_folder}``
            (Zarr).
        """
        if self.backend == "zarr":
            return self._download_from_zarr(
                variable=variable,
                statistic=statistic,
                time=time,
                starting_date=starting_date,
                ending_date=ending_date,
                output_folder=output_folder,
                aoi_extent=aoi_extent,
                ncores=ncores,
                config_key=config_key,
            )

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

    # ------------------------------------------------------------------
    # ECMWF ARCO Zarr backend
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config_key(
        variable: str | list[str],
        statistic: list[str] | None,
        time: list[str] | None,
        config_key: str | None,
    ) -> str:
        """Return the AGERA5_VARIABLE_MAP key for this request.

        Prefers an explicit *config_key*; otherwise matches the CDS request
        fragment (variable + statistic + time) back against the map, which is
        what distinguishes e.g. tmax from tmin — both are ``2m_temperature``.
        """
        if config_key:
            if config_key not in AGERA5_VARIABLE_MAP:
                raise ValueError(
                    f"Unknown AgERA5 config_key '{config_key}'. "
                    f"Valid keys: {sorted(AGERA5_VARIABLE_MAP)}"
                )
            return config_key

        var = variable[0] if isinstance(variable, list) else variable
        for key, spec in AGERA5_VARIABLE_MAP.items():
            if (
                spec["variable"] == var
                and spec.get("statistic") == statistic
                and spec.get("time") == time
            ):
                return key
        raise ValueError(
            f"Cannot map AgERA5 request (variable={var!r}, statistic={statistic!r}, "
            f"time={time!r}) to a known variable. Pass config_key explicitly — "
            f"valid keys: {sorted(AGERA5_VARIABLE_MAP)}"
        )

    @staticmethod
    def _output_path(output_folder: str, short_name: str, date_str: str) -> str:
        """Return the per-day NetCDF path for ``date_str`` (``"YYYYMMDD"``)."""
        return os.path.join(
            output_folder, date_str[:4], f"agera5_{short_name}_{date_str}.nc"
        )

    def _download_from_zarr(
        self,
        variable: str | list[str],
        statistic: list[str] | None,
        time: list[str] | None,
        starting_date: str,
        ending_date: str,
        output_folder: str,
        aoi_extent: list[float],
        ncores: int,
        config_key: str | None,
    ) -> dict[str, str]:
        """Slice the ECMWF ARCO store into per-day NetCDF files."""
        key = self._resolve_config_key(variable, statistic, time, config_key)
        store_var = AGERA5_ARCO_VARIABLES.get(key)
        if store_var is None:
            raise ValueError(
                f"AgERA5 variable '{key}' is not exposed by the ARCO Zarr backend. "
                f"Available: {sorted(AGERA5_ARCO_VARIABLES)}. "
                f"Use backend='cds' for this variable."
            )

        short_name = AGERA5_SHORT_NAMES.get(key, key)
        cf_name = AGERA5_CF_MAP.get(key, short_name)

        # Create the year folders up front so the return value matches the CDS
        # backend's shape even for years the store has no data for.
        year_folders: dict[str, str] = {}
        for year in create_yearly_query(starting_date, ending_date):
            year_folder = os.path.join(output_folder, year)
            Path(year_folder).mkdir(parents=True, exist_ok=True)
            year_folders[year] = year_folder

        download_daily_from_zarr(
            url=self.zarr_url,
            variables={store_var: cf_name},
            extent=aoi_extent,
            starting_date=starting_date,
            ending_date=ending_date,
            path_for=lambda _var, date_str: self._output_path(
                output_folder, short_name, date_str
            ),
            ncores=ncores,
            desc=f"AgERA5 {short_name} (ARCO)",
            storage_options=cds_storage_options(),
            lat_name="latitude",
            lon_name="longitude",
        )
        return year_folders

    # ------------------------------------------------------------------
    # CDS API backend
    # ------------------------------------------------------------------

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
