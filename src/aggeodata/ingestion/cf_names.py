"""CF/xclim variable name mappings for all ingestion sources."""

from __future__ import annotations

import xarray as xr

CF_VARIABLE_MAP: dict[str, str] = {
    # CHIRPS
    "precipitation": "pr",
    # CHIRTS / CHIRTS-ERA5
    "tmax": "tasmax",
    "tmin": "tasmin",
    # NASA POWER (REST API + S3 Zarr)
    "T2M_MAX": "tasmax",
    "T2M_MIN": "tasmin",
    "T2M": "tas",
    "RH2M": "hurs",
    "WS2M": "sfcWind",
    "ALLSKY_SFC_SW_DWN": "rsds",
    "PRECTOTCORR": "pr",
    "PRECTOT": "pr",
    # AgERA5 short names (see AGERA5_SHORT_NAMES in agera5.py)
    "srad": "rsds",
    "ws": "sfcWind",
    "dpt": "tdps",
    "rh06": "rh06",
    "rh09": "rh09",
    "rh12": "rh12",
    "rh15": "rh15",
    "rh18": "rh18",
    "etr": "etr"
}

# AgERA5 config key (from AGERA5_VARIABLE_MAP) -> CF name
AGERA5_CF_MAP: dict[str, str] = {
    "solar_radiation": "rsds",
    "temperature_tmax": "tasmax",
    "temperature_tmin": "tasmin",
    "wind_speed": "sfcWind",
    "dew_point_temperature": "tdps",
    "precipitation": "pr",
}


def to_cf_name(native: str) -> str:
    """Return the CF name for *native*, or *native* unchanged if not mapped."""
    return CF_VARIABLE_MAP.get(native, native)


def rename_cf_vars(ds: xr.Dataset) -> xr.Dataset:
    """Rename all data variables in *ds* to their CF equivalents where known."""
    rename = {v: CF_VARIABLE_MAP[v] for v in ds.data_vars if v in CF_VARIABLE_MAP}
    return ds.rename(rename) if rename else ds
