"""aggeodata MCP server — exposes download tools to Claude agents."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("aggeodata")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _resolve_extent(
    country_code: str,
    feature_name: str,
    adm_level: int,
    bbox: list[float] | None,
) -> list[float]:
    """Return [xmin, ymin, xmax, ymax] from bbox or boundary lookup."""
    if bbox:
        return bbox
    if not country_code:
        raise ValueError("Provide either bbox or country_code.")

    from aggeodata.ingestion.boundaries import get_admin_boundary, _fetch_geojson_cached

    if feature_name:
        gdf = get_admin_boundary(country_code, feature_name, adm_level)
    else:
        try:
            gdf = _fetch_geojson_cached(country_code, 0)
        except Exception:
            gdf = _fetch_geojson_cached(country_code, 1)

    xmin, ymin, xmax, ymax = gdf.total_bounds.tolist()
    return [xmin, ymin, xmax, ymax]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_admin_units(
    country_code: str,
    adm_level: int = 1,
) -> list[str]:
    """Return sorted admin-unit names for a country.

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-3 code, e.g. ``"GHA"``, ``"HND"``.
    adm_level : int
        1 = region/province (default), 2 = district/department.
    """
    from aggeodata.ingestion.boundaries import list_admin_units as _fn
    return _fn(country_code, adm_level)


@mcp.tool()
def download_chirps(
    starting_date: str,
    ending_date: str,
    output_folder: str,
    country_code: str = "",
    feature_name: str = "",
    adm_level: int = 1,
    bbox: list[float] | None = None,
    ncores: int = 2,
) -> dict[str, str]:
    """Download CHIRPS v3 daily precipitation.

    Provide either *bbox* ``[xmin, ymin, xmax, ymax]`` (EPSG:4326) or
    *country_code* with an optional *feature_name* for a sub-national region.
    Workers are capped at 2 to respect UCSB rate limits.

    Returns
    -------
    dict[str, str]
        Mapping ``{date_string: local_path}`` for every downloaded file.
    """
    extent = _resolve_extent(country_code, feature_name, adm_level, bbox)
    ncores = min(ncores, 1)  # UCSB now limits to 1 concurrent connection
    from aggeodata.ingestion.chirps import CHIRPSDownloader
    dl = CHIRPSDownloader()
    return dl.download(
        extent=extent,
        starting_date=starting_date,
        ending_date=ending_date,
        output_folder=output_folder,
        ncores=ncores,
    )


@mcp.tool()
def download_chirts(
    starting_date: str,
    ending_date: str,
    output_folder: str,
    country_code: str = "",
    feature_name: str = "",
    adm_level: int = 1,
    bbox: list[float] | None = None,
    variables: list[str] | None = None,
    chirts_source: str = "era5",
    ncores: int = 2,
) -> dict[str, dict[str, str]]:
    """Download CHIRTS-ERA5 daily Tmax / Tmin.

    Parameters
    ----------
    variables : list[str] | None
        ``["tmax"]``, ``["tmin"]``, or ``["tmax", "tmin"]`` (default = both).
    chirts_source : str
        ``"era5"`` (default, 1983–present) or ``"chirts"`` (1983–2016 only).
    """
    extent = _resolve_extent(country_code, feature_name, adm_level, bbox)
    ncores = min(ncores, 1)  # UCSB now limits to 1 concurrent connection
    from aggeodata.ingestion.chirts import CHIRTSDownloader
    dl = CHIRTSDownloader(variables=variables, source=chirts_source)
    return dl.download(
        extent=extent,
        starting_date=starting_date,
        ending_date=ending_date,
        output_folder=output_folder,
        ncores=ncores,
    )


@mcp.tool()
def download_nasa_power(
    parameters: list[str],
    starting_date: str,
    ending_date: str,
    output_folder: str,
    country_code: str = "",
    feature_name: str = "",
    adm_level: int = 1,
    bbox: list[float] | None = None,
) -> str:
    """Download NASA POWER daily data (no API key needed).

    Routes each parameter to its fastest backend automatically: vars in the
    public S3 Zarr store (``T2M_MAX``, ``T2M_MIN``, ``RH2M``, ``WS2M``) are
    read from S3, while REST-only vars such as ``ALLSKY_SFC_SW_DWN`` (solar
    radiation) are fetched from the POWER regional REST API and merged in.

    Common *parameters*: ``ALLSKY_SFC_SW_DWN`` (solar radiation), ``RH2M``
    (relative humidity), ``T2M_MAX``, ``T2M_MIN``, ``WS2M`` (wind speed).

    Returns
    -------
    str
        Path to the saved NetCDF file.
    """
    extent = _resolve_extent(country_code, feature_name, adm_level, bbox)
    from aggeodata.ingestion.nasa_power import NASAPowerDownloader
    dl = NASAPowerDownloader(parameters=parameters)
    return dl.download(
        extent=extent,
        starting_date=starting_date,
        ending_date=ending_date,
        output_folder=output_folder,
    )


@mcp.tool()
def download_agera5(
    variable: str,
    starting_date: str,
    ending_date: str,
    output_folder: str,
    country_code: str = "",
    feature_name: str = "",
    adm_level: int = 1,
    bbox: list[float] | None = None,
    ncores: int = 4,
) -> dict[str, str]:
    """Download AgERA5 agrometeorological data via the Copernicus CDS API.

    Requires a CDS API key configured in ``~/.cdsapirc``.

    Valid *variable* values: ``wind_speed``, ``vapour_pressure``,
    ``vapour_pressure_defficit``, ``relative_humidity_max``,
    ``relative_humidity_min``, ``reference_evapotranspiration``,
    ``solar_radiation``, ``dew_point_temperature``,
    ``temperature_tmax``, ``temperature_tmin``.

    Returns
    -------
    dict[str, str]
        Mapping ``{year: local_path}``.
    """
    extent = _resolve_extent(country_code, feature_name, adm_level, bbox)
    from aggeodata.ingestion.agera5 import AgEra5Downloader, AGERA5_VARIABLE_MAP

    matched = next(
        (k for k in AGERA5_VARIABLE_MAP if variable == k or k in variable), None
    )
    if matched is None:
        raise ValueError(
            f"Unknown AgERA5 variable '{variable}'. "
            f"Valid keys: {sorted(AGERA5_VARIABLE_MAP)}"
        )
    spec = AGERA5_VARIABLE_MAP[matched]
    dl = AgEra5Downloader()
    return dl.download(
        variable=spec["variable"],
        statistic=spec.get("statistic"),
        time=spec.get("time"),
        starting_date=starting_date,
        ending_date=ending_date,
        output_folder=output_folder,
        aoi_extent=extent,
        ncores=ncores,
    )


@mcp.tool()
def download_gee(
    variables: list[str],
    starting_date: str,
    ending_date: str,
    output_folder: str,
    country_code: str = "",
    feature_name: str = "",
    adm_level: int = 1,
    bbox: list[float] | None = None,
    project: str | None = None,
    ncores: int = 4,
) -> dict[str, dict[str, str]]:
    """Download climate data via Google Earth Engine (no UCSB rate limits).

    Routes each CF variable to its default GEE collection automatically:

    * ``pr``                                → ``UCSB-CHG/CHIRPS/DAILY`` (1981–present)
    * ``tasmax``, ``tasmin``, ``tas``,
      ``tdps``, ``rsds``, ``vp``, ``etr``  → ``projects/climate-engine-pro/assets/ce-ag-era5-v2/daily``

    Requires ``earthengine-api`` and a prior ``earthengine authenticate`` run.

    Parameters
    ----------
    variables : list[str]
        CF variable names, e.g. ``["pr", "tasmax", "rsds", "etr"]``.
    starting_date, ending_date : str
        ISO 8601 date strings ``"YYYY-MM-DD"``.
    output_folder : str
        Root directory; ``{output_folder}/{variable}/{year}/`` sub-folders are
        created automatically.
    project : str | None
        GEE cloud project ID (e.g. ``"my-gee-project"``).  Required for
        post-2023 accounts; omit for legacy accounts.
    ncores : int
        Concurrent day-downloads.  Default: 4 (GEE has no UCSB ban risk).

    Returns
    -------
    dict[str, dict[str, str]]
        ``{cf_variable: {year: year_folder_path}}``
    """
    from aggeodata.ingestion.gee import GEEDownloader
    from aggeodata.pipelines.download import _CF_TO_GEE_DATASET

    extent = _resolve_extent(country_code, feature_name, adm_level, bbox)

    # Group variables by target dataset to minimise GEE initialisation calls
    dataset_vars: dict[str, list[str]] = {}
    for cf_var in variables:
        dataset_id = _CF_TO_GEE_DATASET.get(cf_var)
        if dataset_id is None:
            raise ValueError(
                f"No default GEE dataset for CF variable '{cf_var}'. "
                f"Supported variables: {sorted(_CF_TO_GEE_DATASET)}"
            )
        dataset_vars.setdefault(dataset_id, []).append(cf_var)

    results: dict[str, dict[str, str]] = {}
    for dataset_id, ds_vars in dataset_vars.items():
        dl = GEEDownloader(dataset_id=dataset_id, variables=ds_vars, project=project)
        paths = dl.download(
            extent=extent,
            starting_date=starting_date,
            ending_date=ending_date,
            output_folder=output_folder,
            ncores=ncores,
        )
        results.update(paths)

    return results


@mcp.tool()
def download_soil(
    output_folder: str,
    country_code: str = "",
    feature_name: str = "",
    adm_level: int = 1,
    bbox: list[float] | None = None,
    variables: list[str] | None = None,
    depths: list[str] | None = None,
) -> dict[str, str]:
    """Download SoilGrids GeoTIFF files for a region.

    Downloads physical/chemical variables (clay, sand, bdod, etc.) at 250 m via
    the ISRIC WCS API, and hydraulic variables (wv0010, wv0033, wv1500) at 1 km
    via Google Cloud Storage.

    Parameters
    ----------
    output_folder : str
        Directory where GeoTIFFs will be saved.
    country_code : str
        ISO 3166-1 alpha-3 code (e.g. ``"GHA"``). Used for boundary lookup when
        *bbox* is not provided.
    feature_name : str
        Admin unit name for a sub-national extent (e.g. ``"Zomba"``).
    adm_level : int
        Admin level for feature boundary lookup (default 1).
    bbox : list[float] | None
        ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.  Takes precedence over
        country_code / feature_name.
    variables : list[str] | None
        Soil variables to download.  Defaults to
        ``["clay", "sand", "silt", "bdod", "cfvo", "nitrogen", "phh2o",
           "soc", "wv0010", "wv0033", "wv1500"]``.
    depths : list[str] | None
        Depth intervals.  Defaults to
        ``["0-5", "5-15", "15-30", "30-60", "60-100"]``.

    Returns
    -------
    dict[str, str]
        Mapping ``{filename: local_path}`` for every downloaded GeoTIFF.
    """
    if variables is None:
        variables = ["clay", "sand", "silt", "bdod", "cfvo",
                     "nitrogen", "phh2o", "soc", "wv0010", "wv0033", "wv1500"]
    if depths is None:
        depths = ["0-5", "5-15", "15-30", "30-60", "60-100"]

    extent = _resolve_extent(country_code, feature_name, adm_level, bbox)

    from aggeodata.ingestion.soil import SoilGridsDownloader
    dl = SoilGridsDownloader(soil_layers=variables, depths=depths, output_folder=output_folder)
    downloaded = dl.download(boundaries=extent)
    return {str(k): str(v) for k, v in downloaded.items()}


@mcp.tool()
def build_climate_datacube(
    config_path: str,
) -> str:
    """Assemble downloaded climate files into a multi-variable NetCDF datacube.

    Reads the same YAML config used by the individual download_* tools.  All
    variables listed in ``CLIMATE.variables`` must have been downloaded first.
    The resulting file is saved as ``climate_<suffix>_<year_start>_<year_end>.nc``
    in ``PATHS.output_path``.

    Parameters
    ----------
    config_path : str
        Path to an aggeodata YAML config (``task: datacube`` or any task value).

    Returns
    -------
    str
        Path to the saved NetCDF datacube.
    """
    from aggeodata.pipelines.datacube import run_datacube
    return run_datacube(config_path)


@mcp.tool()
def build_soil_datacube(
    soil_folder: str,
    output_folder: str,
    filename: str | None = None,
    variables: list[str] | None = None,
    reference_variable: str = "wv1500",
    target_crs: str = "EPSG:4326",
) -> str:
    """Convert downloaded SoilGrids GeoTIFFs into a multi-depth NetCDF datacube.

    Reads the GeoTIFF files produced by ``download_soil``, co-registers them
    to the reference variable's grid, stacks them along a ``depth`` dimension,
    and saves the result as a compressed NetCDF ready for ag-cube-cm simulations.

    Parameters
    ----------
    soil_folder : str
        Folder containing the downloaded SoilGrids GeoTIFFs.
    output_folder : str
        Directory where the output NetCDF will be saved.
    filename : str | None
        Output file name.  Defaults to ``soil_<folder_name>.nc``.
    variables : list[str] | None
        Variables to include.  Defaults to all standard DSSAT variables:
        ``["clay", "sand", "silt", "bdod", "cfvo", "nitrogen", "phh2o",
           "soc", "wv0010", "wv0033", "wv1500"]``.
    reference_variable : str
        Variable used as the spatial reference grid.  Default: ``"wv1500"``.
    target_crs : str
        Output CRS.  Default: ``"EPSG:4326"``.

    Returns
    -------
    str
        Path to the saved soil NetCDF datacube.
    """
    if variables is None:
        variables = ["clay", "sand", "silt", "bdod", "cfvo",
                     "nitrogen", "phh2o", "soc", "wv0010", "wv0033", "wv1500"]

    from aggeodata.transform.soil_cube import SoilDataCubeBuilder
    builder = SoilDataCubeBuilder(
        data_folder=soil_folder,
        variables=variables,
        reference_variable=reference_variable,
        target_crs=target_crs,
    )
    return builder.build_and_save(output_path=output_folder, filename=filename)


@mcp.tool()
def reshape_flat_soil_cube(
    input_path: str,
    output_path: str,
) -> str:
    """Reshape a flat-format soil NetCDF into the 3-D depth-dimension format
    required by ag-cube-cm.

    Old pipelines and direct SoilGrids API downloads produce a flat file where
    each depth is a separate variable named ``{var}_{lo}-{hi}cm_mean`` (e.g.
    ``bdod_0-5cm_mean``).  ag-cube-cm expects a single variable per property
    with a ``depth`` coordinate dimension.

    Parameters
    ----------
    input_path : str
        Path to the flat-format soil NetCDF (e.g. ``soil_uruguay.nc``).
    output_path : str
        Path for the converted output NetCDF.

    Returns
    -------
    str
        Path to the saved converted file.
    """
    import xarray as xr
    from aggeodata.transform.soil_cube import reshape_flat_soil_cube as _reshape

    ds = xr.open_dataset(input_path)
    cube = _reshape(ds)
    encoding = {var: {"zlib": True} for var in cube.data_vars if var != "spatial_ref"}
    cube.to_netcdf(output_path, encoding=encoding, engine="netcdf4")
    return output_path


@mcp.tool()
def list_available_climate_indices() -> list[dict]:
    """List every supported climate index with its required variables and default parameters.

    Use the ``name`` values when building the ``indices`` list for
    :func:`compute_climate_features` or a ``PipelineConfig`` YAML.
    """
    return [
        {
            "name": "vpd_lt_15",
            "description": "Percentage of days where VPD < threshold",
            "required_variables": ["vpd"],
            "default_parameters": {"threshold": 1.5},
        },
        {
            "name": "n_vpd_spells",
            "description": "Number of spell events with consecutive days below VPD threshold",
            "required_variables": ["vpd"],
            "default_parameters": {"threshold": 1.5, "min_duration_days": 7},
        },
        {
            "name": "n_wet_spells",
            "description": "Number of wet spell events (consecutive days with precip >= threshold)",
            "required_variables": ["pr"],
            "default_parameters": {"threshold_mm": 1.0, "min_duration_days": 7},
        },
        {
            "name": "n_dry_spells",
            "description": "Number of dry spell events (consecutive days with precip < threshold)",
            "required_variables": ["pr"],
            "default_parameters": {"threshold_mm": 1.0, "min_duration_days": 7},
        },
        {
            "name": "avg_wet_spell_duration",
            "description": "Average length of wet spells exceeding min_duration_days",
            "required_variables": ["pr"],
            "default_parameters": {"threshold_mm": 1.0, "min_duration_days": 7},
        },
        {
            "name": "avg_dry_spell_duration",
            "description": "Average length of dry spells exceeding min_duration_days",
            "required_variables": ["pr"],
            "default_parameters": {"threshold_mm": 1.0, "min_duration_days": 7},
        },
        {
            "name": "heat_wave_duration",
            "description": "Average duration of heat waves (consecutive days above temp threshold)",
            "required_variables": ["tmax"],
            "default_parameters": {"thresh": 28.0, "min_duration_days": 5},
        },
        {
            "name": "cold_wave_duration",
            "description": "Average duration of cold waves (consecutive days below temp threshold)",
            "required_variables": ["tmin"],
            "default_parameters": {"thresh": 5.0, "min_duration_days": 5},
        },
        {
            "name": "rh_85_90_days",
            "description": "Days with RH between 85–90% — one output column per variable listed",
            "required_variables": ["dailyhr", "hr06", "hr09", "hr12", "hr15", "hr18"],
            "default_parameters": {"thresholds": [85, 90], "op_symbols": [">=", "<="]},
        },
        {
            "name": "tmean_25_30_days",
            "description": "Days with mean temperature between 25°C and 30°C",
            "required_variables": ["tmean"],
            "default_parameters": {"thresholds": [25, 30], "op_symbols": [">=", "<="]},
        },
        {
            "name": "max_temp_days",
            "description": "Days where max temperature >= threshold",
            "required_variables": ["tmax"],
            "default_parameters": {"threshold_celsius": 35.0},
        },
        {
            "name": "max_hr_days",
            "description": "Days where daily mean RH >= threshold",
            "required_variables": ["dailyhr"],
            "default_parameters": {"threshold_percent": 80.0},
        },
        {
            "name": "max_hr06_days",
            "description": "Days where 06:00 RH >= threshold",
            "required_variables": ["hr06"],
            "default_parameters": {"threshold_percent": 80.0},
        },
        {
            "name": "max_hr09_days",
            "description": "Days where 09:00 RH >= threshold",
            "required_variables": ["hr09"],
            "default_parameters": {"threshold_percent": 80.0},
        },
        {
            "name": "max_hr12_days",
            "description": "Days where 12:00 RH >= threshold",
            "required_variables": ["hr12"],
            "default_parameters": {"threshold_percent": 80.0},
        },
        {
            "name": "max_hr15_days",
            "description": "Days where 15:00 RH >= threshold",
            "required_variables": ["hr15"],
            "default_parameters": {"threshold_percent": 80.0},
        },
        {
            "name": "max_hr18_days",
            "description": "Days where 18:00 RH >= threshold",
            "required_variables": ["hr18"],
            "default_parameters": {"threshold_percent": 80.0},
        },
        {
            "name": "precip_max_15d",
            "description": "Maximum precipitation accumulated over any 15-day rolling window",
            "required_variables": ["pr"],
            "default_parameters": {},
        },
        {
            "name": "consecutive_dry_days",
            "description": "Longest run of consecutive days below precipitation threshold",
            "required_variables": ["pr"],
            "default_parameters": {"threshold_mm": 1.0},
        },
        {
            "name": "growing_degree_days",
            "description": "Accumulated growing degree days above base temperature",
            "required_variables": ["tmean"],
            "default_parameters": {"base_temperature": 15.0},
        },
        {
            "name": "daily_intensity_index",
            "description": "Mean rainfall on wet days (precip >= threshold)",
            "required_variables": ["pr"],
            "default_parameters": {"threshold_mm": 1.0},
        },
        {
            "name": "disease_pressure_index",
            "description": (
                "Composite disease pressure index: (RH_norm × precip_intensity_norm) / VPD_norm. "
                "Requires max_hr_days and daily_intensity_index to also be listed in indices."
            ),
            "required_variables": ["vpd"],
            "default_parameters": {},
        },
    ]


@mcp.tool()
def compute_climate_features(
    datacube_path: str,
    start_date: str,
    end_date: str,
    summarizations: list[dict],
    indices: list[dict],
    output_path: str | None = None,
) -> dict:
    """Compute climate summaries and indices over a time slice of a NetCDF datacube.

    Parameters
    ----------
    datacube_path : str
        Path to a NetCDF datacube produced by ``run_datacube``.  Must have a
        ``date`` time dimension.  Expected units: temperature in °C,
        precipitation in mm/day, VPD in kPa, relative humidity in %.
    start_date, end_date : str
        ISO 8601 date strings (``"YYYY-MM-DD"``) bounding the time slice.
    summarizations : list[dict]
        Each entry: ``{"meteorological_variable": "tmax", "summary_function": "mean"}``.
        ``summary_function`` must be ``"mean"`` or ``"sum"``.
    indices : list[dict]
        Each entry: ``{"name": "vpd_lt_15", "meteorological_variables": ["vpd"],
        "parameters": {"threshold": 1.5}}``.
        Call :func:`list_available_climate_indices` to discover all supported names.
    output_path : str | None
        If given, the merged result Dataset is saved to this NetCDF path.

    Returns
    -------
    dict
        ``{"output_path": str | None, "variables": list[str], "n_days": int,
        "spatial_mean": {var: float}}``
    """
    import numpy as np
    import pandas as pd
    import xarray
    from aggeodata.config.schemas import ClimateConfig, ClimateIndex, ClimateSummarisation
    from aggeodata.features.climate_indices import calculate_indices
    from aggeodata.features.met_summaries import calculate_meteorological_summaries

    ds = xarray.open_dataset(datacube_path)
    w = ds.sel(date=slice(pd.Timestamp(start_date), pd.Timestamp(end_date)))
    n_days = int(w.sizes.get("date", 0))

    climate_cfg = ClimateConfig(
        climate_data_path=datacube_path,
        summarizations=[ClimateSummarisation(**s) for s in summarizations],
        indices=[ClimateIndex(**i) for i in indices],
    )

    summaries_ds = calculate_meteorological_summaries(w, climate_cfg.summarizations)
    indices_ds = calculate_indices(w, climate_cfg.indices)
    result = xarray.merge([summaries_ds, indices_ds])

    variables = list(result.data_vars)
    spatial_mean = {
        v: float(np.nanmean(result[v].values)) for v in variables
    }

    saved_path = None
    if output_path:
        result.to_netcdf(output_path)
        saved_path = output_path

    return {
        "output_path": saved_path,
        "variables": variables,
        "n_days": n_days,
        "spatial_mean": spatial_mean,
    }


@mcp.tool()
def run_summarization_pipeline(config_path: str) -> dict:
    """Run the full climate summarization and point-extraction pipeline.

    Loads a climate datacube, iterates over unique field-observation dates,
    computes meteorological summaries and climate indices for each temporal
    window (with optional historical lookback), extracts values at observation
    point locations, and writes the results to CSV.

    The YAML must follow the ``PipelineConfig`` schema (lowercase top-level keys:
    ``general_info``, ``climate_config``, ``data_summarization``).

    Parameters
    ----------
    config_path : str
        Path to a ``PipelineConfig`` YAML file.

    Returns
    -------
    dict
        ``{"output_path": str, "n_rows": int, "n_columns": int, "columns": list[str]}``
    """
    import pandas as pd
    import geopandas as gpd
    import xarray
    from pathlib import Path
    from tqdm import tqdm
    from aggeodata.config.loader import load_config
    from aggeodata.config.schemas import PipelineConfig
    from aggeodata.utils.spatial_utils import extracting_using_gpdf, process_temporal_windows

    cfg = load_config(config_path)
    if not isinstance(cfg, PipelineConfig):
        raise ValueError(
            f"Config at '{config_path}' loaded as {type(cfg).__name__}, "
            "expected PipelineConfig. Ensure the YAML uses lowercase section keys "
            "(general_info, climate_config, data_summarization)."
        )

    ds = xarray.open_dataset(cfg.climate_config.climate_data_path)

    gdf = gpd.read_parquet(cfg.data_summarization.field_data_source)
    end_col = cfg.data_summarization.column_ending_date
    gdf["_year"] = gdf[end_col].astype(str).str[:4].astype(int)
    if cfg.general_info.year_oi is not None:
        gdf = gdf.loc[gdf["_year"] == cfg.general_info.year_oi]

    eval_dates = pd.to_datetime(
        gdf[end_col].astype(str), format="%Y%m%d", errors="coerce"
    )
    window_months = int(cfg.data_summarization.temporal_window or 6)
    lookback_months = int(cfg.data_summarization.nmonths_lookback or 0)

    extracted_dfs = []
    for eval_date in tqdm(eval_dates.unique(), desc="Processing dates"):
        merged = process_temporal_windows(
            xrdata=ds,
            climate_config=cfg.climate_config,
            eval_date=eval_date,
            window_months=window_months,
            lookback_months=None if lookback_months == 0 else lookback_months,
        )
        obs_subset = gdf.loc[
            gdf[end_col].astype(str) == pd.Timestamp(eval_date).strftime("%Y%m%d")
        ].copy()
        if obs_subset.empty:
            continue
        extracted_dfs.append(extracting_using_gpdf(obs_subset, merged))

    final_df = pd.concat(extracted_dfs, ignore_index=True)

    out_dir = Path(cfg.general_info.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / cfg.data_summarization.output_filename
    final_df.to_csv(out_file, index=False)

    return {
        "output_path": str(out_file),
        "n_rows": len(final_df),
        "n_columns": len(final_df.columns),
        "columns": final_df.columns.tolist(),
    }


if __name__ == "__main__":
    mcp.run()
