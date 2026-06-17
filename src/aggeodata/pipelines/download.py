"""
aggeodata.pipelines.download
=============================

YAML-driven download pipeline.

Usage
-----
    from aggeodata.pipelines import run_download
    run_download("options/aggeodata_config.yaml")

Or from the command line:
    python -m aggeodata.pipelines.download options/aggeodata_config.yaml
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..config.loader import load_config
from ..config.schemas import AgeodataConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CF variable name → downloader routing tables
# ---------------------------------------------------------------------------

# CF name → AgERA5 config key (from AGERA5_VARIABLE_MAP)
_CF_TO_AGERA5_KEY: dict[str, str] = {
    "pr":      "precipitation",
    "tasmax":  "temperature_tmax",
    "tasmin":  "temperature_tmin",
    "rsds":    "solar_radiation",
    "sfcWind": "wind_speed",
    "hurs_06": "relative_humidity_06",
    "hurs_09": "relative_humidity_09",
    "hurs_12": "relative_humidity_12",
    "hurs_15": "relative_humidity_15",
    "hurs_18": "relative_humidity_18",
    "tdps":    "dew_point_temperature",
    "etr":     "reference_evapotranspiration",
    "vp":      "vapour_pressure",
    "vpd":     "vapour_pressure_defficit",
}

# CF name → CHIRTS native variable name
_CF_TO_CHIRTS_VAR: dict[str, str] = {
    "tasmax": "tmax",
    "tasmin": "tmin",
}

# CF name → NASA POWER parameter code
_CF_TO_NASA_POWER_PARAM: dict[str, str] = {
    "pr":      "PRECTOTCORR",
    "tasmax":  "T2M_MAX",
    "tasmin":  "T2M_MIN",
    "tas":     "T2M",
    "hurs":    "RH2M",
    "sfcWind": "WS2M",
    "rsds":    "ALLSKY_SFC_SW_DWN",  # requires REST backend — not in S3 Zarr
}

# CF variables that require the REST backend (not in S3 Zarr)
_NASA_POWER_REST_ONLY: frozenset[str] = frozenset({"rsds"})

# CF name → default GEE ImageCollection ID
_CF_TO_GEE_DATASET: dict[str, str] = {
    # CHIRPS covers 1981–present
    "pr":     "UCSB-CHG/CHIRPS/DAILY",
    # AgERA5 CE-Pro for temperature — covers 1979–present (CHIRTS/DAILY ends 2016)
    "tasmax": "projects/climate-engine-pro/assets/ce-ag-era5-v2/daily",
    "tasmin": "projects/climate-engine-pro/assets/ce-ag-era5-v2/daily",
    "tas":    "projects/climate-engine-pro/assets/ce-ag-era5-v2/daily",
    "tdps":   "projects/climate-engine-pro/assets/ce-ag-era5-v2/daily",
    "rsds":   "projects/climate-engine-pro/assets/ce-ag-era5-v2/daily",
    "vp":     "projects/climate-engine-pro/assets/ce-ag-era5-v2/daily",
    "etr":    "projects/climate-engine-pro/assets/ce-ag-era5-v2/daily",
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_download(config_path: str | os.PathLike) -> dict[str, dict]:
    """Download all climate and soil data specified in the YAML config.

    Parameters
    ----------
    config_path : str | os.PathLike
        Path to the aggeodata YAML configuration file.

    Returns
    -------
    dict[str, dict]
        ``{cf_variable: {year_or_key: path}}`` for each downloaded variable.
        Soil GeoTIFFs are under key ``"soil"``.
    """
    cfg = load_config(config_path)
    results: dict[str, dict] = {}

    extent = cfg.get_extent()
    start  = cfg.DATES.starting_date
    end    = cfg.DATES.ending_date

    logger.info("aggeodata download  |  %s -> %s  |  extent: %s", start, end, extent)
    logger.info("Output root: %s", cfg.PATHS.output_path)

    # ------------------------------------------------------------------
    # Climate variables
    # ------------------------------------------------------------------
    for cf_var, var_cfg in cfg.CLIMATE.variables.items():
        out_folder = cfg.var_folder(cf_var)
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        source = var_cfg.source

        logger.info("Downloading  %-10s  (source: %s) -> %s", cf_var, source, out_folder)

        try:
            result = _download_variable(
                cf_var=cf_var,
                source=source,
                var_cfg=var_cfg,
                out_folder=out_folder,
                extent=extent,
                start=start,
                end=end,
                cfg=cfg,
            )
            results[cf_var] = result
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED  %s (%s): %s", cf_var, source, exc)
            results[cf_var] = {}

    # ------------------------------------------------------------------
    # Soil layers
    # ------------------------------------------------------------------
    if cfg.SOIL.enabled:
        soil_folder = os.path.join(cfg.PATHS.output_path, f"soil_{cfg.GENERAL.suffix}" if cfg.GENERAL.suffix else "soil")
        logger.info("Downloading soil layers -> %s", soil_folder)
        try:
            from ..ingestion.soil import SoilGridsDownloader
            dl = SoilGridsDownloader(
                soil_layers=cfg.SOIL.layers,
                depths=cfg.SOIL.depths,
                output_folder=soil_folder,
            )
            written = dl.download(boundaries=extent)
            results["soil"] = {os.path.basename(p): p for p in written}
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED  soil: %s", exc)
            results["soil"] = {}

    _log_summary(results)
    return results


# ---------------------------------------------------------------------------
# Per-source download helpers
# ---------------------------------------------------------------------------

def _download_variable(
    cf_var: str,
    source: str,
    var_cfg,
    out_folder: str,
    extent: list[float],
    start: str,
    end: str,
    cfg: AgeodataConfig,
) -> dict:
    if source == "chirps":
        return _download_chirps(out_folder, extent, start, end, cfg)

    if source == "chirts":
        return _download_chirts(cf_var, var_cfg, out_folder, extent, start, end, cfg)

    if source == "agera5":
        return _download_agera5(cf_var, var_cfg, out_folder, extent, start, end, cfg)

    if source == "nasa_power":
        return _download_nasa_power(cf_var, var_cfg, out_folder, extent, start, end, cfg)

    if source == "gee":
        return _download_gee(cf_var, var_cfg, out_folder, extent, start, end, cfg)

    raise ValueError(f"Unknown source '{source}' for variable '{cf_var}'")


def _download_chirps(out_folder, extent, start, end, cfg):
    from ..ingestion.chirps import CHIRPSDownloader
    dl = CHIRPSDownloader()
    return dl.download(
        extent=extent,
        starting_date=start,
        ending_date=end,
        output_folder=out_folder,
        ncores=cfg.GENERAL.ncores,
    )


def _download_chirts(cf_var, var_cfg, out_folder, extent, start, end, cfg):
    from ..ingestion.chirts import CHIRTSDownloader
    chirts_var = _CF_TO_CHIRTS_VAR.get(cf_var)
    if chirts_var is None:
        raise ValueError(f"CF variable '{cf_var}' has no CHIRTS equivalent")
    dl = CHIRTSDownloader(
        variables=[chirts_var],
        source=var_cfg.chirts_source,
    )
    paths = dl.download(
        extent=extent,
        starting_date=start,
        ending_date=end,
        output_folder=out_folder,
        ncores=cfg.GENERAL.ncores,
    )
    # Flatten {variable: {year: folder}} → {year: folder}
    return {yr: folder for var_paths in paths.values() for yr, folder in var_paths.items()}


def _download_agera5(cf_var, var_cfg, out_folder, extent, start, end, cfg):
    from ..ingestion.agera5 import AgEra5Downloader, AGERA5_VARIABLE_MAP
    agera5_key = var_cfg.agera5_key or _CF_TO_AGERA5_KEY.get(cf_var)
    if agera5_key is None or agera5_key not in AGERA5_VARIABLE_MAP:
        raise ValueError(
            f"No AgERA5 mapping for CF variable '{cf_var}'. "
            f"Set agera5_key explicitly in the config."
        )
    spec = AGERA5_VARIABLE_MAP[agera5_key]
    dl = AgEra5Downloader(version=cfg.GENERAL.agera5_version)
    return dl.download(
        variable=spec["variable"],
        statistic=spec.get("statistic"),
        time=spec.get("time"),
        starting_date=start,
        ending_date=end,
        output_folder=out_folder,
        aoi_extent=extent,
        ncores=cfg.GENERAL.ncores,
    )


def _download_nasa_power(cf_var, var_cfg, out_folder, extent, start, end, cfg):
    from ..ingestion.nasa_power import NASAPowerS3Downloader, NASAPowerDownloader
    param = var_cfg.nasa_power_param or _CF_TO_NASA_POWER_PARAM.get(cf_var)
    if param is None:
        raise ValueError(
            f"No NASA POWER parameter for CF variable '{cf_var}'. "
            f"Set nasa_power_param explicitly in the config."
        )

    # rsds (ALLSKY_SFC_SW_DWN) is not in the S3 Zarr store — force REST
    backend = cfg.GENERAL.nasa_power_backend
    if cf_var in _NASA_POWER_REST_ONLY:
        if backend == "s3":
            logger.info(
                "%s is not available in the S3 Zarr store — switching to REST backend", cf_var
            )
        backend = "rest"

    if backend == "s3":
        dl: NASAPowerS3Downloader | NASAPowerDownloader = NASAPowerS3Downloader(parameters=[param])
    else:
        dl = NASAPowerDownloader(parameters=[param])

    nc_path = dl.download(
        extent=extent,
        starting_date=start,
        ending_date=end,
        output_folder=out_folder,
    )
    return {"all": nc_path}


def _download_gee(cf_var, var_cfg, out_folder, extent, start, end, cfg):
    from ..ingestion.gee import GEEDownloader
    dataset_id = var_cfg.gee_dataset_id or _CF_TO_GEE_DATASET.get(cf_var)
    if dataset_id is None:
        raise ValueError(
            f"No default GEE dataset for CF variable '{cf_var}'. "
            f"Set gee_dataset_id explicitly in the config. "
            f"For AgERA5 use: 'projects/climate-engine-pro/assets/ce-ag-era5-v2/daily'."
        )
    dl = GEEDownloader(
        dataset_id=dataset_id,
        variables=[cf_var],
        project=var_cfg.gee_project,
    )
    paths = dl.download(
        extent=extent,
        starting_date=start,
        ending_date=end,
        output_folder=out_folder,
        ncores=cfg.GENERAL.ncores,
    )
    # Flatten {variable: {year: folder}} → {year: folder}
    return {yr: folder for var_paths in paths.values() for yr, folder in var_paths.items()}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_summary(results: dict) -> None:
    print("\n" + "=" * 56)
    print("  DOWNLOAD SUMMARY")
    print("=" * 56)
    for var, paths in results.items():
        status = "OK " if paths else "---"
        count = len(paths) if isinstance(paths, dict) else 0
        print(f"  {status}  {var:<14}  ({count} entries)")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m aggeodata.pipelines.download <config.yaml>")
        sys.exit(1)
    run_download(sys.argv[1])
