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
    start  = cfg.dates.starting_date
    end    = cfg.dates.ending_date
    # The AOI tag in yearly store names is the config file's stem.
    aoi = Path(os.fspath(config_path)).stem

    logger.info("aggeodata download  |  %s -> %s  |  extent: %s", start, end, extent)
    logger.info("Output root: %s", cfg.paths.output_path)

    # ------------------------------------------------------------------
    # Climate variables
    # ------------------------------------------------------------------
    for cf_var, var_cfg in cfg.climate.variables.items():
        out_folder = cfg.var_folder(cf_var)
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        source = var_cfg.source
        backend = "zarr" if cfg.use_zarr_for(cf_var) else "official"

        logger.info(
            "Downloading  %-10s  (source: %s, backend: %s) -> %s",
            cf_var, source, backend, out_folder,
        )

        # With yearly Zarr the daily files are deleted, so the downloaders'
        # own resume check can't see them — narrow the request to what the
        # existing stores do not already cover.
        dl_start, dl_end = start, end
        if cfg.general.download_format == "zarr" and source != "nasa_power":
            from ..transform.yearly_zarr import find_yearly_stores, pending_range

            pending = pending_range(out_folder, cf_var, start, end)
            if pending is None:
                logger.info("  %s: already covered by yearly Zarr cubes — skipping", cf_var)
                results[cf_var] = {
                    y: str(p) for y, p in find_yearly_stores(out_folder, cf_var).items()
                }
                continue
            dl_start, dl_end = pending
            if (dl_start, dl_end) != (start, end):
                logger.info("  %s: fetching only %s -> %s", cf_var, dl_start, dl_end)

        fast = _try_yearly_fast_path(
            cf_var, source, var_cfg, out_folder, extent, dl_start, dl_end, cfg)
        if fast is not None:
            # Already yearly stores on disk — no daily files to consolidate.
            results[cf_var] = fast
            continue

        try:
            result = _download_variable(
                cf_var=cf_var,
                source=source,
                var_cfg=var_cfg,
                out_folder=out_folder,
                extent=extent,
                start=dl_start,
                end=dl_end,
                cfg=cfg,
            )
            results[cf_var] = result
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED  %s (%s): %s", cf_var, source, exc)
            results[cf_var] = {}
            continue

        if cfg.general.download_format == "zarr":
            try:
                results[cf_var] = _consolidate_yearly(
                    cf_var, source, out_folder, aoi, start, end, cfg
                )
            except Exception as exc:  # noqa: BLE001
                # The raw files are still on disk, so this is recoverable.
                logger.error("FAILED  %s: yearly Zarr consolidation: %s", cf_var, exc)

    # ------------------------------------------------------------------
    # Soil layers
    # ------------------------------------------------------------------
    if cfg.soil.enabled:
        soil_folder = os.path.join(cfg.paths.output_path, f"soil_{cfg.general.suffix}" if cfg.general.suffix else "soil")
        logger.info("Downloading soil layers -> %s", soil_folder)
        try:
            from ..ingestion.soil import SoilGridsDownloader
            dl = SoilGridsDownloader(
                soil_layers=cfg.soil.variables,
                depths=cfg.soil.depths,
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
# Yearly Zarr consolidation
# ---------------------------------------------------------------------------

def _zarr_store_spec(cf_var: str, source: str, var_cfg, cfg: AgeodataConfig):
    """Everything needed to read one variable straight from its Zarr store.

    ``None`` when this variable has no Zarr store, which sends the caller back
    to the per-day path. The details mirror each downloader's own call to
    :func:`download_daily_from_zarr`, so the two paths read exactly the same
    bytes with the same coordinate names.
    """
    # zarr_url_for() returns only a config OVERRIDE, which is None unless the
    # user set one; each downloader falls back to its own class default. Mirror
    # that, or the fast path is handed url=None and dies with the unhelpful
    # "No group found in store None at path ''".
    if source == "chirps":
        from ..ingestion.chirps import CHIRPSDownloader
        return {
            "url": cfg.zarr_url_for(cf_var) or CHIRPSDownloader.URL_HF,
            "variables": {CHIRPSDownloader.HF_VARIABLE: cf_var},
            "desc": "CHIRPS (HF)",
        }

    if source == "chirts":
        from ..ingestion.chirts import CHIRTSDownloader, _HF_VAR
        native = _CF_TO_CHIRTS_VAR.get(cf_var)
        if native is None or var_cfg.chirts_source != "era5":
            return None          # only the era5 mirror has a Zarr store
        return {
            "url": cfg.zarr_url_for(cf_var) or CHIRTSDownloader.URL_HF,
            "variables": {_HF_VAR[native]: cf_var},
            "desc": "CHIRTS (HF)",
        }

    if source == "agera5":
        from ..ingestion.agera5 import (
            AGERA5_ARCO_GEO_URL, AGERA5_ARCO_TIME_URL, AGERA5_ARCO_VARIABLES,
        )
        from ..ingestion.zarr_backend import cds_storage_options
        key = var_cfg.agera5_key or _CF_TO_AGERA5_KEY.get(cf_var)
        # Only the keys present in the ARCO store have a direct read; the rest
        # still go through the CDS API day by day.
        store_var = AGERA5_ARCO_VARIABLES.get(key) if key else None
        if not store_var:
            return None
        # The two ARCO stores differ only in chunking; keep the downloader's choice.
        default_url = (AGERA5_ARCO_GEO_URL if var_cfg.agera5_chunking == "geo"
                       else AGERA5_ARCO_TIME_URL)
        return {
            "url": cfg.zarr_url_for(cf_var) or default_url,
            "variables": {store_var: cf_var},
            "storage_options": cds_storage_options(),
            "lat_name": "latitude",
            "lon_name": "longitude",
            "desc": f"AgERA5 {cf_var} (ARCO)",
        }

    return None                  # nasa_power and gee are not Zarr-backed


def _try_yearly_fast_path(cf_var, source, var_cfg, out_folder, extent, start, end, cfg):
    """Write the yearly Zarr store straight from the remote store, or return None.

    When the source is a Zarr store and the destination is Zarr too, the per-day
    round trip is pure overhead: the slice is exploded into one NetCDF per day
    and immediately read back to be merged into a store per year. Measured on one
    CHIRPS season over a district, that round trip was 93.8 s against 2.9 s going
    direct — a 32x difference for byte-identical output, because the remote stores
    are chunked by time and a season is a handful of chunk reads.

    Anything unsupported returns None and the caller falls through to the
    existing path, so a source without a Zarr store behaves exactly as before.
    """
    if cfg.general.download_format != "zarr" or not cfg.use_zarr_for(cf_var):
        return None
    try:
        spec = _zarr_store_spec(cf_var, source, var_cfg, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.debug("  %s: no Zarr fast path (%s)", cf_var, exc)
        return None
    if not spec or not spec.get("url"):
        return None

    from ..ingestion.zarr_backend import fetch_zarr_years
    from ..transform.yearly_zarr import CubeGeometry

    geometry = CubeGeometry(sharded=cfg.general.zarr_sharded)
    if cfg.general.zarr_chunks:
        t, y, x = cfg.general.zarr_chunks
        geometry = CubeGeometry(
            time_chunk=t, lat_chunk=y, lon_chunk=x,
            time_shard=geometry.time_shard, lat_shard=geometry.lat_shard,
            lon_shard=geometry.lon_shard, sharded=cfg.general.zarr_sharded,
        )

    try:
        written = fetch_zarr_years(
            url=spec["url"],
            variables=spec["variables"],
            extent=extent,
            starting_date=start,
            ending_date=end,
            var_folder=out_folder,
            source=source,
            aoi=cfg.general.suffix,
            geometry=geometry,
            quantize=cfg.general.zarr_quantize,
            codec=cfg.general.zarr_codec,
            storage_options=spec.get("storage_options"),
            lat_name=spec.get("lat_name", "lat"),
            lon_name=spec.get("lon_name", "lon"),
            desc=spec.get("desc", source),
        )
    except Exception as exc:  # noqa: BLE001
        # The daily path still works; losing the speed-up beats losing the run.
        logger.warning("  %s: direct yearly read failed (%s) — using daily files",
                       cf_var, exc)
        return None

    stores = written.get(cf_var) or {}
    if not stores:
        return None
    logger.info("  %s: wrote %d yearly store(s) directly, skipping daily files",
                cf_var, len(stores))
    return stores


def _consolidate_yearly(
    cf_var: str,
    source: str,
    out_folder: str,
    aoi: str,
    start: str,
    end: str,
    cfg: AgeodataConfig,
) -> dict:
    """Collapse a variable's daily files into one Zarr store per year.

    NASA POWER is skipped: it already writes a single multi-day NetCDF, so
    there are no per-day files to consolidate.
    """
    if source == "nasa_power":
        logger.info("  %s: NASA POWER is already a single file — not consolidating", cf_var)
        return {"all": out_folder}

    from ..transform.yearly_zarr import (
        CubeGeometry,
        consolidate_variable_to_yearly_zarr,
    )

    geometry = CubeGeometry(sharded=cfg.general.zarr_sharded)
    if cfg.general.zarr_chunks:
        t, y, x = cfg.general.zarr_chunks
        geometry = CubeGeometry(
            time_chunk=t, lat_chunk=y, lon_chunk=x,
            time_shard=geometry.time_shard, lat_shard=geometry.lat_shard,
            lon_shard=geometry.lon_shard, sharded=cfg.general.zarr_sharded,
        )

    logger.info("Consolidating %s into yearly Zarr cubes", cf_var)
    return consolidate_variable_to_yearly_zarr(
        cf_var=cf_var,
        source=source,
        var_folder=out_folder,
        aoi=aoi,
        start=start,
        end=end,
        geometry=geometry,
        quantize=cfg.general.zarr_quantize,
        codec=cfg.general.zarr_codec,
    )


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
        return _download_chirps(cf_var, out_folder, extent, start, end, cfg)

    if source == "chirts":
        return _download_chirts(cf_var, var_cfg, out_folder, extent, start, end, cfg)

    if source == "agera5":
        return _download_agera5(cf_var, var_cfg, out_folder, extent, start, end, cfg)

    if source == "nasa_power":
        return _download_nasa_power(cf_var, var_cfg, out_folder, extent, start, end, cfg)

    if source == "gee":
        return _download_gee(cf_var, var_cfg, out_folder, extent, start, end, cfg)

    raise ValueError(f"Unknown source '{source}' for variable '{cf_var}'")


def _download_chirps(cf_var, out_folder, extent, start, end, cfg):
    from ..ingestion.chirps import CHIRPSDownloader
    dl = CHIRPSDownloader(
        backend="hf" if cfg.use_zarr_for(cf_var) else "ucsb",
        hf_url=cfg.zarr_url_for(cf_var),
    )
    return dl.download(
        extent=extent,
        starting_date=start,
        ending_date=end,
        output_folder=out_folder,
        ncores=cfg.general.ncores,
    )


def _download_chirts(cf_var, var_cfg, out_folder, extent, start, end, cfg):
    from ..ingestion.chirts import CHIRTSDownloader
    chirts_var = _CF_TO_CHIRTS_VAR.get(cf_var)
    if chirts_var is None:
        raise ValueError(f"CF variable '{cf_var}' has no CHIRTS equivalent")
    use_zarr = cfg.use_zarr_for(cf_var)
    if use_zarr and var_cfg.chirts_source != "era5":
        raise ValueError(
            f"Variable '{cf_var}': the Hugging Face mirror holds CHIRTS-ERA5 only. "
            f"Set chirts_source: era5, or disable use_zarr for this variable."
        )
    dl = CHIRTSDownloader(
        variables=[chirts_var],
        source=var_cfg.chirts_source,
        backend="hf" if use_zarr else "ucsb",
        hf_url=cfg.zarr_url_for(cf_var),
    )
    paths = dl.download(
        extent=extent,
        starting_date=start,
        ending_date=end,
        output_folder=out_folder,
        ncores=cfg.general.ncores,
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
    use_zarr = cfg.use_zarr_for(cf_var)
    dl = AgEra5Downloader(
        version=cfg.general.agera5_version,
        backend="zarr" if use_zarr else "cds",
        chunking=var_cfg.agera5_chunking,
        zarr_url=cfg.zarr_url_for(cf_var),
    )
    return dl.download(
        variable=spec["variable"],
        statistic=spec.get("statistic"),
        time=spec.get("time"),
        starting_date=start,
        ending_date=end,
        output_folder=out_folder,
        aoi_extent=extent,
        ncores=cfg.general.ncores,
        config_key=agera5_key,
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
    backend = cfg.general.nasa_power_backend
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
        ncores=cfg.general.ncores,
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
