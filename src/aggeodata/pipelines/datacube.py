"""
aggeodata.pipelines.datacube
=============================

YAML-driven datacube stacking pipeline.

Reads the same config used by ``run_download``, discovers all downloaded
files, and assembles a spatially co-registered multi-variable temporal cube
saved as a compressed NetCDF.

Usage
-----
    from aggeodata.pipelines import run_datacube
    path = run_datacube("options/aggeodata_config.yaml")

Or from the command line:
    python -m aggeodata.pipelines.datacube options/aggeodata_config.yaml

Expected file layout (created by run_download):
    <output_path>/<cf_var>_<suffix>_raw/
      CHIRPS    → <year>/chirps_pr_YYYYMMDD.nc
      CHIRTS    → tmax/<year>/chirts_tmax_YYYYMMDD.nc
      NASA POWER→ nasa_power_<start>_<end>.nc   (single multi-day file)
      AgERA5    → <year>.zip  (extracted automatically)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..config.loader import load_config
from ..config.schemas import AgeodataConfig
from ..ingestion.files_manager import find_date_instring, uncompress_zip_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_datacube(config_path: str | os.PathLike) -> str:
    """Build a multi-variable temporal datacube from downloaded files.

    Discovers all downloaded files for each variable in the config, finds
    common dates, resamples every variable to the reference variable's spatial
    grid, and concatenates along the time dimension.

    Parameters
    ----------
    config_path : str | os.PathLike
        Path to the aggeodata YAML configuration file.

    Returns
    -------
    str
        Path to the saved NetCDF datacube.
    """
    cfg = load_config(config_path)
    start  = cfg.DATES.starting_date
    end    = cfg.DATES.ending_date
    ref_var = cfg.GENERAL.reference_variable
    suffix  = cfg.GENERAL.suffix

    logger.info("Building datacube  |  ref=%s  |  %s -> %s", ref_var, start, end)

    # ------------------------------------------------------------------
    # 1. Collect date → filepath for every variable
    # ------------------------------------------------------------------
    var_files: dict[str, dict[str, str]] = {}
    nasa_power_ds: dict[str, xr.Dataset] = {}   # multi-day NC opened once

    for cf_var, var_cfg in cfg.CLIMATE.variables.items():
        var_folder = cfg.var_folder(cf_var)
        source = var_cfg.source

        if not os.path.isdir(var_folder):
            logger.warning("Folder missing for %s (%s) — skipping", cf_var, var_folder)
            continue

        if source == "nasa_power":
            nc_files = glob(os.path.join(var_folder, "nasa_power_*.nc"))
            if not nc_files:
                logger.warning("No NASA POWER NC found in %s", var_folder)
                continue
            ds = xr.open_dataset(nc_files[0])
            # normalise time coord name
            if "time" not in ds.dims and "date" in ds.dims:
                ds = ds.rename({"date": "time"})
            nasa_power_ds[cf_var] = ds
            # Build date-set for common-date intersection; handle cftime safely
            import pandas as pd
            try:
                time_vals = pd.to_datetime(ds.time.values)
            except Exception:
                time_vals = pd.DatetimeIndex(
                    [pd.Timestamp(str(t)) for t in ds.time.values]
                )
            dates = {t.strftime("%Y%m%d"): nc_files[0] for t in time_vals}
            var_files[cf_var] = dates
        else:
            files = _collect_per_day_files(cf_var, source, var_folder, start, end)
            if not files:
                logger.warning("No files found for %s in %s", cf_var, var_folder)
                continue
            var_files[cf_var] = files
            logger.info("  %-10s  %d days found", cf_var, len(files))

    if not var_files:
        raise RuntimeError("No variable files found. Run run_download() first.")

    # ------------------------------------------------------------------
    # 2. Common dates across all variables
    # ------------------------------------------------------------------
    date_sets = [set(files.keys()) for files in var_files.values()]
    common_dates = sorted(set.intersection(*date_sets))

    if not common_dates:
        raise RuntimeError(
            "No common dates found across all variables. "
            "Check that all variables are downloaded for the same period."
        )
    logger.info("%d common dates (%s … %s)", len(common_dates), common_dates[0], common_dates[-1])

    # ------------------------------------------------------------------
    # 3. Build per-date datasets and stack
    # ------------------------------------------------------------------
    from ..ingestion.utils import resample_variables
    from ..ingestion.gis_functions import read_raster_data

    ncores = cfg.GENERAL.ncores
    target_crs = cfg.GENERAL.target_crs

    def _process_date(date_str: str) -> tuple[str, xr.Dataset]:
        ds = _build_single_date(
            date_str=date_str,
            var_files=var_files,
            nasa_power_ds=nasa_power_ds,
            ref_var=ref_var,
            resample_variables_fn=resample_variables,
            read_raster_data_fn=read_raster_data,
            target_crs=target_crs,
        )
        return date_str, ds

    results: dict[str, xr.Dataset] = {}

    with ThreadPoolExecutor(max_workers=ncores) as pool:
        futures = {pool.submit(_process_date, d): d for d in common_dates}
        for future in tqdm(
            as_completed(futures), total=len(common_dates),
            desc="Stacking dates", unit="day",
        ):
            date_str = futures[future]
            try:
                _, ds_date = future.result()
                results[date_str] = ds_date
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping date %s: %s", date_str, exc)

    # Reassemble in original chronological order
    per_date      = [results[d] for d in common_dates if d in results]
    succeeded_dates = [d for d in common_dates if d in results]

    if not per_date:
        raise RuntimeError("All dates failed during loading. Check warnings above.")

    # ------------------------------------------------------------------
    # 4. Concatenate along time
    # ------------------------------------------------------------------
    import rioxarray  # noqa: F401 — registers .rio accessor

    logger.info("Concatenating %d dates ...", len(per_date))
    timestamps = [datetime.strptime(d, "%Y%m%d") for d in succeeded_dates]
    cube = xr.concat(per_date, dim="time")
    cube["time"] = timestamps

    # ------------------------------------------------------------------
    # 4b. Fix CRS encoding
    # After xr.concat, the scalar `spatial_ref` coordinate from each
    # per-date dataset is demoted to a data variable and its attributes
    # (including crs_wkt) are lost, breaking QGIS georeferencing.
    # Re-establish a clean CF-compliant CRS on the concatenated cube.
    # ------------------------------------------------------------------
    # Remove spatial_ref if concat turned it into a data variable
    if "spatial_ref" in cube.data_vars:
        cube = cube.drop_vars("spatial_ref")

    # Drop any other leftover grid-mapping coordinate variables
    gm_var_names = [
        name for name, var in cube.variables.items()
        if "grid_mapping_name" in var.attrs
    ]
    if gm_var_names:
        cube = cube.drop_vars(gm_var_names, errors="ignore")

    # Wipe stale grid_mapping pointers from attrs and encoding
    for name in list(cube.variables):
        cube.variables[name].attrs.pop("grid_mapping", None)
        cube.variables[name].encoding.pop("grid_mapping", None)

    # Write a fresh spatial_ref coordinate with crs_wkt
    cube = cube.rio.write_crs(target_crs)

    # Explicitly set grid_mapping on all climate data variables.
    # rioxarray's write_crs does this in-memory but the attribute can be
    # missing after to_netcdf round-trips or when concat shuffles variables.
    for var in list(cube.data_vars):
        cube[var].attrs["grid_mapping"] = "spatial_ref"

    # Add CF axis/units to spatial coordinate variables so QGIS can
    # georeference even without parsing the grid_mapping variable.
    x_dim = cube.rio.x_dim
    y_dim = cube.rio.y_dim
    if cube.rio.crs and cube.rio.crs.is_geographic:
        if x_dim in cube.coords:
            cube[x_dim].attrs.update({
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
                "axis": "X",
            })
        if y_dim in cube.coords:
            cube[y_dim].attrs.update({
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
                "axis": "Y",
            })

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    ys = start[:4]
    ye = end[:4]
    fname = f"climate_{suffix}_{ys}_{ye}.nc" if suffix else f"climate_{ys}_{ye}.nc"
    out_path = os.path.join(cfg.PATHS.output_path, fname)
    Path(cfg.PATHS.output_path).mkdir(parents=True, exist_ok=True)

    data_vars = [v for v in cube.data_vars if v != "spatial_ref"]
    encoding = {v: {"zlib": True, "complevel": 4} for v in data_vars}
    cube.to_netcdf(out_path, encoding=encoding, engine="netcdf4")
    logger.info("Datacube saved -> %s", out_path)

    _log_cube_summary(cube, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Per-day file collection
# ---------------------------------------------------------------------------

def _collect_per_day_files(
    cf_var: str,
    source: str,
    var_folder: str,
    start: str,
    end: str,
) -> dict[str, str]:
    """Return {date_str: full_filepath} for daily NetCDF files."""
    ys = int(start[:4])
    ye = int(end[:4])
    start_int = int(start.replace("-", ""))
    end_int   = int(end.replace("-", ""))

    # CHIRTS nests files in a variable subfolder (tmax/ or tmin/)
    if source == "chirts":
        chirts_native = "tmax" if cf_var == "tasmax" else "tmin"
        search_root = os.path.join(var_folder, chirts_native)
    else:
        search_root = var_folder

    # For AgERA5 extract zip archives first so globbing finds the daily NCs
    if source == "agera5":
        for year in range(ys, ye + 1):
            zip_files = glob(os.path.join(search_root, f"*{year}*.zip"))
            if zip_files:
                uncompress_zip_path(search_root, str(year))

    nc_files = sorted(glob(os.path.join(search_root, "**", "*.nc"), recursive=True))

    result: dict[str, str] = {}
    year_prefix = str(ys)[:3]          # e.g. "202"
    for filepath in nc_files:
        fname = os.path.basename(filepath)
        try:
            date_str = find_date_instring(fname, pattern=year_prefix)
            date_int = int(date_str)
            if start_int <= date_int <= end_int:
                result[date_str] = filepath
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Single-date dataset builder
# ---------------------------------------------------------------------------

def _build_single_date(
    date_str: str,
    var_files: dict[str, dict[str, str]],
    nasa_power_ds: dict[str, xr.Dataset],
    ref_var: str,
    resample_variables_fn,
    read_raster_data_fn,
    target_crs: str | None = None,
) -> xr.Dataset:
    """Load all variables for one date and resample to the reference grid."""
    dict_xr: dict[str, xr.Dataset] = {}

    for cf_var, files in var_files.items():
        filepath = files.get(date_str)
        if filepath is None:
            raise KeyError(f"No file for {cf_var} on {date_str}")

        if cf_var in nasa_power_ds:
            ds = nasa_power_ds[cf_var]
            ts = datetime.strptime(date_str, "%Y%m%d")
            slice_ds = ds.sel(time=str(ts.date()), method="nearest")
            # Drop time dim so resample_variables sees a 2D field
            if "time" in slice_ds.dims:
                slice_ds = slice_ds.isel(time=0, drop=True)
            dict_xr[cf_var] = slice_ds
        else:
            ds = read_raster_data_fn(filepath)
            # Drop any extra time/band dim — keep spatial only
            for dim in ("time", "date", "band"):
                if dim in ds.dims:
                    ds = ds.isel({dim: 0}, drop=True)
            # AgERA5 NetCDFs have no embedded CRS — write EPSG:4326 before resampling
            if ds.rio.crs is None:
                ds = ds.rio.write_crs("EPSG:4326")
            dict_xr[cf_var] = ds

    merged = resample_variables_fn(dict_xr, reference_variable=ref_var, target_crs=target_crs)
    return merged


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_cube_summary(cube: xr.Dataset, path: str) -> None:
    size_mb = os.path.getsize(path) / 1e6
    print("\n" + "=" * 56)
    print("  DATACUBE SUMMARY")
    print("=" * 56)
    print(f"  File   : {path}  ({size_mb:.1f} MB)")
    print(f"  Shape  : {dict(cube.dims)}")
    print(f"  Vars   : {list(cube.data_vars)}")
    t = cube.indexes.get("time")
    if t is not None and len(t):
        print(f"  Period : {t[0]} … {t[-1]}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m aggeodata.pipelines.datacube <config.yaml>")
        sys.exit(1)
    run_datacube(sys.argv[1])
