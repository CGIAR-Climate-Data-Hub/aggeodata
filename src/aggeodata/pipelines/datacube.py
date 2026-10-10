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
from ..transform.yearly_zarr import (
    collect_daily_files,
    find_yearly_stores,
    open_yearly_stores,
)

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
    start  = cfg.dates.starting_date
    end    = cfg.dates.ending_date
    ref_var = cfg.general.reference_variable
    suffix  = cfg.general.suffix

    logger.info("Building datacube  |  ref=%s  |  %s -> %s", ref_var, start, end)

    # ------------------------------------------------------------------
    # 1. Collect date → filepath for every variable
    # ------------------------------------------------------------------
    var_files: dict[str, dict[str, str]] = {}
    # Variables served from an already-open multi-day Dataset rather than one
    # file per date: NASA POWER's single NetCDF, and yearly Zarr cubes.
    multiday_ds: dict[str, dict[str, xr.Dataset]] = {}
    opened_stores: list[xr.Dataset] = []

    for cf_var, var_cfg in cfg.climate.variables.items():
        var_folder = cfg.var_folder(cf_var)
        source = var_cfg.source

        if not os.path.isdir(var_folder):
            logger.warning("Folder missing for %s (%s) — skipping", cf_var, var_folder)
            continue

        # Yearly Zarr cubes (written by run_download with download_format=zarr)
        # take precedence over loose daily files.
        stores = find_yearly_stores(var_folder, cf_var)
        if stores:
            by_date, opened = open_yearly_stores(stores, start, end)
            if by_date:
                multiday_ds[cf_var] = by_date
                var_files[cf_var] = {d: str(stores[d[:4]]) for d in by_date}
                opened_stores.extend(opened)
                logger.info("  %-10s  %d days from %d yearly Zarr cube(s)",
                            cf_var, len(by_date), len(stores))
                continue
            for ds in opened:
                ds.close()
            logger.warning("Yearly Zarr cubes for %s hold no dates in range", cf_var)

        if source == "nasa_power":
            nc_files = glob(os.path.join(var_folder, "nasa_power_*.nc"))
            if not nc_files:
                logger.warning("No NASA POWER NC found in %s", var_folder)
                continue
            ds = xr.open_dataset(nc_files[0])
            # normalise time coord name
            if "time" not in ds.dims and "date" in ds.dims:
                ds = ds.rename({"date": "time"})
            # Build date-set for common-date intersection; handle cftime safely
            import pandas as pd
            try:
                time_vals = pd.to_datetime(ds.time.values)
            except Exception:
                time_vals = pd.DatetimeIndex(
                    [pd.Timestamp(str(t)) for t in ds.time.values]
                )
            dates = {t.strftime("%Y%m%d"): nc_files[0] for t in time_vals}
            multiday_ds[cf_var] = {d: ds for d in dates}
            var_files[cf_var] = dates
        else:
            files = collect_daily_files(cf_var, source, var_folder, start, end)
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
    # 3. Build per-date datasets and stack in batches
    # Processing all dates at once causes linear memory growth (every dataset
    # stays in RAM until the final xr.concat). Batching keeps peak memory
    # proportional to batch_size × n_vars × spatial_pixels instead of
    # n_dates × n_vars × spatial_pixels.
    # ------------------------------------------------------------------
    from ..ingestion.utils import resample_variables
    from ..ingestion.gis_functions import read_raster_data

    ncores     = cfg.general.ncores
    target_crs = cfg.general.target_crs
    batch_size = max(ncores * 8, 30)   # process ~30 dates at a time, free RAM between batches

    def _process_date(date_str: str) -> tuple[str, xr.Dataset]:
        ds = _build_single_date(
            date_str=date_str,
            var_files=var_files,
            multiday_ds=multiday_ds,
            ref_var=ref_var,
            resample_variables_fn=resample_variables,
            read_raster_data_fn=read_raster_data,
            target_crs=target_crs,
        )
        return date_str, ds

    batch_slabs: list[xr.Dataset] = []   # one concat per batch — tiny list
    succeeded_dates: list[str]    = []
    n_skipped = 0

    batches = [common_dates[i:i + batch_size] for i in range(0, len(common_dates), batch_size)]

    with tqdm(total=len(common_dates), desc="Stacking dates", unit="day") as pbar:
        for batch in batches:
            batch_results: dict[str, xr.Dataset] = {}

            with ThreadPoolExecutor(max_workers=ncores) as pool:
                futures = {pool.submit(_process_date, d): d for d in batch}
                for future in as_completed(futures):
                    date_str = futures[future]
                    try:
                        _, ds_date = future.result()
                        batch_results[date_str] = ds_date
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Skipping date %s: %s", date_str, exc)
                        n_skipped += 1
                    pbar.update(1)

            # Concat this batch in chronological order and free per-date datasets
            batch_dates = [d for d in batch if d in batch_results]
            if batch_dates:
                slab = xr.concat([batch_results[d] for d in batch_dates], dim="time")
                slab = slab.load()   # materialise before freeing per-date refs
                batch_slabs.append(slab)
                succeeded_dates.extend(batch_dates)
                del batch_results   # release per-date datasets immediately

    # Every slab is materialised, so the source stores can be released now.
    for _store in opened_stores:
        try:
            _store.close()
        except Exception:  # noqa: BLE001
            pass

    if not batch_slabs:
        raise RuntimeError("All dates failed during loading. Check warnings above.")

    if n_skipped:
        logger.warning("%d dates skipped — check warnings above", n_skipped)

    # Final concat across batch slabs (small list — one entry per batch)
    per_date = batch_slabs

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
    # 5. Upsample to target resolution (optional)
    # ------------------------------------------------------------------
    if cfg.general.target_resolution is not None:
        res = cfg.general.target_resolution
        cube = cube.rio.reproject(cfg.general.target_crs, resolution=res)
        logger.info("Resampled to %.4f° -> shape %s", res, dict(cube.sizes))

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    ys = start[:4]
    ye = end[:4]
    stem = f"climate_{suffix}_{ys}_{ye}" if suffix else f"climate_{ys}_{ye}"
    Path(cfg.paths.output_path).mkdir(parents=True, exist_ok=True)
    fmt = cfg.general.output_format

    zarr_path: str | None = None
    if fmt in {"zarr", "both"}:
        zarr_path = _save_zarr(cube, os.path.join(cfg.paths.output_path, stem + ".zarr"), cfg)

    if fmt == "zarr":
        _log_cube_summary(cube, zarr_path)
        return zarr_path

    out_path = os.path.join(cfg.paths.output_path, stem + ".nc")
    data_vars = [v for v in cube.data_vars if v != "spatial_ref"]
    encoding = {v: {"zlib": True, "complevel": 4} for v in data_vars}
    cube.to_netcdf(out_path, encoding=encoding, engine="netcdf4")
    logger.info("Datacube saved -> %s", out_path)

    _log_cube_summary(cube, out_path)
    if zarr_path:
        print(f"  Zarr   : {zarr_path}")
    return out_path


def _save_zarr(cube: xr.Dataset, path: str, cfg: AgeodataConfig) -> str:
    """Write the cube as a Zarr v3 store using the configured geometry."""
    from ..transform.zarr_export import CubeGeometry, export_datacube_to_zarr

    chunks = cfg.general.zarr_chunks
    geometry = CubeGeometry(sharded=cfg.general.zarr_sharded)
    if chunks:
        t, y, x = chunks
        geometry = CubeGeometry(
            time_chunk=t, lat_chunk=y, lon_chunk=x,
            time_shard=geometry.time_shard, lat_shard=geometry.lat_shard,
            lon_shard=geometry.lon_shard, sharded=cfg.general.zarr_sharded,
        )

    logger.info("Writing Zarr datacube -> %s", path)
    export_datacube_to_zarr(
        cube, path,
        geometry=geometry,
        quantize=cfg.general.zarr_quantize,
        codec=cfg.general.zarr_codec,
    )
    return path


# ---------------------------------------------------------------------------
# Single-date dataset builder
# ---------------------------------------------------------------------------

def _build_single_date(
    date_str: str,
    var_files: dict[str, dict[str, str]],
    multiday_ds: dict[str, dict[str, xr.Dataset]],
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

        if cf_var in multiday_ds:
            import rioxarray  # noqa: F401 — registers the .rio accessor

            ds = multiday_ds[cf_var][date_str]
            ts = datetime.strptime(date_str, "%Y%m%d")
            slice_ds = ds.sel(time=str(ts.date()), method="nearest")
            # Drop time dim so resample_variables sees a 2D field
            if "time" in slice_ds.dims:
                slice_ds = slice_ds.isel(time=0, drop=True)
            # Read the CRS before dropping spatial_ref: a yearly Zarr cube is
            # opened with decode_coords="all", which makes spatial_ref a
            # coordinate, so dropping it would take the CRS with it.
            crs = slice_ds.rio.crs
            slice_ds = slice_ds.drop_vars(["spatial_ref", "crs"], errors="ignore")
            data_vars = [v for v in slice_ds.data_vars]
            if data_vars and cf_var not in data_vars:
                slice_ds = slice_ds.rename({data_vars[0]: cf_var})
            slice_ds = slice_ds.rio.write_crs(crs or "EPSG:4326")
            dict_xr[cf_var] = slice_ds.load()
        else:
            ds = read_raster_data_fn(filepath)
            # Drop any extra time/band dim — keep spatial only
            for dim in ("time", "date", "band"):
                if dim in ds.dims:
                    ds = ds.isel({dim: 0}, drop=True)
            # AgERA5 NetCDFs have no embedded CRS — write EPSG:4326 before resampling
            if ds.rio.crs is None:
                ds = ds.rio.write_crs("EPSG:4326")
            # Rename internal variable to CF name to prevent duplicates in the merged cube
            data_vars = [v for v in ds.data_vars if v not in ("crs", "spatial_ref")]
            if data_vars and data_vars[0] != cf_var:
                ds = ds.rename({data_vars[0]: cf_var})
            dict_xr[cf_var] = ds

    merged = resample_variables_fn(dict_xr, reference_variable=ref_var, target_crs=target_crs)
    return merged


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_cube_summary(cube: xr.Dataset, path: str) -> None:
    # A Zarr store is a directory, so sum it rather than stat it.
    if os.path.isdir(path):
        size_mb = sum(
            f.stat().st_size for f in Path(path).rglob("*") if f.is_file()
        ) / 1e6
    else:
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
