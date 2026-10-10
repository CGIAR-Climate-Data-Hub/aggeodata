"""
aggeodata.transform.yearly_zarr
================================

Consolidate downloaded daily NetCDFs into one Zarr datacube per year per
variable, and read those cubes back.

A multi-year download leaves tens of thousands of single-day NetCDFs, which is
slow to glob, wasteful on disk (one header per day) and awkward to move around.
This module collapses each year into a single store named

    ``{source}_{aoi}_{cf_var}_{year}.zarr``

for example ``chirps_hnd_test_pr_2010.zarr``, where *aoi* is the stem of the
configuration file that defined the area of interest.  The stores use the same
chunk geometry, codec and integer quantisation as
:mod:`aggeodata.transform.zarr_export`, so they match the published
GeoSPOptimizer archives.

:func:`aggeodata.pipelines.datacube.run_datacube` prefers these stores over
loose daily files when both are present, so a folder can be migrated without
rebuilding anything.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..ingestion.files_manager import find_date_instring, uncompress_zip_path
from .zarr_export import CubeGeometry, export_datacube_to_zarr, open_cube

logger = logging.getLogger(__name__)

__all__ = [
    "yearly_store_name",
    "collect_daily_files",
    "consolidate_variable_to_yearly_zarr",
    "find_yearly_stores",
    "open_yearly_stores",
    "covered_dates",
    "pending_range",
]

#: ``{source}_{aoi}_{cf_var}_{year}.zarr`` — the trailing year is what the
#: reader keys on, so the leading tokens may contain underscores.
_STORE_RE = re.compile(r"^(?P<source>[^_]+)_(?P<rest>.+)_(?P<year>\d{4})\.zarr$")


def yearly_store_name(source: str, aoi: str, cf_var: str, year: int | str) -> str:
    """Return the store filename for one source/AOI/variable/year.

    Parameters
    ----------
    source : str
        Data provider, e.g. ``"chirps"``.
    aoi : str
        Area-of-interest tag — the configuration file stem.
    cf_var : str
        CF variable name, e.g. ``"pr"``.
    year : int | str
        Calendar year.

    Returns
    -------
    str
        e.g. ``"chirps_hnd_test_pr_2010.zarr"``.
    """
    return f"{source}_{aoi}_{cf_var}_{year}.zarr"


# ---------------------------------------------------------------------------
# Daily file discovery
# ---------------------------------------------------------------------------

def collect_daily_files(
    cf_var: str,
    source: str,
    var_folder: str,
    start: str,
    end: str,
) -> dict[str, str]:
    """Return ``{YYYYMMDD: filepath}`` for the daily NetCDFs of one variable.

    Parameters
    ----------
    cf_var : str
        CF variable name — selects the CHIRTS sub-folder (``tmax``/``tmin``).
    source : str
        Data provider; ``"chirts"`` nests per variable and ``"agera5"`` may
        still hold un-extracted yearly zips.
    var_folder : str
        Download folder for this variable.
    start, end : str
        ISO 8601 ``"YYYY-MM-DD"`` bounds, both inclusive.

    Returns
    -------
    dict[str, str]
        Date string to full path, restricted to the requested range.
    """
    ys, ye = int(start[:4]), int(end[:4])
    start_int, end_int = int(start.replace("-", "")), int(end.replace("-", ""))

    # CHIRTS nests files in a variable subfolder (tmax/ or tmin/)
    if source == "chirts":
        chirts_native = "tmax" if cf_var == "tasmax" else "tmin"
        search_root = os.path.join(var_folder, chirts_native)
    else:
        search_root = var_folder

    # AgERA5 CDS downloads arrive as yearly zips; extract so globbing finds them
    if source == "agera5":
        for year in range(ys, ye + 1):
            if glob(os.path.join(search_root, f"*{year}*.zip")):
                uncompress_zip_path(search_root, str(year))

    result: dict[str, str] = {}
    year_prefix = str(ys)[:3]  # e.g. "202"
    for filepath in sorted(glob(os.path.join(search_root, "**", "*.nc"), recursive=True)):
        fname = os.path.basename(filepath)
        try:
            date_str = find_date_instring(fname, pattern=year_prefix)
            if start_int <= int(date_str) <= end_int:
                result[date_str] = filepath
        except Exception:  # noqa: BLE001 — a file we cannot date is not ours
            pass
    return result


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def consolidate_variable_to_yearly_zarr(
    cf_var: str,
    source: str,
    var_folder: str,
    aoi: str,
    start: str,
    end: str,
    *,
    geometry: CubeGeometry | None = None,
    quantize: bool = True,
    codec: str = "blosc",
    remove_daily: bool = True,
    extra_attrs: dict | None = None,
) -> dict[str, str]:
    """Collapse one variable's daily NetCDFs into one Zarr store per year.

    Years already holding a store are skipped, so re-running after a partial
    download only does the new work.  Daily files are removed **only** after
    their year's store has been written and reopened successfully.

    Parameters
    ----------
    cf_var : str
        CF variable name; also the data-variable name inside each store.
    source : str
        Data provider, used as the filename prefix.
    var_folder : str
        Download folder for this variable; stores are written here.
    aoi : str
        Area-of-interest tag — the configuration file stem.
    start, end : str
        ISO 8601 ``"YYYY-MM-DD"`` bounds, both inclusive.
    geometry : CubeGeometry | None
        Chunk geometry; defaults to the reference geometry clamped to the cube.
    quantize : bool
        Store as scaled integers.  Default: True.
    codec : str
        ``"blosc"`` (default) or ``"zstd"``.
    remove_daily : bool
        Delete the daily NetCDFs once their year is consolidated.  Default: True.
    extra_attrs : dict | None
        Extra root-group attributes.

    Returns
    -------
    dict[str, str]
        ``{year: store_path}`` for every year now on disk.

    Notes
    -----
    One year is held in memory at a time, so peak use is roughly
    ``8 * 366 * y * x`` bytes.
    """
    daily = collect_daily_files(cf_var, source, var_folder, start, end)
    stores: dict[str, str] = {
        year: str(path) for year, path in find_yearly_stores(var_folder, cf_var).items()
    }

    by_year: dict[str, dict[str, str]] = {}
    for date_str, path in daily.items():
        by_year.setdefault(date_str[:4], {})[date_str] = path

    for year in sorted(by_year):
        out_path = os.path.join(var_folder, yearly_store_name(source, aoi, cf_var, year))
        days = by_year[year]
        existing = _store_dates(out_path) if os.path.isdir(out_path) else None

        if existing is not None:
            new_days = {d: p for d, p in days.items() if d not in existing}
            if not new_days:
                logger.info("  %s %s: store already covers these %d day(s) — skipping",
                            cf_var, year, len(days))
                stores[year] = out_path
                if remove_daily:
                    _remove_daily(days.values())
                continue
            logger.info("  %s %s: merging %d new day(s) into the existing store",
                        cf_var, year, len(new_days))
            cube = _merge_with_store(out_path, new_days, cf_var)
            expected = len(existing) + len(new_days)
        else:
            logger.info("  %s %s: consolidating %d day(s) -> %s",
                        cf_var, year, len(days), os.path.basename(out_path))
            cube = _stack_days(days, cf_var)
            expected = len(days)

        attrs = {"source_product": source, "aoi": aoi, "year": int(year),
                 **(extra_attrs or {})}
        # Write beside the target and swap, so an interrupted merge cannot
        # leave a half-written store where a complete one used to be.
        tmp_path = out_path + ".tmp"
        shutil.rmtree(tmp_path, ignore_errors=True)
        export_datacube_to_zarr(
            cube, tmp_path, geometry=geometry, quantize=quantize, codec=codec,
            extra_attrs=attrs,
        )
        del cube

        try:
            with open_cube(tmp_path) as check:
                n = int(check.sizes["time"])
        except Exception as exc:  # noqa: BLE001
            logger.error("  %s %s: store failed to reopen (%s) — keeping daily files",
                         cf_var, year, exc)
            shutil.rmtree(tmp_path, ignore_errors=True)
            continue
        if n != expected:
            logger.error("  %s %s: store holds %d day(s), expected %d — keeping daily files",
                         cf_var, year, n, expected)
            shutil.rmtree(tmp_path, ignore_errors=True)
            continue

        shutil.rmtree(out_path, ignore_errors=True)
        os.rename(tmp_path, out_path)
        stores[year] = out_path
        if remove_daily:
            _remove_daily(days.values())

    if remove_daily:
        _prune_empty_dirs(var_folder)
    return stores


def _store_dates(store_path: str) -> set[str] | None:
    """Return the ``YYYYMMDD`` dates held by a store, or None if unreadable."""
    try:
        with open_cube(store_path) as ds:
            return {t.strftime("%Y%m%d") for t in pd.DatetimeIndex(ds["time"].values)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s (%s) — treating it as absent",
                       os.path.basename(store_path), exc)
        return None


def _merge_with_store(store_path: str, new_days: dict[str, str], cf_var: str) -> xr.Dataset:
    """Concatenate an existing year's store with newly downloaded days."""
    import rioxarray  # noqa: F401

    with open_cube(store_path) as prev:
        prev = prev.load()
    rename = {a: b for a, b in (("lat", "y"), ("lon", "x")) if a in prev.dims}
    if rename:
        prev = prev.rename(rename)
    crs = prev.rio.crs
    prev = prev[[cf_var]].drop_vars(["spatial_ref", "crs"], errors="ignore")

    fresh = _stack_days(new_days, cf_var)
    merged = xr.concat([prev, fresh[[cf_var]]], dim="time").sortby("time")
    return merged.rio.write_crs(crs or "EPSG:4326")


def _stack_days(days: dict[str, str], cf_var: str) -> xr.Dataset:
    """Read one year of daily NetCDFs and concatenate them along time."""
    from ..ingestion.gis_functions import read_raster_data

    slices: list[xr.Dataset] = []
    stamps: list[pd.Timestamp] = []
    for date_str in sorted(days):
        ds = read_raster_data(days[date_str])
        # Daily files carry a singleton date/band/time axis from their writer.
        for dim in ("time", "date", "band"):
            if dim in ds.dims:
                ds = ds.isel({dim: 0}, drop=True)
        ds = ds.drop_vars(["spatial_ref", "crs"], errors="ignore")
        data_vars = [v for v in ds.data_vars]
        if not data_vars:
            raise ValueError(f"{days[date_str]} holds no data variable")
        if data_vars[0] != cf_var:
            ds = ds.rename({data_vars[0]: cf_var})
        slices.append(ds[[cf_var]].load())
        stamps.append(pd.Timestamp(date_str))

    cube = xr.concat(slices, dim="time")
    cube["time"] = stamps
    # Daily writers drop the CRS into an undecoded variable; re-assert it so the
    # export resolves a geographic grid rather than falling back to y/x in metres.
    import rioxarray  # noqa: F401

    if cube.rio.crs is None:
        cube = cube.rio.write_crs("EPSG:4326")
    return cube


def _remove_daily(paths) -> None:
    for p in paths:
        try:
            os.remove(p)
        except OSError as exc:  # noqa: PERF203
            logger.debug("could not remove %s: %s", p, exc)


def _prune_empty_dirs(root: str) -> None:
    """Remove directories left empty after the daily files were deleted."""
    for dirpath, _, _ in sorted(
        (w for w in os.walk(root)), key=lambda w: len(w[0]), reverse=True
    ):
        if dirpath == root or dirpath.endswith(".zarr"):
            continue
        if ".zarr" in Path(dirpath).parts[-3:]:
            continue
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def find_yearly_stores(var_folder: str, cf_var: str | None = None) -> dict[str, Path]:
    """Return ``{year: store_path}`` for the yearly Zarr cubes in *var_folder*.

    Parameters
    ----------
    var_folder : str
        Download folder for one variable.
    cf_var : str | None
        Keep only stores whose variable token matches.  ``None`` accepts any.

    Returns
    -------
    dict[str, Path]
    """
    out: dict[str, Path] = {}
    if not os.path.isdir(var_folder):
        return out
    for entry in sorted(Path(var_folder).glob("*.zarr")):
        if not entry.is_dir():
            continue
        m = _STORE_RE.match(entry.name)
        if not m:
            continue
        if cf_var is not None and not m.group("rest").endswith(f"_{cf_var}"):
            continue
        out[m.group("year")] = entry
    return out


def covered_dates(var_folder: str, cf_var: str, start: str, end: str) -> set[str]:
    """Return the ``YYYYMMDD`` dates in ``[start, end]`` already held by stores."""
    out: set[str] = set()
    start_int, end_int = int(start.replace("-", "")), int(end.replace("-", ""))
    for path in find_yearly_stores(var_folder, cf_var).values():
        dates = _store_dates(str(path))
        if dates:
            out |= {d for d in dates if start_int <= int(d) <= end_int}
    return out


def pending_range(
    var_folder: str, cf_var: str, start: str, end: str
) -> tuple[str, str] | None:
    """Return the sub-range still to download, or None if fully covered.

    With ``download_format: zarr`` the daily files are deleted after each year
    is consolidated, so the downloaders' own "is the file already on disk?"
    resume check always misses.  Narrowing the request to the span that the
    yearly stores do not already cover restores it — re-running an unchanged
    config becomes a no-op, and extending the date range only fetches the new
    days.

    The span is contiguous: an interior gap re-fetches the days around it,
    which the merge step then folds into the existing store.

    Parameters
    ----------
    var_folder : str
        Download folder for this variable.
    cf_var : str
        CF variable name.
    start, end : str
        ISO 8601 ``"YYYY-MM-DD"`` bounds, both inclusive.

    Returns
    -------
    tuple[str, str] | None
        ``(start, end)`` of the uncovered span, or None when nothing is left.
    """
    have = covered_dates(var_folder, cf_var, start, end)
    if not have:
        return start, end
    wanted = pd.date_range(start, end, freq="D")
    missing = [t for t in wanted if t.strftime("%Y%m%d") not in have]
    if not missing:
        return None
    return missing[0].strftime("%Y-%m-%d"), missing[-1].strftime("%Y-%m-%d")


def open_yearly_stores(
    stores: dict[str, Path], start: str, end: str
) -> tuple[dict[str, xr.Dataset], list[xr.Dataset]]:
    """Open yearly cubes and map every in-range date to its store.

    Parameters
    ----------
    stores : dict[str, Path]
        ``{year: store_path}`` from :func:`find_yearly_stores`.
    start, end : str
        ISO 8601 ``"YYYY-MM-DD"`` bounds, both inclusive.

    Returns
    -------
    tuple
        ``({YYYYMMDD: dataset}, [opened datasets])`` — the second element is
        for the caller to close.
    """
    start_int, end_int = int(start.replace("-", "")), int(end.replace("-", ""))
    by_date: dict[str, xr.Dataset] = {}
    opened: list[xr.Dataset] = []

    for year in sorted(stores):
        ds = open_cube(stores[year])
        # Match the dimension names the per-day reader produces.
        rename = {a: b for a, b in (("lat", "y"), ("lon", "x"),
                                    ("latitude", "y"), ("longitude", "x"))
                  if a in ds.dims}
        if rename:
            ds = ds.rename(rename)
        opened.append(ds)
        for t in pd.DatetimeIndex(ds["time"].values):
            date_str = t.strftime("%Y%m%d")
            if start_int <= int(date_str) <= end_int:
                by_date[date_str] = ds
    return by_date, opened
