"""
aggeodata.ingestion.hf_zarr
============================

Hugging Face Zarr backend shared by the CHIRPS and CHIRTS downloaders.

The official UCSB endpoints serve one GeoTIFF per day, which makes long time
series slow to retrieve.  The same archives are mirrored as consolidated Zarr
datacubes on Hugging Face, where a bounding box + date range is a single
indexed read:

* CHIRPS v3 daily  — ``aaguilar90/chirps-v3-daily-rnl``   (variable ``precip``)
* CHIRTS-ERA5 daily — ``aaguilar90/chirts-era5-daily``    (variables ``Tmax``, ``Tmin``)

Both stores are global, 0.05 deg, EPSG:4326, with ``(time, lat, lon)`` dims and
descending latitude.

This module slices a store and writes the result as **one NetCDF per day**, in
exactly the same layout the UCSB downloaders produce, so
:func:`aggeodata.pipelines.datacube.run_datacube` consumes either backend
without changes.

Rule: this module ONLY downloads files to disk.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "HF_CHIRPS_URL",
    "HF_CHIRTS_URL",
    "open_hf_zarr",
    "subset_hf_zarr",
    "download_daily_from_hf",
]

HF_CHIRPS_URL: str = (
    "https://huggingface.co/datasets/aaguilar90/chirps-v3-daily-rnl/"
    "resolve/main/chirps-v3-daily-rnl.zarr"
)
HF_CHIRTS_URL: str = (
    "https://huggingface.co/datasets/aaguilar90/chirts-era5-daily/"
    "resolve/main/chirts-era5-daily.zarr"
)

_NC_ENCODING = {"zlib": True, "complevel": 4}

# Each worker holds one loaded time-batch in memory; keep a batch small enough
# that ncores batches fit comfortably in RAM.
_TARGET_BATCH_BYTES = 256 * 1024**2
_MAX_BATCH_DAYS = 128  # matches the stores' time chunk size


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------

def open_hf_zarr(url: str) -> xr.Dataset:
    """Open a consolidated Zarr store hosted on Hugging Face.

    Parameters
    ----------
    url : str
        Full ``https://huggingface.co/datasets/.../resolve/main/<name>.zarr`` URL.

    Returns
    -------
    xarray.Dataset
        Lazily-opened dataset (no data transferred yet).
    """
    try:
        import zarr  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The Hugging Face backend requires zarr: pip install 'aggeodata[download]'"
        ) from exc

    logger.info("HF Zarr: opening %s", url)
    return xr.open_zarr(url, consolidated=True, chunks={})


def subset_hf_zarr(
    ds: xr.Dataset,
    variables: list[str],
    extent: list[float],
    starting_date: str,
    ending_date: str,
) -> xr.Dataset:
    """Slice *ds* to the requested variables, bounding box and date range.

    Handles descending latitude (and descending longitude, should a future
    store use it) so the caller can always pass ``[xmin, ymin, xmax, ymax]``.

    Parameters
    ----------
    ds : xarray.Dataset
        Store opened by :func:`open_hf_zarr`.
    variables : list[str]
        Variable names as they appear in the store (e.g. ``["Tmax"]``).
    extent : list[float]
        ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
    starting_date, ending_date : str
        ISO 8601 ``"YYYY-MM-DD"`` bounds, both inclusive.

    Returns
    -------
    xarray.Dataset
        Lazy subset containing only *variables*.

    Raises
    ------
    ValueError
        If a variable is absent from the store, or the extent / date range
        falls outside its coverage.
    """
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Variables {missing} are not in the Hugging Face store. "
            f"Available: {sorted(str(v) for v in ds.data_vars)}"
        )

    xmin, ymin, xmax, ymax = extent
    lat = ds["lat"].values
    lon = ds["lon"].values
    lat_slice = slice(ymax, ymin) if lat[0] > lat[-1] else slice(ymin, ymax)
    lon_slice = slice(xmax, xmin) if lon[0] > lon[-1] else slice(xmin, xmax)

    sub = ds[variables].sel(
        time=slice(starting_date, ending_date),
        lat=lat_slice,
        lon=lon_slice,
    )

    if sub.sizes["time"] == 0:
        t = ds["time"].values
        raise ValueError(
            f"No dates between {starting_date} and {ending_date} in the Hugging Face store "
            f"(coverage: {pd.Timestamp(t[0]).date()} to {pd.Timestamp(t[-1]).date()})."
        )
    if sub.sizes["lat"] == 0 or sub.sizes["lon"] == 0:
        raise ValueError(
            f"Extent {extent} does not intersect the Hugging Face store "
            f"(lat {lat.min():.2f}..{lat.max():.2f}, lon {lon.min():.2f}..{lon.max():.2f})."
        )

    # The store's grid-mapping variable is re-written per day on export.
    return sub.drop_vars("spatial_ref", errors="ignore")


# ---------------------------------------------------------------------------
# Daily NetCDF export
# ---------------------------------------------------------------------------

def download_daily_from_hf(
    url: str,
    variables: dict[str, str],
    extent: list[float],
    starting_date: str,
    ending_date: str,
    path_for: Callable[[str, str], str],
    ncores: int = 1,
    batch_days: int | None = None,
    desc: str = "HF Zarr",
) -> dict[str, list[str]]:
    """Slice a Hugging Face Zarr store and write one NetCDF per day.

    Days whose output file already exists are skipped, so an interrupted run
    resumes where it stopped.  Each file is written to a temporary name and
    atomically renamed, so a cancelled run never leaves a truncated NetCDF
    behind for the datacube stage to find.

    Parameters
    ----------
    url : str
        Hugging Face Zarr URL.
    variables : dict[str, str]
        ``{store_variable: output_variable}`` — e.g. ``{"Tmax": "tasmax"}``.
        The output name is both the NetCDF variable name and the key passed to
        *path_for*.
    extent : list[float]
        ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
    starting_date, ending_date : str
        ISO 8601 ``"YYYY-MM-DD"`` bounds, both inclusive.
    path_for : callable
        ``path_for(output_variable, "YYYYMMDD") -> output NetCDF path``.
        Parent folders are created automatically.
    ncores : int
        Number of time-batches fetched concurrently.  Default: 1.
    batch_days : int | None
        Days loaded per request.  ``None`` (default) sizes batches from the
        AOI so a batch stays near 256 MB, capped at the store's 128-day chunk.
    desc : str
        Progress-bar label.

    Returns
    -------
    dict[str, list[str]]
        ``{output_variable: [written NetCDF paths]}`` (sorted by date).
    """
    from tqdm import tqdm

    ds = open_hf_zarr(url)
    sub = subset_hf_zarr(ds, list(variables), extent, starting_date, ending_date)

    time_index = pd.DatetimeIndex(sub["time"].values)
    date_strs = [t.strftime("%Y%m%d") for t in time_index]

    # Resume: keep only days with at least one output file still missing.
    pending: list[tuple[int, str]] = []
    cached_sample: str | None = None
    for i, d in enumerate(date_strs):
        for out_var in variables.values():
            path = path_for(out_var, d)
            if os.path.exists(path):
                cached_sample = cached_sample or path
            else:
                pending.append((i, d))
                break

    n_cached = len(date_strs) - len(pending)
    if n_cached:
        logger.info("%s: %d/%d days already on disk", desc, n_cached, len(date_strs))
    if cached_sample is not None:
        _warn_on_grid_mismatch(cached_sample, sub, desc)

    written: dict[str, list[str]] = {out_var: [] for out_var in variables.values()}
    if not pending:
        return written

    if batch_days is None:
        batch_days = _auto_batch_days(sub, len(variables))
    logger.info(
        "%s: fetching %d days in batches of %d (%d workers)",
        desc, len(pending), batch_days, max(ncores, 1),
    )

    batches = [pending[i:i + batch_days] for i in range(0, len(pending), batch_days)]

    def _load(batch: list[tuple[int, str]]) -> xr.Dataset:
        return sub.isel(time=[i for i, _ in batch]).load()

    pbar = tqdm(total=len(pending), desc=desc, unit="day")
    try:
        with ThreadPoolExecutor(max_workers=max(ncores, 1)) as pool:
            futures = {pool.submit(_load, b): b for b in batches}
            # Writes stay on the main thread: HDF5/netCDF4 is not thread-safe.
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    loaded = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "%s: batch %s..%s failed: %s", desc, batch[0][1], batch[-1][1], exc
                    )
                    pbar.update(len(batch))
                    continue

                for pos, (_, date_str) in enumerate(batch):
                    day = loaded.isel(time=pos, drop=True)
                    for store_var, out_var in variables.items():
                        try:
                            path = _write_day(day, store_var, out_var, date_str, path_for, url)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("%s: %s %s failed: %s", desc, out_var, date_str, exc)
                            continue
                        if path is not None:
                            written[out_var].append(path)
                    pbar.update(1)
                del loaded
    finally:
        pbar.close()

    for out_var in written:
        written[out_var].sort()
    return written


def _warn_on_grid_mismatch(existing_path: str, sub: xr.Dataset, desc: str) -> None:
    """Warn when files already in the folder sit on a different grid.

    The UCSB backend clips with rasterio, which can keep one extra pixel along
    an edge that the Zarr coordinate slice drops.  Mixing both backends in one
    variable folder therefore yields per-day files of differing shape, and
    ``run_datacube`` concatenates those along time into a NaN-padded union.
    Re-download the folder with a single backend instead.
    """
    try:
        with xr.open_dataset(existing_path) as existing:
            same = (
                existing.sizes.get("y") == sub.sizes["lat"]
                and existing.sizes.get("x") == sub.sizes["lon"]
            )
            shape = (existing.sizes.get("y"), existing.sizes.get("x"))
    except Exception:  # noqa: BLE001 — a diagnostic must never break the download
        return

    if not same:
        logger.warning(
            "%s: %s is %s but this request produces (%d, %d). The folder mixes files "
            "from the UCSB and Hugging Face backends, whose edge pixels differ; "
            "the datacube step will pad the mismatch with NaNs. Delete the folder "
            "and re-download it with one backend.",
            desc, os.path.basename(existing_path), shape,
            sub.sizes["lat"], sub.sizes["lon"],
        )


def _auto_batch_days(sub: xr.Dataset, n_vars: int) -> int:
    """Pick a per-request day count that keeps one batch near the memory target."""
    pixels = sub.sizes["lat"] * sub.sizes["lon"] * max(n_vars, 1)
    bytes_per_day = max(pixels * 8, 1)  # stores are float64
    return max(1, min(_MAX_BATCH_DAYS, int(_TARGET_BATCH_BYTES // bytes_per_day)))


def _write_day(
    day: xr.Dataset,
    store_var: str,
    out_var: str,
    date_str: str,
    path_for: Callable[[str, str], str],
    url: str,
) -> str | None:
    """Write one variable for one date; return the path, or None if cached."""
    import rioxarray  # noqa: F401 — registers the .rio accessor

    out_path = path_for(out_var, date_str)
    if os.path.exists(out_path):
        return None

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    out = (
        day[store_var]
        .rename({"lat": "y", "lon": "x"})
        .to_dataset(name=out_var)
        .drop_vars("spatial_ref", errors="ignore")
    )
    out = out.rio.write_crs("EPSG:4326")
    out[out_var].attrs["grid_mapping"] = "spatial_ref"
    out.attrs = {"source": url, "date": date_str}

    tmp_path = f"{out_path}.part"
    out.to_netcdf(tmp_path, encoding={out_var: dict(_NC_ENCODING)}, engine="netcdf4")
    os.replace(tmp_path, out_path)
    logger.debug("HF Zarr saved: %s", out_path)
    return out_path
