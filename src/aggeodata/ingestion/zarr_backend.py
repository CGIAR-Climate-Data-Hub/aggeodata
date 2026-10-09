"""
aggeodata.ingestion.zarr_backend
=================================

Analysis-ready Zarr backend shared by the CHIRPS, CHIRTS and AgERA5 downloaders.

The official endpoints serve one file per day, which makes long time series slow
to retrieve.  The same archives are published as consolidated Zarr datacubes,
where a bounding box plus a date range is a single indexed read:

================  ==========================================================
CHIRPS v3 daily   ``aaguilar90/chirps-v3-daily-rnl`` on Hugging Face
                  (variable ``precip``; 0.05 deg; no credentials)
CHIRTS-ERA5 daily ``aaguilar90/chirts-era5-daily`` on Hugging Face
                  (variables ``Tmax``, ``Tmin``; 0.05 deg; no credentials)
AgERA5            ECMWF ARCO datastore (0.1 deg; **needs a CDS API key**)
================  ==========================================================

This module slices a store and writes the result as **one NetCDF per day**, in
the same layout the per-day downloaders produce, so
:func:`aggeodata.pipelines.datacube.run_datacube` consumes either backend
without changes.  Output is always north-up with ``y``/``x`` coordinates,
whatever orientation and coordinate names the source store uses.

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
    "AGERA5_ARCO_GEO_URL",
    "AGERA5_ARCO_TIME_URL",
    "read_cds_api_key",
    "cds_storage_options",
    "open_remote_zarr",
    "subset_zarr",
    "download_daily_from_zarr",
]

HF_CHIRPS_URL: str = (
    "https://huggingface.co/datasets/aaguilar90/chirps-v3-daily-rnl/"
    "resolve/main/chirps-v3-daily-rnl.zarr"
)
HF_CHIRTS_URL: str = (
    "https://huggingface.co/datasets/aaguilar90/chirts-era5-daily/"
    "resolve/main/chirts-era5-daily.zarr"
)

#: AgERA5 ARCO store chunked for time series at a location — chunks are
#: ``(2112 time, 16 lat, 32 lon)``.  Best for many days over a modest AOI,
#: which is the usual aggeodata request.
AGERA5_ARCO_GEO_URL: str = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-001/arco/"
    "sis_agrometeorological_indicators/all/geoChunked.zarr"
)
#: AgERA5 ARCO store chunked for a map at a single time step — chunks are
#: ``(1 time, 1024 lat, 1024 lon)``.  Best for a few days over a large AOI.
AGERA5_ARCO_TIME_URL: str = (
    "https://arco.datastores.ecmwf.int/cadl-arco-time-001/arco/"
    "sis_agrometeorological_indicators/all/timeChunked.zarr"
)

_NC_ENCODING = {"zlib": True, "complevel": 4}

# Each worker holds one loaded time-batch in memory; keep a batch small enough
# that ncores batches fit comfortably in RAM.
_TARGET_BATCH_BYTES = 256 * 1024**2
_MAX_BATCH_DAYS = 128


# ---------------------------------------------------------------------------
# Credentials (AgERA5 / ECMWF ARCO)
# ---------------------------------------------------------------------------

def read_cds_api_key(path: str | os.PathLike | None = None) -> str:
    """Return the CDS API key from ``~/.cdsapirc``.

    The ECMWF ARCO datastore authenticates with the same key the CDS API uses,
    passed as a bearer token.

    Parameters
    ----------
    path : str | os.PathLike | None
        Location of the rc file.  Defaults to ``~/.cdsapirc``.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    FileNotFoundError
        If the rc file does not exist.
    ValueError
        If the file has no ``key:`` entry.
    """
    rc = Path(path) if path else Path.home() / ".cdsapirc"
    if not rc.exists():
        raise FileNotFoundError(
            f"{rc} not found. AgERA5 needs a free CDS account: register at "
            "https://cds.climate.copernicus.eu/ and write your key to this file as:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <YOUR-API-KEY>"
        )
    for line in rc.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("key:"):
            key = line.split(":", 1)[1].strip()
            if key:
                return key
    raise ValueError(f"No 'key:' entry found in {rc}")


def cds_storage_options(path: str | os.PathLike | None = None) -> dict:
    """Return fsspec ``storage_options`` carrying the CDS bearer token."""
    return {"headers": {"Authorization": f"Bearer {read_cds_api_key(path)}"}}


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------

def open_remote_zarr(url: str, storage_options: dict | None = None) -> xr.Dataset:
    """Open a consolidated remote Zarr store.

    Parameters
    ----------
    url : str
        Full ``https://`` URL of the ``.zarr`` store.
    storage_options : dict | None
        Passed to fsspec — e.g. ``cds_storage_options()`` for the ECMWF ARCO
        datastore.  ``None`` for public stores.

    Returns
    -------
    xarray.Dataset
        Lazily-opened dataset (no array data transferred yet).
    """
    try:
        import zarr  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The Zarr backend requires zarr: pip install 'aggeodata[download]'"
        ) from exc

    logger.info("Zarr: opening %s", url)
    return xr.open_zarr(
        url, consolidated=True, chunks={}, storage_options=storage_options
    )


def subset_zarr(
    ds: xr.Dataset,
    variables: list[str],
    extent: list[float],
    starting_date: str,
    ending_date: str,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.Dataset:
    """Slice *ds* to the requested variables, bounding box and date range.

    Slice bounds are ordered to match each coordinate's own direction, so the
    caller always passes ``[xmin, ymin, xmax, ymax]`` regardless of whether the
    store stores latitude ascending (AgERA5) or descending (CHIRPS/CHIRTS).

    Parameters
    ----------
    ds : xarray.Dataset
        Store opened by :func:`open_remote_zarr`.
    variables : list[str]
        Variable names as they appear in the store (e.g. ``["Tmax"]``).
    extent : list[float]
        ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.
    starting_date, ending_date : str
        ISO 8601 ``"YYYY-MM-DD"`` bounds, both inclusive.
    lat_name, lon_name : str
        Latitude / longitude coordinate names in the store.  CHIRPS and CHIRTS
        use ``lat``/``lon``; AgERA5 uses ``latitude``/``longitude``.

    Returns
    -------
    xarray.Dataset
        Lazy subset containing only *variables*.

    Raises
    ------
    ValueError
        If a variable or coordinate is absent from the store, or the extent /
        date range falls outside its coverage.
    """
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Variables {missing} are not in the Zarr store. "
            f"Available: {sorted(str(v) for v in ds.data_vars)}"
        )
    for coord in (lat_name, lon_name, "time"):
        if coord not in ds.coords:
            raise ValueError(
                f"Coordinate '{coord}' is not in the Zarr store. "
                f"Available: {sorted(str(c) for c in ds.coords)}"
            )

    xmin, ymin, xmax, ymax = extent
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    lat_slice = slice(ymax, ymin) if lat[0] > lat[-1] else slice(ymin, ymax)
    lon_slice = slice(xmax, xmin) if lon[0] > lon[-1] else slice(xmin, xmax)

    sub = ds[variables].sel(
        time=slice(starting_date, ending_date),
        **{lat_name: lat_slice, lon_name: lon_slice},
    )

    if sub.sizes["time"] == 0:
        t = ds["time"].values
        raise ValueError(
            f"No dates between {starting_date} and {ending_date} in the Zarr store "
            f"(coverage: {pd.Timestamp(t[0]).date()} to {pd.Timestamp(t[-1]).date()})."
        )
    if sub.sizes[lat_name] == 0 or sub.sizes[lon_name] == 0:
        raise ValueError(
            f"Extent {extent} does not intersect the Zarr store "
            f"(lat {lat.min():.2f}..{lat.max():.2f}, lon {lon.min():.2f}..{lon.max():.2f})."
        )

    # Grid-mapping / CRS variables are re-written per day on export.
    return sub.drop_vars(["spatial_ref", "crs"], errors="ignore")


# ---------------------------------------------------------------------------
# Daily NetCDF export
# ---------------------------------------------------------------------------

def download_daily_from_zarr(
    url: str,
    variables: dict[str, str],
    extent: list[float],
    starting_date: str,
    ending_date: str,
    path_for: Callable[[str, str], str],
    ncores: int = 1,
    batch_days: int | None = None,
    desc: str = "Zarr",
    storage_options: dict | None = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> dict[str, list[str]]:
    """Slice a remote Zarr store and write one NetCDF per day.

    Days whose output file already exists are skipped, so an interrupted run
    resumes where it stopped.  Each file is written to a temporary name and
    atomically renamed, so a cancelled run never leaves a truncated NetCDF
    behind for the datacube stage to find.

    Parameters
    ----------
    url : str
        Zarr store URL.
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
        AOI so a batch stays near 256 MB, capped at 128 days.
    desc : str
        Progress-bar label.
    storage_options : dict | None
        fsspec options — e.g. ``cds_storage_options()`` for AgERA5.
    lat_name, lon_name : str
        Latitude / longitude coordinate names in the store.

    Returns
    -------
    dict[str, list[str]]
        ``{output_variable: [written NetCDF paths]}`` (sorted by date).
    """
    from tqdm import tqdm

    ds = open_remote_zarr(url, storage_options=storage_options)
    sub = subset_zarr(
        ds, list(variables), extent, starting_date, ending_date,
        lat_name=lat_name, lon_name=lon_name,
    )

    # Normalise to north-up so every backend writes the same orientation.
    if sub[lat_name].values[0] < sub[lat_name].values[-1]:
        sub = sub.isel({lat_name: slice(None, None, -1)})

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
        _warn_on_grid_mismatch(cached_sample, sub, lat_name, lon_name, desc)

    written: dict[str, list[str]] = {out_var: [] for out_var in variables.values()}
    if not pending:
        return written

    if batch_days is None:
        batch_days = _auto_batch_days(sub, lat_name, lon_name, len(variables))
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
                            path = _write_day(
                                day, store_var, out_var, date_str, path_for, url,
                                lat_name, lon_name,
                            )
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


def _warn_on_grid_mismatch(
    existing_path: str, sub: xr.Dataset, lat_name: str, lon_name: str, desc: str
) -> None:
    """Warn when files already in the folder sit on a different grid.

    The per-day backends clip with rasterio or a server-side ``area`` request,
    which can keep one extra pixel along an edge that the Zarr coordinate slice
    drops.  Mixing both backends in one variable folder therefore yields
    per-day files of differing shape, and ``run_datacube`` concatenates those
    along time into a NaN-padded union.  Re-download with a single backend.
    """
    try:
        with xr.open_dataset(existing_path) as existing:
            same = (
                existing.sizes.get("y") == sub.sizes[lat_name]
                and existing.sizes.get("x") == sub.sizes[lon_name]
            )
            shape = (existing.sizes.get("y"), existing.sizes.get("x"))
    except Exception:  # noqa: BLE001 — a diagnostic must never break the download
        return

    if not same:
        logger.warning(
            "%s: %s is %s but this request produces (%d, %d). The folder mixes files "
            "from the per-day and Zarr backends, whose edge pixels differ; the "
            "datacube step will pad the mismatch with NaNs. Delete the folder "
            "and re-download it with one backend.",
            desc, os.path.basename(existing_path), shape,
            sub.sizes[lat_name], sub.sizes[lon_name],
        )


def _auto_batch_days(sub: xr.Dataset, lat_name: str, lon_name: str, n_vars: int) -> int:
    """Pick a per-request day count that keeps one batch near the memory target."""
    pixels = sub.sizes[lat_name] * sub.sizes[lon_name] * max(n_vars, 1)
    bytes_per_day = max(pixels * 8, 1)
    return max(1, min(_MAX_BATCH_DAYS, int(_TARGET_BATCH_BYTES // bytes_per_day)))


def _write_day(
    day: xr.Dataset,
    store_var: str,
    out_var: str,
    date_str: str,
    path_for: Callable[[str, str], str],
    url: str,
    lat_name: str,
    lon_name: str,
) -> str | None:
    """Write one variable for one date; return the path, or None if cached."""
    import rioxarray  # noqa: F401 — registers the .rio accessor

    out_path = path_for(out_var, date_str)
    if os.path.exists(out_path):
        return None

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    da = day[store_var]
    units = da.attrs.get("units")
    out = (
        da.rename({lat_name: "y", lon_name: "x"})
        .to_dataset(name=out_var)
        .drop_vars(["spatial_ref", "crs"], errors="ignore")
    )
    # Clear any grid-mapping pointer inherited from the store (AgERA5/ARCO names
    # its grid-mapping variable "crs"), otherwise write_crs reuses that name and
    # the attr set below would dangle.
    out[out_var].attrs.pop("grid_mapping", None)
    out[out_var].encoding.pop("grid_mapping", None)

    out = out.rio.write_crs("EPSG:4326")
    out[out_var].attrs["grid_mapping"] = "spatial_ref"
    if units:
        out[out_var].attrs["units"] = units
    out.attrs = {"source": url, "date": date_str}

    tmp_path = f"{out_path}.part"
    out.to_netcdf(tmp_path, encoding={out_var: dict(_NC_ENCODING)}, engine="netcdf4")
    os.replace(tmp_path, out_path)
    logger.debug("Zarr backend saved: %s", out_path)
    return out_path
