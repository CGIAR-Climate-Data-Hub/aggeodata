"""
cdh.transform.cog_export
=========================

Export xarray Datasets to Cloud Optimized GeoTIFF (COG).

Two export modes are supported:

* **One file per variable** — ``export_cog(ds, output_dir, mode='per_variable')``
  writes one multi-band COG per variable (bands = time steps).
* **One file per time step** — ``export_cog(ds, output_dir, mode='per_timestep')``
  writes one single-band COG per variable per day, organised in sub-folders.

In both modes the output files use Deflate + predictor-2 compression
(lossless, widely compatible) and include overview levels (2, 4, 8, 16)
so they are readable as true COGs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_OVERVIEW_LEVELS = [2, 4, 8, 16]
_DEFLATE_PROFILE = {
    "driver": "GTiff",
    "compress": "deflate",
    "predictor": 2,
    "zlevel": 6,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "BIGTIFF": "IF_SAFER",
}


def _write_cog(
    data: np.ndarray,
    out_path: str,
    transform,
    crs: str,
    nodata: float = float("nan"),
    band_descriptions: list[str] | None = None,
) -> None:
    """Write a numpy array as a COG GeoTIFF with overview levels."""
    import rasterio
    from rasterio.enums import Resampling as _R
    from rasterio.transform import from_bounds

    if data.ndim == 2:
        data = data[np.newaxis, ...]  # ensure (bands, y, x)
    bands, height, width = data.shape

    profile = {
        **_DEFLATE_PROFILE,
        "count": bands,
        "height": height,
        "width": width,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        if band_descriptions:
            for i, desc in enumerate(band_descriptions, start=1):
                dst.update_tags(i, description=desc)
        overview_levels = [l for l in _OVERVIEW_LEVELS if l < min(height, width)]
        if overview_levels:
            dst.build_overviews(overview_levels, _R.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    # Re-write to finalise COG internal structure (GDAL copy trick)
    _finalise_cog(out_path)


def _finalise_cog(path: str) -> None:
    """Use rasterio copy with COPY_SRC_OVERVIEWS=YES to produce a valid COG layout."""
    import rasterio
    import rasterio.shutil as rio_shutil

    tmp_path = path + ".tmp.tif"
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        profile.update({"COPY_SRC_OVERVIEWS": "YES", **_DEFLATE_PROFILE})
        rio_shutil.copy(src, tmp_path, copy_src_overviews=True, **_DEFLATE_PROFILE)
    os.replace(tmp_path, path)


def export_cog(
    ds: xr.Dataset,
    output_dir: str,
    mode: Literal["per_variable", "per_timestep"] = "per_variable",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    time_dim: str = "time",
    nodata: float = float("nan"),
    prefix: str = "",
) -> dict[str, list[str]]:
    """Export a climate Dataset to COG GeoTIFF files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset with ``lat``, ``lon``, and optionally ``time`` dimensions.
        Must have EPSG:4326 CRS or be interpretable as geographic coordinates.
    output_dir : str
        Root directory for output files.
    mode : str
        ``'per_variable'``  — one multi-band COG per variable (bands = time steps).
        ``'per_timestep'``  — one single-band COG per variable per time step.
    lat_dim : str
        Name of the latitude dimension.  Default: ``'lat'``.
    lon_dim : str
        Name of the longitude dimension.  Default: ``'lon'``.
    time_dim : str
        Name of the time dimension.  Default: ``'time'``.
    nodata : float
        No-data value written into the COG header.  Default: NaN.
    prefix : str
        Optional filename prefix (e.g. country code).

    Returns
    -------
    dict[str, list[str]]
        ``{variable_name: [file_path, ...]}`` for every exported file.
    """
    import rioxarray  # noqa: F401
    from rasterio.transform import from_bounds

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Infer affine transform from coordinate arrays
    lats = ds[lat_dim].values
    lons = ds[lon_dim].values
    if len(lons) < 2 or len(lats) < 2:
        raise ValueError("Dataset must have at least 2 lat and 2 lon points to infer transform.")

    dlat = abs(float(lats[1] - lats[0]))
    dlon = abs(float(lons[1] - lons[0]))
    xmin = float(lons.min()) - dlon / 2
    xmax = float(lons.max()) + dlon / 2
    ymin = float(lats.min()) - dlat / 2
    ymax = float(lats.max()) + dlat / 2
    height, width = len(lats), len(lons)
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    crs = "EPSG:4326"
    try:
        if ds.rio.crs is not None:
            crs = str(ds.rio.crs)
    except Exception:  # noqa: BLE001
        pass

    has_time = time_dim in ds.dims
    results: dict[str, list[str]] = {}

    for var in ds.data_vars:
        da = ds[var]
        var_dir = os.path.join(output_dir, var) if mode == "per_timestep" else output_dir
        Path(var_dir).mkdir(parents=True, exist_ok=True)
        results[var] = []

        if mode == "per_variable":
            if has_time:
                arr = da.values  # (time, lat, lon)
                if lat_dim == "lat" and lats[0] > lats[-1]:
                    arr = arr[:, ::-1, :]  # flip to S→N for standard rasterio
                time_labels = [str(t)[:10] for t in ds[time_dim].values]
            else:
                arr = da.values[np.newaxis, ...]
                time_labels = [var]

            out_path = os.path.join(output_dir, f"{prefix}{var}.tif" if prefix else f"{var}.tif")
            _write_cog(
                arr.astype(np.float32),
                out_path,
                transform,
                crs,
                nodata=nodata,
                band_descriptions=time_labels,
            )
            results[var].append(out_path)
            logger.info("COG saved: %s (%d bands)", out_path, arr.shape[0])

        elif mode == "per_timestep":
            if not has_time:
                arr = da.values
                if arr.ndim == 2 and lats[0] > lats[-1]:
                    arr = arr[::-1, :]
                out_path = os.path.join(var_dir, f"{prefix}{var}.tif" if prefix else f"{var}.tif")
                _write_cog(arr.astype(np.float32), out_path, transform, crs, nodata=nodata)
                results[var].append(out_path)
            else:
                for t_idx, t_val in enumerate(ds[time_dim].values):
                    t_label = str(t_val)[:10].replace("-", "")
                    arr = da.isel({time_dim: t_idx}).values
                    if lats[0] > lats[-1]:
                        arr = arr[::-1, :]
                    fname = f"{prefix}{var}_{t_label}.tif" if prefix else f"{var}_{t_label}.tif"
                    out_path = os.path.join(var_dir, fname)
                    _write_cog(arr.astype(np.float32), out_path, transform, crs, nodata=nodata)
                    results[var].append(out_path)
                logger.info("COG exported: %s  %d files", var, len(results[var]))

    return results
