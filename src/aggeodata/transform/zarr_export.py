"""
aggeodata.transform.zarr_export
================================

Export an aggeodata datacube as a Zarr v3 store.

Structure, chunk geometry, codec and integer quantisation follow the
GeoSPOptimizer datacubes (``transformer/zarr_utils.py``,
``transformer/agera5.py``) so a cube built here is laid out like the published
CHIRPS / CHIRTS / AgERA5 archives and reads back the same way:

* **Chunks** ``(128 time, 100 lat, 100 lon)`` — the unit of decompression and
  of one HTTP range request once the store is hosted remotely.  Clamped down to
  the cube's own shape, since an AOI cube is usually smaller than one chunk.
* **Codec** Blosc/zstd level 5 with bitshuffle, which beat raw zstd by ~12% on
  the reference benchmarks and level 9 by only 5% at 14x the write cost.
  ``codec="zstd"`` writes plain Zstandard instead, which GDAL can read.
* **Quantisation** each variable is stored as a scaled integer with a
  ``_FillValue``, roughly halving the float32 payload.  The quanta mirror the
  reference project: 0.1 for temperature and precipitation, 10 kJ for solar
  radiation — all far finer than the products' own uncertainty.  The scale is
  validated against the cube's actual range, and a variable that would not fit
  falls back to float32 rather than being silently clipped.
* **Sharding** available but **off by default**, unlike the global archives.
  Sharding exists to keep a global store at a few thousand files; an AOI cube
  has few files anyway, and GDAL cannot read sharded stores at all
  (``Unsupported codec: sharding_indexed``), which would cost QGIS and R
  compatibility for no gain.

Rule: this module ONLY writes a store from an in-memory cube.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "CubeGeometry",
    "VariableEncoding",
    "CF_ENCODINGS",
    "compressors",
    "export_datacube_to_zarr",
    "open_cube",
]

TIME_UNITS = "days since 1980-01-01"
EPOCH = np.datetime64("1980-01-01", "D")


# ---------------------------------------------------------------------------
# Chunk / shard geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CubeGeometry:
    """Chunk and shard geometry of the exported datacube.

    Chunks are the unit of decompression and of a single HTTP range request.
    Shards bundle many chunks into one file; each shard extent must be an exact
    multiple of the corresponding chunk extent.

    Defaults match the GeoSPOptimizer archives, except ``sharded`` which is off
    here — see the module docstring.
    """

    time_chunk: int = 128
    lat_chunk: int = 100
    lon_chunk: int = 100
    time_shard: int = 128
    lat_shard: int = 600
    lon_shard: int = 600
    sharded: bool = False

    @property
    def chunks(self) -> tuple[int, int, int]:
        return (self.time_chunk, self.lat_chunk, self.lon_chunk)

    @property
    def shards(self) -> tuple[int, int, int] | None:
        return (self.time_shard, self.lat_shard, self.lon_shard) if self.sharded else None

    def validate(self, n_lat: int, n_lon: int) -> None:
        """Check shard/chunk divisibility; warn on a ragged trailing shard."""
        if not self.sharded:
            return
        for name, shard, chunk in (
            ("time", self.time_shard, self.time_chunk),
            ("lat", self.lat_shard, self.lat_chunk),
            ("lon", self.lon_shard, self.lon_chunk),
        ):
            if shard % chunk:
                raise ValueError(
                    f"{name} shard ({shard}) must be a whole multiple of the "
                    f"{name} chunk ({chunk})"
                )
        for name, size, shard in (("lat", n_lat, self.lat_shard),
                                  ("lon", n_lon, self.lon_shard)):
            if size % shard:
                warnings.warn(
                    f"grid {name} size ({size}) is not a whole number of "
                    f"{name} shards ({shard}); the trailing shard holds "
                    f"{size % shard} row(s)", stacklevel=2)

    def clamp(self, n_time: int, n_lat: int, n_lon: int) -> CubeGeometry:
        """Return a copy with chunks (and shards) clipped to the cube's shape.

        An AOI cube is typically far smaller than one global chunk; leaving the
        nominal chunk in the metadata would advertise a block the store does
        not have.
        """
        tc = max(1, min(self.time_chunk, n_time))
        yc = max(1, min(self.lat_chunk, n_lat))
        xc = max(1, min(self.lon_chunk, n_lon))
        if not self.sharded:
            return CubeGeometry(tc, yc, xc, tc, yc, xc, sharded=False)
        # Keep shards a whole multiple of the clamped chunks.
        ts = max(tc, (min(self.time_shard, n_time) // tc) * tc)
        ys = max(yc, (min(self.lat_shard, n_lat) // yc) * yc)
        xs = max(xc, (min(self.lon_shard, n_lon) // xc) * xc)
        return CubeGeometry(tc, yc, xc, ts, ys, xs, sharded=True)


def compressors(codec: str = "blosc") -> list:
    """Compressor chain for a data array.

    ``blosc`` (zstd-5 + bitshuffle) is the reference default and smallest for
    sparse, zero-heavy fields such as precipitation.  ``zstd`` is plain
    Zstandard, which GDAL can read — most GDAL builds ship without Blosc.
    """
    if codec == "zstd":
        from zarr.codecs import ZstdCodec
        return [ZstdCodec(level=5)]
    if codec == "blosc":
        from zarr.codecs import BloscCodec, BloscShuffle
        return [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.bitshuffle)]
    raise ValueError(f"unknown codec {codec!r}; expected 'blosc' or 'zstd'")


# ---------------------------------------------------------------------------
# Per-variable integer encoding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariableEncoding:
    """How one CF variable is stored: ``stored = round((physical - offset) / scale)``.

    ``kelvin_offset`` marks variables whose offset depends on the units the cube
    actually carries.  aggeodata cubes mix sources — CHIRTS writes degC while
    AgERA5 writes K — so the offset is resolved per cube rather than baked in.
    """

    dtype: str
    scale: float
    fill: int
    standard_name: str = ""
    long_name: str = ""
    #: Subtract 273.15 before scaling when the variable's units are Kelvin.
    kelvin_offset: bool = False
    #: Alternative scale used when units are a large-magnitude energy flux
    #: (AgERA5 J m-2 d-1) rather than a small one (MJ m-2 d-1, W m-2).
    joule_scale: float | None = None

    def resolve(self, units: str | None, vmin: float, vmax: float) -> tuple[float, float]:
        """Return the ``(scale, offset)`` to use for this cube."""
        u = (units or "").strip().lower()
        offset = 273.15 if (self.kelvin_offset and u in {"k", "kelvin"}) else 0.0
        scale = self.scale
        if self.joule_scale is not None and "j m-2" in u:
            scale = self.joule_scale
        return scale, offset

    def bounds(self) -> tuple[int, int]:
        """Integer range usable for data, with ``fill`` excluded."""
        info = np.iinfo(self.dtype)
        lo, hi = int(info.min), int(info.max)
        if self.fill <= lo:
            lo += 1
        elif self.fill >= hi:
            hi -= 1
        return lo, hi


_I16 = -32768
_U16 = 65535

#: Storage spec per CF variable.  Quanta mirror the GeoSPOptimizer archives.
CF_ENCODINGS: dict[str, VariableEncoding] = {
    "pr":      VariableEncoding("int16", 0.1, -9999, "precipitation_amount",
                                "daily precipitation"),
    "tasmax":  VariableEncoding("int16", 0.1, _I16, "air_temperature",
                                "daily maximum air temperature", kelvin_offset=True),
    "tasmin":  VariableEncoding("int16", 0.1, _I16, "air_temperature",
                                "daily minimum air temperature", kelvin_offset=True),
    "tas":     VariableEncoding("int16", 0.1, _I16, "air_temperature",
                                "daily mean air temperature", kelvin_offset=True),
    "tdps":    VariableEncoding("int16", 0.1, _I16, "dew_point_temperature",
                                "dew point temperature", kelvin_offset=True),
    "rsds":    VariableEncoding("uint16", 0.01, _U16,
                                "surface_downwelling_shortwave_flux_in_air",
                                "surface downwelling shortwave radiation",
                                joule_scale=10000.0),
    "sfcWind": VariableEncoding("int16", 0.01, _I16, "wind_speed",
                                "near-surface wind speed"),
    "hurs":    VariableEncoding("int16", 0.01, _I16, "relative_humidity",
                                "near-surface relative humidity"),
    "vp":      VariableEncoding("int16", 0.01, _I16, "water_vapor_partial_pressure_in_air",
                                "vapour pressure"),
    "vpd":     VariableEncoding("int16", 0.01, _I16, "water_vapor_saturation_deficit_in_air",
                                "vapour pressure deficit"),
    "etr":     VariableEncoding("int16", 0.01, _I16, "water_evapotranspiration_flux",
                                "reference evapotranspiration"),
}
# The hourly relative-humidity variants share the hurs spec.
for _h in ("hurs_06", "hurs_09", "hurs_12", "hurs_15", "hurs_18"):
    CF_ENCODINGS[_h] = CF_ENCODINGS["hurs"]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_datacube_to_zarr(
    cube: xr.Dataset,
    path: str | Path,
    geometry: CubeGeometry | None = None,
    *,
    overwrite: bool = True,
    quantize: bool = True,
    codec: str = "blosc",
    extra_attrs: dict | None = None,
) -> dict:
    """Write *cube* as a Zarr v3 store laid out like the GeoSPOptimizer archives.

    Parameters
    ----------
    cube : xarray.Dataset
        Datacube with a ``time`` dimension and two spatial dimensions
        (``y``/``x`` or ``lat``/``lon``), as produced by
        :func:`aggeodata.pipelines.datacube.run_datacube`.
    path : str | Path
        Destination ``.zarr`` directory.
    geometry : CubeGeometry | None
        Chunk/shard geometry.  Defaults to the reference geometry, clamped to
        the cube's shape.
    overwrite : bool
        Replace an existing store.  Default: True.
    quantize : bool
        Store variables as scaled integers per :data:`CF_ENCODINGS`.  False
        writes float32 instead — larger, but exact.  Default: True.
    codec : str
        ``"blosc"`` (default) or ``"zstd"`` (GDAL-readable).
    extra_attrs : dict | None
        Extra root-group attributes, e.g. a CDH metadata record.

    Returns
    -------
    dict
        ``{"path", "days", "variables", "bytes", "files", "encodings"}``.

    Raises
    ------
    ValueError
        If the cube has no time dimension or no recognisable spatial dims.

    Notes
    -----
    Each variable is materialised as float64 once while encoding, so peak
    memory is roughly ``8 * time * y * x`` bytes on top of the cube itself.
    That matches what :func:`~aggeodata.pipelines.datacube.run_datacube`
    already holds in memory; building a continental, multi-decade cube would
    need a block-wise writer instead.
    """
    try:
        import zarr
        from zarr.codecs import BytesCodec
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Zarr export requires zarr>=3: pip install 'aggeodata[download]'"
        ) from exc

    dest = Path(path)
    y_dim, x_dim = _spatial_dims(cube)
    if "time" not in cube.dims:
        raise ValueError("cube has no 'time' dimension")

    geographic = _is_geographic(cube)
    y_name, x_name = ("lat", "lon") if geographic else ("y", "x")

    times = np.asarray(cube["time"].values, dtype="datetime64[D]")
    yv = np.asarray(cube[y_dim].values, dtype="float64")
    xv = np.asarray(cube[x_dim].values, dtype="float64")
    n_time, n_y, n_x = len(times), len(yv), len(xv)

    geom = (geometry or CubeGeometry()).clamp(n_time, n_y, n_x)
    geom.validate(n_y, n_x)

    data_vars = [v for v in cube.data_vars if v not in ("spatial_ref", "crs")]
    if not data_vars:
        raise ValueError("cube has no data variables to write")

    root = zarr.create_group(store=str(dest), overwrite=overwrite)
    _write_coords(root, times, yv, xv, y_name, x_name, geographic, cube)

    encodings: dict[str, dict] = {}
    for name in data_vars:
        da = cube[name].transpose("time", y_dim, x_dim)
        values = np.asarray(da.values, dtype="float64")
        units = da.attrs.get("units")
        spec = CF_ENCODINGS.get(str(name)) if quantize else None

        if spec is not None:
            spec, scale, offset = _fit_encoding(str(name), spec, units, values)
        if spec is None:
            arr = root.create_array(
                name=str(name), shape=(n_time, n_y, n_x),
                chunks=geom.chunks, shards=geom.shards,
                dtype="float32", fill_value=np.float32("nan"),
                serializer=BytesCodec(), compressors=compressors(codec),
                dimension_names=("time", y_name, x_name),
            )
            arr[...] = values.astype("float32")
            attrs = {"grid_mapping": "spatial_ref", "coordinates": f"time {y_name} {x_name}"}
            if units:
                attrs["units"] = units
            arr.attrs.update(_cf_attrs(str(name), units) | attrs)
            encodings[str(name)] = {"dtype": "float32"}
            continue

        stored = _encode(values, spec, scale, offset)
        arr = root.create_array(
            name=str(name), shape=(n_time, n_y, n_x),
            chunks=geom.chunks, shards=geom.shards,
            dtype=spec.dtype, fill_value=spec.fill,
            serializer=BytesCodec(), compressors=compressors(codec),
            dimension_names=("time", y_name, x_name),
        )
        arr[...] = stored
        arr.attrs.update({
            **_cf_attrs(str(name), units),
            "scale_factor": scale,
            "add_offset": offset,
            "_FillValue": spec.fill,
            "grid_mapping": "spatial_ref",
            "coordinates": f"time {y_name} {x_name}",
        })
        encodings[str(name)] = {
            "dtype": spec.dtype, "scale_factor": scale,
            "add_offset": offset, "_FillValue": spec.fill,
        }
        logger.info(
            "  %-8s -> %-7s scale=%g offset=%g (max quantisation error %g %s)",
            name, spec.dtype, scale, offset, scale / 2, units or "",
        )

    root.attrs.update({
        "Conventions": "CF-1.10",
        "title": "aggeodata multi-variable climate datacube",
        "source": "aggeodata (CHIRPS / CHIRTS / AgERA5 / NASA POWER / GEE)",
        "variables": [str(v) for v in data_vars],
        "time_coverage_start": str(times[0]),
        "time_coverage_end": str(times[-1]),
        "zarr_geometry": asdict(geom),
        **_spatial_attrs(yv, xv, geographic, cube),
        **(extra_attrs or {}),
    })
    zarr.consolidate_metadata(str(dest))

    files = [f for f in dest.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    logger.info("Zarr datacube saved -> %s (%.2f MB, %d files)",
                dest, total / 1e6, len(files))
    return {
        "path": str(dest), "days": n_time, "variables": [str(v) for v in data_vars],
        "bytes": total, "files": len(files), "encodings": encodings,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spatial_dims(cube: xr.Dataset) -> tuple[str, str]:
    for y, x in (("y", "x"), ("lat", "lon"), ("latitude", "longitude")):
        if y in cube.dims and x in cube.dims:
            return y, x
    raise ValueError(
        f"cube has no recognisable spatial dims (got {list(cube.dims)}); "
        "expected y/x, lat/lon or latitude/longitude"
    )


def _resolve_crs(cube: xr.Dataset):
    """Return the cube's CRS, or None.

    ``cube.rio.crs`` only sees a grid mapping that xarray decoded into a
    coordinate, which needs ``decode_coords="all"``.  A cube reopened from one
    of our NetCDFs with plain ``open_dataset`` carries ``spatial_ref`` as an
    ordinary data variable instead, so fall back to reading its WKT directly.
    """
    try:
        import rioxarray  # noqa: F401
        crs = cube.rio.crs
        if crs is not None:
            return crs
    except Exception:  # noqa: BLE001
        pass

    try:
        from rasterio.crs import CRS
    except ImportError:  # pragma: no cover
        return None

    for name in ("spatial_ref", "crs"):
        if name not in cube.variables:
            continue
        attrs = cube[name].attrs
        for key in ("crs_wkt", "spatial_ref", "epsg_code"):
            wkt = attrs.get(key)
            if not wkt:
                continue
            try:
                return CRS.from_user_input(wkt)
            except Exception:  # noqa: BLE001 — try the next attribute
                continue
    return None


def _is_geographic(cube: xr.Dataset) -> bool:
    crs = _resolve_crs(cube)
    if crs is not None:
        return bool(crs.is_geographic)
    # No CRS to go on — fall back to the dimension-name convention.
    return any(d in cube.dims for d in ("lat", "latitude"))


def _cf_attrs(name: str, units: str | None) -> dict:
    spec = CF_ENCODINGS.get(name)
    attrs: dict = {}
    if spec is not None:
        if spec.standard_name:
            attrs["standard_name"] = spec.standard_name
        if spec.long_name:
            attrs["long_name"] = spec.long_name
    if units:
        attrs["units"] = units
    return attrs


def _fit_encoding(
    name: str, spec: VariableEncoding, units: str | None, values: np.ndarray
) -> tuple[VariableEncoding | None, float, float]:
    """Resolve scale/offset and verify the cube's range actually fits.

    Returns ``(None, 0, 0)`` to signal a float32 fallback — better than
    silently clipping real values into the integer range.
    """
    finite = np.isfinite(values)
    if not finite.any():
        logger.warning("%s: all values are missing — writing float32", name)
        return None, 0.0, 0.0

    vmin = float(values[finite].min())
    vmax = float(values[finite].max())
    scale, offset = spec.resolve(units, vmin, vmax)
    lo, hi = spec.bounds()
    q_lo = (vmin - offset) / scale
    q_hi = (vmax - offset) / scale

    if q_lo < lo or q_hi > hi:
        logger.warning(
            "%s: range [%g, %g] %s does not fit %s at scale %g "
            "(needs %.0f..%.0f, have %d..%d) — writing float32 instead",
            name, vmin, vmax, units or "", spec.dtype, scale, q_lo, q_hi, lo, hi,
        )
        return None, 0.0, 0.0
    return spec, scale, offset


def _encode(values: np.ndarray, spec: VariableEncoding, scale: float, offset: float) -> np.ndarray:
    """Physical floats -> stored integers; non-finite becomes ``fill``."""
    missing = ~np.isfinite(values)
    q = (values - offset) / scale
    np.rint(q, out=q)
    lo, hi = spec.bounds()
    # Zero the holes before casting: NaN -> int is undefined in numpy.
    q[missing] = 0
    np.clip(q, lo, hi, out=q)
    out = q.astype(spec.dtype)
    out[missing] = spec.fill
    return out


def _write_coords(root, times, yv, xv, y_name, x_name, geographic, cube) -> None:
    days = (times - EPOCH).astype("int32")
    t = root.create_array(name="time", shape=days.shape, chunks=days.shape,
                          dtype="int32", dimension_names=("time",))
    t[:] = days
    t.attrs.update(units=TIME_UNITS, calendar="proleptic_gregorian",
                   standard_name="time", axis="T")

    if geographic:
        meta = ((y_name, yv, "degrees_north", "latitude", "Y"),
                (x_name, xv, "degrees_east", "longitude", "X"))
    else:
        meta = ((y_name, yv, "m", "projection_y_coordinate", "Y"),
                (x_name, xv, "m", "projection_x_coordinate", "X"))
    for nm, vals, unit, std, axis in meta:
        a = root.create_array(name=nm, shape=vals.shape, chunks=vals.shape,
                              dtype="float64", dimension_names=(nm,))
        a[:] = vals
        a.attrs.update(units=unit, standard_name=std, long_name=std, axis=axis)

    crs_attrs = {"grid_mapping_name": "latitude_longitude" if geographic else "unknown",
                 "semi_major_axis": 6378137.0,
                 "inverse_flattening": 298.257223563}
    crs = _resolve_crs(cube)
    if crs is not None:
        wkt = crs.to_wkt()
        crs_attrs["crs_wkt"] = wkt
        crs_attrs["spatial_ref"] = wkt
        epsg = crs.to_epsg()
        if epsg:
            crs_attrs["epsg_code"] = f"EPSG:{epsg}"
    else:
        logger.warning(
            "No CRS found on the cube; writing the store without a crs_wkt. "
            "Set one with cube.rio.write_crs(...) before exporting."
        )

    ref = root.create_array(name="spatial_ref", shape=(), chunks=(), dtype="int32")
    ref[...] = 0
    ref.attrs.update(crs_attrs)


def _spatial_attrs(yv, xv, geographic, cube) -> dict:
    res_y = abs(float(yv[1] - yv[0])) if len(yv) > 1 else 0.0
    res_x = abs(float(xv[1] - xv[0])) if len(xv) > 1 else 0.0
    prefix = "lat" if geographic else "y"
    xprefix = "lon" if geographic else "x"
    return {
        "spatial_resolution": round(res_x, 8),
        f"geospatial_{prefix}_min": round(float(yv.min()) - res_y / 2, 8),
        f"geospatial_{prefix}_max": round(float(yv.max()) + res_y / 2, 8),
        f"geospatial_{xprefix}_min": round(float(xv.min()) - res_x / 2, 8),
        f"geospatial_{xprefix}_max": round(float(xv.max()) + res_x / 2, 8),
    }


def open_cube(path: str | Path, chunks: str | dict | None = {}) -> xr.Dataset:
    """Open an exported datacube with every data variable decoded to float32.

    Zarr attributes are JSON, so ``scale_factor`` round-trips as a Python float
    and xarray's CF decoder promotes the variable to float64 — doubling every
    payload for no added information.  The cast back happens here, once.

    ``decode_coords="all"`` promotes ``spatial_ref`` from a data variable to a
    coordinate, which is what makes ``ds.rio.crs`` resolve instead of returning
    ``None``.
    """
    ds = xr.open_zarr(str(path), consolidated=True, chunks=chunks,
                      decode_timedelta=False, decode_coords="all")
    for name in ds.data_vars:
        if name != "spatial_ref" and np.issubdtype(ds[name].dtype, np.floating):
            ds[name] = ds[name].astype("float32")
    return ds
