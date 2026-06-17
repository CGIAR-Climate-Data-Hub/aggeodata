"""
aggeodata.transform.soil_cube
================================

Transform Layer — Soil Datacube Builder.

Converts downloaded SoilGrids GeoTIFF files (per variable + depth) into a
single multi-depth NetCDF datacube suitable for consumption by ag-cube-cm
crop model simulations.

Key exports
-----------
* :data:`TEXTURE_CLASSES`                      — USDA texture class lookup
* :func:`find_soil_textural_class_in_nparray`  — vectorized USDA classification
* :func:`calculate_rgf`                         — Root Growth Factor per layer
* :func:`create_depth_dimension`                — stack datasets along depth dim
* :func:`get_layer_texture`                     — add USDA texture layer
* :class:`SoilDataCubeBuilder`                  — multi-depth soil cube builder
"""

from __future__ import annotations

import glob as _glob
import logging
import os
import re as _re
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Texture classification
# ---------------------------------------------------------------------------

TEXTURE_CLASSES: dict[int, str] = {
    0: "unknown", 1: "sand", 2: "loamy sand", 3: "sandy loam",
    4: "loam", 5: "silt loam", 6: "silt", 7: "sandy clay loam",
    8: "clay loam", 9: "silty clay loam", 10: "sandy clay",
    11: "silty clay", 12: "clay",
}


def find_soil_textural_class_in_nparray(
    sand: np.ndarray, clay: np.ndarray
) -> np.ndarray:
    """Vectorised USDA soil texture classification on 2-D NumPy arrays."""
    if not isinstance(sand, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(sand)}")

    silt = 100 - sand - clay
    silt[silt == 100] = 0

    cond1  = (sand >= 85) & ((silt + clay * 1.5) < 15)
    cond2  = (sand > 70) & (sand < 91) & ((silt + 1.5 * clay) >= 15) & ((silt + 2 * clay) < 30)
    cond3  = (
        ((clay >= 7) & (clay < 20) & (sand > 52) & ((silt + 2 * clay) >= 30))
        | ((clay < 7) & (silt < 50) & (sand > 43))
    )
    cond4  = (clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52)
    cond5  = ((silt >= 50) & (clay >= 12) & (clay < 27)) | ((silt >= 50) & (silt < 80) & (clay < 12))
    cond6  = (silt >= 80) & (clay < 12)
    cond7  = (clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45)
    cond8  = (clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45)
    cond9  = (clay >= 27) & (clay < 40) & (sand <= 20)
    cond10 = (clay >= 35) & (sand > 45)
    cond11 = (clay >= 40) & (silt >= 40)
    cond12 = (clay >= 40) & (sand <= 45) & (silt < 40)

    result = np.zeros(clay.shape, dtype=int)
    result[clay == 0] = -1
    for code, cond in enumerate(
        [cond1, cond2, cond3, cond4, cond5, cond6,
         cond7, cond8, cond9, cond10, cond11, cond12], start=1,
    ):
        result[(result == 0) & cond] = code
    result[result == -1] = 0
    return result


# ---------------------------------------------------------------------------
# Root Growth Factor
# ---------------------------------------------------------------------------

def calculate_rgf(depths: list[int]) -> list[float]:
    """Root Growth Factor (RGF) for soil layers, ranging 0.0–1.0."""
    arr = np.array(depths)
    if len(arr) > 1:
        centres: list[float] = [float(arr[0] / 2)] + (
            ((arr[1:] - arr[:-1]) / 2 + arr[:-1]).tolist()
        )
    else:
        centres = list(arr.astype(float))
    return [1.0 if c <= 15 else float(np.exp(-0.02 * c)) for c in centres]


# ---------------------------------------------------------------------------
# Folder manager
# ---------------------------------------------------------------------------

class _SoilFolderManager:
    """Discover and sort SoilGrids GeoTIFF files by variable and depth."""

    def __init__(self, path: str, variables: list[str], raster_extension: str = ".tif") -> None:
        self.path = path
        self.variables = variables
        self._extension = raster_extension
        self.depths: list[str] | None = None

    @staticmethod
    def _extract_depth(paths: list[str], variable: str, units: str = "cm") -> list[str]:
        depths = []
        for path in paths:
            matches = list(_re.finditer(variable, path))[-1]
            depths.append(path[matches.end() + 1 : path.index(units + "_")])
        return depths

    @staticmethod
    def _sort_depths(depths: list[str]) -> tuple[list[str], list[int]]:
        init_depths = [d.split("-")[0] for d in depths]
        sort_idx = list(np.argsort(np.array(init_depths).astype(int)))
        return [str(depths[i]) for i in sort_idx], sort_idx

    def _check_variable_paths(self, variable: str) -> list[str]:
        return _glob.glob(self.path + "/*{}*{}".format(variable, self._extension))

    def _extract_depths(self, variable: str, units_string: str = "cm") -> list[str]:
        variable_paths = self._check_variable_paths(variable)
        if variable_paths:
            depths = self._extract_depth(variable_paths, variable, units=units_string)
            self.depths, self._depthssorted = self._sort_depths(depths)
        else:
            self._depthssorted = []
        return self.depths or []

    def variable_path(self, variable: str, units_string: str = "cm") -> list[str] | None:
        variable_paths = self._check_variable_paths(variable)
        if not variable_paths:
            logger.warning("No data found for variable: %s", variable)
            return None
        self._extract_depths(variable, units_string=units_string)
        return [variable_paths[i] for i in self._depthssorted]

    def get_all_paths(self, units_string: str = "cm", by: str = "depth") -> dict:
        paths_dict: dict[str, list[str]] = {}
        for var in self.variables:
            varinfo = self.variable_path(var, units_string=units_string)
            if varinfo is not None:
                paths_dict[var] = varinfo

        depths_available = [
            self._extract_depth(v, k, units=units_string) for k, v in paths_dict.items()
        ]
        self.depths = self._sort_depths(np.unique(depths_available))[0]

        if by != "depth":
            return paths_dict

        paths_by_depth: dict[str, dict[str, str]] = {}
        for i, depth in enumerate(self.depths):
            varpaths: dict[str, str] = {}
            for k, v in paths_dict.items():
                if i < len(v):
                    varpaths[k] = v[i]
            paths_by_depth[depth] = varpaths
        return paths_by_depth


# ---------------------------------------------------------------------------
# Depth stacking
# ---------------------------------------------------------------------------

def create_depth_dimension(
    xrdata_dict: dict[str, xr.Dataset],
    dim_name: str = "depth",
) -> xr.Dataset:
    """Concatenate a per-depth dict of datasets along a new depth dimension."""
    first_ds = list(xrdata_dict.values())[0]
    reference_crs = None
    try:
        import rioxarray  # noqa: F401
        reference_crs = first_ds.rio.crs
    except Exception:
        pass

    slices: list[xr.Dataset] = []
    for label, ds in tqdm.tqdm(xrdata_dict.items(), desc="Stacking soil depths"):
        ds_exp = ds.expand_dims(dim=[dim_name])
        ds_exp[dim_name] = [label]
        slices.append(ds_exp)

    cube = xr.concat(slices, dim=dim_name)

    if reference_crs is not None:
        try:
            cube.rio.write_crs(reference_crs, inplace=True)
            for var in cube.data_vars:
                if var != "spatial_ref":
                    cube[var].attrs["grid_mapping"] = "spatial_ref"
        except Exception:
            pass

    if "band_data" in cube.data_vars:
        cube = cube.drop_vars("band_data")

    return cube


# ---------------------------------------------------------------------------
# Texture layer helper
# ---------------------------------------------------------------------------

def get_layer_texture(
    soil_layer: xr.Dataset,
    texture_name: str = "texture",
) -> xr.Dataset:
    """Add a USDA soil textural class layer to a soil dataset."""
    sand = soil_layer["sand"].values
    clay = soil_layer["clay"].values
    if np.nanmax(sand) > 100:
        sand = sand * 0.1
        clay = clay * 0.1

    texture_map = find_soil_textural_class_in_nparray(sand, clay).astype(float)
    texture_map[texture_map == 0] = np.nan

    ref_dims = soil_layer.sizes
    xrimg = xr.DataArray(texture_map)
    new_dims: dict = {}
    for keyval in xrimg.sizes:
        pos_dims = [
            j for j, k in enumerate(ref_dims.keys())
            if xrimg.sizes[keyval] == ref_dims[k]
        ]
        new_dims[keyval] = pos_dims
    k0, k1 = list(new_dims.keys())[0], list(new_dims.keys())[1]
    ref_keys = list(ref_dims.keys())
    if len(new_dims[k1]) > 1:
        new_dims[k1] = ref_keys[1]
        new_dims[k0] = ref_keys[0]
    else:
        new_dims[k1] = ref_keys[new_dims[k1][0]]
        new_dims[k0] = ref_keys[new_dims[k0][0]]
    xrimg.name = texture_name
    xrimg = xrimg.rename(new_dims)
    return xr.merge([soil_layer.copy(), xrimg])


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class SoilDataCubeBuilder:
    """Build a multi-depth soil NetCDF datacube from raw SoilGrids GeoTIFFs.

    Parameters
    ----------
    data_folder : str
        Folder containing downloaded SoilGrids GeoTIFFs (one file per
        variable + depth combination, as produced by ``download_soil``).
    variables : list[str]
        Soil variable names to include, e.g. ``["clay", "sand", "wv1500"]``.
    extent : list[float] | None
        Optional ``[xmin, ymin, xmax, ymax]`` spatial clip.
    reference_variable : str
        Variable used as the spatial reference for co-registration.
    crs : str
        Native CRS of the SoilGrids files.  Default: ``"ESRI:54052"``.
    target_crs : str | None
        Reproject the final cube to this CRS.  Default: ``"EPSG:4326"``.
    """

    def __init__(
        self,
        data_folder: str,
        variables: list[str],
        extent: list[float] | None = None,
        reference_variable: str = "wv1500",
        crs: str = "ESRI:54052",
        target_crs: str | None = "EPSG:4326",
    ) -> None:
        self.data_folder = data_folder
        self.variables = variables
        self._extent = extent
        self.reference_variable = reference_variable
        self.crs = crs
        self.target_crs = target_crs

    def build(self, verbose: bool = True) -> xr.Dataset:
        """Build the multi-depth soil datacube."""
        folder_manager = _SoilFolderManager(self.data_folder, self.variables)
        query_paths = folder_manager.get_all_paths(by="depth")

        xr_by_depth: dict[str, xr.Dataset] = {}
        for depth_label, var_paths in tqdm.tqdm(
            query_paths.items(), desc="Building soil cube", disable=not verbose
        ):
            xr_by_depth[depth_label] = self._stack_depth_layer(var_paths)

        cube = create_depth_dimension(xr_by_depth, dim_name="depth")

        if self.target_crs is not None:
            try:
                import rioxarray  # noqa: F401
                cube.rio.write_crs(self.target_crs, inplace=True)
            except Exception as exc:
                logger.warning("Could not write target CRS: %s", exc)

        return cube

    def build_and_save(
        self,
        output_path: str,
        filename: str | None = None,
        verbose: bool = True,
    ) -> str:
        """Build the soil datacube and save it to a NetCDF file.

        Returns
        -------
        str
            Path to the saved NetCDF file.
        """
        import rioxarray  # noqa: F401

        cube = self.build(verbose=verbose)

        if self.target_crs:
            try:
                cube = cube.rio.write_crs(self.target_crs, grid_mapping_name="spatial_ref")
                cube.attrs["crs"] = self.target_crs
            except Exception:
                pass

        # Ensure CRS consistency
        try:
            if cube.rio.crs is not None:
                cube = cube.rio.write_crs(cube.rio.crs, inplace=True)
                for var in cube.data_vars:
                    if var != "spatial_ref":
                        cube[var].attrs["grid_mapping"] = "spatial_ref"
        except Exception:
            pass

        if filename is None:
            folder_name = Path(self.data_folder).name
            filename = f"soil_{folder_name}.nc"

        out_file = os.path.join(output_path, filename)
        encoding = {var: {"zlib": True} for var in cube.data_vars}
        cube.to_netcdf(out_file, encoding=encoding, engine="netcdf4")
        logger.info("Soil datacube saved → %s", out_file)
        return out_file

    def _stack_depth_layer(self, var_paths: dict[str, str]) -> xr.Dataset:
        import rioxarray  # noqa: F401

        data_arrays: dict[str, xr.DataArray] = {}
        ref_da: xr.DataArray | None = None

        for var, fp in var_paths.items():
            try:
                da = rioxarray.open_rasterio(fp, masked=True)
                if "band" in da.dims and da.sizes["band"] == 1:
                    da = da.squeeze("band", drop=True)

                if da.rio.crs is None:
                    x_mag = float(abs(da.x.values[0])) if da.x.size > 0 else 0.0
                    native_crs = self.crs if x_mag > 1000 else "EPSG:4326"
                    da = da.rio.write_crs(native_crs)

                if self.target_crs:
                    current = da.rio.crs
                    from pyproj import CRS as _CRS
                    if current and _CRS(str(current)) != _CRS(self.target_crs):
                        da = da.rio.reproject(self.target_crs)

                da.name = var
                data_arrays[var] = da
                if var == self.reference_variable:
                    ref_da = da
            except Exception as exc:
                logger.warning("Could not read %s (%s): %s", var, fp, exc)

        if not data_arrays:
            raise ValueError(f"No soil variables loaded. Files: {list(var_paths.values())}")

        if ref_da is None:
            ref_da = next(iter(data_arrays.values()))
            logger.warning("Reference variable '%s' not found; using '%s'.",
                           self.reference_variable, ref_da.name)

        merged: dict[str, xr.DataArray] = {}
        for var, da in data_arrays.items():
            try:
                merged[var] = ref_da if var == ref_da.name else da.rio.reproject_match(ref_da)
            except Exception as exc:
                logger.warning("Could not resample %s: %s", var, exc)

        ds = xr.Dataset(merged)
        if self.target_crs:
            try:
                ds.rio.write_crs(self.target_crs, inplace=True)
            except Exception:
                pass
        return ds


def reshape_flat_soil_cube(ds: xr.Dataset) -> xr.Dataset:
    """Reshape a flat-format soil NetCDF into the 3-D depth-dimension format
    expected by ag-cube-cm.

    Flat format (produced by old pipelines or direct SoilGrids API downloads)
    stores one variable per depth as ``{var}_{lo}-{hi}cm_mean``, e.g.
    ``bdod_0-5cm_mean``, ``clay_5-15cm_mean``.

    This function converts that to a dataset with a ``depth`` coordinate
    dimension and clean variable names (``bdod``, ``clay``, …), which is the
    format ``SoilDataCubeBuilder`` produces and ``ag-cube-cm`` expects.

    Parameters
    ----------
    ds : xr.Dataset
        Flat-format soil dataset.

    Returns
    -------
    xr.Dataset
        Dataset with a ``depth`` dimension and one variable per soil property.

    Examples
    --------
    >>> flat = xr.open_dataset("soil_uruguay.nc")
    >>> cube = reshape_flat_soil_cube(flat)
    >>> cube  # dims: (depth, y, x)
    """
    import re

    pattern = re.compile(r"^(.+?)_(\d+-\d+)cm_mean$")

    var_by_depth: dict[str, dict[str, xr.DataArray]] = {}
    for vname in ds.data_vars:
        m = pattern.match(vname)
        if m:
            base, depth = m.group(1), m.group(2)
            var_by_depth.setdefault(base, {})[depth] = ds[vname]

    if not var_by_depth:
        logger.warning("reshape_flat_soil_cube: no flat-format variables found — returning unchanged")
        return ds

    all_depths = sorted(
        {d for depths in var_by_depth.values() for d in depths},
        key=lambda d: int(d.split("-")[0]),
    )

    slices: list[xr.Dataset] = []
    for depth in all_depths:
        slice_vars: dict[str, xr.DataArray] = {}
        for base, depth_dict in var_by_depth.items():
            if depth in depth_dict:
                da = depth_dict[depth].copy()
                da.name = base
                slice_vars[base] = da
        slice_ds = xr.Dataset(slice_vars).expand_dims({"depth": [depth]})
        slices.append(slice_ds)

    cube = xr.concat(slices, dim="depth")

    if "spatial_ref" in ds:
        cube["spatial_ref"] = ds["spatial_ref"]

    try:
        import rioxarray  # noqa: F401
        if ds.rio.crs is not None:
            cube = cube.rio.write_crs(ds.rio.crs, inplace=True)
    except Exception:
        pass

    return cube


__all__ = [
    "SoilDataCubeBuilder",
    "create_depth_dimension",
    "calculate_rgf",
    "find_soil_textural_class_in_nparray",
    "get_layer_texture",
    "reshape_flat_soil_cube",
    "TEXTURE_CLASSES",
]
