"""
aggeodata.ingestion.utils
==========================

Generic download and IO utilities.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import copy

import os
import requests
import json
import pandas as pd
import tqdm

import zipfile
import numpy as np
import glob

import xarray
import rioxarray  # noqa: F401 — registers .rio accessor; must be imported before any .rio call

from rasterio.enums import Resampling


def download_file(start_date:str,
                end_date:str,
                xmin:float,
                xmax:float,
                ymin:float,
                ymax:float,
                url:str,
                variable:str,
                download_path:str
                ):

    """
    Downloads a file from a specified URL using a POST request.

    Parameters
    ----------
    start_date : str
        The start date for the data download, in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data download, in 'YYYY-MM-DD' format.
    xmin : float
        The minimum longitude for the bounding box.
    xmax : float
        The maximum longitude for the bounding box.
    ymin : float
        The minimum latitude for the bounding box.
    ymax : float
        The maximum latitude for the bounding box.
    url : str
        The URL to download the file from.
    variable : str
        The name of the variable to download.
    download_path : str
        The path to the directory where the file will be saved.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    
    headers = {'Accept': 'application/json'}
    
    url_params = {"startDt":start_date,
                "endDt":end_date,
                "xmin":xmin,
                "xmax":xmax,
                "ymin":ymin,
                "ymax":ymax,
                "variableName":variable
    }
    
    response = requests.post(url,
                            headers = headers,
                            data=json.dumps(url_params),
                            stream=True)
    if response.status_code == 200:
        file_name = f"{variable}_{start_date}_to_{end_date}.nc"
        file_path = os.path.join(download_path, file_name)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return file_path
    else:
        response.raise_for_status()

def check_depth_name_dims(xrdata):
    """
    Selects the first slice of a multi-dimensional xarray Dataset.

    This function is used to reduce a Dataset to a 2D array by selecting the
    first slice along the time or band dimension.

    Parameters
    ----------
    xrdata : xarray.Dataset
        The input Dataset.

    Returns
    -------
    xarray.Dataset
        The 2D Dataset.
    """
    
    if 'date' in list(xrdata.sizes.keys()):
        xrdata = xrdata.isel({'date': 0})
        xrdata = xrdata.drop_vars(['date'], errors='ignore')
    elif 'time' in list(xrdata.sizes.keys()):
        xrdata = xrdata.isel({'time': 0})
        xrdata = xrdata.drop_vars(['time'], errors='ignore')
    elif 'band' in list(xrdata.sizes.keys()):
        xrdata = xrdata.isel({list(xrdata.sizes.keys())[0]: 0})
        xrdata = xrdata.drop_vars(['band'], errors='ignore')
    else:
        raise ValueError('check depth order')
    
    
    return xrdata


def set_xr_attributes(xrdata, xdimref_name = 'x', ydimref_name = 'y'):
    if 'transform' not in list(xrdata.attrs.keys()):
        xrdata.attrs['transform'] = xrdata.rio.transform()
    if 'crs' not in list(xrdata.attrs.keys()):
        xrdata.attrs['crs'] = xrdata.rio.crs
    if 'width' not in list(xrdata.attrs.keys()):
        xrdata.attrs['width'] = len(xrdata[xdimref_name].values)
    if 'height' not in list(xrdata.attrs.keys()):
        xrdata.attrs['height'] = len(xrdata[ydimref_name].values)

    return xrdata

def stack_xrdata_variable(xrdata, xrrefence, xrefdim_name, yrefdim_name,method, target_crs, only_use_first_date =True):
    from .gis_functions import resample_xarray
    if len(xrdata.sizes.keys()) >= 3 and only_use_first_date:
        xrdata = check_depth_name_dims(xrdata)

    resampled_data = resample_xarray(xrdata, xrrefence, xrefdim_name=xrefdim_name, yrefdim_name=yrefdim_name, method = method, target_crs = target_crs)
    variable_name = list(resampled_data.data_vars.keys())[0]
    resampled_data = resampled_data[variable_name].values
    return resampled_data
    
def set_crs(xrdata, crs):

    if hasattr(crs, "to_epsg") and crs.to_epsg() is not None:
        xrdata.attrs["crs"] = f"EPSG:{crs.to_epsg()}"
    elif hasattr(crs, "to_string"):
        xrdata.attrs["crs"] = crs.to_string()
    return xrdata


def resample_variables(dict_xr,reference_variable = None, 
                       only_use_first_date = True, 
                       verbose = False, method: str = 'linear', target_crs = None):
    
    resampling_methods = {
        'nearest': Resampling.nearest,
        'linear': Resampling.bilinear,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'average': Resampling.average
    }
    
    resampling_method = resampling_methods.get(method.lower(), Resampling.nearest)

    listvariables = list(dict_xr.keys())
    if not listvariables:
        raise ValueError(
            "resample_variables: no variables were loaded successfully. "
            "Check that the raster files exist and that any crop_extent is in "
            "the same CRS as the raster files (SoilGrids uses ESRI:54052)."
        )
    if reference_variable is None:
        reference_variable = listvariables[0]
    if reference_variable not in listvariables:
        raise ValueError(
            f"resample_variables: reference variable '{reference_variable}' was not "
            f"loaded successfully. Available variables: {listvariables}"
        )
    listvariables.remove(reference_variable)

    # 1. Prepare Reference
    xr_ref = dict_xr[reference_variable].copy()
    
    if len(xr_ref.sizes.keys()) >= 3 and only_use_first_date:
        xr_ref = check_depth_name_dims(xr_ref)

    if isinstance(xr_ref, xarray.Dataset):
        for i in list(xr_ref.data_vars.keys()):
            if i != 'spatial_ref':
                variable_name = i
                break
        xr_ref = xr_ref[variable_name]
    
    xr_ref.name = variable_name
    
    if target_crs is not None:
        if str(target_crs) != str(xr_ref.rio.crs):
            #xr_reference = reproject_xrdata(xr_reference, target_crs)
            xr_ref = xr_ref.rio.reproject(target_crs, resampling=resampling_method)


    xr_ref.rio.write_crs(xr_ref.rio.crs, inplace=True)

    #resampled_list = [xr_reference[variable_name].values]
    
    ref_x_dim = xr_ref.rio.x_dim
    ref_y_dim = xr_ref.rio.y_dim
    if xr_ref.rio.nodata is None:
        xr_ref.rio.write_nodata(np.nan, encoded=True, inplace=True)

    processed_xarrays = {variable_name: xr_ref}

    for var_name, xr_data in dict_xr.items():
        if var_name == variable_name:
            continue
        
        if isinstance(xr_data, xarray.Dataset):
            for i in list(xr_data.data_vars.keys()):
                if i != 'spatial_ref':
                    variable_name = i
                    break
            xr_data = xr_data[variable_name]

        if len(xr_data.sizes.keys()) >= 3 and only_use_first_date:
            xr_data = check_depth_name_dims(xr_data)
        

        is_native_soilgrids = abs(xr_data.rio.bounds()[0]) > 1000 

        if is_native_soilgrids:
            # Tell rioxarray the TRUTH: This data is in Goode Homolosine
            homolosine_crs = '+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs'
            xr_data.rio.write_crs(homolosine_crs, inplace=True)
            if verbose: print(f"Detected native SoilGrids projection for {var_name}")
        else:
            # If it's already in small numbers (degrees), assume it matches the reference
            xr_data.rio.write_crs(xr_ref.rio.crs, inplace=True)


        data_x_dim = xr_data.rio.x_dim
        data_y_dim = xr_data.rio.y_dim
        
        rename_dict = {}
        if data_x_dim != ref_x_dim: rename_dict[data_x_dim] = ref_x_dim
        if data_y_dim != ref_y_dim: rename_dict[data_y_dim] = ref_y_dim
        
        if rename_dict:
            xr_data = xr_data.rename(rename_dict)

        if xr_data.rio.nodata is None:
            xr_data.rio.write_nodata(np.nan, encoded=True, inplace=True)

        if xr_ref.rio.nodata is None:
            xr_ref.rio.write_nodata(np.nan, encoded=True, inplace=True)

        if xr_data.rio.nodata is None:
            xr_data.rio.write_nodata(np.nan, encoded=True, inplace=True)

        # Derive and store the affine transform from coordinate arrays so
        # rasterio does not fall back to the identity matrix during reproject.
        xr_data.rio.write_transform(inplace=True)

        resampled_data = xr_data.rio.reproject_match(xr_ref, resampling=resampling_method)
   
        
        resampled_data.name = var_name
        processed_xarrays[var_name] = resampled_data


    dataouput = xarray.merge(list(processed_xarrays.values()), compat='override')

    # rioxarray raises "Multiple grid mappings exist" when data variables carry
    # *different* grid_mapping values (checked in both .encoding AND .attrs).
    # After xarray.merge, sources with different grid mapping coordinate names
    # (e.g. "spatial_ref" from CHIRPS vs "crs" from AgEra5) produce this state.
    # Fix: drop all grid-mapping coordinate variables, then wipe grid_mapping
    # from both attrs AND encoding on every remaining variable so rioxarray
    # falls back to the single DEFAULT_GRID_MAP ("spatial_ref") and write_crs
    # can establish a clean single mapping.
    gm_var_names = [
        name for name, var in dataouput.variables.items()
        if "grid_mapping_name" in var.attrs
    ]
    if gm_var_names:
        dataouput = dataouput.drop_vars(gm_var_names, errors="ignore")
    for name in list(dataouput.variables):
        dataouput.variables[name].attrs.pop("grid_mapping", None)
        dataouput.variables[name].encoding.pop("grid_mapping", None)

    dataouput.rio.write_crs(xr_ref.rio.crs, inplace=True)
    return dataouput
    


def compress_file(input_filepath: str, output_zip_filepath: str) -> None:
    """
    Compresses a single file into a new ZIP archive.

    Parameters
    ----------
    input_filepath : str
        The path to the file to compress.
    output_zip_filepath : str
        The path to the output ZIP archive.
    """
    
    with zipfile.ZipFile(output_zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(input_filepath, os.path.basename(input_filepath))
    #print(f"File '{input_filepath}' compressed to '{output_zip_filepath}'")


def set_encoding(xrdata: xarray.Dataset, compress_method: str = 'zlib') -> dict:
    """
    Creates an encoding dictionary for compressing an xarray Dataset.

    Parameters
    ----------
    xrdata : xarray.Dataset
        The input Dataset.
    compress_method : str, optional
        The compression method to use, by default 'zlib'

    Returns
    -------
    dict
        The encoding dictionary.
    """
    return {k: {compress_method: True} for k in list(xrdata.data_vars.keys())}


def compress_xarray_dataset(xrdata: xarray.Dataset, ouput_filename: str, to_zipfile: bool = True, engine: str = 'netcdf4',scale_value: int = 100, nodata: int = -9999) -> xarray.Dataset:
    """
    Compresses an xarray Dataset and saves it to a NetCDF file.

    This function uses masked arrays to handle NoData values, which is a more
    robust method than replacing them with a placeholder value.

    Parameters
    ----------
    xrdata : xarray.Dataset
        The input Dataset.
    ouput_filename : str
        The path to the output NetCDF file.
    to_zipfile : bool, optional
        Whether to compress the NetCDF file into a ZIP archive, by default True
    engine : str, optional
        The NetCDF engine to use, by default 'netcdf4'
    scale_value : int, optional
        The scale factor to apply to the data before saving, by default 100
    nodata : int, optional
        The NoData value to use, by default -9999

    Returns
    -------
    xarray.Dataset
        The compressed Dataset.
    """
    data = copy.deepcopy(xrdata)
    data = data * scale_value

    # Use a masked array to handle NoData values
    for k in data.data_vars.keys():
        data[k] = data[k].where(~np.isnan(data[k]), nodata)
    data = data.astype(np.int32)
    data.attrs['nodata'] = nodata
    data.attrs['dtype'] = 'int'
    data.attrs['width'] = data.x.values.shape[0]
    data.attrs['height'] = data.y.values.shape[0]
    data.attrs['count'] = len(list(data.data_vars.keys()))
    data.attrs['scale_values_factor'] = scale_value
    
    encoding = set_encoding(data)
    data.to_netcdf(ouput_filename, encoding = encoding, engine = engine)
    if to_zipfile:
        compress_file(ouput_filename, ouput_filename.replace('.nc','.zip'))
        data = None
        os.remove(ouput_filename)
    return data

def read_compressed_xarray(input_filepath: str, output_path:str = None, engine: str = 'netcdf4', update_file = True) -> xarray.Dataset:
    """
    Reads a compressed xarray Dataset from a NetCDF or ZIP file.

    Parameters
    ----------
    input_filepath : str
        The path to the input file.
    output_path : str, optional
        The path to extract the file to if it is a ZIP archive, by default None
    engine : str, optional
        The NetCDF engine to use, by default 'netcdf4'

    Returns
    -------
    xarray.Dataset
        The loaded Dataset.
    """
    input_path = os.path.dirname(input_filepath)
    input_filepath = os.path.basename(input_filepath)
    output_path = output_path if output_path else input_path
    
    if input_filepath.endswith('zip'):
        with zipfile.ZipFile(os.path.join(input_path,input_filepath), 'r') as zip_object:
            zip_object.extractall(output_path)        
        input_filepath = input_filepath.replace('.zip', '.nc')

    output_filepath = os.path.join(output_path, input_filepath)
    
    with xarray.open_dataset(output_filepath, engine = engine) as ds:
        data = ds.copy()
    
    del ds    
    
    data = data.astype(float)
    scale_values_factor = data.attrs['scale_values_factor']

    if data.attrs.get('dtype', None):
        data.attrs['dtype'] = 'float'
    
    data = data/scale_values_factor
    nodata = data.attrs.get('nodata', None)
    if nodata:
        data.attrs['nodata'] = nodata/scale_values_factor
        for k in data.data_vars.keys():
            data[k] = data[k].where(data[k] != nodata/scale_values_factor, np.nan)
    if update_file:
        encoding = set_encoding(data)
        data.to_netcdf(output_filepath, encoding = encoding, engine = engine)
        
    del data
