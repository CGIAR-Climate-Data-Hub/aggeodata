"""
Meteorological summary statistics for Witches' Broom disease modelling.

Computes mean and accumulated statistics over a time window from an
xarray Dataset.  Also adds 6 monthly diurnal temperature difference
bands when requested.

Usage example
-------------
from src.features.met_summaries import calculate_meteorological_summaries

variables_dict = {
    "temp_accum":     ['tmean'],
    "hr_accum":       ['dailyhr'],
    "hr06_accum":     ['hr06'],
    "hr09_accum":     ['hr09'],
    "hr12_accum":     ['hr12'],
    "hr15_accum":     ['hr15'],
    "hr18_accum":     ['hr18'],
    "dewtemp_accum":  ['dpt'],
    "vpd_accum":      ['vpd'],
    "etr_accum":      ['etr'],
    "srad_accum":     ['srad'],
    "prec_accum":     ['precipitation'],
    "dtd_monthly":    ['tmax', 'tmin'],   # optional: 6 monthly bins
}

summary_ds = calculate_meteorological_summaries(ds, variables_dict)
"""

import numpy as np
import pandas as pd
import xarray


def calculate_meteorological_summaries(xrdata, summaries_config):
    """
    Compute per-pixel mean and accumulated statistics over the time window.

    Parameters
    ----------
    xrdata : xarray.Dataset
        Requires a 'time' dimension.
    summaries_config : dict
        Keys follow the pattern ``<name>_avg`` (time-mean) or
        ``<name>_accum`` (time-sum).  The value is a list with the
        variable name(s) from *xrdata*.

        Special key ``dtd_monthly`` expects ``[tmax_var, tmin_var]`` and
        produces six monthly diurnal temperature difference bands
        (``dtd_m1`` … ``dtd_m6``).

    Returns
    -------
    xarray.Dataset
    """
    summaries = {}

    for cgf in summaries_config:

        var_name = cgf.meteorological_variable[0] if isinstance(cgf.meteorological_variable, list) else cgf.meteorological_variable
        summary_function = cgf.summary_function
        if summary_function == 'mean':
            colname = f"{var_name}_avg"
            summaries[colname] = xrdata[var_name].mean(dim='date', keep_attrs=True)
            
        elif summary_function == 'sum':
            colname = f"{var_name}_accum"
            summaries[colname] = xrdata[var_name].sum(dim='date', keep_attrs=True)
        else:
            raise ValueError(f"Unsupported summary_function '{summary_function}' in config")
    
    # Unwrap single-variable Datasets returned by some xarray operations
    for key, value in summaries.items():
        if isinstance(value, xarray.Dataset):
            inner = list(value.data_vars)[0]
            summaries[key] = value[inner]

    summaries_ds = xarray.Dataset(summaries)


    return summaries_ds
