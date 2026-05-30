"""
Climate indices for Witches' Broom disease modelling.

Functions here compute threshold-based and spell-length indicators from
xarray Datasets produced by the climate data pipeline.  The expected
time dimension is named 'time'.

Usage example
-------------
from src.features.climate_indices import calculate_indices

indices_dict = {
    "vpd_lt_15":            ['vpd', 1.5],
    "n_vpd_spells":         ['vpd', 1.5, 7],
    "n_wet_spells":         ['pr', 1.0, 7],
    "n_dry_spells":         ['pr', 1.0, 7],
    "avg_wet_spell_duration": ['pr', 1.0, 7],
    "avg_dry_spell_duration": ['pr', 1.0, 7],
    "rh_85_90_days":        [['dailyhr','hr06','hr09','hr12','hr15','hr18'], [85, 90], ['>=','<=']],
    "tmean_25_30_days":     ['tmean', [25, 30], ['>=','<=']],
    "max_temp_days":        ['tmax', 32],
    "precip_max_15d":       ['pr'],
    "max_hr_days":          ['dailyhr', 80],
    "consecutive_dry_days": ['pr', 1.0],
    "growing_degree_days":  ['tmean', 15],
    "daily_intensity_index":['pr', 1.0],
    "disease_pressure_index": ['vpd'],
}

indices_ds = calculate_indices(ds, indices_dict)
"""

import operator
import numpy as np
import xarray
import xclim
from xclim.indices import run_length


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize(da):
    """Min-max normalisation to [0, 1]."""
    return (da - da.min()) / (da.max() - da.min())


def set_variable_units(xrdata, variable):
    """Attach CF-compliant unit string to a DataArray's attrs."""
    unit_map = {
        'temp': 'degC',
        'hr':   '%',
        'prec': 'mm/d',
        'vpd':  'kPa',
    }
    xrdata.attrs['units'] = unit_map[variable]
    return xrdata


def mask_operation(xrdata, op, value, unit):
    """Return a boolean mask after unit-aware comparison via xclim."""
    ops = {
        '>':  operator.gt,
        '<':  operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
    }
    thresh = xclim.core.units.convert_units_to(f"{value} {unit}", xrdata)
    return ops[op](xrdata, thresh)


def threshold_days(xrclimatedata, value, meteorological_var, op='>='):
    """Count days where *xrclimatedata* satisfies *op* against *value*."""
    xrclimatedata = set_variable_units(xrclimatedata, meteorological_var)
    mask = mask_operation(xrclimatedata, op, value, xrclimatedata.attrs['units'])
    return mask.sum(dim='date')


def multiple_threshold_days(xrclimatedata, values, op_symbols, meteorological_var):
    """Count days satisfying multiple simultaneous threshold conditions."""
    xrclimatedata = set_variable_units(xrclimatedata, meteorological_var)
    mask = xarray.ones_like(xrclimatedata)
    op_symbols = list(op_symbols) if not isinstance(op_symbols, list) else op_symbols
    values = list(values) if not isinstance(values, list) else values
    for op, v in zip(op_symbols, values):
        mask = mask * mask_operation(xrclimatedata, op, v, xrclimatedata.attrs['units'])
    return mask.sum(dim='date')


def consecutive_days(xrclimatedata, value, meteorological_var, op='>='):
    """Longest run of consecutive days satisfying the threshold condition."""
    xrclimatedata = set_variable_units(xrclimatedata, meteorological_var)
    mask = mask_operation(xrclimatedata, op, value, xrclimatedata.attrs['units'])
    return run_length.longest_run(mask, dim='date')


def get_avg_spell_length(precip_1d, threshold=1.0, window=5,
                         cond_op='>=', window_op='>='):
    """
    Average length of spells (runs) exceeding *window* days.
    Returns 0.0 instead of np.nan if no spells are found.
    """
    ops = {
        '>':  operator.gt,
        '<':  operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
    }
    mask = ops[cond_op](precip_1d, threshold)
    padded = np.pad(mask, (1, 1), mode='constant', constant_values=False)
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    lengths = ends - starts
    valid = lengths[ops[window_op](lengths, window)]
    return np.mean(valid) if len(valid) > 0 else 0.0

def calculate_indices(xrdata, climate_indices):
    """
    Compute a set of climate indices from a meteorological xarray Dataset.
    """
    out = {}
    idx_map = {idx.name: idx for idx in climate_indices}
    
    # Track the total number of valid days in this specific slice
    total_days = len(xrdata.date)

    if "vpd_lt_15" in idx_map:
        idx = idx_map["vpd_lt_15"]
        var = idx.meteorological_variables[0]
        thresh = idx.parameters.get("threshold", 1.5) if idx.parameters else 1.5
        
        if total_days > 0:
            out["vpd_lt_15"] = (
                threshold_days(xrdata[var], thresh, meteorological_var='vpd', op='<=')
                / total_days
            ) * 100
        else:
            out["vpd_lt_15"] = xarray.full_like(xrdata[var]._pick_first_or_similar(), np.nan)

    if "n_vpd_spells" in idx_map:
        idx = idx_map["n_vpd_spells"]
        var = idx.meteorological_variables[0]
        params = idx.parameters or {}
        thresh = params.get("threshold", 1.5)
        window = params.get("min_duration_days", 7)
        
        # Guard 1: Ensure array length meets minimum spell window requirements
        if total_days >= window:
            is_humid = xrdata[var] <= thresh
            out["n_vpd_spells"] = run_length.windowed_run_events(
                is_humid, window=window, dim='date'
            )
        else:
            out["n_vpd_spells"] = xarray.zeros_like(xrdata[var].isel(date=0, drop=True), dtype=float)

    if "n_wet_spells" in idx_map:
        idx = idx_map["n_wet_spells"]
        var = idx.meteorological_variables[0]
        params = idx.parameters or {}
        thresh = params.get("threshold_mm", 1.0)
        min_duration = params.get("min_duration_days", 7)
        
        # Guard 2: Protect against ZeroDivisionError
        if total_days >= min_duration:
            out["n_wet_spells"] = run_length.windowed_run_events(
                xrdata[var] >= thresh, window=min_duration, dim='date'
            )
        else:
            out["n_wet_spells"] = xarray.zeros_like(xrdata[var].isel(date=0, drop=True), dtype=float)

    if "n_dry_spells" in idx_map:
        idx = idx_map["n_dry_spells"]
        var = idx.meteorological_variables[0]
        params = idx.parameters or {}
        thresh = params.get("threshold_mm", 1.0)
        min_duration = params.get("min_duration_days", 7)
        
        # Guard 3: Protect against ZeroDivisionError
        if total_days >= min_duration:
            out["n_dry_spells"] = run_length.windowed_run_events(
                xrdata[var] < thresh, window=min_duration, dim='date'
            )
        else:
            out["n_dry_spells"] = xarray.zeros_like(xrdata[var].isel(date=0, drop=True), dtype=float)
    
    if "heat_wave_duration" in idx_map:
        idx = idx_map["heat_wave_duration"]
        var = idx.meteorological_variables[0]
        params = idx.parameters or {}
        thresh = params.get("thresh", 28.0)
        min_duration_days = params.get("min_duration_days", 5)
        
        if total_days >= min_duration_days:
            out["heat_wave_duration"] = xarray.apply_ufunc(
                get_avg_spell_length,
                xrdata[var],
                kwargs={'threshold': thresh, 'window': min_duration_days,
                        'cond_op': '>=', 'window_op': '>='},
                input_core_dims=[['date']],
                output_core_dims=[[]],
                vectorize=True,
                dask='allowed',
                output_dtypes=[float],
            )
        else:
            # Ensures 0.0 is passed up if the temporal slice is completely empty
            out["heat_wave_duration"] = xarray.zeros_like(xrdata[var].isel(date=0, drop=True), dtype=float)

    if "cold_wave_duration" in idx_map:
        idx = idx_map["cold_wave_duration"]
        var = idx.meteorological_variables[0]
        params = idx.parameters or {}
        thresh = params.get("thresh", 5.0)
        min_duration_days = params.get("min_duration_days", 5)
        
        if total_days >= min_duration_days:
            out["cold_wave_duration"] = xarray.apply_ufunc(
                get_avg_spell_length,
                xrdata[var],
                kwargs={'threshold': thresh, 'window': min_duration_days,
                        'cond_op': '<=', 'window_op': '>='},
                input_core_dims=[['date']],
                output_core_dims=[[]],
                vectorize=True,
                dask='allowed',
                output_dtypes=[float],
            )
        else:
            out["cold_wave_duration"] = xarray.zeros_like(xrdata[var].isel(date=0, drop=True), dtype=float)

    if "avg_wet_spell_duration" in idx_map:
        idx = idx_map["avg_wet_spell_duration"]
        var = idx.meteorological_variables[0]
        params = idx.parameters or {}
        thresh = params.get("threshold_mm", 1.0)
        min_duration = params.get("min_duration_days", 7)
        
        if total_days >= min_duration:
            out["avg_wet_spell_duration"] = xarray.apply_ufunc(
                get_avg_spell_length,
                xrdata[var],
                kwargs={'threshold': thresh, 'window': min_duration,
                        'cond_op': '>=', 'window_op': '>='},
                input_core_dims=[['date']],
                output_core_dims=[[]],
                vectorize=True,
                dask='allowed',
                output_dtypes=[float],
            )
        else:
            out["avg_wet_spell_duration"] = xarray.zeros_like(xrdata[var].isel(date=0, drop=True), dtype=float)

    if "avg_dry_spell_duration" in idx_map:
        idx = idx_map["avg_dry_spell_duration"]
        var = idx.meteorological_variables[0]
        params = idx.parameters or {}
        thresh = params.get("threshold_mm", 1.0)
        window = params.get("min_duration_days", 7)
        
        if total_days >= window:
            out["avg_dry_spell_duration"] = xarray.apply_ufunc(
                get_avg_spell_length,
                xrdata[var],
                kwargs={'threshold': thresh, 'window': window,
                        'cond_op': '<', 'window_op': '>='},
                input_core_dims=[['date']],
                output_core_dims=[[]],
                vectorize=True,
                dask='allowed',
                output_dtypes=[float],
            )
        else:
            out["avg_dry_spell_duration"] = xarray.zeros_like(xrdata[var].isel(date=0, drop=True), dtype=float)
    
    if "rh_85_90_days" in idx_map:
        idx = idx_map["rh_85_90_days"]
        var_names = idx.meteorological_variables
        params = idx.parameters or {}
        thresholds = params.get("thresholds", [85, 90])
        op_symbols = params.get("op_symbols", [">=", "<="])
        
        for var in var_names:
            out[f'{var}_85_90_days'] = multiple_threshold_days(
                xrdata[var], values=thresholds,
                op_symbols=op_symbols, meteorological_var='hr'
            )

    if "tmean_25_30_days" in idx_map:
        idx = idx_map["tmean_25_30_days"]
        var = idx.meteorological_variables[0]
        params = idx.parameters or {}
        thresholds = params.get("thresholds", [25, 30])
        op_symbols = params.get("op_symbols", [">=", "<="])
        
        out["tmean_25_30_days"] = multiple_threshold_days(
            xrdata[var], values=thresholds,
            op_symbols=op_symbols, meteorological_var='hr'
        )

    if "max_temp_days" in idx_map:
        idx = idx_map["max_temp_days"]
        var = idx.meteorological_variables[0]
        thresh = idx.parameters.get("threshold_celsius", 35.0) if idx.parameters else 35.0
        
        out["max_temp_days"] = threshold_days(
            xrdata[var], thresh, meteorological_var='temp', op='>='
        )

    rhoptions = ["hr", "hr06", "hr09", "hr12", "hr15", "hr18"]
    for option in rhoptions:
        option_name = f"max_{option}_days"
        if option_name in idx_map:
            idx = idx_map[option_name]
            var = idx.meteorological_variables[0]
            thresh = idx.parameters.get("threshold_percent", 80.0) if idx.parameters else 80.0
            
            out[option_name] = threshold_days(
                xrdata[var], thresh, meteorological_var='hr', op='>='
            )

    if "precip_max_15d" in idx_map:
        idx = idx_map["precip_max_15d"]
        var = idx.meteorological_variables[0]
        if total_days >= 15:
            out["precip_max_15d"] = xrdata[var].rolling(date=15).sum().max(dim='date')
        else:
            out["precip_max_15d"] = xarray.zeros_like(xrdata[var].isel(date=0, drop=True), dtype=float)

    if "consecutive_dry_days" in idx_map:
        idx = idx_map["consecutive_dry_days"]
        var = idx.meteorological_variables[0]
        thresh = idx.parameters.get("threshold_mm", 1.0) if idx.parameters else 1.0
        
        if total_days > 0:
            out["consecutive_dry_days"] = consecutive_days(
                xrdata[var], value=thresh, meteorological_var='prec', op='<'
            )
        else:
            out["consecutive_dry_days"] = xarray.zeros_like(xrdata[var].isel(date=0, drop=True), dtype=float)

    if "growing_degree_days" in idx_map:
        idx = idx_map["growing_degree_days"]
        var = idx.meteorological_variables[0]
        tbase = idx.parameters.get("base_temperature", 15.0) if idx.parameters else 15.0
        
        out["growing_degree_days"] = (xrdata[var] - tbase).sum(
            dim='date', keep_attrs=True
        )

    if "daily_intensity_index" in idx_map:
        idx = idx_map["daily_intensity_index"]
        var = idx.meteorological_variables[0]
        thresh = idx.parameters.get("threshold_mm", 1.0) if idx.parameters else 1.0
        
        out["daily_intensity_index"] = xrdata[var].where(
            xrdata[var] >= thresh
        ).mean(dim='date', keep_attrs=True)

    if "disease_pressure_index" in idx_map:
        idx = idx_map["disease_pressure_index"]
        vpd_var = idx.meteorological_variables[0]
        
        if "max_hr_days" in out and "daily_intensity_index" in out:
            rh_norm  = normalize(out["max_hr_days"])
            pi_norm  = normalize(out["daily_intensity_index"])
            vpd_norm = normalize(xrdata[vpd_var].mean(dim='date', keep_attrs=True))
            out["disease_pressure_index"] = (
                (rh_norm * pi_norm) / vpd_norm
            ).clip(min=0, max=1)

    # Clean up single-variable nested sub-datasets safely
    for key, value in out.items():
        if isinstance(value, xarray.Dataset):
            inner = list(value.data_vars)[0]
            out[key] = value[inner]

    return xarray.Dataset(out)