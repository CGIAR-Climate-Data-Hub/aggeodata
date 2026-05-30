"""Spatial extraction and raster utilities."""
from datetime import datetime, timedelta
from aggeodata.config.schemas import ClimateConfig
from aggeodata.features.climate_indices import calculate_indices
from aggeodata.features.met_summaries import calculate_meteorological_summaries
import pandas as pd
import xarray


def extracting_using_gpdf(gpdf, xrdata):
    """
    Extract xarray values at point locations defined by a GeoDataFrame.

    Nearest-neighbor lookup in x/y.  Returns a DataFrame with the
    original GeoDataFrame columns joined to the extracted climate values.
    """
    x_coords = xarray.DataArray(gpdf.geometry.x, dims='z')
    y_coords = xarray.DataArray(gpdf.geometry.y, dims='z')
    extracted = xrdata.sel(x=x_coords, y=y_coords, method='nearest')
    df_extracted = extracted.to_dataframe().reset_index(drop=True)
    return pd.concat([gpdf.reset_index(drop=True), df_extracted], axis=1)


def process_temporal_windows(
    xrdata: xarray.Dataset,
    climate_config: ClimateConfig,
    eval_date: datetime,
    window_months: int,
    lookback_months: int | None,
):

    window_days = window_months * 30
    windows = {}
    
    if lookback_months and lookback_months < window_months:
        lookback_days = lookback_months * 30
        
        label_primary = f"m1_m{lookback_months}"
        label_historical = f"m{lookback_months + 1}_m{window_months}"
        
        windows[label_primary] = (
            eval_date - timedelta(days=window_days), 
            eval_date - timedelta(days=lookback_days)
        )
        windows[label_historical] = (
            eval_date - timedelta(days=lookback_days-1), 
            eval_date 
        )
    else:
        label_single = f"m1_m{window_months}"
        windows[label_single] = (
            eval_date - timedelta(days=window_days), 
            eval_date
        )

    processed_blocks = {}

    for label, (wstart, wend) in windows.items():
        print(label, (wstart, wend))
        w = xrdata.sel(date=slice(wstart, wend))
        summaries = calculate_meteorological_summaries(w, climate_config.summarizations)
        indices = calculate_indices(w, climate_config.indices)
        merged = xarray.merge([summaries, indices])
        merged = merged.rename({v: f"{label}_{v}" for v in merged.data_vars})
        processed_blocks[label] = merged

    return xarray.merge(list(processed_blocks.values()))