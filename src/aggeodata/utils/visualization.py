"""Plotting utilities for climate indices and disease data."""

import math
import matplotlib.pyplot as plt


def plot_facet_grid(xrdata, ncols=3, figsize=None):
    """
    Grid of spatial maps, one panel per variable in *xrdata*.

    Parameters
    ----------
    xrdata : xarray.Dataset
    ncols : int
    figsize : tuple, optional
    """
    variables = list(xrdata.data_vars)
    n_vars = len(variables)
    nrows = math.ceil(n_vars / ncols)
    if figsize is None:
        figsize = (ncols * 5, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, var_name in enumerate(variables):
        ax = axes[i]
        xrdata[var_name].plot(
            ax=ax,
            robust=True,
            cmap='YlGnBu',
            cbar_kwargs={'label': getattr(xrdata[var_name], 'units', var_name)},
        )
        ax.set_title(var_name.upper())
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_facet_grid_with_points(xr_ds, points_gdf=None, ncols=3, figsize=(15, 12)):
    """
    Grid of spatial maps with optional disease observation points overlaid.

    Parameters
    ----------
    xr_ds : xarray.Dataset
    points_gdf : geopandas.GeoDataFrame, optional
        Observation locations drawn as red dots.
    ncols : int
    figsize : tuple
    """
    variables = list(xr_ds.data_vars)
    variables.remove('spatial_ref') if 'spatial_ref' in variables else None
    n_vars = len(variables)
    nrows = math.ceil(n_vars / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, var_name in enumerate(variables):
        
        ax = axes[i]
        xr_ds[var_name].plot(
            ax=ax,
            robust=True,
            cmap='YlGnBu',
            cbar_kwargs={'label': getattr(xr_ds[var_name], 'units', '')},
        )
        if points_gdf is not None:
            points_gdf.plot(
                ax=ax,
                color='red',
                markersize=15,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8,
            )
        ax.set_title(var_name.upper())
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_climate_timeseries(xrdata, method: str = "spatial_mean", x_loc: float = None, y_loc: float = None):
    """
    Generates a high-quality timeseries plot for tmax and tmin from an xarray Dataset.

    Parameters
    ----------
    xrdata : xarray.Dataset
        Input dataset with 'date', 'x', and 'y' dimensions.
    method : str
        Either 'spatial_mean' (average over all x/y grid cells) or 'point' (extract specific coordinate).
    x_loc : float, optional
        Target X coordinate value (required if method='point').
    y_loc : float, optional
        Target Y coordinate value (required if method='point').
    """
    # 1. Clean slicing to isolate only the target variables
    ds_subset = xrdata[["tmax", "tmin"]]
    
    # 2. Extract time series depending on selection strategy
    if method == "spatial_mean":
        # Average over spatial dimensions, maintaining the date tracking timeline
        ts_data = ds_subset.mean(dim=["x", "y"])
        title_suffix = " (Regional Spatial Mean)"
        
    elif method == "point":
        if x_loc is None or y_loc is None:
            raise ValueError("You must provide x_loc and y_loc when method='point'")
        # Select the nearest grid pixel match to your target location coordinates
        ts_data = ds_subset.sel(x=x_loc, y=y_loc, method="nearest")
        title_suffix = f" at Location (X: {x_loc}, Y: {y_loc})"
        
    else:
        raise ValueError("Invalid method. Choose 'spatial_mean' or 'point'.")

    # 3. Render the graphics canvas
    plt.figure(figsize=(12, 5), dpi=100)
    
    # Plot maximum and minimum paths dynamically tracing the 'date' coordinate index
    plt.plot(ts_data.date, ts_data["tmax"], label="Max Temperature (tmax)", color="#d95f02", linewidth=1.5)
    plt.plot(ts_data.date, ts_data["tmin"], label="Min Temperature (tmin)", color="#7570b3", linewidth=1.5)
    
    # Style allocations (Clean layout best practices)
    plt.title(f"Temperature Profile Over Time{title_suffix}", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Timeline (Date)", fontsize=11, labelpad=10)
    plt.ylabel("Temperature (°C)", fontsize=11, labelpad=10)
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right", fontsize=10)
    plt.tight_layout()
    
    # Render view output box frame
    plt.show()
