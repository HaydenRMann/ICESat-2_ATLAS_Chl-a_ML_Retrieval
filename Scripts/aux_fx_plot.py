"""

Home to auxilary plotting functions

"""
import os
import sys
import glob
import shutil
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd


import geopandas as gpd
import rasterio
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.features import rasterize
from shapely.geometry import LineString, Polygon
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis, pearsonr


import xarray as xr
import earthaccess as ea
from sliderule import icesat2, sliderule
from harmony import (
    BBox,
    CapabilitiesRequest,
    Client,
    Collection,
    JobsRequest,
    LinkType,
    Request,
)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import contextily as cx
import seaborn as sns
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


from aux_fx_process import (
    grid_data,
    add_lon_lat,
    get_photons,
    water_mask,
    water_surface_retrieval_and_binning,
    get_segment_vars,
    get_PACE_time_windows,
    get_PACE_data,
    gridded_chl,
    get_segment_centers,
    match_chl_to_segments,
)
from aux_fx_plot import ATLAS_overlays_TX, make_scatter_plot


ea.login()
# ea.login(strategy="interactive", persist=True)

THRESHOLD_LIMIT = 200




""" 

ATLAS Variable Overlays

"""
# 

def ATLAS_overlays_TX(matchup_df, variable, label, norm_col, filename, params):
    """"
    Plots ATLAS variable overlays over Texas
    """
    # extract plotting parameters
    pLAT_MIN, pLAT_MAX, pLON_MIN, pLON_MAX, resolution, date_array = params

    # Set xlim and ylim
    xlim = ([pLON_MIN + resolution[0], pLON_MAX - resolution[0]])
    ylim = ([pLAT_MIN + resolution[1], pLAT_MAX - resolution[1]])

    # Get all PACE files for the date
    dict_date_files = {}

    for date_str in date_array:
        # Convert to timestamp to do safe calendar math
        current_date = pd.to_datetime(date_str)
        pre_date_dt = current_date - pd.Timedelta(days=1)
        post_date_dt = current_date + pd.Timedelta(days=1)
        
        # Format them back to strings matching pace
        date_pace = current_date.strftime("%Y%m%d")
        pre_pace  = pre_date_dt.strftime("%Y%m%d")
        post_pace = post_date_dt.strftime("%Y%m%d")

        # +/- 12 hours
        patterns = [
            f"DATA/PACE_l2_Texas/*{date_pace}*.nc",
            f"DATA/PACE_l2_Texas/*{pre_pace}T1[2-9]*.nc",
            f"DATA/PACE_l2_Texas/*{pre_pace}T2[0-4]*.nc",
            f"DATA/PACE_l2_Texas/*{post_pace}T0[0-9]*.nc",
            f"DATA/PACE_l2_Texas/*{post_pace}T1[0-2]*.nc"
        ]
        
        matched_files = []
        for pattern in patterns:
            matched_files.extend(glob.glob(pattern))
            
        dict_date_files[date_str] = matched_files


    # instantiate plot
    fig, ax = plt.subplots(
        figsize=(8, 5.5),
        constrained_layout=True
    )

    # plot parameters so they are standardized
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    # Loop through all marked PACE files
    i = 0
    for date_key in dict_date_files:
        axis = ax

        gridded_list = []

        # format and grid PACE l2m data
        for file in dict_date_files[date_key]:
            dt = xr.open_datatree(file)
            ds = xr.merge(dt.to_dict().values())
            ds = ds.set_coords(("longitude", "latitude"))

            ds_gridded = grid_data(ds, resolution)

            # Drop the original swath dimensions so xarray won't try to align them. Otherwise my code was getting screwed up. Thanks Claude for the help with this line
            dims_to_drop = [d for d in ["number_of_lines", "pixels_per_line"] if d in ds_gridded.dims]
            coords_to_drop = [c for c in ["number_of_lines", "pixels_per_line"] if c in ds_gridded.coords]
            
            ds_gridded = ds_gridded.drop_dims(dims_to_drop).drop_vars(coords_to_drop, errors="ignore")

            gridded_list.append(ds_gridded)

        # one object for PACE data for the day
        combined = xr.concat(gridded_list, dim="swath")
        ds_gridded = combined.mean(dim="swath", skipna=True)

        # map PACE data
        chl_norm = LogNorm(vmin=0.01, vmax=10.0)
        im = ds_gridded["chlor_a"].plot(
            ax=axis,
            cmap="viridis",
            norm=chl_norm,
            add_colorbar=False,
        )


        # LIDAR overlay
        df_day_segments = matchup_df[
            pd.to_datetime(matchup_df['date']).dt.strftime("%Y-%m-%d") == date_key
        ]
        
        if len(df_day_segments) > 0:
            df_day_segments.plot(
            ax=axis,
            column=variable,
            cmap="magma",
            linewidth=5,
            norm=norm_col,
            legend=False,
        )
        

        # Land mask
        land = gpd.read_file("ne_10m_land.shp")
        land = land.to_crs(epsg=4326)
        land.plot(ax=axis, facecolor="lightgray", edgecolor="black", linewidth=0.5, zorder=2)


        # Variable Colorbar
        cax1 = inset_axes(
            axis, width="3%", height="100%",    
            loc="center left",
            bbox_to_anchor=(1.02, 0.0, 1, 1),  
            bbox_transform=axis.transAxes,
            borderpad=0,
        )
        cb1 = fig.colorbar(
            cm.ScalarMappable(norm=norm_col, cmap="magma"),
            cax=cax1
        )
        cb1.set_label(label)


        # PACE Chl-A Colorbar
        cax2 = inset_axes(
            axis, width="3%", height="100%",
            loc="center left",
            bbox_to_anchor=(1.22, 0.0, 1, 1),  
            bbox_transform=axis.transAxes,
            borderpad=0,
        )
        cb2 = fig.colorbar(cm.ScalarMappable(norm=chl_norm, cmap="viridis"), cax=cax2)
        cb2.set_label("Chl-a [mg m$^{-3}$]")

        # format
        axis.set_xlim([xlim[0] + resolution[0], xlim[1] - resolution[0]])
        axis.set_ylim([ylim[0] + resolution[1], ylim[1] - resolution[1]])
        axis.set_title("")
        axis.set_xlabel("Longitude", fontsize=16)
        axis.set_ylabel("Latitude", fontsize=16)

        # increment loop
        i = i+1


    # save and plot

    fig.canvas.draw()

    fig.savefig(
        filename,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="white"
    )

    plt.show()
    return None




def make_scatter_plot(df, x_col, xlabel, xlim, ylim, savepath, text_xy=(0.66, 0.95), x_is_log=False):
    """
    ATLAS variable vs PACE chl-a scatter plots.
    Parameter-focused fx.
    """

    # Plot parameters
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    # instantiate plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    # regression plot of variable vs PACE log chl-a
    sns.regplot(
        data=df,
        x=x_col,
        y="log_chl_a",
        scatter_kws={"alpha": 0.005},
        line_kws={"color": "red"},
        truncate=True
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PACE l2m Chl-a [mg m$^{-3}$]")

    # format y-axes (Chl-a)
    y_tick_values = [0.01, 0.1, 1, 10, 100]
    y_tick_labels = ["0.01", "0.1", "1.0", "10", "100"]
    ax.set_yticks(np.log10(y_tick_values))
    ax.set_yticklabels(y_tick_labels)

    # Format x-axes
    # x_is_log is true for R_sw only.
    if x_is_log:
        x_tick_values = np.array([0.01, 0.1, 1, 10, 100, 1000])
        x_tick_positions = np.log10(x_tick_values)
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(["0.01", "0.1", "1.0", "10", "100", "1000"])

    plt.xlim(xlim)
    plt.ylim(ylim)

    # Calculate r, p 
    x = df[x_col].values
    y = df["log_chl_a"].values
    mask = np.isfinite(x) & np.isfinite(y)

    corr, p_value = pearsonr(x[mask], y[mask])

    # Display r, p
    ax.text(
        *text_xy,
        f"Correlation = {corr:.2f}\n"
        f"N = {len(df['log_chl_a'])}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )

    # Hides labels on mini lines
    ax.xaxis.set_minor_formatter(ticker.NullFormatter()) 

    # Find minor values for ticklines
    minor_values = []
    for exponent in [-2, -1, 0, 1, 2]:  #  0.01  to 100
        # gets 2-10
        minor_values.extend([i * (10 ** exponent) for i in range(2, 10)])

    # take log. so now we have small minor ticks at -[2,-1,0,1,2]*[0.01, 0.1, 1, 10] (stops at 100)
    minor_positions = np.log10(minor_values)

    # shows minor_x on x axis when log
    if x_is_log:
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs="auto", numticks=12))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_positions))
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
    # otherwise, basic located minor positions (none)
    ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_positions))

    # get rid of box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plot and save

    plt.savefig(savepath, dpi=600, facecolor="white")
    plt.show()

    # return r and p in case this is useful in the future
    return corr, p_value