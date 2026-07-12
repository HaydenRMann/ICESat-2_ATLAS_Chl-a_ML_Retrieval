"""

Home to auxilary functions

"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import xarray as xr
import rasterio
from rasterio.enums import Resampling
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from matplotlib.ticker import FuncFormatter
import numpy as np
import earthaccess as ea
import pandas as pd
import glob
import xarray as xr
from datetime import datetime
from pathlib import Path
import rasterio
from IPython.display import JSON
from rasterio.enums import Resampling



""" 

Grid L2m data

"""
# Via: https://nasa.github.io/oceandata-notebooks/notebooks/oci/subsetting_with_harmony-py.html

def grid_data(src, resolution, dst_crs="epsg:4326", resampling=Resampling.nearest):
    """
    Reproject a L2 dataset to match an input grid. Makes sure 3D variables are
        in (Z, Y, X) dimension order, and all variables have spatial dims/crs 
        assigned.
    Args:
        src - an xarray dataset or dataarray to reproject
        resolution - resolution of the output grid, in dst_crs units
        dst_crs - CRS of the output data
        resampling - resampling method (see rasterio.enums)
    Returns:
        dst - projected xr dataset
    """
    if (len(list(src.dims)) == 3) and (list(src.dims)[0] != "wavelength_3d"):
        src = src.transpose("wavelength_3d", ...)
    src = src.rio.set_spatial_dims("pixels_per_line", "number_of_lines")
    src = src.rio.write_crs("epsg:4326")

    # Calculating the default affine transform
    defaults = rasterio.warp.calculate_default_transform(
        src.rio.crs,
        dst_crs,
        src.rio.width,
        src.rio.height,
        left=src.attrs["geospatial_lon_min"],
        bottom=src.attrs["geospatial_lat_min"],
        right=src.attrs["geospatial_lon_max"],
        top=src.attrs["geospatial_lat_max"],
    )
    # Aligning that transform to our desired resolution
    transform, width, height = rasterio.warp.aligned_target(*defaults, resolution)
    
    # Run projection
    dst = src.rio.reproject(
        dst_crs=dst_crs,
        shape=(height, width),
        transform=transform,
        src_geoloc_array=(
            src["longitude"],
            src["latitude"],
        ),
        nodata=np.nan,
        resample=resampling,
    )
    dst["x"] = dst["x"].round(9)
    dst["y"] = dst["y"].round(9)
    
    return dst.rename({"x": "longitude", "y": "latitude"})


""" 

ATLAS Variable Overlays

"""
def ATLAS_overlays(matchup_df, variable, label, norm_col, filename, params):
    pLAT_MIN, pLAT_MAX, pLON_MIN, pLON_MAX, resolution, date_array = params

    xlim = ([pLON_MIN + resolution[0], pLON_MAX - resolution[0]])
    ylim = ([pLAT_MIN + resolution[1], pLAT_MAX - resolution[1]])
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

        patterns = [
            f"DATA/PACE_l2_NC/*{date_pace}*.nc",
            f"DATA/PACE_l2_NC/*{pre_pace}T1[2-9]*.nc",
            f"DATA/PACE_l2_NC/*{pre_pace}T2[0-4]*.nc",
            f"DATA/PACE_l2_NC/*{post_pace}T0[0-9]*.nc",
            f"DATA/PACE_l2_NC/*{post_pace}T1[0-2]*.nc"
        ]
        
        matched_files = []
        for pattern in patterns:
            matched_files.extend(glob.glob(pattern))
            
        dict_date_files[date_str] = matched_files

    fig, ax = plt.subplots(
        figsize=(8, 5.5),
        constrained_layout=True
    )

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

    i = 0
    for date_key in dict_date_files:
        axis = ax

        gridded_list = []

        for file in dict_date_files[date_key]:
            dt = xr.open_datatree(file)
            ds = xr.merge(dt.to_dict().values())
            ds = ds.set_coords(("longitude", "latitude"))

            ds_gridded = grid_data(ds, resolution)

            # Drop the original swath dimensions so xarray won't try to align them
            dims_to_drop = [d for d in ["number_of_lines", "pixels_per_line"] if d in ds_gridded.dims]
            coords_to_drop = [c for c in ["number_of_lines", "pixels_per_line"] if c in ds_gridded.coords]
            
            ds_gridded = ds_gridded.drop_dims(dims_to_drop).drop_vars(coords_to_drop, errors="ignore")

            gridded_list.append(ds_gridded)

        combined = xr.concat(gridded_list, dim="swath")
        ds_gridded = combined.mean(dim="swath", skipna=True)

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

        # vmin = df_day_segments[variable].min()
        # vmax = df_day_segments[variable].max()

        if len(df_day_segments) > 0:
            df_day_segments.plot(
            ax=axis,
            column=variable,
            cmap="magma",
            linewidth=6,
            norm=norm_col,
            legend=False,
        )

        

        land = gpd.read_file("ne_10m_land.shp")
        land = land.to_crs(epsg=4326)
        land.plot(ax=axis, facecolor="lightgray", edgecolor="black", linewidth=0.5, zorder=2)

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

        cax2 = inset_axes(
            axis, width="3%", height="100%",
            loc="center left",
            bbox_to_anchor=(1.22, 0.0, 1, 1),  
            bbox_transform=axis.transAxes,
            borderpad=0,
        )

        cb2 = fig.colorbar(cm.ScalarMappable(norm=chl_norm, cmap="viridis"), cax=cax2)
        cb2.set_label("Chl-a [mg m$^{-3}$]")

        axis.set_xlim([xlim[0] + resolution[0], xlim[1] - resolution[0]])
        axis.set_ylim([ylim[0] + resolution[1], ylim[1] - resolution[1]])
        axis.set_title("")
        axis.set_xlabel("Longitude", fontsize=16)
        axis.set_ylabel("Latitude", fontsize=16)
        i = i+1

    fig.canvas.draw()

    bbox = fig.get_tightbbox(fig.canvas.get_renderer())

    fig.savefig(
        filename,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="white"
    )

    plt.show()
    return None