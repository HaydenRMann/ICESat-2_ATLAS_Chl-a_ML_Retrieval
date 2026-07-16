"""

Home to auxilary functions

"""
"""
Standard imports for [SCRIPT NAME]
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





""" 

Grid PACE L2m data: Credit: The Ocean Ecology Laboratory at NASA Goddard

Directly Via: https://nasa.github.io/oceandata-notebooks/notebooks/oci/subsetting_with_harmony-py.html 
"""

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

FX: add_lon_lat

"""
def add_lon_lat(gdf):
    """Return a copy with lon/lat columns derived from GeoDataFrame geometry."""
    out = gdf.copy()
    if "geometry" in out.columns:
        out["lon"] = out.geometry.x
        out["lat"] = out.geometry.y
    return out



"""

FX: Get Photons

"""
def get_photons(window_start_str, window_end_str, aoi):
    """ 
    Calls sliderule to retrieve photons of a given time period and aoi. 
    Returns a dataframe of photons
    """
    base_parms = {
            "poly": aoi,
            "t0": window_start_str + "T00:00:00Z",
            "t1": window_end_str + "T23:59:59Z",
            "cnf": 1,
            "srt": 1,
            "atl03_ph_fields": ["pce_mframe_cnt"],
            "atl03_geo_fields": ["solar_elevation"],
    }     
    print("")
    print("")
    print(window_start_str)
    print(window_end_str)
    

    try:
        photons = sliderule.run("atl03x", base_parms)
    except Exception as e:
        return None

    print("Photons Retrieved")
    print(f"rows: {len(photons):,}")
    print(f"columns: {len(photons.columns)}")

    if len(photons) == 0:
        return None

    # df of all photons
    photons_xy = add_lon_lat(photons) 
    return photons_xy


""" 

Water Mask

"""
def water_mask(photons_xy, LAND):
    """ 
    Masks LiDAR photon data to only be over water + ocean.
    Returns LiDAR photon data, limited to water/ocean.

    Land mask: https://www.arcgis.com/home/item.html?id=595533fecdb0472db4b4b8e3ca8d9e42#overview
    """
    LAND = LAND.to_crs(photons_xy.crs)

    # define resolution (0.01° ~ 1km, adjust as needed)
    res = 0.01

    # Determine bounding box for the land mask
    # I could probably just do this for my bbox, but I just copied code from a previous project of mine for this
    minx, miny, maxx, maxy = -180, -90, 180, 90

    # calc number of raster pixels
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)

    # coodrinate transformation from pixel loc <--> lat/lon
    transform = rasterio.transform.from_bounds(
    minx, miny, maxx, maxy, width, height
    )

    # rasterize land polygon
    land_shapes = [(geom, 1) for geom in LAND.geometry]

    # put it onto an array/raster (bunch of 0s and 1s showing water or land status per pixel)
    land_raster = rasterize(
        land_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    # get photon coordinates
    xs = photons_xy.geometry.x.values
    ys = photons_xy.geometry.y.values

    # convert photon coordinates into raster coordinates
    col = ((xs - minx) / res).astype(np.int32)
    row = ((maxy - ys) / res).astype(np.int32)

    # Checks whether photon locs fall in water/land
    valid = (
    (col >= 0) & (col < width) &
    (row >= 0) & (row < height)
    )

    # array of all land vals
    on_land = np.zeros(len(photons_xy), dtype=bool)

    # mask where row and column (rasterized lat/lon) are both over land
    on_land[valid] = land_raster[row[valid], col[valid]] == 1

    # limit photons_xy to be oceans only
    photons_xy_ocean = photons_xy.loc[~on_land].copy()
    return photons_xy_ocean



"""

Water Surface Retrieval and Binning

"""

def water_surface_retrieval_and_binning(photons_xy_ocean, SEGMENT_LENGTH):
    """ 
    
    Credit: Dr. Kelsey Bisson & Eidam, E. F., Bisson, K., Wang, C., Walker, C., & Gibbons, A. (2024). 
    ICESat-2 and ocean particulates: A roadmap for calculating Kd from space-based lidar photon profiles.
    Remote Sensing of Environment, 311, 114222.
      
      
    Adapted by current author to retrieve water height for each individual ground track / beam, to
    account for variations in sea surface height.


    Via Eidam et al., 2024: For a lightweight ATL03-only first pass, estimate the water surface from the photon-height
      histogram. The densest height bin is treated as the candidate surface return, then `water_surface` keeps photons 
      within an adaptive vertical window around that modal height. This is an exploratory detector: inspect the histogram
        diagnostics and tune the binning/window before using the result quantitatively.

    If the water-surface plot looks like only one or two points, first check the printed counts. A sparse or clustered-looking
    plot can happen even with thousands of selected photons when only a few ICESat-2 pass/beam segments cross the small AOI 
    and `x_atc` is plotted as an absolute along-track coordinate.
    """

    water_surface_list = []
    below_surface_list = []

    for (rgt_val, cycle_val, gt_val), gt_photons in photons_xy_ocean.groupby(["rgt", "cycle", "gt"]):

        # get rid of -infinity and +infinity height readings
        height_values = gt_photons["height"].replace([np.inf, -np.inf], np.nan).dropna()

        # # skip passes with too few photons to build a meaningful histogram
        # if len(height_values) < 30:
        #     continue

        # build histogram range + bins + counts + edges
        hist_range = tuple(height_values.quantile([0.01, 0.99]))
        hist_bins = int(np.clip(np.sqrt(len(height_values)), 40, 160))
        hist_counts, hist_edges = np.histogram(height_values, bins=hist_bins, range=hist_range)

        # water surface bin + bin centers
        water_surface_bin = int(np.argmax(hist_counts))
        water_surface_center = 0.5 * (hist_edges[water_surface_bin] + hist_edges[water_surface_bin + 1])

        # determine peak count in histogram + bin width
        hist_peak_count = int(hist_counts[water_surface_bin])
        hist_bin_width_m = float(hist_edges[1] - hist_edges[0])

        # use bin width to determine the 'half window' for the water surface
        water_surface_half_window_m = max(0.75, 1.5 * hist_bin_width_m)

        # Form a mask for the water surface
        water_surface_mask = gt_photons["height"].between(
                water_surface_center - water_surface_half_window_m,
                water_surface_center + water_surface_half_window_m,
        )

        # assign along-track segment IDs now, while we still have just this GT's
        # photons. Grouping by rgt+cycle (gt is already fixed by the outer loop)
        # handles repeat satellite passes over the same ground track.
        gt_photons = gt_photons.copy()

        ## yes, this starts from 0 for each rgt/cycle track, but the binning variable code below works around this!
        gt_photons["segment_id"] = (
                gt_photons.groupby(["rgt", "cycle", "gt"])["x_atc"]
                .transform(lambda x: ((x - x.min()) // SEGMENT_LENGTH).astype(int))
        )

        # separate photons by height
        water_surface = gt_photons[water_surface_mask].copy()
        below_surface = gt_photons[gt_photons["height"] < water_surface_center - water_surface_half_window_m].copy()

        # determine water surface height resuidual
        water_surface["height_residual_m"] = water_surface["height"] - water_surface_center
        below_surface["height_residual_m"] = below_surface["height"] - water_surface_center


        # determine water surface fraction:
        water_surface_fraction = len(water_surface) / max(len(gt_photons), 1)

        water_surface_list.append(water_surface)
        below_surface_list.append(below_surface)

    # combine all gts back together
    water_surface = pd.concat(water_surface_list) if water_surface_list else pd.DataFrame()
    below_surface = pd.concat(below_surface_list) if below_surface_list else pd.DataFrame()
    return water_surface, below_surface


def get_segment_vars(below_surface, water_surface, THRESHOLD_LIMIT):
    """
    Returns the segment stats of the segment photon 'cloud':
        D10: 10th percentile photon depth
        D25: 25th percentile photon depth
        D50: 50th percentile photon depth
        D75: 75th percentile photon depth
        D90: 90th percentile photon depth
        N_subsurface: number of subsurface-designated photons
        N_surface: number of surface-designated photons
        R_sw: N_subsurface / N_surface
        Z_max: deepest photon
        mean_depth: mean photon depth
        std_of_depth: photon depth standard deviation
        skewness: photon depth cloud skewness
        kurtosis: photon depth cloud kurtosis
        geo: geographic center

        Calculated later:
            log_R_sw = np.log10(R_sw)
            D75/D25

    
    """

    def get_segment_stats(below_photons, surface_photons):
        """Returns variables listed above PER segment"""

        # no photons --> no stats :(
        if len(below_photons) == 0:
                return {
                    "D10": np.nan,
                    "D25": np.nan,
                    "D50": np.nan,
                    "D75": np.nan,
                    "D90": np.nan,
                    "N_subsurface": 0,
                    "N_surface": len(surface_photons),
                    "R_sw": 0 if len(surface_photons) > 0 else np.nan,
                    "z_max": np.nan,
                    "mean_depth": np.nan,
                    "std_of_depth": np.nan,
                    "skewness": np.nan,
                    "kurtosis": np.nan,
                    "geo": None
                }


        # photon_depth = np.asarray(below_photons["height"])
        photon_depth = np.asarray(below_photons["height_residual_m"])


        # inverse since height is inverted
        D90, D75, D50, D25, D10 = np.percentile(photon_depth, [10, 25, 50, 75, 90])

        # get stats
        N_subsurface = len(below_photons)
        N_surface = len(surface_photons)
        z_max = abs(np.min(photon_depth))
        R_sw = N_subsurface/N_surface if N_surface > 0 else np.nan
        mean_depth = np.mean(photon_depth)
        std_of_depth = np.std(photon_depth)

        skewness = skew(photon_depth)
        kurtosis_val = kurtosis(photon_depth)

        # segment_geo
        segment = below_photons.sort_values("x_atc")

        # convert photons geo into a linestring for plotting later
        segment_geo = LineString([
                segment.geometry.iloc[0],
                segment.geometry.iloc[-1]
        ])

        return {
                "D10": D10,
                "D25": D25,
                "D50": D50,
                "D75": D75,
                "D90": D90,

                "N_subsurface": N_subsurface,
                "N_surface": N_surface,

                "R_sw": R_sw,
                "z_max": z_max,

                "mean_depth": mean_depth,
                "std_of_depth": std_of_depth,
                "skewness": skewness,
                "kurtosis": kurtosis_val,
                "geo": segment_geo,
                "time": below_photons.index.mean()
        }


    rows = []

    # columns to differentiate each individual 1 km segment
    group_cols = ["rgt", "cycle", "gt", "segment_id"]
    below_groups = below_surface.groupby(group_cols)
    surface_groups = water_surface.groupby(group_cols)

    rows = []

    # loop through all groups/segments and get their stats
    for key, below_group in below_groups:

        # was running into error here, claude helped fix. just to make sure that groups exist 
        #   (isn't needed in the latest iteration, but did help as a safeguard once, so was kept in here)
        try:
                surface_group = surface_groups.get_group(key)
        except KeyError:
                surface_group = water_surface.iloc[0:0]  

        stats = get_segment_stats(below_group, surface_group)
        rows.append(stats)

    segment_vars = pd.DataFrame(rows)
    segment_vars = gpd.GeoDataFrame(segment_vars, geometry="geo", crs=below_surface.crs)

    type_a = segment_vars.loc[segment_vars["N_subsurface"] < THRESHOLD_LIMIT]
    """ ALERT """
    segment_vars = segment_vars.loc[segment_vars["N_subsurface"] >= THRESHOLD_LIMIT]

    return segment_vars



""" 

Get time windows for PACE data

"""

def get_PACE_time_windows(segment_vars):
    """
    Takes ATLAS segment data and creates windows of +/- 12 hours around
    those segment times. Returned time windows are later used to
    query for PACE granules within the windows.
    """

    # All LiDAR segments
    df = segment_vars.copy()

    # covert lidar time to pandas datetime
    df["time"] = pd.to_datetime(df["time"])

    # tolerance for PACE windows is +/- 12 hours
    delta = pd.Timedelta(hours=12)

    # make windows
    windows = pd.DataFrame({
        "start": df["time"] - delta,
        "end": df["time"] + delta
    }).sort_values("start")


    # If windows overlap, merge them
    merged_windows = []
    current_start = windows.iloc[0]["start"]
    current_end = windows.iloc[0]["end"]

    for i in range(1, len(windows)):
        start = windows.iloc[i]["start"]
        end = windows.iloc[i]["end"]

        if start <= current_end:  
                # overlap → extend window
                current_end = max(current_end, end)
        else:
                # no overlap → save previous
                merged_windows.append((current_start, current_end))
                current_start = start
                current_end = end

    # append last window
    merged_windows.append((current_start, current_end))
    merged_windows_df = pd.DataFrame(merged_windows, columns=["start", "end"])

    return merged_windows_df


def get_PACE_data(merged_windows_df, window_start_str, window_end_str, LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, harmony_client, PACE_ROOT):
    """
    Downloads PACE l2m Chl-a, given input parameters for time, bounding box, client, and directory to download to.
    """

    requests = []


    BOUNDING_BOX = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX) 

    for _, row in merged_windows_df.iterrows():
        req = Request(
            collection=Collection(id="PACE_OCI_L2_BGC"),
            spatial=BBox(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
            temporal={
                "start": pd.to_datetime(row["start"]).to_pydatetime(),
                "stop": pd.to_datetime(row["end"]).to_pydatetime()
            },
            variables=["geophysical_data/chlor_a"],
            labels={}
        )
        requests.append(req)

    len(requests)


    ### submit requests:
    jobs = []

    jobs = []
    failed = []

    for r in requests:
        try:
            job = harmony_client.submit(r)
            jobs.append(job)
        except Exception as e:
            failed.append((r, str(e)))

    print(str(len(failed)) + " out of " + str(len(requests)) + " time windows (+/- 12 hrs) did not have PACE flyover")


    ## Download PACE data ###

    window_tag = f"{window_start_str}_{window_end_str}"
    pace_window_dir = os.path.join(PACE_ROOT, window_tag)
    os.makedirs(pace_window_dir, exist_ok=True)

    downloaded_paths = []

    for job_id in jobs:
        try:
                harmony_client.wait_for_processing(job_id, show_progress=True)
                futures = harmony_client.download_all(job_id, directory=pace_window_dir, overwrite=True)

                for future in futures:
                    filename = future.result()   # blocks until this file finishes downloading
                    downloaded_paths.append(filename)

        except Exception as e:
                print(f"Job {job_id} failed:", e)

    print(f"Downloaded {len(downloaded_paths)} files")


    ### get rid of non-unique values

    seen = set()
    unique_files = []

    for path in downloaded_paths:
        name = Path(path).name

        if name not in seen:
            seen.add(name)
            unique_files.append(path)
    return pace_window_dir



""" 

Form a gridded geodataframe of chlorophyll-a from PACE

"""

def gridded_chl(pace_window_dir):
    """
    Grid chl data downloaded in a certain window..
    Returns a gridded geodataframe of chl-a
    """

    import xarray as xr

    # retreive all files
    files = glob.glob(os.path.join(pace_window_dir, "*.nc"))

    # open data, call NASA's grid_data function, convert to df
    dfs = []
    for file in files:
        try:
            swath_time = pd.to_datetime(
                Path(file).stem.split(".")[1],
                format="%Y%m%dT%H%M%S"
            )

            dt = xr.open_datatree(file)
            ds = xr.merge(dt.to_dict().values())
            ds = ds.set_coords(("longitude", "latitude"))

            x = grid_data(ds, 0.02)

            # Convert to DataFrame
            df = (
                x["chlor_a"]
                .to_dataframe()
                .reset_index()
                .dropna(subset=["chlor_a"])
            )

            df["swath_time"] = swath_time

            dfs.append(df)

        except Exception as e:
                print(f"Skipping {file}: {e}")

    if not dfs:
        return None

    # load all individual dataframes into one big dataframe
    chl_df = pd.concat(dfs, ignore_index=True)

    # Convert to GeoDataFrame
    chl_a_gdf = gpd.GeoDataFrame(
        chl_df,
        geometry=gpd.points_from_xy(chl_df.longitude, chl_df.latitude),
        crs="EPSG:4326",
    )

    # return the complete geodataframe
    return chl_a_gdf


"""

Find Segment Center Points

"""

def get_segment_centers(segment_vars):
    """
    Takes lidar segments and calculates the midpoints.
    Adds a lat, lon, and lidar_date column to the lidar segments data. Returns this updated data.
    """
    segment_vars_centered = segment_vars.to_crs(epsg=4326)
    midpoints = segment_vars_centered.geometry.interpolate(0.5, normalized=True)
    segment_vars_centered["lon"] = midpoints.x
    segment_vars_centered["lat"] = midpoints.y
    segment_vars_centered["lidar_date"] = pd.to_datetime(segment_vars_centered["time"])
    
    return segment_vars_centered




def match_chl_to_segments(segment_vars, chl_a_gdf, tolerance_deg, tolerance_time):
    """ 
    
    Matchup points using nearest neighbors
    
    """

    # get the segment variables with a consitently-calculated lon/lat/time
    segment_vars_centered = get_segment_centers(segment_vars)

    # prep PACE chl-a points
    valid = chl_a_gdf["chlor_a"].notna()

    # if there are no valid chlorophyll points, then return early
    if valid.sum() == 0:
            print("No valid chl_a points in this window's PACE data; skipping matchup.")
            segment_vars_centered["chl_a"] = np.nan
            segment_vars_centered["chl_lon_dist"] = np.nan
            segment_vars_centered["chl_lat_dist"] = np.nan
            segment_vars_centered["chl_time_dist"] = np.nan
            return segment_vars_centered

    # arrays of respective lat/lon/swathtime/chlor_a for each PACE data
    chl_lon = chl_a_gdf.loc[valid, "longitude"].values
    chl_lat = chl_a_gdf.loc[valid, "latitude"].values
    chl_time = pd.to_datetime(chl_a_gdf.loc[valid, "swath_time"]).values
    chl_val  = chl_a_gdf.loc[valid, "chlor_a"].values

    # convert time to seconds (float) relative to some epoch, so it can share
    # a distance metric with degrees -- scale it so the time tolerance
    # corresponds to a comparable "distance" unit
    epoch = chl_time.min()
    chl_time_sec = (chl_time - epoch) / np.timedelta64(1, "s")

    # make tolerance_time equivalent to tolerance_deg in the tree metric
    tol_time_sec = pd.Timedelta(tolerance_time).total_seconds()
    time_scale = tolerance_deg / tol_time_sec

    # create a 2d array where each point has lat, lon, and scaled time value
    chl_points = np.column_stack([chl_lon, chl_lat, chl_time_sec * time_scale])

    # build a k-dimensional tree (3d since we have lat/lon/time)
    # enables future KNN lookup
    tree = cKDTree(chl_points)

    # prep segment query points. extract lon/lat/time. Convert time to same epoch as the PACE data
    seg_lon = segment_vars_centered["lon"].values
    seg_lat = segment_vars_centered["lat"].values
    seg_time = pd.to_datetime(segment_vars_centered["lidar_date"]).values
    seg_time_sec = (seg_time - epoch) / np.timedelta64(1, "s")

    # creates a 2d for LIDAR points where each point has lat, lon, and scaled time value. same size as PACE Chl-a chl_points
    query_points = np.column_stack([seg_lon, seg_lat, seg_time_sec * time_scale])

    # single nearest neighbor in true joint space. 
    """ This could be improved. I just didn't have enough time to look more into it"""
    """ I used logic from one of my basic school projects"""
    dist, idx = tree.query(query_points, k=1)

    # Retrieve the matched chlorophyll observation for each lidar point.
    matched_lon = chl_lon[idx]
    matched_lat = chl_lat[idx]
    matched_time = chl_time[idx]
    matched_val = chl_val[idx]

    # Calculate distances (longitudinal, latitudinal, and time)
    """ Should be updated if not in low latitudes """
    lon_dist = np.abs(seg_lon - matched_lon)
    lat_dist = np.abs(seg_lat - matched_lat)
    time_dist = np.abs(seg_time - matched_time)

    # Create a mask for values that are too far apart (spatiotemporally)
    too_far = (
            (lon_dist > tolerance_deg)
            | (lat_dist > tolerance_deg)
            | (time_dist > np.timedelta64(int(tol_time_sec), "s"))
    )

    # print how many segments are masked
    n_too_far = too_far.sum()
    if n_too_far > 0:
            print(f"Masking {n_too_far} segments beyond tolerance ({tolerance_deg}° or {tolerance_time})")

    # Convert chlorophyll values to float so too-far matches can be assigned NaN
    matched_val = matched_val.astype(float)

    # Remove matches that didn't match the spatiotemporal tolerance.
    matched_val[too_far] = np.nan

    # Store matched chlorophyll data and matching uncertainties in the dataset.
    segment_vars_centered["chl_a"] = matched_val
    segment_vars_centered["chl_lon_dist"] = lon_dist
    segment_vars_centered["chl_lat_dist"] = lat_dist
    segment_vars_centered["chl_time_dist"] = time_dist

    # Return the matchup points!
    return segment_vars_centered