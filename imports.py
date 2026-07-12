""" 

All necessary imports and PARAMS:

"""
from pyproj import Transformer
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import StrMethodFormatter
import warnings
warnings.filterwarnings("ignore")
import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sliderule import icesat2, sliderule
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.features import rasterize
import numpy as np
from scipy.stats import skew, kurtosis
import geopandas as gpd
from shapely.geometry import LineString


def params():

    aoi = [
        {"lat": 33.70, "lon": -78.30},
        {"lat": 33.70, "lon": -75.30},
        {"lat": 34.80, "lon": -75.30},
        {"lat": 34.80, "lon": -78.30},
        {"lat": 33.70, "lon": -78.30},
    ]

    resources = None

    base_parms = {
    "poly": aoi,
    "t0": "2024-05-01T00:00:00Z",
    "t1": "2026-05-01T23:59:59Z",
    "cnf": 1,
    "srt": 1,
    "atl03_ph_fields": ["pce_mframe_cnt"],
    "atl03_geo_fields": ["solar_elevation"],
}

    return aoi, resources, base_parms