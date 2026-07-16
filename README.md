# Bridging Passive Remote Sensing Gaps: A novel machine learning approach to chlorophyll-a estimation using ICESat-2 ATL03 geolocated photon data trained using PACE OCI

<img align = "left" src="https://science.nasa.gov/wp-content/uploads/2023/11/sarp-patch.jpeg?w=1280&format=webp" alt="drawing" width="200"/> This project was completed during the 2026 [NASA Student Airborne Research Program (SARP)](https://science.nasa.gov/earth-science/early-career-opportunities/student-airborne-research-program/). For two weeks, I participated in NASA flight campaigns out of Ellington Field at Johnson Space Center in Houston, TX. I then relocated to the 'Ocean's West' group at San Diego State University in San Diego, CA, where I completed my independent research. I researched chlorophyll-a estimation using ICESat-2 ATLAS ATL03 geolocated photon data, by training it on PACE l2m chlorophyll-a product. The first step was to process and derive statistics from the raw photon data. Then, random forest regression was used to predict chlorophyll-a (R<sup>2</sup> = 0.78) trained on PACE observations. <br><br><br>

**Code Author, Project Lead**: Hayden Mann (Bowdoin College '27)<br> 
&emsp;&emsp;&emsp;- Code Contact: hmann@bowdoin.edu <br><br>
**Graduate Mentor**: Dorothy Grimmer (Texas A&M University)<br> 
**Faculty Advisor**: Dr. Henry Houskeeper (Woods Hole Oceanographic Institute)<br> 
**Contributing Code**: Dr. Kelsey Bisson (NASA HQ)
<br>


## Abstract
Phytoplankton are the foundation of marine food webs and the biological carbon pump. Validated algorithms have been developed for the retrieval of chlorophyll-a, a proxy for phytoplankton, from passive ocean remote sensing instruments, including NASA’s Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) satellite’s Ocean Color Instrument (OCI). However, optical passive remote sensing sensors cannot observe the ocean over dark areas, because they rely on solar illumination. Thus, observational gaps exist during nighttime and within high-latitude regions during polar night. Conversely, active remote sensing instruments provide their own light source and measure the reflected signal, allowing them to function independently of illumination state. Although active lidars, including NASA’s ATLAS (Advanced Topographic Laser Altimeter System) aboard ICESat-2 (Ice, Cloud, and land Elevation Satellite), can operate within many passive remote sensing gaps, methods of chlorophyll-a retrieval from ATLAS have yet to be widely implemented. Here, the depth profiles of photons from ICESat-2 ATLAS observations are assessed at 1 km along-track resolution to extract depth percentiles, bin-depth ratios, and other depth-based statistics relevant to phytoplankton distributions. The engineered photon segment depth features were trained on PACE chlorophyll-a data using Random Forest Regression. The ATLAS chlorophyll-a estimation agrees with the PACE chlorophyll-a product (R2 = 0.78), suggesting ATLAS supports opportunities to mitigate PACE observational gaps where ATLAS matchups are available. The random forest estimation compresses the chlorophyll-a range, modestly overestimating low values and underestimating high values. The model is a regional proof of concept but omits geographic input variables, such as bathymetric depth and location, suggesting broader applicability of this approach to other regions. Future improvements should focus on refining photon cloud data processing, tuning feature engineering, and training on more globally-representative datasets. The agreement between ATLAS and PACE chlorophyll-a estimation encourages future investigation into advancing lidar chlorophyll-a retrieval to extend passive remote sensing observations of high latitude or nighttime ocean environments. 

# Code Overview
## Workflow Overview: details below
1. Set aoi, time, and photon-input parameters in 01_match_maker.ipynb
2. Run 01_match_maker.ipynb. Sliderule requests scales up quickly.
3. Locate saved matchup point files.
4. Run 02_RF_implementation.ipynb to train model.

## Scripts
### *Runners + Main Plots*
#### 01_match_maker.ipynb
- Processing workflow
  1. Install and Imports
  2. Configure Parameters
  3. Prepare for For Loop
  4. For Loop to make matchup points
     - Import functions (noted in italics below) from _aux_fx_process.ipynb_
     - Start Loop
       - Set up time-window dates and loop timestep
       - Get photons using _get_photons()_
       - Mask photons to be over water only using _water_mask()_
       - Retrieve Water Surface (Eidam et al., 2024) per cycle/ground track (author update) and segment binning (author update) using _water_surface_retrieval_and_binning()_
       - Segment variable retrieval using _get_segment_vars_
       - Get PACE data for time window using _get_PACE_data()_
       - Form a gridded geodataframe of PACE chl-a data using _gridded_chl_
       - Filtered the matchup dataframe to keep rows where PACE chl-a is not NaN
       - Print descriptive statistics
       - Save matchups in parquet
       - Delete PACE data for this time window
       - Increment loop <br>

- Variable overlay plot workflow (fx in _aux_fx_plot.py_)
  1. Open matchup data
  2. Select date of choice for plot
  3. Download PACE l2m chlorophyll-a data for that date
  4. Select variables for ploting; plot. <br><br>
  
#### 01_match_maker.ipynb
- Processing workflow: Steps 3-6 based on workflow from Corcoran and Parrish (2021)
  1. Install and imports
  2. Import matchup points
  3. Preliminary RF run on all features
  4. Determine feature importance based on the first RF run
  5. Create a correlation matrix of all features
  6. Correlated feature trimming, mathematically
  7. Compute trimmed feature correlation with chlorophyll
  8. Final RF run on trimmed features
- Plot workflow
  1. PACE chl-a vs ATLAS chl-a scatter plot (toggle outlier removal for regression fit line)
  2. Map PACE chl-a under ATLAS chl-a for a single date (toggle colormap / lidar track outline)
  3. Variable scatter plots
  4. Regional time series <br><br>

### *Auxilary Plots*

#### aux_Strip_plots.ipynb
- Similar workflow to _01_match_maker.ipynb_, but just for one day
- Plots a 3-part strip plot of ATLAS chl-a, PACE chl-a, and their percent error.

#### aux_photoncloudplot.ipynb
- Similar (preliminary version of) workflow to _01_match_maker.ipynb_, but for a small area offshore of NC
- Purpose is to plot a photon segment cloud example for SARP presentation.

### *Auxilary Fxs*
#### aux_fx_plot.py
#### aux_fx_process.py

## Ne_10m_land
- https://www.arcgis.com/home/item.html?id=595533fecdb0472db4b4b8e3ca8d9e42#overview
- Navigated to via USGS website. Referred to in Corcoran and Parrish (2021).
  
## environment.yml
- All dependencies should be contained in here. Email H.M. if any are missing.
- *Code was run on python3.14 (no conda), so it was not initially configured using an environment.*

## SARP26_FINAL_FIGS


## References
- Initial photon cloud processing + water surface retrieval (both updated by H.M.)
  - Eidam E.F., Bisson, K., Wang, C., Walker, C., Gibbons, A. (2024). ICESat-2 and ocean particulates: A roadmap for calculating K<sub>d</sub>  from space-based lidar photon profiles. *Remote Sensing of Environment, 311*. doi:10.1016/j.rse.2024.114222
- Workflow for Random Forest Regression feature trimming.
  - Corcoran, F. and Parrish, C.E. (2021). Diffuse attenuation coefficient (K<sub>d</sub>) from ICESat-2 ATLAS spaceborne lidar using random-forest regression. _Photogrammetric Engineering and Remote Sensing, 87_(11). doi:10.14358/PERS.21-00013R2
- Sliderule API JOSS Article
  - Shean, D., Swinski, J.P., Smith, B., Sutterley, T., Henderson, S., Ugarte, C., Lidwa, E., and Neumann, T. (2023). SlideRule: Enabling rapid, scalable, open science for the NASA ICESat-2 mission and beyond. _The Journal of Open Source Software 8_(81). doi:10.21105/joss.04982
- Sliderule versino used to prepare data products
  - Shean, D., Swinski, J.P., Ugarte, C., Lidwa, E., Smith, B., Sutterley, T., Henderson, S., and Neumann, T. (2026). Sliderule. doi:10.5281/zenodo.4660020
  - Published July 10, 2026
  - Used July 14, 2026
