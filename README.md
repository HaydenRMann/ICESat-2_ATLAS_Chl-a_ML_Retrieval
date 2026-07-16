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

## Scripts
### *Runners + Main Plots*
- 01_match_maker.ipynb
- 02_RF_implementation.ipynb
### *Auxilary Plots*
- aux_Strip_plots.ipynb
- aux_photoncloudplot.ipynb\
### *Auxilary Fxs*
- aux_fx_plot.py
- aux_fx_process.py

## Ne_10m_land
- ...
  
## environment.yml
- All dependencies should be contained in here. Email H.M. if any are missing.
- *Code was run on python3.14 (no conda), so it was not initially configured using an environment.*

## SARP26_FINAL_FIGS
