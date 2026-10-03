# Getting started

Choose a page from the sidebar to explore the included seawater isotope and
hydrographic data. A desktop or laptop browser is recommended; mobile and
tablet layouts may have limited space for interactive figures and controls.

## A first workflow

1. Start with **User Data Check Quick Visualizer** to inspect a CSV/XLSX file,
   review missing values and quality flags, and make a simple 2D--4D plot.
2. On that page, and on the supported specialist pages, use **Data filtering**
   to choose a reference dataset and, when appropriate, select **Uploaded
   data**. Change filters, then select **Apply settings** to update the figure.
3. Use the specialist pages for the bundled reference data or the optional
   local **User Excel data**: **2Dplus Visualizer** for a spatial overview;
   **3D 4D Visualizer** for longitude, latitude, depth, and variable
   relationships; or the salinity--d18O, mapping, T--S, and depth-profile
   pages for focused hydrographic and isotope exploration.

## User data

`local_data/user_data.xlsx` is a zero-value public sample supporting the
always-loaded **User Excel data** workflow. For researcher-owned CSV, XLSX, or
XLS data outside the repository, set `ENVGEO_LOCAL_USER_DATA_PATH` before
starting the app. Uploaded files are used only for the current browser session
and are not written into the bundled reference datasets. Browser uploads are
available in **User Data Check Quick Visualizer** and as overlays on the
salinity--d18O, mapping, T--S, custom-parameter, depth-profile, and
vertical-section pages. They are not currently available in 2Dplus or 3D 4D.

## Page status and interpretation

The **Vertical Section Visualizer** is an experimental workflow. Check
vertical-section interpolation against observed points and data coverage before
using it as an analysis result. The
**Correlation Overview** page is a preserved exploratory/archive workflow; it
is not a target for new feature development.

Reference data sources, citation guidance, and known limitations are available
in the Home-page **Data Sources**, **About**, and **Updates** tabs and in the
repository documentation.
