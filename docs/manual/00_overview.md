# Overview

## What This App Does

EnvGeo-Seawater is an interactive application for exploring seawater isotope
and hydrographic datasets.

It supports maps, temperature-salinity diagrams, salinity--d18O relationships,
depth profiles, and 2D/3D/4D visualizations.

The current collection includes the multi-year Kodama et al. (2024) core
dataset, cited regional datasets, and global reference datasets including NASA
GISS and PAGES CoralHydro2k. It contains approximately 50,000 records. Cite
the original sources shown in **Data Sources** and **Filtered dataset** when
using results.

## Target Users

- Marine geochemistry researchers
- Oceanography students
- Users who want to compare local seawater data with curated reference datasets

## Local Installation

Once the v1.3.4 package is published to PyPI, install and start the local app
without cloning the source repository:

```bash
python -m pip install envgeo-seawater
envgeo-seawater
```

On macOS Apple Silicon, install `proj`, `pyproj`, and `cartopy` from
conda-forge in a Python 3.12 Conda environment before the pip command. The
[stable README](https://github.com/envgeo/seawater_map/blob/main/README.md)
contains the complete platform-specific instructions.

## Basic Workflow

1. Choose a page from the sidebar. For a one-time file, open **User Data Check
   & Quick Visualizer** and upload CSV/XLSX data. Files remain only in the
   current browser session.
2. To load a persistent local table, set `ENVGEO_LOCAL_USER_DATA_PATH` to an
   external CSV, XLSX, or XLS file before starting the app. The bundled
   `local_data/user_data.xlsx` is a zero-value public sample; do not place
   personal data in the repository or app folder.
3. In **Data filtering**, select the reference datasets and adjust the desired
   conditions. `User Excel data` appears when a persistent local table is
   configured; `Uploaded data` appears when a session upload is available.
4. Select **Apply settings** after changing filters, then adjust figure
   settings as needed.
5. Inspect maps, plots, tables, and quality flags.
6. Download figures or filtered-data summaries where a page provides that
   control.

## Main Data Types

- δ18O (`d18O`)
- δD (`dD`)
- d-excess
- Salinity
- Temperature
- Water depth
- Latitude and longitude
- Sampling year and month

## Notes And Limitations

- Some pages are labeled beta because their workflow or scientific design is still being refined.
- Very large global selections may make 3D/4D visualizations slow.
- Browser-uploaded data remain in memory for the current Streamlit session and
  are not saved by the app. Uploads are supported by **User Data Check & Quick
  Visualizer** and the salinity--d18O, mapping, T--S, custom-parameter,
  depth-profile, and vertical-section pages. The 2Dplus and 3D/4D pages do not
  currently accept browser uploads.
- `User Excel data` is a persistent local dataset loaded from an external path;
  it is separate from session-only `Uploaded data`.
- Vertical Section interpolation remains experimental; inspect observed points,
  settings, and data coverage before using a section as an analysis result.
- Custom Parameter Plot and Vertical Section are beta workflows. Correlation
  Overview is a retained exploratory archive and does not receive new features.
