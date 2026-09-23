# Overview

## What This App Does

EnvGeo-Seawater is an interactive application for exploring seawater isotope and hydrographic datasets.

It supports maps, temperature-salinity diagrams, salinity-d18O relationships, depth profiles, and 3D/4D visualizations.

## Target Users

- Marine geochemistry researchers
- Oceanography students
- Users who want to compare local seawater data with curated reference datasets
- Developers maintaining or extending EnvGeo-Seawater

## Basic Workflow

1. For an always-loaded local table, place the file at
   `local_data/user_data.xlsx` before starting the app. For one-time data,
   upload a CSV/XLSX file in **User Data Check & Quick Visualizer**.
2. Select a dataset. The local table appears as `User Excel data`; browser
   uploads appear as `Uploaded data`.
3. Choose a visualization page.
4. Adjust figure settings.
5. Inspect maps, plots, tables, and quality flags.
6. Download figures or filtered-data summaries when needed.

## Main Data Types

- d18O
- dD
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
  are not saved by the app. The public **User Data Check & Quick Visualizer**
  provides upload-first quality review and simple 2D--4D exploration.
- `User Excel data` is the optional, always-loaded local dataset. It is read
  from the Git-ignored local path at startup and appended to each reference
  source. It is not the same as session-only `Uploaded data`.
- Vertical Section interpolation remains experimental; inspect observed points,
  settings, and data coverage before using a section as an analysis result.
