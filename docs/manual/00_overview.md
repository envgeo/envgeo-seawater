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

1. Upload a CSV/XLSX file in **User Data Check & Quick Visualizer** when you
   want to examine your own data, or choose a specialist page for a focused
   figure.
2. Select a dataset, including `Uploaded data` where it is available.
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
- Vertical Section interpolation remains experimental; inspect observed points,
  settings, and data coverage before using a section as an analysis result.
