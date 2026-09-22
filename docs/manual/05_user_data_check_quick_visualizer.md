# User Data Check & Quick Visualizer

## Purpose

Use this page as the public, upload-first entry point for checking a CSV/XLSX
file before making specialist figures. It combines in-memory upload, editable
column recognition, quality and missing-value review, shared Data filtering,
and simple 2D--4D exploration.

## Basic Workflow

1. Upload a CSV or XLSX file in the sidebar.
2. Review detected columns and correct assignments when necessary.
3. Keep **Comparison data source** as `None` for an upload-only view, or add a
   curated reference source for context.
4. In Data filtering, choose `Uploaded data`, reference datasets, or both, and
   click **Apply settings**.
5. Review the Overview & Quality tab, then use the plot and map tabs.
6. Download the filtered CSV when needed.

## Tabs

- **Overview & Quality**: row counts, missing/valid availability, quality flags,
  and upload preview.
- **2D Explore**: arbitrary X-Y scatter, Salinity-d18O, and
  Temperature-Salinity views.
- **3D / 4D**: selectable X, Y, Z, and color dimensions.
- **2D Map** and **3D Map**: geographic inspection with palette, region, and
  viewpoint controls.
- **Data & Export**: filtered table and CSV download.

## Performance And Data Handling

The default **Maximum plotted rows** is 5,000 for responsive Plotly figures.
The information message states when this cap is active; raise it to at most
50,000 when the browser and dataset permit. Uploaded files and temporary
reference-plus-upload data remain in the current Streamlit session only and
are not saved by the app.
