# User Data Check & Quick Visualizer

## Purpose

Use this page as the public, upload-first entry point for checking a CSV, XLSX,
or XLS file before making specialist figures. It combines in-memory upload,
editable coordinate-column assignment, quality and missing-value review,
shared Data filtering, and simple 2D--4D exploration.

## Basic Workflow

1. Upload a CSV, XLSX, or XLS file in the **Uploaded data overlay** sidebar
   panel. A CSV template is available before a file is loaded.
2. Review the recognized columns. Any numeric columns can be used as 2D/3D
   axes; assign longitude, latitude, and depth in **Uploaded data columns**
   when a geographic map or geographic 3D view is needed.
3. Keep **Comparison data source** as `None` for an upload-only view, or add a
   curated reference source for context.
4. In Data filtering, choose `Uploaded data`, reference datasets, or both, and
   click **Apply settings**.
5. Review the Overview & Quality tab, then use the plot and map tabs.
6. Download filtered CSV data or interactive figures where the relevant
   control is provided.

## 3D Camera Controls

The **3D / 4D** and **3D Map** tabs use the same camera controls as the main
3D/4D visualizer: drag to rotate and use the Plotly toolbar at the upper right
to switch rotation, pan, zoom, or reset the camera. Holding **Shift**,
**Control**, or **Command** can change drag behavior; the exact behavior
depends on the browser and operating system.

## Upload Handling

The application recognizes common English and unambiguous Japanese column
names. Automatic recognition remains editable. Numeric values that cannot be
interpreted after standard cleanup are treated as missing and appear in the
quality review. Internal application metadata is excluded from previews,
downloads, and axis menus.

Use **Clear shared uploaded data** in the upload panel to remove the shared
browser-session upload. It does not modify any local file or bundled dataset.

## Tabs

- **Overview & Quality**: row counts, missing/valid availability, quality flags,
  and upload preview.
- **2D Explore**: arbitrary X-Y scatter, Salinity-d18O, and
  Temperature-Salinity views.
- **3D / 4D**: selectable X, Y, Z, and color dimensions.
- **2D Map** and **3D Map**: geographic inspection with palette, region, and
  viewpoint controls. Geographic 3D requires assigned longitude, latitude,
  and depth columns.
- **Data & Export**: filtered table and CSV download.

## Performance And Data Handling

The default **Maximum plotted rows** is 5,000 for responsive Plotly figures.
The information message states when this cap is active; raise it to at most
50,000 when the browser and dataset permit. Uploaded files and temporary
reference-plus-upload data remain in the current Streamlit session only and
are not saved by the app. Uploading does not alter bundled reference datasets
or optional persistent **User Excel data**.
