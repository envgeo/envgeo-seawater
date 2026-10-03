# Interactive 3D/4D Visualizer

## What This Page Does

This page visualizes selected reference data using 3D scatter plots with a
fourth variable shown by color. It includes hydrographic and map-depth views.

## Basic Workflow

1. Select a reference data source and set the sidebar filters.
2. Select **Apply settings** after changing the filters.
3. Choose one of the six standard **4D view** options or **Custom 4D view
   beta**.
4. For map-depth views, adjust the map center, region preset, depth range, and
   marker size as needed.
5. Choose a colormap and adjust the colorbar range for the selected variable.
6. Inspect the 3D/4D figure and linked sampling-location map.
7. Download interactive HTML or current-view CSV data where provided.

## Standard Views

- Salinity--δ18O--depth, colored by temperature (Fig. 1)
- Salinity--temperature--depth, colored by δ18O (Fig. 2)
- Map--depth views colored by δ18O, temperature, salinity, or d-excess
  (Fig. 3--6)

**Custom 4D view beta** provides salinity--δ18O, temperature--salinity,
map--depth, and fully custom X--Y--Z--color layouts. The full custom layout
requires different X, Y, and Z variables and at least four numeric columns.

## Main Controls

- **4D view**
- **Custom view type**
- **Map center for map-depth views**
- **Map-depth region preset**
- **Colormap**
- **Colorbar range**
- **Map controls / Map style**

## Outputs

- Six standard 4D views
- Custom 4D view beta
- Sampling location map
- Self-contained interactive HTML for the selected figure and location map
- Current-view data table and CSV download

## Notes And Limitations

- This page can be slow for large global selections.
- Map-depth views use geographic scaling and coastline overlays.
- Custom 4D view beta is experimental and may change.
- This page does not accept browser uploads. An optional persistent **User
  Excel data** table can be supplied from an external path before app startup;
  see [Data Filtering](01_data_filtering.md).
