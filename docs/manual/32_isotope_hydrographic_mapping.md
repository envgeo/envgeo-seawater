# Isotope & Hydrographic Mapping

## What This Page Does

This page maps isotope and hydrographic parameters such as δ18O, δD, d-excess,
salinity, and temperature.

## Basic Workflow

1. Select a reference data source and set the sidebar filters.
2. Optionally upload session-only data in **Uploaded data overlay**. Assign
   longitude and latitude columns to show uploaded points on maps.
3. Choose the mapped parameter and either **Scatter Map** or **Contour Map**.
4. Select **Apply settings** after changing filters.
5. Adjust map center, display extent, colormap, color range, and colorbar
   settings in the sidebar.
6. Review the static parameter map and the interactive sampling-location map.
7. Download the scatter or contour PNG where provided.

## Map Types

- **Scatter Map** plots each available observation at its location.
- **Contour Map** interpolates available longitude, latitude, and parameter
  values onto a grid, then overlays the observations. It attempts linear
  interpolation when possible and falls back to nearest-neighbour interpolation
  for sparse, duplicate, or collinear locations.

## Main Controls

- **Mapped parameter**
- **Map type**
- **Map center**
- **Colormap**
- **Map longitude / latitude range**
- **Colorbar range**
- **Colorbar thickness / length / font size**
- **Map controls / Map style**

## Outputs

- Scatter map
- Contour-style map
- Sampling location map
- Downloadable map images
- Filtered dataset table
- Uploaded-data quality check and foreground overlay when applicable

## Notes And Limitations

- Contour maps are exploratory visual aids, not independently validated fields.
  Interpret them only together with the plotted observations, data coverage,
  and the selected interpolation behavior.
- Map display settings are independent from Data filtering settings.
- Static scatter and contour maps use bundled coastline and land data, so they
  do not download Natural Earth files at runtime. For the interactive sampling
  location map, select **Coastline (offline)** to use no external map tiles.
- Browser uploads remain in the current Streamlit session and do not alter
  reference data or persistent **User Excel data**.
