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
5. Adjust map center, static-map projection, display extent, colormap, color range, and colorbar
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
- **Map projection / 投影法**
- **Colormap**
- **Map longitude / latitude range**
- **Colorbar range**
- **Colorbar thickness / length / font size**
- **Map controls / Map style**

## Static Map Projections

The projection selector applies only to the downloadable Cartopy **Scatter Map**
and **Contour Map**. It does not change the interactive sampling-location map
or its web-map background style.

For near-global views, the displayed land mask and coastline use a lighter
rendering representation only; the bundled Natural Earth source files and the
selected observations are not changed. Static figures are sent as PNG bytes and
their Matplotlib canvases are released promptly to limit Cloud memory use.

- **Plate Carrée (longitude–latitude)** is the default and keeps the familiar
  longitude–latitude layout.
- **Mercator** is useful at low to middle latitudes. It cannot show the poles,
  so the static map is limited to ±84.5° latitude.
- **Robinson** is useful for global figure layouts. Gridlines are shown without
  edge labels because their boundary is curved.
- **Azimuthal Equidistant** is useful for a centred regional or polar view. It
  preserves distance and direction from the map centre, but not area or shape
  everywhere on the map. Use a regional extent for distance-and-direction
  interpretation. The full-longitude Arctic Ocean and Southern Ocean presets
  are centred on their respective pole while retaining their selected regional
  latitude range. A global **Scatter Map** is shown as a circular world view.
  Global exploratory contours are available with Plate Carrée and Mercator.
  Robinson and globe-style Azimuthal Equidistant and Orthographic views show
  observations for a global selection because their curved boundary, outer
  edge, or hidden hemisphere cannot represent one continuous contour field.
- **Orthographic (map centre)** provides a globe-style view centred on the
  midpoint of the map-display extent, corresponding to ODV's oblique
  orthographic view. It is useful for an Atlantic-, Pacific-, or other
  basin-centred overview.
- **Orthographic (South Polar)** and **Orthographic (North Polar)** provide a
  globe-style view centred on the selected pole, like ODV's polar orthographic
  maps. Orthographic views are useful for visual context, but do not preserve
  distance, direction, area, or shape away from the centre.

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
- Projection changes the appearance of the static figure, not the selected
  observations or the interpolation input coordinates.
- Static scatter and contour maps use bundled coastline and land data, so they
  do not download Natural Earth files at runtime. For the interactive sampling
  location map, select **Coastline (offline)** to use no external map tiles.
- Browser uploads remain in the current Streamlit session and do not alter
  reference data or persistent **User Excel data**.
