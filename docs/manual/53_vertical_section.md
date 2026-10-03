# Vertical Section Visualizer beta

## What This Page Does

This experimental page produces an interpolated vertical section along a selected
axis or A–B line. It is intended for exploratory inspection, not final scientific
interpretation without independent validation.

## Basic Workflow

1. Select a reference dataset and set the shared filters.
2. Choose **Axis-based** mode, or choose **A-B section** and enter endpoints or
   draw a line on the selector map.
3. Set the corridor half-width. In A-B mode, only observations inside this corridor
   are projected onto the section.
4. Select a target parameter, bathymetry source, and display settings.
5. Inspect the colour and line section views, the Section Map, and the section table.
6. Narrow the filters or corridor if too many rows remain for interpolation.

## Uploaded Data In Section Calculations

Upload or reuse session data with longitude, latitude, depth, and the selected
target parameter. In A-B mode, uploaded samples inside the active corridor are
projected onto the section line. In Axis-based mode, they use the same axis
coordinate and distance origin as the reference data. Select **Uploaded data** in
the shared Data filtering form to include valid uploaded rows in section projection,
interpolation, contouring, and the observed-depth seafloor fallback. Deselecting it
removes those rows from the calculation and display. Uploaded samples remain
distinct foreground markers in the selector map, section plots, and Section Map.
The page reports valid and missing/invalid counts for the selected target parameter.

## Main Controls

- Shared user-data upload, column assignment, and marker-style panels
- Common Data filtering form used by the other visualization pages
- **Section mode** — Axis-based (`Longitude_degE`, `Latitude_degN`, or `Distance_km`)
  or A-B section.
- **A-B endpoints / Draw on map** — manual endpoints always work; drawing requires
  `folium` and `streamlit-folium` and is unavailable in **Coastline (offline)** mode.
- **Half-width of section corridor** and **Show section corridor**.
- **Bathymetry source** — built-in GEBCO, an uploaded CSV/Excel table, or deepest
  observations. GEBCO is used only for A-B sections; a readable fallback is shown
  if it cannot supply a profile.
- **Resolution**, **Smoothing (visual only)**, and the maximum row count used for
  interpolation. The page enforces an internal hard cap to protect shared/Cloud
  environments.
- Target-parameter range, colormap, and horizontal colour-bar controls.

## Outputs

- Colour-filled and contour-line section views
- A Section Map showing the line, corridor, and observations used
- Expandable table of the rows used for the section calculation

## Notes And Limitations

- This page is beta. Vertical-section interpolation is scientifically and
  computationally sensitive; validate the method and display settings before using
  an output as a final analysis figure.
- The interpolation uses a safe fallback sequence (cubic, then linear, then nearest
  neighbour when needed). Cells filled only by nearest-neighbour extrapolation are
  visually marked as lower-confidence areas; they are not observations.
- Smoothing changes the displayed interpolated grid only. It does not add observations
  or demonstrate physical continuity.
- The seafloor is a display constraint, not a bathymetric analysis. Where a selected
  source cannot provide a usable profile, the page falls back to observed maximum depth.
- A line that directly crosses the antimeridian (180°/−180°) uses a local planar
  approximation not validated for that geometry. Treat such output as a rough visual
  reference only.
- Browser uploads are session-only and never modify reference datasets or persistent
  **User Excel data**.
