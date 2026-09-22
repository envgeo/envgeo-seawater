# Vertical Section Visualizer beta

## What This Page Does

This beta page explores vertical sections along selected transects.

## Basic Workflow

1. Select or define a section path.
2. Choose data and bathymetry source.
3. Adjust interpolation and display settings.
4. Inspect the vertical section output.
5. Refine settings if needed.

## Uploaded Data In Section Calculations

Upload or reuse session data with longitude, latitude, depth, and the selected
target parameter. In A-B mode, uploaded samples inside the active corridor are
projected onto the section line. In Axis-based mode, they use the same axis
coordinate and distance origin as the reference data. When **Uploaded data**
is selected in Data filtering, valid uploaded rows are combined locally with
the selected reference rows for section projection, interpolation, contouring,
and the observed-depth seafloor fallback. Deselecting it removes those rows
from the calculation and display. Uploaded samples remain distinct foreground
markers in the A-B selector map, section plots, and Section Map. The page
reports valid and missing/invalid counts for the selected target parameter.

## Main Controls

- Shared user-data upload, column assignment, and marker-style panels
- Common Data filtering form used by the other visualization pages
- Section path settings
- Bathymetry source
- Uploaded bathymetry file
- Interpolation and plotting settings, including colormap and horizontal
  colorbar length, thickness, font-size, and tick-count controls

## Outputs

- Vertical section plot
- Section map or supporting view
- Filtered or section-selected data table, where available

## Notes And Limitations

- This page is still beta.
- Vertical-section interpolation is scientifically and computationally sensitive.
- Further validation is needed before treating outputs as final analysis figures.
