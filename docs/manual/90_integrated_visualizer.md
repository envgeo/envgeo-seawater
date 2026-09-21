# Integrated Visualizer beta

## What This Page Does

This beta page tests integrated workflows that combine existing visualization pages, shared filtering, upload comparison, and quick visual checks.

## Basic Workflow

1. Choose workflow mode.
2. Select or upload data if needed.
3. Apply shared filters.
4. Explore maps, T-S diagrams, salinity-d18O views, custom plots, summaries, and quality flags.
5. Download summaries or quality-flag tables where available.

## Main Controls

- Workflow mode
- Full existing page workflows
- Shared-filter beta workflow
- Uploaded data overlay controls
- Marker style controls
- Map controls
- Quality Flags tab

## Outputs

- Integrated map
- T-S diagram
- Salinity-d18O plot
- Custom 2D/3D quick views
- Uploaded-data overlays
- Reference and uploaded-data quality summaries
- Downloadable CSV summaries

## Notes And Limitations

- This page is the main test bed for future integration.
- Uploaded data are intended to remain in memory only during the Streamlit session.
- Some workflows call existing page files for compatibility.
- Individual visualization pages will remain first-class workflows. Validation and reference-data comparison will move to an independent User Data Validator page.
- Full existing page compatibility mode is transitional and will be retired after supported pages have native upload overlays.
- Integrated Visualizer will remain available during migration and may then be kept as a hidden development archive.
- See the [Integrated Visualizer strategy](../integrated_visualizer_strategy.md) for the accepted architecture and migration plan.
