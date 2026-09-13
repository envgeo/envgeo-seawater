# Update History

Detailed development log for recent EnvGeo-Seawater updates.

## Unreleased

### Version 1.3.0 concise summary - 2026-09-11

- Refined the EnvGeo-Seawater interface for clearer public testing, including updated page titles, sidebar labels, map guidance, and figure-control wording.
- Expanded plotting options for seawater parameters, including d18O, dD, d-excess, salinity, temperature, depth, latitude, and longitude.
- Improved map and Plotly visualization workflows with shared map presets, API-key-free map backgrounds, cmocean/EnvGeo colormap options, and cleaner color controls.
- Added and improved user-upload support in the integrated beta workflow, including overlay styling, shared colorbar use, and uploaded-data quality summaries.
- Added reusable data-quality handling for invalid depth, temperature, and salinity values, preserving original values through quality-flag columns.
- Centralized common utilities such as d-excess calculation, filename generation, UI labels, map styles, colormap options, and filtered-data summaries in `envgeo_utils.py`.
- Added CSV/PDF export support for environment diagnostics and CSV export for filtered-data statistics.
- Improved project documentation, Japanese README content, update logs, and repository cleanup toward a future public release.
- Expanded pytest coverage for public structure, data loading, quality rules, filename helpers, and core utility behavior.

### 2026-09-11

- Renamed `32_d18O_mapping.py` to `32_Isotope_Hydrographic_Mapping.py` and updated the page title to `Isotope & Hydrographic Mapping` because the page now maps multiple isotope and hydrographic parameters.
- Renamed the Depth Profile page file from `37_Depth_Profile_(T,S,d18O).py` to `37_Depth_Profile.py` because the page now supports additional parameters.
- Added a `Size contrast` control to Custom Parameter Plot beta so marker-size differences can be emphasized more strongly.
- Added colormap selection for the Custom Parameter Plot beta colorbar.
- Added legend on/off and regression-line on/off controls to the Custom Parameter Plot beta page.
- Added a legend on/off control next to the background-data option in the Temperature-Salinity Diagram page.
- Simplified README content by removing internal project-management notes and keeping only public-facing setup, usage, data, and citation guidance.
- Added fallback UI labels in active pages to reduce errors when a test deployment has an older `envgeo_utils.py`.
- Changed the default background-data setting in the Salinity-d18O Relationship page to `No`.
- Added numeric figure-size, tick-count, and font-size controls to the Salinity-d18O Relationship page.
- Added parameter-based colorbar support to the Salinity-d18O Relationship page for filtered data points.
- Removed implementation-oriented `Auto-Zoom` wording from visible plot and map headings while keeping the existing map update behavior.
- Changed the 3D Visualizer Plotly color controls from radio buttons to `Color filtered` selectors with expanded available parameter options.
- Improved remaining sidebar and map guidance labels by replacing decorated legacy phrases with plain shared UI text.
- Improved figure-control wording by replacing old red map-area notes with a quieter shared caption and unified sidebar label.
- Improved figure download filename handling by adding shared filename helpers in `envgeo_utils.py` and applying them to active plotting pages.
- Renamed the Correlation Overview page title from `Compiled figs` to `Correlation Overview`.
- Added `35_Custom_Parameter_Plot_beta.py` as an experimental T-S-style custom 2D plotting page with selectable X axis, Y axis, color, marker size, numeric plot controls, and missing-value counts.
- Changed paired font-size and tick-count controls from range sliders to separate numeric inputs in the Temperature-Salinity Diagram and Depth Profile pages.
- Corrected current app and page version displays to `1.3.0`.
- Added dD and d-excess as target parameters in the Depth Profile page, with missing-value counts and selected-parameter map coloring.
- Changed the isotope and hydrographic mapping page into a broader workflow with selectable d18O, dD, d-excess, salinity, and temperature map parameters.
- Added a filtered-data color-by selector to the Temperature-Salinity Diagram page, with support for depth, latitude, longitude, year, month, d18O, dD, and d-excess.
- Added user-adjustable colorbar ranges for each selected T-S color parameter.
- Added figure-size, tick-count, and font-size controls to the Temperature-Salinity Diagram page.
- Simplified the Temperature-Salinity Diagram controls by removing the separate T-S colormap selector and keeping only the color parameter and color range controls.
- Changed the default background-data setting in the Temperature-Salinity Diagram page to `No`.

### 2026-09-10

- Added adjustable Matplotlib colorbar thickness, length, and font-size controls to the isotope and hydrographic mapping page.
- Improved the Home main-tab guidance, About text, Data Sources headings, and Manual starting-point notes.
- Improved Home/About/Manual wording for dataset scope, device guidance, and figure-use citation guidance.
- Removed the old heavy-traffic warning from the Home page main tab.
- Improved the Home page tabs with compact styling, clearer active-tab highlighting, and readable Title Case labels.
- Changed Vertical Section Visualizer to a beta-labeled workflow while section interpolation and display behavior are still being refined.
- Improved the integrated beta Map tab with `st.fragment` so map-control changes can rerun only the map section instead of the full page.
- Improved the integrated beta Map tab controls with color and region settings on the left and Map Style in the right one-third column.
- Improved the integrated beta Shared-filter tab labels and tab CSS with a clearer style similar to the earthquake Advanced page.
- Added a compact quality-flag criteria note below the `Sidebar-filtered dataset (CSV)` table.
- Added CSV export for `Details and statistics of sidebar-filtered data`, including filter conditions, selected-data counts, row counts, quality-flag counts, and summary statistics.
- Changed the shared standard map background from `carto-positron` to API-key-free `open-street-map` because CARTO basemaps now require API keys.
- Added CSV and PDF report export to the environment checker for runtime, dependency, and project-file diagnostics.
- Kept the Streamlit environment checker implementation in `tools/env_check_streamlit.py` and added `pages/99_Environment_Check.py` as a local-development sidebar wrapper.
- Updated `requirements.txt` to match the current Anaconda `envgeo_streamlit142` environment and document the verified Python 3.10 dependency set.
- Expanded shared ocean-region map presets for Japan-adjacent seas, Kuroshio/Oyashio regions, North Pacific, tropical Pacific, Indian Ocean, Atlantic Ocean, Mediterranean Sea, Arctic Ocean, and Southern Ocean sectors.
- Restored `Jet` as the default colormap for the isotope and hydrographic mapping page while keeping EnvGeo and cmocean options selectable.
- Added cmocean/EnvGeo colormap selection to the isotope and hydrographic mapping page for both Matplotlib Cartopy maps and Plotly Mapbox maps.
- Adopted cmocean colormap options for oceanographic variables while keeping `EnvGeo variable default` as the initial selection for continuity with existing figures.
- Added shared colormap-selection helpers for Plotly figures.
- Imported the latest working-page revisions for 4D Visualizer and 3D/4D Uploader, and added Correlation Overview and Vertical Section Visualizer as active candidate pages.
- Added automatic standardization for common uploaded-data column aliases such as lon, lat, Depth, Temp, S, delta18O, and delta_D.
- Improved the integrated beta quick-view color settings so numeric Plotly views use the shared EnvGeo-Seawater color-scale function.
- Added uploaded-data quality-check summaries to the integrated beta Summary, Upload, and Quality tabs.
- Improved the integrated beta salinity-d18O color selector so numeric colorbar-compatible columns are listed before categorical columns.
- Added help text for Map marker offset and a marker outline-width control for uploaded-data overlays in the integrated beta page.
- Added a marker color mode that lets uploaded data share the active colorbar in the integrated beta Map, T-S, and salinity-d18O views when the selected color column is numeric.
- Fixed uploaded-data overlays in the integrated beta T-S and salinity-d18O Plotly views so they use WebGL traces and stay visible above dense reference-data plots.
- Added uploaded-data marker style controls for size, color, shape, opacity, Mapbox top-layer rendering, and optional map offset in the integrated beta page.
- Added shared ocean-region map presets and connected them to the integrated beta Map view.
- Improved the shared sidebar-filter summary with row count, quality-flag count, and parameter statistics for d18O, dD, d-excess, salinity, temperature, and depth.
- Added `90_Integrated_Visualizer_beta.py` as an experimental integrated visualizer with full existing-page compatibility mode and shared-filter beta mode.
- Added uploaded-data support to the integrated beta page for selected original visualization workflows, map, T-S, salinity-d18O, and custom 2D/3D plots.
- Removed the standalone 3D/4D uploader from the integrated beta workflow selector because it uses a separate upload-first workflow.
- Changed the shared app version metadata to `1.3.0`.
- Added Japanese explanations to `envgeo_utils.py` for quality normalization and d-excess calculation.
- Added reusable quality-rule metadata for invalid depth, temperature, and salinity values.
- Centralized d-excess calculation in `envgeo_utils.py` for reuse across Streamlit pages.
- Added quality flag columns to preserve original invalid values after NaN conversion.
- Prepared repository cleanup for future public releases.
- Merged the former standalone about page into `home.py`.
- Removed retired navigation pages, duplicate page copies, and obsolete beta pages from the current source tree.
- Added `.gitignore` and cleaned generated local files such as `.DS_Store`, `__pycache__`, and `.pytest_cache`.
- Added a Japanese README.
- Simplified public-facing repository wording in README and app update history.

## 1.0.1 - 2026-03-24

- Improved stable Streamlit app structure for the EnvGeo-Seawater public release.
- Updated home, about, data-source, manual, update-log, and Japanese information pages.
- Prepared repository materials for future public releases.

## 1.0.0 - 2026-03-18

- Changed the seawater isotope and hydrographic visualization app with a major update.
- Added updated 3D/4D, 2D mapping, T-S diagram, depth-profile, and salinity-δ18O workflows.
- Improved data filtering, figure output, and source-aware display.

## 0.2.0 - 2026-02-18

- Changed the integrated Streamlit app with a major pre-1.0 update.
- Improved page organization and visualizer behavior.

## b20 - 2024-12-14

- Added Excel upload support and custom plotting.
- Added datasets from additional references.
- Expanded the "Including data from other papers" section with new reference data.
- Improved and optimized visualizers.

## Public release - 2024-05-15

- Released the public Streamlit app.

## Maintenance - 2023-07-22

- Fixed general bugs.

## b03 - 2023-05-22

- Added pre-release version b03.
