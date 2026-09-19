# Update History

Detailed development log for recent EnvGeo-Seawater updates.

## Unreleased

### 2026-09-19

- Updated the development version to 1.3.1 for the Python 3.10-3.12 and Streamlit 1.42-1.63 compatibility cycle.
- Set the test-site Streamlit requirement range to 1.42-1.63 while retaining Plotly 5.24 as the release baseline.
- Fixed Matplotlib/Cartopy figure-state conflicts in Correlation Overview and Salinity-d18O Relationship by drawing on explicit GeoAxes, saving explicit figures, and closing completed figures.
- Disabled exploratory `print()` output in Correlation Overview to keep Streamlit server logs readable.
- Restored automatic Custom Parameter Plot axis and color ranges when switching data sources by keeping widget state separate for each dataset.
- Replaced Custom Parameter Plot mathtext isotope labels with Unicode labels to avoid a Matplotlib parsing error on Streamlit Cloud.
- Matched the Vertical Section Visualizer page-title size to the other main visualization pages.
- Adopted a staged Plotly migration policy: move to MapLibre APIs while still using Plotly 5.24, then verify the same code with Plotly 6.7 and 7.1.
- Confirmed the long-term plan to add memory-only user-data upload and overlay plotting to individual pages through shared utility functions and staged tests.
- Defined Correlation Overview as an archive display of the original hand-written exploratory workflow; it is excluded from new-feature and upload integration work.

### 2026-09-18

- Started compatibility testing with a separate Python 3.12.14 / Streamlit 1.63.0 Conda environment while retaining the verified Streamlit 1.42 environment.
- Confirmed dependency consistency, 57 passing Seawater tests, 9 passing Earthquake tests with 4 optional skips, and successful Seawater Home startup on the new environment.
- Added a separate Streamlit 1.63 / Plotly 5.24.1 comparison environment after identifying Plotly 7 Mapbox API removal as the main source of interactive map errors; documented the migration policy and results in dedicated development notes.
- Added shared compatibility handling for full-width Streamlit elements and Pandas future options, removing repeated deprecation warnings while retaining Streamlit 1.42 support.
- Re-ran 57 Seawater tests in both Streamlit 1.42 and 1.63 environments and confirmed clean initial rendering of all pages in Streamlit 1.63.

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

### 2026-09-17

- Clarified that the Interactive 2D/2.5D and 3D/4D Visualizers are Plotly-based exploration tools, while publication- and presentation-ready static figures should be created with the corresponding individual pages.
- Added `TODO_Japanese.md` and linked the English and Japanese ToDo files for easier local development tracking.
- Reduced `91_EnvGeo_Earthquake.py` to a lightweight redirect page because the active Earthquake implementation is maintained in the dedicated application.

### 2026-09-16

- Renamed the former 3D/4D pages to `Interactive 2D/2.5D Visualizer` and `Interactive 3D/4D Visualizer`, with shorter page filenames.
- Added `dD-δ18O relationship` and `Custom 2D/2.5D plot beta` options to the Interactive 2D/2.5D Visualizer, reusing Box/Lasso selection and linked sampling-location maps.
- Added a single-color option to the Custom 2D/2.5D plot so it can be used as either a pure 2D plot or a color-coded 2.5D plot.
- Renamed the 2D/2.5D page file to `03_2Dplus_Visualizer.py` and aligned Custom plot color controls horizontally; the default Custom color mode now uses a data parameter instead of single color.
- Added a `Full custom X-Y-Z-color` mode to the 4D Visualizer custom view so users can choose all three axes and the color parameter.
- Clarified 3D/4D Visualizer labels so map-depth scale settings are identified as Fig.3-Fig.6 controls and sampling-location map settings are labeled separately.
- Standardized data-table wording: sidebar-filtered results are labeled `Filtered dataset`, while Plotly Box/Lasso outputs remain `Box/Lasso-selected dataset`.
- Applied low-risk cleanup from an external code review, including clearer radio-widget calls, idiomatic empty `else` blocks, top-level imports, and removal of a no-op uploaded-marker colorscale setting.
- Added Claude review follow-up items to `TODO.md` for future data-source selector, auto-zoom, month-display, XY scatter, legacy-variable, and upload-loader refactoring.
- Applied additional low-risk cleanup from the full-file Claude review, including boolean empty-data checks, removal of unused 4D variables, corrected Custom plot exclusion counts, removal of unused month-display variables, and safer Vertical Section color-scale/import handling.
- Added publication/package follow-up notes to `TODO.md` for research-impact citations, paper figures, packaging, development requirements, dependency pins, and future refactoring.

### 2026-09-14

- Fixed a Streamlit Cloud Cartopy error in Isotope & Hydrographic Mapping by avoiding exact full-globe longitude bounds when changing map centers.
- Removed duplicate-looking coastline outlines in Isotope & Hydrographic Mapping by using the land layer only as a fill and keeping coastline lines separate.
- Added a `Region preset` control to Isotope & Hydrographic Mapping map display settings so the figure extent can be changed without changing the filtered dataset.
- Refined the Isotope & Hydrographic Mapping page by moving the mapped-parameter selector next to the map-type control and removing redundant parameter captions.
- Tested collapsible sidebar panels in Custom Parameter Plot beta, then restored the standard bordered sidebar layout because nested expanders are not suitable for the current filtering UI.
- Updated the shared map guidance text to mention map center, extent, colormap, and figure settings in the sidebar.
- Added colormap selection beside `Color filtered` in the 3D Visualizer Plotly views and connected the selected colormap to both the scatter plot and matching map.
- Added an optional regression line to the 3D Visualizer salinity-d18O Plotly view, including compact regression statistics beside the control.
- Kept the Temperature-Salinity view in the 3D Visualizer free of regression-line controls after testing the feature.
- Fixed Box/Lasso selection in the 3D Visualizer so added regression-line traces do not interfere with highlighting matching sampling locations on the map.
- Added `Custom 4D plot beta` to the 4D Visualizer while preserving Fig.1-Fig.6.
- Added custom 4D templates for `Salinity-d18O-[custom]-[custom]`, `T-S-[custom]-[custom]`, and `Lon-Lat-depth-[custom]`.
- Adjusted the `Lon-Lat-depth-[custom]` template to behave like the existing Fig.3-Fig.6 map-depth views, including map-centered longitude handling, coastline traces, and geographic aspect scaling.
- Removed an implementation-oriented custom-beta caption from the 4D Visualizer UI.
- Added a shared `Area filter preset` control to the common Data filtering sidebar so users can initialize longitude and latitude filters from familiar ocean-region presets and still fine-tune the sliders manually.
- Refined 4D Visualizer UI wording for the main view selector, custom view controls, map-depth settings, colorbar range controls, and sampling-location map labels.
- Harmonized visible UI wording across the 3D Visualizer, T-S, salinity-d18O, isotope/hydrographic mapping, Depth Profile, and Custom Parameter Plot pages, including map labels, map-style controls, color-parameter controls, background-data toggles, and selection-table labels.
- Moved 2D figure download buttons below their corresponding figures in the T-S, salinity-d18O, Depth Profile, isotope/hydrographic mapping, and Custom Parameter Plot pages.
- Added concise help text to common user controls, including color-parameter selectors, background-data toggles, regression-line controls, map-style selectors, 4D view selectors, and profile-parameter controls.
- Adjusted the Depth Profile figure title wrapping and top margin so long filter-condition titles fit better in downloaded images.
- Changed Depth Profile figure width and height controls from a paired slider to numeric inputs for more precise layout adjustment.
- Standardized precise figure controls by using numeric inputs for figure size, font size, tick counts, and the mapping-page colorbar font size where applicable.
- Updated the English and Japanese README files to reflect the current page structure, beta/local-development page roles, environment-check workflow, user-data integration status, and a more cautious reproducibility description.
- Added `docs/README.md` and `docs/release_checklist.md` to separate release, deployment, Zenodo, and internal planning notes from the top-level README.
- Added `docs/testing.md` and `docs/testing_Japanese.md` to explain the current pytest suite, its scope, limitations, and planned expansion in a public-facing format.
- Added English and Japanese user-manual skeletons under `docs/manual/` and `docs/manual_Japanese/`, including overview, shared filtering, and page-by-page manual templates.
- Revised the testing documentation to keep it public-facing, moving internal planning out of `testing.md` and `testing_Japanese.md`.
- Added project ToDo notes for future user-data upload support in individual pages, Streamlit submit-button key cleanup after upgrade, Integrated Visualizer publication strategy, and the likely private/development-only role of the standalone 3D/4D uploader.

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
- Added a compact quality-flag criteria note below the `Filtered dataset (CSV)` table.
- Added CSV export for `Details and statistics of filtered data`, including filter conditions, filtered-data counts, row counts, quality-flag counts, and summary statistics.
- Changed the shared standard map background from `carto-positron` to API-key-free `open-street-map` because CARTO basemaps now require API keys.
- Added CSV and PDF report export to the environment checker for runtime, dependency, and project-file diagnostics.
- Kept the Streamlit environment checker implementation in `tools/env_check_streamlit.py` and added `pages/99_Environment_Check.py` as a local-development sidebar wrapper.
- Updated `requirements.txt` to match the current Anaconda `envgeo_st142_py310_plotly5` environment and document the verified Python 3.10 dependency set.
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
