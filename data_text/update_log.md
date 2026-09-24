# Update History

Detailed development log for recent EnvGeo-Seawater updates.

## Unreleased documentation maintenance

- Clarified that `local_data/user_data.xlsx` is a tracked zero-value public
  sample. It may be edited for local researcher use, or an external file may be
  selected with `ENVGEO_LOCAL_USER_DATA_PATH`; restore the zero-value sample
  before committing or synchronizing a public copy.
- Added English and Japanese citation, license, and release-preparation
  checklists.

## 1.3.3 - 2026-09-23

### Release summary

- Added safe degraded-network map behaviour: Plotly maps retain a bundled
  coastline overlay, and `Coastline (offline)` provides a tile-free local map.
- Bundled Natural Earth 50m land polygons for page 32 static Cartopy maps so
  correct land masking does not require a first-run external download.
- Made interactive HTML downloads self-contained on pages 03, 04, and 05.
  Saved figures embed Plotly.js and retain zoom, the mode bar, and responsive
  sizing; online basemap tiles remain external assets.
- Completed the persistent local `User Excel data` workflow, while browser
  uploads remain session-only, and clarified data origin in Quick Visualizer.
- Retained Vertical Section as a beta workflow with its scientific and
  offline-input limitations documented separately.
- Hardened uploaded-data-only workflows in pages 31, 34, and 37: single-value
  coordinate filters no longer crash, invalid coordinates degrade safely, and
  a valid uploaded location can render a map without reference rows.
- Verified the full suite in the supported Python 3.12 / Streamlit 1.63 /
  Plotly 5.24 environment: **302 passed**. One pytest deprecation warning is
  recorded for future fixture cleanup; it does not indicate an app failure.

### T-S Density Contours Stage 1 (2026-09-24)

- Renamed internal variable `sigma_theta` to `sigma0_approx` in
  `pages/34_T-S_diagram.py` to reflect that the quantity is an approximate
  σ0 value, not σθ.
- Changed the density-contour grid from data-extrema ±5 to the displayed
  axis range (`lim_min_X/max_X`, `lim_min_Y/max_Y`) so contours are always
  bounded by the selected salinity and temperature axes.
- Added domain clipping: the contour grid is clipped to the quality-checked
  valid input range (Salinity 0–50; Temperature −5–45 °C), so negative salinity
  values selectable on the Global axis are never passed to `gsw.sigma0`.
- Added **Density contour interval (approx. σ0)** selectbox inside Figure controls
  (options: 0.2, 0.5, 1.0 kg m⁻³; default 0.5 kg m⁻³). Replaced the previous
  `levels=50` fixed level count with an explicit contour-value array derived from
  the selected interval. Degenerate ranges or missing valid values are handled
  safely without raising an exception.
- Added an inline `st.caption` below the T-S figure stating the contours are
  approximate σ0 reference contours (Practical Salinity ≈ Absolute Salinity;
  in-situ temperature ≈ Conservative Temperature) and not pointwise sample density.
- Updated README.md and README_Japanese.md: replaced σθ with σ0 approximation wording.
- Updated English and Japanese T-S manuals: added contour interval to Main Controls;
  corrected approximation note to state differences vary by source, location, and
  depth (removed earlier audit-specific 0.4 kg m⁻³ figure).
- Added focused test `test/test_ts_density_contour.py` verifying axis-limit
  grid bounds, domain clipping, absence of `levels=50`, contour interval options
  and default, explicit level generation, and approximation wording.
- Stage 2 (TEOS-10 data transformation) and Stage 3 (SA–CT mode) remain
  pending; design record: `docs/ts_density_contour_review_and_plan.md`.

### Approximate σ0 Reference Contour Overlay — Page 03 Pilot (2026-09-24)

- Added a **Show density contours** checkbox so the approximate reference
  layer can be hidden without changing the data points. Lightened the contour
  lines and labels so they remain visual context rather than a dominant layer.
- Added approximate σ0 reference contour overlay to the T–S scatter plot in
  `pages/03_[Interactive]_2Dplus_Visualizer.py` as a limited pilot.
  The overlay applies to the Temperature–Salinity view only; Salinity-d18O,
  Custom, and dD views are unaffected.
- Used `go.Contour` (physical isoline of a 2-D scalar field) rather than
  `px.density_contour` (point kernel density). The grid is computed with
  `gsw.sigma0(S_grid, T_grid)` where the salinity and temperature grid spans
  the current scatter-plot axis range, clipped to the quality-checked domain
  (Salinity 0–50; Temperature −5–45 °C). Negative salinity is never passed to
  `gsw.sigma0`.
- Contour trace is prepended to `fig.data` (behind scatter points). No fill,
  no extra colorbar (`showscale=False`), no hover (`hoverinfo="none"`), no
  legend entry. Line labels are drawn in grey.
- Added **Density contour interval (approx. σ0)** selectbox in the T–S
  section (options: 0.2, 0.5, 1.0 kg m⁻³; default 1.0 kg m⁻³). The
  `contours.size` parameter is used; Plotly draws one contour line per
  interval step.
- Updated `selected_point_indices` to accept a `scatter_curve_number`
  parameter (default 0). When the contour trace is prepended the scatter trace
  shifts to `curveNumber=1`; the function receives the correct index so
  Box/Lasso selection is not broken.
- Added an inline `st.caption` below the T–S figure stating the contours are
  approximate σ0 reference contours and not pointwise sample density.
- Updated English and Japanese manuals for page 03
  (`docs/manual/03_2dplus_visualizer.md`,
  `docs/manual_Japanese/03_2dplus_visualizer.md`): added density contour
  interval to Main Controls; added approximation note to Notes And Limitations.
- Added focused static-analysis tests in `test/test_p03_ts_density_contour.py`
  covering: `go.Contour` usage, absence of `px.density_contour`, domain
  clipping constants, selectbox options and default, `showscale=False`,
  approximation caption wording, and `scatter_curve_number` parameter.

### Detailed development log

- Replaced the former fixed local workbook with an always-loaded
  `local_data/user_data.xlsx` workflow. Its rows receive the `User Excel data`
  dataset label and are appended to the Japan Sea, Around Japan, and Global
  reference selections. Browser `Uploaded data` remain session-only and
  independent.
- Matched the former workbook ingestion order: read the local table, assign its
  dataset category, concatenate it with the selected reference source, and then
  apply the shared cleaning and quality pipeline.
- Fixed spreadsheet numbers copied with invisible Unicode whitespace. Regular,
  non-breaking, narrow non-breaking, and full-width spaces are removed before
  numeric conversion; Unicode minus signs are normalized. Literal replacement
  is used for compatibility with both Python- and PyArrow-backed Pandas strings.
- Improved Salinity-d18O diagnostics. The sub-dataset panel reports the number
  of always-loaded local rows, plotting reports local rows excluded by filters
  or missing required axes, and the Filtered dataset table now retains every
  row that passed Data filtering rather than showing only plot-ready rows.
- Changed Custom Parameter Plot so valid X/Y rows remain visible with a fixed
  fallback color when the selected color parameter is missing.
- Added regression coverage for invisible spreadsheet spaces, PyArrow strings,
  configured local-table merging, and non-fatal missing local files.
- Added a shared English coordinate-entry note below browser upload controls:
  use decimal degrees rather than degrees–minutes–seconds, with valid latitude
  and longitude ranges and concrete decimal-degree examples. The same note
  appears in Integrated Visualizer.
- Clarified that browser uploads are session-only and that `User Excel data`
  is a separate always-loaded local table. The upload panel directs users to
  Home → Show README → User Data Integration for the full workflow; the
  compact upload-panel guidance is English-only.
- Fixed the Home-page Japanese README link. Streamlit rendered the relative
  Markdown link as a browser URL, so Home now provides a dedicated
  `日本語版 README` expander instead.
- Fixed mouse-wheel zoom in the `User Data Check & Quick Visualizer` 2-D map.
  The Plotly map now enables `scrollZoom`; a regression test protects this
  setting.
- Reordered the Quick Visualizer sidebar so `Uploaded marker style` appears
  before Data filtering, matching the other upload-enabled pages. Marker
  controls now remain available even when current filters exclude all uploads.
- Clarified the Quick Visualizer empty-state message: users should use the
  sidebar to upload a file or select comparison data.

## 1.3.2 - 2026-09-22

### Release summary

- From version 1.3 onward, development has substantially adopted AI coding assistants (OpenAI Codex, Anthropic Claude Code) for code review, implementation drafting, refactoring, tests, bug investigation, and documentation; every adopted change is reviewed, edited, and verified by the human author. See `docs/development_notes.md` and the project README for the full policy.
- Established the shared browser-upload workflow across the active specialist pages and the new public `User Data Check & Quick Visualizer`.
- Made selected `Uploaded data` available through common Data filtering and integrated it into compatible calculations while preserving foreground marker rendering.
- Improved Vertical Section upload handling and colorbar controls; its interpolation remains an experimental workflow requiring scientific validation.
- Restored consistent tab styling under Streamlit 1.63 and documented the planned retirement of the legacy local user-data workbook.
- Verified the current 1.3.2 consolidation in the Streamlit 1.63 / Plotly 5.24 environment: 108 pytest tests passed.

### Detailed development log

### 2026-09-22

- Renamed page 05 to `User Data Check & Quick Visualizer` and its source file to `05_User_Data_Check_Quick_Visualizer.py`; removed the beta label because the page is now the public user-data entry point.
- Replaced the fixed legacy workbook with a configurable always-loaded local user table. Researcher-owned CSV/XLSX/XLS files use the same preparation path as browser uploads, receive the `User Excel data` dataset label, and are appended to each selected reference source; browser uploads remain a separate session-only category.
- Changed page 05 from upload-only `User Data Quick Visualizer` into `User Data Check & Quick Visualizer`. It now places the shared reference-plus-upload Data filtering workflow after the upload controls and applies the selected combined dataframe to Overview & Quality, arbitrary 2D, Salinity-d18O, Temperature-Salinity, arbitrary 3D/4D, 2D Map, Geographic 3D, and filtered CSV export. Summary, upload preview, data table, and quality checks formerly separated across Integrated Visualizer tabs are consolidated into this user-facing entry page; page 90 remains available during migration.
- Improved page 05 with shared ocean-region presets, Atlantic/Pacific-centred Geographic 3D, common color-palette selection, page-04-style depth-map presentation, bounded rich hover metadata, and Arrow-safe mixed identifier previews. The map and geographic 3D view use the same filtered integrated dataframe.
- Improved: unified all tab interfaces in Home, Integrated Visualizer beta, User Data Quick Visualizer, and Vertical Section Visualizer with the blue card-style selected-tab treatment from Earthquake Advanced. The styling supports light/dark themes and narrow screens; Vertical Section Color/Line tabs now also include purpose icons.
- Changed the former 3D/4D Visualizer Uploader into the public `User Data Quick Visualizer`: it now provides shared-session CSV/XLSX upload, recognized/editable columns, quality review, arbitrary numeric-column 2D and 3D/4D scatter plots, longitude-latitude-depth geographic 3D, optional reference context, a plotted-row cap, and interactive HTML export. It remains the dedicated upload-first route for 4D exploration.
- Maintenance: standardized the leading headers of every Python source, helper tool, and test. Existing module explanations remain; each now explicitly identifies its Python 3 executable declaration, UTF-8 declaration, known creation/author metadata or a maintainer when creation metadata was not recorded, and its last-updated date.
- Changed the selected-data regressions in Salinity-d18O Relationship and Custom Parameter Plot beta, and the Scatter/Contour calculations in Isotope & Hydrographic Mapping, to use the local integrated dataframe of reference rows plus the `Uploaded data` rows selected in Data filtering. Uploaded-only selections are therefore calculation inputs too. When Mapping cannot perform linear contour interpolation because fewer than three points are selected or locations are collinear/duplicated, it now safely falls back to nearest-neighbour interpolation. The frontmost uploaded-marker redraw remains in place.
- Restored the prior Integrated-style workflow for Vertical Section Visualizer: `Uploaded data` appears in Data filtering → Select sub-dataset. When selected there, valid uploaded rows follow the common filters and are combined locally with reference rows for section projection, interpolation, contours, and observed-depth seafloor fallback; deselecting it removes them from both section calculation and display. The maximum valid-row safety limit applies to the combined section input.
- Fixed Uploaded data-only selection in Vertical Section Visualizer. The common filter now receives a local reference-plus-upload dataframe, so selecting only `Uploaded data` no longer produces a false “no data found” state. Optional absent upload columns use safe filter defaults; the section still requires valid longitude, latitude, depth, and target values.
- Added the filtered Uploaded data count in parentheses beside the main `data found` total when Vertical Section Visualizer uses its Uploaded data sub-dataset.
- Extended the local reference-plus-upload Data filtering workflow to Salinity-d18O Relationship, Isotope & Hydrographic Mapping, T-S Diagram, Custom Parameter Plot beta, and Depth Profile. Each now exposes `Uploaded data` in Select sub-dataset and applies the common filters to selected uploaded rows without changing source files. Mapping keeps uploaded rows out of contour interpolation to avoid silently changing the calculated field.
- Ensured uploaded rows are always rendered in the foreground across all upload-enabled figures. In particular, Vertical Section now redraws selected uploaded section points as the final outlined trace even when they are included in the interpolation input.
- Reduced the default uploaded-marker size in Depth Profile from 140 to 10, with a 1-unit minimum, so dense uploaded profiles begin with unobtrusive markers while retaining manual size control.
- Fixed uploaded-data-only filtering in Depth Profile: valid selected uploaded rows now satisfy the profile drawing check even when no reference rows remain. Added uploaded-only AppTests across pages 31, 32, 34, 35, 37, and 53; also made Salinity-d18O Relationship skip its selected-reference regression safely when no reference rows are selected.
- Added an `Uploaded data` item inside the shared Data filtering form for all native-overlay pages (31, 32, 34, 35, 37, and 53). It provides overlay visibility and optional application of the common time, position, depth, salinity, isotope, and temperature ranges without merging uploaded rows into reference data. Vertical Section now passes only these filtered uploaded rows to its A-B selector map before the existing 3,000-point display cap is applied.
- Added selected-target availability reporting to Vertical Section Visualizer: it now shows valid versus missing/invalid values for the active target parameter, alongside the existing count of rows ready for section plotting. Uploaded markers are now also visible while drawing an A-B line on the Folium selector map; both the selector and the Section Map report coordinate-based overlay counts and exclusions.
- Expanded Vertical Section Visualizer color controls with the shared EnvGeo colormap choices plus adjustable horizontal colorbar thickness, length, font size, and approximate tick count. Colorbar ticks now use horizontal, human-readable rounded values instead of densely angled decimal labels.
- Matched the Vertical Section Visualizer data-source selector to the other pages by displaying its three choices horizontally on one line.
- Aligned the Vertical Section Visualizer sidebar with the other visualization pages: the three shared user-data upload panels now appear first, followed by the common Data filtering form with the same dataset/transect selectors, Month segmented control, area preset, range sliders, Apply buttons, and filtered-data summary. Section-specific, bathymetry, and display controls follow afterward.
- Added native shared-upload overlays to Vertical Section Visualizer beta. Uploaded longitude, latitude, depth, and selected target values are projected onto the active A-B corridor or the matching Axis-based coordinate and drawn as distinct foreground markers on both section plots and the section map; they remain excluded from reference filtering, interpolation, and seafloor estimation. Registered page 53 as a native overlay owner in Integrated Visualizer and added standalone/embedded AppTests.
- Confirmed 87 passing tests in the Streamlit 1.42 baseline environment and 12 targeted upload-overlay AppTests in the Streamlit 1.63 environment after the page 53 rollout, sidebar alignment, and colorbar controls.
- Expanded uploaded-location map hover text to show available uploaded metadata and experimental fields, including Year, Month, Cruise, Station, and arbitrary user columns. Quality-report internals remain hidden; hover content is bounded to avoid slow maps for unusually wide tables.
- Added Month alias recognition for `month`, `sampling_month`, `sample_month`, and the Japanese label `月`, plus an editable Month assignment in Depth Profile so uploaded profiles retain month-based coloring when labels differ.
- Added a shared optional line-style extension to uploaded marker controls and enabled it for Depth Profile. Users can now adjust uploaded profile line width and choose dotted, dashed, solid, or dash-dot lines while preserving the previous dotted 2.0-width default.
- Added the uploaded-data quality-check expander to Depth Profile and placed it above the profile figure, consistent with the other upload-enabled pages.
- Extended Custom Parameter Plot beta so uploaded-only numeric fields can be selected as axes, values missing the selected shared color parameter use the fixed marker color instead of being excluded, and figure-size, font-size, and tick-count controls use a compact two-column layout.
- Expanded common upload AppTests for the five upload-enabled pages, Integrated embedding ownership, Depth Profile line controls, and uploaded-only Custom Parameter Plot axes. Confirmed 83 passing tests in the Streamlit 1.42 baseline environment and 8 targeted AppTests in the Streamlit 1.63 environment.

### 2026-09-21

- Fixed Integrated Visualizer so `uses_native_upload_overlay` recognizes every Full-existing-page workflow that has its own upload panel (Salinity-d18O Relationship, Isotope & Hydrographic Mapping, T-S Diagram, Custom Parameter Plot beta, Depth Profile) instead of only T-S Diagram. Previously, opening any of the other pages through Integrated's Full existing page mode with uploaded data present silently merged the uploaded rows into the reference dataset via the legacy `load_isotope_data` patch (affecting background statistics and, for the Mapping page, the contour interpolation) while also rendering a duplicate native upload panel. Added a regression test asserting `NATIVE_UPLOAD_OVERLAY_PAGES` stays in sync with pages that actually implement `envgeo_user_data.render_upload_panel` and the `INTEGRATED_EMBEDDED_PAGE_KEY` check.
- Registered Custom Parameter Plot beta in Integrated Visualizer's Full-existing-page workflow list so its already-implemented upload overlay is reachable from Integrated, not only as a standalone page.
- Fixed Depth Profile so the uploaded-overlay caption is always shown once required columns are assigned, including when every uploaded row is excluded (missing/invalid X parameter or depth); it now reports "0 / N plotted (N excluded due to missing values)" instead of showing nothing, matching the other upload-enabled pages.
- Added the shared uploaded-location overlay to the Isotope & Hydrographic Mapping page: longitude and latitude columns (the only required roles) are auto-detected or manually assigned, the currently selected parameter (d18O, dD, d-excess, Salinity, Temperature) drives shared-colorbar coloring when available, frontmost outlined markers (zorder=10) appear on both the Scatter Map and the Contour Map without being mixed into the griddata interpolation, the Plotly Sampling Location Map gains an `add_uploaded_map_overlay` overlay, automatic map framing includes uploaded locations, and a quality-check expander is shown when uploaded data is present.
- Added the shared uploaded-location overlay to the Salinity-d18O Relationship map, including frontmost outlined markers, shared d18O colors or fixed-color fallback, automatic map framing, and an explanation when coordinates are unavailable.
- Fixed quality flags being cleared when T-S or another individual page reconfirmed automatically recognized upload columns; existing flags are now preserved and merged with any new flags found after manual column assignment.
- Added uploaded sampling locations to the T-S Diagram map when valid longitude and latitude are available, using frontmost outlined markers, shared d18O map colors when possible, fixed-color fallback, and uploaded locations in automatic map framing.
- Extracted the shared upload, editable column-assignment, and marker-style sidebar panels into `envgeo_user_data.py`, retaining page-specific plotting in each visualization page.
- Added uploaded-data support to Salinity-d18O Relationship with automatic/manual Salinity and d18O assignment, shared-colorbar or fixed-color markers, quality reporting, and frontmost Matplotlib overlay rendering.
- Added a pilot `Uploaded data columns` panel between upload and marker controls in T-S Diagram, with editable automatic assignments and explicit manual selection for unknown temperature and salinity labels.
- Added a shared manual column-mapping helper that retains original experimental columns and reapplies standard numeric conversion and quality checks.
- Clarified the final user-data workflow: every supported page registers uploads in shared session state, User Data Validator applies the same core quality rules regardless of upload origin, and Integrated's uploader is retired only after all target-page overlays and equivalent Validator checks are verified.
- Replaced the active 50m and 110m coastline Excel assets with CSV files and centralized CSV loading in `envgeo_utils.py`.
- Removed the legacy direct Japan-coastline Excel dependency from the local 3D/4D uploader and moved the superseded 10m, 50m, 110m, and Japan coastline workbooks to the workspace archive.
- Refined the upload migration strategy: extract Shared-filter beta into an independent User Data Validator, keep individual visualization pages as first-class workflows, and retain Integrated Visualizer as a working migration fallback until it can become a hidden development archive.
- Added incremental migration rules so each change is limited to one shared component or one page, with the existing workflow retained until its replacement passes tests and screen-level checks.
- Planned a focused `envgeo_user_data.py` module for shared upload processing and UI instead of continuing to enlarge `envgeo_utils.py`.

### 2026-09-20

- Added shared, memory-only upload state so prepared user data can be reused across Integrated Visualizer and individual pages during the same Streamlit session.
- Moved CSV/Excel reading, upload preparation, template generation, quality-row extraction, numeric conversion, quality normalization, and d-excess calculation into reusable `envgeo_utils.py` functions.
- Added initial user-data upload, quality review, marker styling, shared-colorbar coloring, and frontmost overlay plotting to Temperature-Salinity Diagram.
- Grouped upload and uploaded-marker controls into two collapsed panels at the top of the T-S Diagram sidebar, separate from reference-data filters and figure controls.
- Prevented duplicate upload controls and double plotting when T-S Diagram is opened inside Integrated Visualizer: Integrated owns file upload, while the native T-S page owns marker styling and overlay rendering.
- Documented the accepted Integrated Visualizer architecture and migration plan in dedicated English and Japanese strategy files, retaining individual pages as first-class workflows.
- Added unambiguous Japanese aliases for longitude, latitude, depth, temperature, and salinity upload columns.
- Expanded upload tests and confirmed 65 passing tests plus successful T-S Diagram AppTest runs with and without shared uploaded data.

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
- Clarified sidebar update behavior with red captions for settings that require an Apply button and blue captions for settings that update automatically.
- Changed Interactive 3D/4D figure-scale controls and Isotope & Hydrographic Mapping display controls to update automatically, removing mixed manual and automatic behavior within those sections.
- Added a trial responsive layout to Interactive 2D/2.5D plots: desktop width remains capped at 850 px while the plots shrink to the available width on narrow screens.
- Renamed the current interactive page files to include `[Interactive]`, making their exploratory Plotly role clear in the Streamlit page list and repository.

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
