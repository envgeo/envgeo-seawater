# Integrated Visualizer Strategy

Decision date: 2026-09-20  
Last updated: 2026-09-22

## 1.3.2 Implementation Update

The planned independent User Data Validator is now implemented as the public
**User Data Check & Quick Visualizer** (page 05). It provides the upload-first
quality review, missing-value checks, reference comparison, shared Data
filtering, simple 2D--4D plots, maps, and filtered CSV export. Page 90 remains
a transitional integration/development page; do not move new public workflow
features back into it.

The active specialist pages 31, 32, 34, 35, 37, and 53 expose `Uploaded data`
in common Data filtering. Page 53 uses selected valid uploaded rows in its
local section calculation. Pages 03 and 04 remain outside the upload rollout.

## Decision

Individual visualization pages remain first-class EnvGeo-Seawater workflows.
Uploaded-data validation and comparison are provided by the public User Data
Check & Quick Visualizer. Integrated Visualizer remains a transitional beta test bed and
will become a hidden development archive after its useful workflows have moved
to the validator, shared core, and individual pages.

Upload support will use one shared implementation. Common upload processing and
UI should move out of page files into a focused core module, provisionally
`envgeo_user_data.py`, rather than continuing to enlarge `envgeo_utils.py`.

## Why

T-S diagrams, depth profiles, maps, vertical sections, and custom plots have
different required columns, scientific checks, and figure controls. Keeping
their plotting logic in focused pages makes the behavior easier to understand,
test, document, and maintain. Moving all of those responsibilities into one
large page would increase coupling and make scientific changes harder to review.

The independent User Data Validator will provide cross-view quality review and
rapid comparison without mixing those tasks with publication-oriented figure
controls. Shared session data will let users validate once and then move to any
individual visualization page.

## Current Transitional State

- Upload files and prepared data remain in memory for the current Streamlit
  session only.
- CSV/Excel reading, column-name standardization, numeric conversion, seawater
  quality rules, d-excess calculation, and shared session storage are provided
  by `envgeo_utils.py`.
- Shared upload, column-assignment, and marker-style panels are now provided by
  `envgeo_user_data.py` and used by T-S Diagram and Salinity-d18O Relationship.
- T-S Diagram and Salinity-d18O Relationship retain only their page-specific
  required-column definitions and Matplotlib overlay rendering.
- Integrated Visualizer uses the same shared processing and session data but
  currently owns separate Streamlit controls and Plotly overlay code.
- When T-S Diagram is opened inside Integrated Visualizer, Integrated owns file
  upload while T-S Diagram owns marker styling and overlay rendering. This
  prevents duplicate controls and double plotting.
- Other Full existing page workflows still use temporary data-loader
  replacement until their native upload overlays are implemented.
- Shared-filter beta currently combines validation and comparison functions
  that will be extracted into the independent User Data Validator.

This mixed ownership is a migration state, not the final architecture.

## Target Responsibilities

| Layer | Responsibility |
| --- | --- |
| Shared upload core | In-memory file reading, normalized column mapping, manual mapping support, numeric conversion, validation results, quality flags, derived values, and session state |
| Seawater upload rules | Seawater aliases, required-column profiles, valid ranges, and d-excess configuration |
| Shared Streamlit upload UI | Upload, template download, detected-column report, manual correction, quality summary, clear action, and common marker-style model |
| User Data Validator | Column review and correction, quality reporting, exclusion reasons, reference-data comparison, and simple diagnostic plots |
| Individual page | Required columns for that figure, exclusion summary, plot-specific conversion, and final overlay rendering |
| Integrated Visualizer | Temporary migration test bed; hidden development archive after migration |

Marker settings should use one backend-neutral model: size multiplier, fixed
color or shared-colorbar mode, shape name, opacity, outline color, and outline
width. Individual Matplotlib and Plotly pages should only translate that model
to backend-specific marker arguments.

## Final User-Data Workflow

1. Every supported individual page provides the shared upload panel near the
   top of its sidebar and registers prepared data in the same in-memory session
   state.
2. An individual page provides a compact quality summary, the marker settings
   needed by that figure, and its own overlay rendering. It does not duplicate
   the full validation interface.
3. User Data Validator can open the same session data regardless of which page
   originally uploaded it. The Validator provides detailed column mapping,
   missing-value and range checks, quality flags, exclusion counts and reasons,
   reference-data comparison, and optional quality-report export.
4. Quality rules and derived values are implemented in the shared core, not in
   the Validator page. Individual pages and the Validator therefore report the
   same result for the same uploaded data.
5. Integrated Visualizer keeps its uploader while any supported individual
   page still depends on it. The Integrated uploader is removed only after all
   target pages have verified native upload and overlay support and the
   independent Validator provides equivalent Shared-filter quality checks.

Uploaded data remains available only within the current Streamlit session and
is not written to local or server storage.

## Column Assignment UI

Individual-page upload controls use three collapsed sidebar panels in this
order: `Uploaded data overlay`, `Uploaded data columns`, and `Uploaded marker
style`. Known aliases are assigned automatically and shown as editable
selections. Unknown labels remain unassigned until the user explicitly chooses
their plotting role; they must not be accepted by speculative matching.

Scientific pages label selectors by meaning, such as `Temperature column` and
`Salinity column`, rather than only `X` and `Y`. Generic custom visualizers may
instead expose X, Y, Z, and color roles. Original source columns are retained
when a role is assigned so new elements and experimental parameters remain
available for later custom plots. Missing required roles open the column panel
and prevent the overlay until the user completes the assignment.

When an individual page includes a sampling-location map, upload support must
cover both the main figure and that map. Valid uploaded longitude/latitude rows
are drawn last with visually distinct outlined markers, included in automatic
map framing, and colored with the existing map color scale when compatible.
Rows without a compatible color value use the fixed upload color. If location
columns are missing or invalid, the page reports why no uploaded locations are
shown and displays plotted/excluded counts.

## Planned Validator And Integrated Roles

Move to the independent User Data Validator:

- Shared-filter exploration across maps and related plots.
- Uploaded-data quality review and quick visual checks.
- Detected-column reporting and future manual column correction.
- Exclusion counts and reasons.
- Reference-data comparison using simple diagnostic plots.
- Session handoff to individual visualization pages.

Retire or hide with Integrated Visualizer after migration:

- Integrated-specific upload preparation code.
- Temporary monkey-patching of `load_isotope_data`.
- Full existing page embedding as a compatibility mechanism.
- Duplicated T-S, salinity-d18O, and other validation plots after Validator
  equivalents are verified.

Integrated Visualizer source may remain as a development archive, but it should
not be presented as the normal public workflow after migration.

## Migration Sequence

1. Extract upload processing, shared upload UI, session state, quality reporting, required-column profiles, and the marker-style model into a focused core module.
2. Replace the T-S Diagram's local upload controls with that shared component and verify the pilot again. Completed on 2026-09-21.
3. Extract Shared-filter beta into an independent User Data Validator page.
4. Apply the shared component to Salinity-d18O Relationship, Mapping, Depth Profile, Custom Parameter Plot, Interactive 2D/3D/4D pages, and Vertical Section. Pages 31, 32, 34, 35, and 37 are complete; Vertical Section (page 53) was completed on 2026-09-22. Interactive pages 03 and 04 remain pending.
5. Remove each workflow from Integrated's temporary loader-replacement path after its native overlay is verified.
6. Hide Integrated Visualizer from the normal public page list after Validator and individual-page migration are complete.

Each migration step requires focused pytest coverage and an AppTest or visual
check before moving to the next page.

## Incremental Migration Rules

- Limit one work unit to either one shared component or one individual page.
- Do not remove the current Integrated implementation until its replacement has
  passed tests and a screen-level check.
- Keep Shared-filter beta until User Data Validator provides equivalent
  validation and comparison coverage.
- Keep each Full existing page path until that individual page has a verified
  native upload overlay.
- End every migration step with a usable application so development can pause
  safely between steps.

## Future envgeo4d Direction

The eventual module boundary should separate generic upload mechanics from
domain rules and plotting:

- Current transition: use a focused module such as `envgeo_user_data.py` for
  shared upload processing and UI instead of placing every new function in
  `envgeo_utils.py`.
- `envgeo4d/common`: file input, session state, column-mapping framework,
  validation result structures, and shared UI models.
- `envgeo4d/seawater`: seawater aliases, ranges, d-excess, and figure-specific
  requirements.
- `envgeo4d/earthquake` and future domains: their own schemas and quality rules
  while reusing the common upload framework.

### Shared map and offline-asset direction

Map controls, map layout, and offline geographic assets are also shared-core
concerns, but they must remain separate layers rather than becoming one large
map function. The common layer should provide: (1) a reusable Streamlit map
controls component with page-specific widget keys, (2) Plotly/Folium layout
helpers, (3) path-safe loading and caching of bundled coastline CSV/GeoJSON
assets, (4) online/offline capability and asset-availability checks, and (5)
generic construction of online-tile or local-coastline layers. The current
`apply_map_style()` and `apply_standard_map_layout()` are small transitional
helpers in that direction; the `Map controls` widget itself is still copied
per page and is a future extraction candidate.

The common layer must not contain scientific interpretation of its data.
GEBCO sampling, seafloor interpolation, and oceanographic fallback policy
remain in `envgeo4d/seawater`; earthquake catalog acquisition, plate-boundary
semantics, and magnitude/depth rules remain in `envgeo4d/earthquake`. External
services do not argue against common code: their configuration, capability
status, attribution hooks, and safe local fallback belong in the common layer,
while each domain decides which sources are scientifically appropriate.

Start the migration with bundled 50 m/110 m coastline assets and their loader,
cache, resolution selection, licence/attribution metadata, and tests. Then
extract map controls and layout, followed by longitude/extent helpers. Adopt
one component in one page at a time; retain each app's local implementation
until both Seawater and Earthquake have been visually checked and tested.

Unknown column names must never be accepted solely through speculative matching.
Automatic recognition should be followed by a visible mapping report and, when
needed, explicit user correction.
