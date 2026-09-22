# EnvGeo-Seawater ToDo

Japanese version: [TODO_Japanese.md](TODO_Japanese.md)

This file records development notes that should not be forgotten during local
cleanup and refactoring. Items here can be moved into README, documentation, or
GitHub issues when they become stable enough.

As a general project policy, keep user-facing UI text, README files, manuals,
testing guides, release notes, and major development policies available in both
English and Japanese. Keep this file aligned with `TODO_Japanese.md`.

## Priority Roadmap

Updated: 2026-09-19

Work in this order to avoid repeating page-level changes.

1. Complete the compatibility patch releases.
   - Finish manual checks with Python 3.10 / Streamlit 1.42 / Plotly 5.24 and
     Python 3.12 / Streamlit 1.63 / Plotly 5.24.
   - Check Plotly selection, Cartopy maps, forms, uploads/downloads, Vertical
     Section, and Earthquake Simple/Advanced interactions.
   - Fix only reproducible compatibility problems and add focused tests.
   - Treat Seawater 1.3.2 and Earthquake 0.3.2 as the current local release boundary; complete the remaining visual checks before any public release/tag.
2. Migrate maps from Mapbox to MapLibre.
   - Inventory shared and page-specific Mapbox usage.
   - Convert one representative page while still using Plotly 5.24.
   - Move reusable map behavior into shared helpers.
   - Verify the same implementation with Plotly 5.24, 6.7, and 7.1, including
     Python 3.10 / Streamlit 1.63 / Plotly 7.
   - Review native `st.plotly_chart` selection as a replacement for
     `streamlit-plotly-events`.
3. Build the shared user-data core.
   - Add pure, tested CSV/XLSX loading, column auto-detection and manual mapping,
     validation, quality flags, d-excess, source labels, and memory-only state.
   - Keep Streamlit widgets separate from the reusable data-processing logic.
4. Roll user-data overlay plotting out to active individual pages.
   - Start with T-S, Salinity-d18O, Mapping, and Depth Profile.
   - Continue with Custom Parameter Plot, Interactive Visualizers, and Vertical
     Section after the first pages are stable.
   - Keep Correlation Overview as an archive display and exclude it from new
     feature work.
5. Consolidate the public application and documentation.
   - Decide the final Integrated Visualizer role and which individual pages are
     public, advanced, beta, local-only, or retired.
   - Complete English/Japanese UI review, help text, manuals, screenshots,
     installation documentation, packaging, and the JOSS resubmission checklist.

Do not combine MapLibre migration, all-page upload rollout, navigation changes,
and broad refactoring in one release.

## Documentation and Test Strategy for JOSS

Date adopted: 2026-09-19

Use the useful parts of external review suggestions as a continuing development
policy rather than treating a particular documentation tool as a submission
requirement.

Documentation policy:
- Improve the content of the existing English and Japanese Markdown manuals
  first: installation, quick start, page workflows, user-data import, quality
  control, troubleshooting, testing, contribution, and support information.
- Use MkDocs as the leading candidate for a future standalone documentation
  site on GitHub Pages. Sphinx is not currently necessary for this
  Markdown-centered, user-facing Streamlit project.
- Introduce MkDocs after the manual structure is reasonably stable. A generated
  website should organize good documentation rather than replace missing
  content.
- Add API reference pages for the stable public functions in the future shared
  core (`envgeo4d` / `envgeo_utils`), especially data loading, validation,
  quality flags, d-excess, region presets, and export helpers. Do not attempt to
  document every page script as a public API.
- Keep `51_Correlation_Overview.py` documented as an intentionally preserved
  exploratory/archive workflow, not as a target for API cleanup.

Testing policy:
- Maintain three complementary layers: pure-function pytest tests, Streamlit
  `AppTest` workflow tests, and a short manual visual QA checklist.
- Convert the current ad hoc Streamlit startup checks into persistent pytest
  tests using `streamlit.testing.v1.AppTest`.
- Begin AppTest coverage with representative active workflows: Home, Mapping,
  T-S, Depth Profile, and Integrated Visualizer upload/quality checks. Expand to
  the remaining active pages in small steps.
- Test normal input and failure states, including empty filtering results,
  missing columns, invalid CSV/XLSX files, NaN and out-of-range values, quality
  warnings, and download controls.
- Keep numerical and scientific behavior in ordinary unit tests. Use AppTest
  for page startup, widgets, forms, messages, and session-state behavior.
- Keep Plotly Box/Lasso interaction, map tiles, Cartopy rendering, colorbars,
  and downloaded-figure layout in the manual visual checklist where AppTest
  cannot verify browser-rendered behavior reliably.
- Add a minimal GitHub Actions workflow now for the existing pytest suite. Add
  persistent AppTest cases and a broader compatibility matrix later, after the
  representative AppTest set and Streamlit/Plotly baselines are stable.

JOSS note:
- Documentation and tests are two major workstreams, but the resubmission also
  needs packaging/installability, CI evidence, contribution/support guidance,
  research-impact references, public development history, tagged releases, and
  an appropriate disclosure of AI-assisted development.

## High Priority

### Claude review follow-ups

Date added: 2026-09-17

Current status:
- Claude review comments were checked against the current codebase.
- Low-risk cleanup was applied to active pages:
  - Removed misleading `args=[1, 0]` from `st.radio()` calls where it was not
    being used as a callback argument.
  - Replaced non-idiomatic `else:()` blocks with explicit `else: pass` in the
    main active plotting pages touched by the review.
  - Moved local `io` / `textwrap` imports in T-S, salinity-d18O, and Depth
    Profile pages to the file-level import area.
  - Removed a no-op `colorscale=None if use_colorbar else None` line from the
    integrated visualizer upload-map trace.
- `51_Correlation_Overview.py` is an intentional archive of the original
  hand-written exploratory workflow. No new features, upload integration, or
  structural refactoring are planned.
- This is intentional preservation, not simple neglect. The page documents the
  development process and remains available for display. Maintenance is limited
  to changes required to keep the page opening and its existing figures visible.

Planned direction:
- Add `envgeo_utils.render_data_source_selector()` to reduce repeated data
  source radio/reference blocks across pages.
- Add a shared map auto-zoom helper, such as
  `envgeo_utils.auto_zoom_from_extent()`, and gradually replace repeated zoom
  calculations.
- Add a shared longitude-centering helper, currently repeated as
  `normalize_lon_to_center()`, to `envgeo_utils.py`.
- Consider extracting the repeated T-S / salinity-d18O scatter workflow in the
  integrated visualizer into a reusable XY scatter helper.
- Move repeated month-range display logic into
  `envgeo_utils.format_month_selection()`.
- Refactor repeated coastline trace code in `04_[Interactive]_3D_4D_Visualizer.py` into a
  helper such as `add_coastline_traces()`.
- Revisit dead legacy variables such as `X_Y = 1`, `sheet_num = [2]`, and
  unused color variables page by page, starting with Depth Profile and
  Salinity-d18O Relationship.
- Refactor `sidebar_filter_and_display()` gradually. It is currently a large
  mixed UI/filter/statistics function with a long tuple return value.
- Long term: replace the upload-data monkey-patch in the integrated visualizer
  with an explicit loader override or injected data object so multi-user
  behavior is easier to reason about.

JOSS/package follow-ups from the full-file review:
- Add research-impact citations to `paper.md` and `paper.bib`, including Aono,
  Sakamoto, and Kuroki studies that used seawater d18O data.
- Add at least one figure/screenshot to the JOSS paper.
- Revisit the State of the field section so the difference from ODV and other
  oceanographic tools is specific to isotope/hydrographic workflows, d-excess,
  and EnvGeo-Seawater's web-first use case.
- Plan the packaging route for `pyproject.toml`, PyPI, and possibly
  conda-forge. This is required before JOSS resubmission if installation via a
  package manager is expected.
- Create `requirements-dev.txt` for pytest-related development dependencies.
- Revisit strict `==` dependency pins after the Streamlit 1.6x migration test.

### JOSS readiness audit follow-up

Date added: 2026-09-19

An external review identified four critical release items. The review counted
four test files and about 46 tests, but the current project has five
`test_*.py` files and 60 tests. `envgeo_utils.py` is currently about 78 KB.

Complete the critical items in dependency order:

1. Add a minimal GitHub Actions workflow that runs the current pytest suite on
   pushes and pull requests. Do not wait for AppTest expansion before adding
   basic CI.
2. Create `CITATION.cff` with the current project metadata and version. Leave
   the DOI absent or clearly pending until Zenodo issues the final identifier.
3. Add the canonical repository URL and software version to `paper.md`; complete
   the Availability wording and research-impact references before release.
4. Finish the 1.3.2 compatibility checks and create the final tagged GitHub
   release.
5. Archive that release with Zenodo, then add the issued DOI consistently to
   `paper.md`, `CITATION.cff`, README citation guidance, and release records.

Also measure test coverage for planning purposes, but improve tests according
to scientific and workflow risk rather than pursuing a coverage percentage by
itself.

### User-data upload support for individual pages

Date added: 2026-09-13; architecture confirmed: 2026-09-20

Current status:
- Keep the existing implementation in `90_Integrated_Visualizer_beta.py` as the
  main test bed for user-data upload workflows.
- The approved long-term direction is to support user-data upload and overlay
  plotting on the individual visualization pages.
- Complete the shared loader and schema workflow before adding separate upload
  implementations to every page.
- Keep individual visualization pages as first-class public workflows. Move
  validation and reference-data comparison to an independent User Data
  Validator. Keep Integrated Visualizer during migration, then retain it only
  as a hidden development archive. See `docs/integrated_visualizer_strategy.md`.

Planned direction:
- Move the common upload workflow into a focused module such as
  `envgeo_user_data.py` before expanding it to individual pages. Reuse existing
  `envgeo_utils.py` functions without continuing to enlarge that module.
- Treat user-data upload support as a core EnvGeo utility, not only as a
  seawater-specific feature. The same foundation should eventually support other
  EnvGeo applications, such as earthquake and other geoscience visualizers.
- The shared workflow should handle CSV/XLSX reading, column-name
  auto-detection, manual column correction, required-column checks, d-excess
  calculation, quality flags, source labels, and session-only memory handling.
- Use consistent overlay controls for marker size, color or shared colorbar,
  symbol, outline, opacity, and foreground order.
- Extract Shared-filter beta into an independent User Data Validator and keep
  the existing Integrated workflow until equivalent checks are verified.
- Make one shared-component or one-page change at a time; keep the application
  usable and tested at the end of every migration step.
- Roll out and test the shared workflow in stages:
  - Temperature-Salinity Diagram
  - Salinity-d18O Relationship
  - Mapping / isotope map pages
  - Depth Profile
  - Custom Parameter Plot
  - Interactive 2D/2.5D and 3D/4D Visualizers
  - Vertical Section beta
- Exclude Correlation Overview from the upload rollout because it is retained
  as an archive page rather than an actively developed workflow.
- Add tests for CSV/XLSX parsing, alias and Japanese-label recognition, manual
  mapping, validation failures, quality flags, and memory-only handling before
  completing the rollout.
- Maintain `User Data Check & Quick Visualizer` as the public upload-first
  route for quality review and simple arbitrary-column 2D--4D exploration.
  Keep its session-only data model, shared Data filtering, and clear division
  of responsibility from the specialist analysis pages while page 90 remains
  available during the migration.
- Retire `dataset/91_USER_UPLOAD_UNPUB.xlsx` only after auditing and removing
  its live `envgeo_utils.py` loader path, its `Unpublished dataset` inclusion,
  any test/sample dependency, and obsolete documentation. Browser CSV/XLSX
  upload is now the normal user-data workflow; do not delete the workbook
  before this reference audit is complete.

Notes:
- Uploaded user data should remain in memory only and should not be saved to the
  local machine or server.
- Uploaded user data should be visually distinguishable from reference data.
- Uploaded-data quality flags should be visible and downloadable where relevant.
- Do not save uploaded source files or merged datasets to local or server
  storage unless the user explicitly chooses a future export action.

## Streamlit Upgrade Follow-ups

### Streamlit 1.63 / Python 3.12 migration test

Date started: 2026-09-18

Detailed record: [docs/streamlit_migration.md](docs/streamlit_migration.md)

Current status:
- Created the separate Conda environment `envgeo_st163_py312_plotly7`; the existing
  `envgeo_st142_py310_plotly5` environment remains the stable comparison environment.
- Added `envgeo_st163_py312_plotly5` to isolate Streamlit 1.63 compatibility
  from the separate Plotly 7 / MapLibre migration.
- Confirmed Python 3.12.14, Streamlit 1.63.0, Cartopy 0.26.0, Pandas 3.0.6,
  NumPy 2.5.3, Plotly 7.1.0, Matplotlib 3.11.2, SciPy 1.18.1,
  scikit-learn 1.9.1, and gsw 3.6.23.
- `pip check` reports no broken requirements.
- EnvGeo-Seawater tests: 57 passed.
- EnvGeo-Earthquake tests: 9 passed and 4 skipped.
- The Seawater Home page starts successfully with Streamlit 1.63 at
  `http://localhost:8503` and returns HTTP 200.
- The current `requirements.txt` allows Streamlit 1.42-1.63 while retaining
  Plotly 5.24 as the verified plotting baseline. A fresh test-site deployment
  resolves to Streamlit 1.63.

Remaining checks:
- Visually test representative Seawater pages, including Plotly selection,
  Cartopy mapping, file upload/download, forms, and Vertical Section tools.
- Visually test the Earthquake Simple and Advanced pages.
- The repeated Pandas future-option and Streamlit full-width deprecation
  warnings have been resolved with shared compatibility helpers.
- Follow the Plotly 5.24-to-MapLibre strategy recorded in
  `docs/streamlit_migration.md`, then verify the same code with Plotly 6.7 and
  7.1.
- Test `st.Page` / `st.navigation` separately before changing public navigation.
- After visual verification, create a reproducible migration requirements file
  and decide whether this environment becomes the new development baseline.

### Form submit button keys

Date added: 2026-09-13

Current status:
- The current local/test environment uses Streamlit 1.42.0.
- In this version, `st.form_submit_button()` does not support the `key`
  argument.
- `51_Correlation_Overview.py` therefore uses two different labels,
  `Apply settings` and `Apply settings!`, for the top and bottom submit buttons
  inside the same form.

Planned direction:
- After updating Streamlit to a version where `st.form_submit_button()` supports
  `key`, revisit the two-button form layout.
- Use matching button labels, such as `Apply settings`, with separate keys for
  the top and bottom buttons.
- Apply the same pattern to other long sidebar forms if needed.

### Managed page navigation

Date added: 2026-09-17

Planned direction:
- Test migration from automatic `pages/` directory navigation to the
  `st.Page` / `st.navigation` system during the Streamlit 1.6x migration.
- Keep short, stable source filenames while defining clearer user-facing page
  titles and icons separately.
- Organize the sidebar into groups such as interactive exploration,
  publication-ready static figures, beta tools, and development utilities.
- Register only pages intended for the public interface. Keep development and
  diagnostic files in the repository without automatically exposing them in
  the public sidebar.
- Prefer restrained, consistent Material icons or simple symbols instead of
  adding Unicode emoji directly to Python filenames.
- Confirm page URLs, `st.switch_page` / `st.page_link` references, integrated
  visualizer routing, and Streamlit Cloud behavior before adopting the new
  navigation as the default.

## User-facing UI Polish

### Reduce small points of confusion in the plotting workflow

Date added: 2026-09-14

Current status:
- Major plotting pages now use more consistent labels such as `Sampling Location
  Map`, `Map style`, `Color parameter`, and `Show background data`.
- 2D figure download buttons have been moved below the corresponding figures.
- `Data filtering` includes a short note explaining that users should click
  `Apply settings` after changing filter conditions.

Planned direction:
- Make the `Data filtering` apply workflow even clearer if users still miss it.
  A possible wording is: `Change filters, then click Apply settings to refresh
  all figures.`
- Add short help text to `Color parameter` controls where the same label has
  page-specific meaning, such as marker color in T-S plots, mapped parameter in
  map views, or color axis in 3D/4D views.
- Review whether `Map controls` popovers are discoverable enough. If needed,
  add concise help text so users know they can change the map background there.
- Keep improving the distinction between `Sidebar-filtered dataset` and
  `Box/Lasso-selected dataset` in 3D/interactive pages. A short caption may help
  first-time users understand that Box/Lasso selection is an additional
  interactive subset.
- Decide how to handle Plotly map/figure downloads. Static Matplotlib figures
  now have clear `Download image` buttons below the figures, while Plotly views
  currently rely more on the Plotly modebar camera/export behavior.
- Before public release, explain the role of `beta` pages clearly. Some beta
  pages are active research/development tools, while others may become advanced
  or private workflows.
