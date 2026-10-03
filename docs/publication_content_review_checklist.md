# Publication Content Review Checklist

[日本語版](publication_content_review_checklist_Japanese.md)

## Purpose and status

Use this checklist for the deliberate, file-by-file review before a stable
release, GitHub Release, Zenodo archive, or JOSS submission. It is a content
and readability checklist; build, installation, CI, and release actions remain
in `release_checklist.md`.

- `[x]` content review completed for this file or group
- `[ ]` not yet reviewed in this pass
- “Stable only” means `seawater_map`; “development only” means the canonical
  working copy and/or `envgeo-seawater`, not the stable release repository.

For every reviewed item, check: scientific and user-facing wording; current
version and scope; concise bilingual comments where helpful; absence of local
paths, personal data, tokens, and internal conversation text; and agreement
between the canonical working copy and the stable repository when the item is
shared.

## 1. Shared Python modules

- [x] `envgeo_utils.py` — section headings, comments, obsolete commented-out
  code, and configuration readability reviewed. Functional refactoring is
  deferred; final regression evidence remains a separate release task.
- [x] `envgeo_assets.py` — role, public-path safety boundary, docstrings, and
  section formatting reviewed; canonical asset-path tests passed.
- [x] `envgeo_user_data.py` — session-only upload boundary, column controls,
  hover-text bounds, map-overlay formatting, and focused tests reviewed (4 passed).
- [x] `envgeo_launcher.py` — entry-script resolution, installed-launcher import
  boundary, bilingual section formatting, and packaging proof reviewed (7 passed).
- [x] `envgeo_diagnostic_launcher.py` — explicit local diagnostic entry point,
  package command definition, public-navigation separation, and packaging proof reviewed (7 passed).
- [x] `__init__.py` — package metadata has no import side effects; version 1.3.4
  matches `pyproject.toml` and the focused version/import tests passed.
- [x] `home.py` — public wording, version display, bundled-document paths,
  external-reference labels, safe load failures, and focused public-surface tests reviewed (4 passed).
- [x] `bathymetry/make_lightweight_gebco.py` — source-only derivation record
  for the bundled GEBCO grid; it has no runtime reference and is deliberately
  excluded from wheels.

## 2. Page scripts

### Stable release pages (`seawater_map`)

- [x] `pages/03_[Interactive]_2Dplus_Visualizer.py` — imports, bilingual
  section/comment format, historical commented code, T–S σ0-reference notes,
  linked-map controls, and the focused contour test reviewed (32 passed).
- [x] `pages/04_[Interactive]_3D_4D_Visualizer.py` — imports, bilingual
  section/comment format, Fig.1–Fig.6 and custom-view structure, map-depth
  controls, historical commented code, and focused tests reviewed.
- [x] `pages/05_User_Data_Check_Quick_Visualizer.py` — session-only upload
  boundary, internal-origin exclusion, 2D/3D/map helpers, bilingual
  docstrings/section format, and focused tests reviewed (7 + 5 passed).
- [x] `pages/31_Salinity-d18O_Relationship.py` — imports, bilingual
  section/comment format, Salinity–δ18O regression and uploaded-overlay
  boundaries, image-download handling, sampling map, and focused tests
  reviewed (3 + 52 passed).
- [x] `pages/32_Isotope_Hydrographic_Mapping.py` — imports, bilingual
  section/comment format, bundled offline land-mask handling, scatter/contour
  map safeguards, uploaded overlays, and focused non-Cartopy tests reviewed
  (106 passed; 4 Cartopy-dependent overlay tests skipped locally).
- [x] `pages/34_T-S_diagram.py` — imports, bilingual section/comment format,
  approximate-σ0 reference contours, uploaded overlays, image-download
  handling, sampling map, and focused tests reviewed (37 + 11 + 52 passed).
- [x] `pages/35_Custom_Parameter_Plot.py` — numeric-parameter helpers,
  bilingual section/comment format, uploaded overlays, regression boundary,
  image/table export, and focused tests reviewed (6 + 52 passed).
- [x] `pages/37_Depth_Profile.py` — imports, bilingual section/comment format,
  depth-direction and gap-row handling, month-band and uploaded overlays,
  image/table export, sampling map, and focused tests reviewed
  (4 + 52 passed).
- [x] `pages/53_Vertical_Section_Visualizer.py` — imports, bilingual
  section/comment format, A–B and axis-based section safeguards, GEBCO and
  uploaded bathymetry fallbacks, interpolation confidence overlay, offline
  map fallback, uploaded overlays, and focused tests reviewed
  (51 + 7 + 52 passed).
- [x] `pages/80_Correlation_Overview.py` — archive policy confirmed: the
  historical hand-written workflow is retained; runtime data/media paths are
  package-safe, active debug output and sensitive paths are absent, and only a
  bilingual archive-status caption was added.

### Development-only pages and diagnostics

- [x] `pages/90_Integrated_Visualizer_beta.py` — reviewed in the development
  repository only; stable-release/JOSS exclusion, historical wording,
  in-memory uploads, fixed embedded-page registry, and focused tests verified
  (7 + 4 passed).
- [x] `pages/91_EnvGeo_Earthquake.py` — development-only redirect reviewed;
  it has no Seawater code/data dependency and remains outside the Sprint 3
  package and stable release pending a separate Earthquake audit (4 passed).
- [x] `pages/99_Environment_Check.py` — retained only as a local-development
  wrapper; excluded from the stable clone, public navigation, wheel, release,
  and archive. Diagnostics are provided by `tools/env_check_streamlit.py` and
  `envgeo-seawater-check` (11 passed).

## 3. Tests, tools, and package configuration

- [x] `test/conftest.py` and `test/README.md` / `test/README_Japanese.md` — reviewed; the import-path rationale and test/CI scope are current and bilingual.
- [x] `test_basic.py` and `test_data_integrity.py` — reviewed; duplicated path setup was removed in favour of `conftest.py`, and current v1.3.4 metadata plus public-dataset integrity checks passed (8 passed).
- [x] `test_envgeo_assets.py` and `test_envgeo_user_data.py` — reviewed; duplicated path setup was removed, bilingual scope comments were clarified, and asset-security plus session-only upload checks passed (29 passed).
- [x] `test_envgeo_utils.py` — reviewed; duplicated path setup and unused imports were removed, scope is bilingual, and shared data, map, upload, coastline, and export regressions passed (55 passed).
- [x] `test_natural_earth_land.py` and `test_offline_map.py` — reviewed; bundled land assets, no-download fallback, and offline-map behaviour are documented bilingually. Local Cartopy-dependent checks skip only when Cartopy is absent; all remaining checks passed (67 passed, 10 skipped).
- [x] `test_p03_ts_density_contour.py`, `test_ts_density_contour.py`, and `test_packaging_proof.py` — reviewed; static checks preserve approximate-σ0 wording and GSW input bounds, while package checks verify release data and development-page exclusions (76 passed).
- [x] `test_public_surface.py`, `test_quick_visualizer.py`, and `test_self_contained_html.py` — reviewed; public page inventory, session-only upload origin labels, and CDN-free Plotly HTML export passed (40 passed).
- [x] `test_repository_health.py`, `test_uploaded_page_overlays.py`, and `test_vertical_section_visualizer.py` — reviewed; source integrity, session-only upload overlays, and Vertical Section safeguards passed. The all-NA-column concat warning was fixed and protected by a regression test (canonical 115 passed / 4 skipped; stable 106 passed / 13 skipped; overlay rerun warning-free).
- [ ] Remaining test modules:
  none.
- [x] `tools/env_check_streamlit.py` — local-only runtime/package diagnostic;
  explicit local-path and package-inventory output is documented as excluded
  from shared records. It performs no external communication or persistent
  write, and its `pip list` action is read-only (7 + AppTest passed).
- [x] `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, and
  `runtime.txt` — Python >=3.10 declaration, Python 3.10/3.12 CI coverage,
  Cloud Python 3.12 runtime, version 1.3.4, dependency source of truth,
  package data, and console commands reviewed (26 focused tests passed).
- [x] `.github/workflows/ci.yml` — Python 3.10/3.12 matrix, local-user-data
  isolation, temporary dependency caches, network-independent test suite,
  wheel build, isolated install, public-page checks, and artifacts reviewed
  (113 focused tests passed; both GitHub CI jobs previously passed).
- [x] `.gitignore`, `LICENSE`, and `CONTRIBUTING.md` /
  `CONTRIBUTING_Japanese.md` — generated, secret, private-data, diagnostic,
  internal-record, and screenshot exclusions; MIT scope; and bilingual safe
  contribution guidance reviewed (30 focused tests passed).

## 4. User-facing root and data text

- [x] `README.md` / `README_Japanese.md` — stable-page scope, approximately
  50,000 cited records, user-data boundaries, bilingual public wording, and
  concise data/software/geospatial acknowledgements reviewed.
- [x] `TODO.md` / `TODO_Japanese.md` — separated current v1.3.4 release work
  from post-release development, marked older review/JOSS notes as historical,
  retained the EnvGeo Data plan, and removed obsolete current-version claims.
- [x] `local_data/README.md` / `local_data/README_Japanese.md` — zero-row
  public sample, external-path option, session-only browser uploads, and the
  no-private-data release boundary reviewed.
- [x] `data_text/about.md`, `japanese.md`, `manual.md`, `manual_Japanese.md`,
  `main_references.md`, `other_references.md`, `NASA_references.md`, and
  `CoralHydro2_references.md` — Home text, bilingual guidance, cited-source
  labels, approximately 50,000-record scope, and user-data boundaries reviewed.
- [x] `data_text/update_log.md` / `update_log_Japanese.md` — added a concise
  current v1.3.4 release-candidate summary, retained older work as explicitly
  historical preparation notes, and checked English/Japanese correspondence.

## 5. User manuals

- [x] `docs/manual/README.md` / `docs/manual_Japanese/README.md` — stable
  public scope, navigation, and historical-page boundary reviewed.
- [x] Overview and filtering: `00_overview.md`, `01_data_filtering.md`.
- [x] Page manuals: `03_2dplus_visualizer.md`, `04_3d_4d_visualizer.md`,
  `05_user_data_check_quick_visualizer.md`, `31_salinity_d18o.md`,
  `32_isotope_hydrographic_mapping.md`, `34_ts_diagram.md`, `35_custom_parameter_plot.md`,
  `37_depth_profile.md`, and `53_vertical_section.md`, each with its matching
  Japanese file — implementation boundaries and user-facing wording reviewed.
- [x] `90_integrated_visualizer.md` and Japanese counterpart — retained as
  development history; stable-release and static-website exclusion made unambiguous.

## 6. Technical, provenance, and historical documents

- [x] Documentation index: `docs/README.md` / `docs/README_Japanese.md` —
  English/Japanese links, listed documents, development-history wording,
  stable-release scope, and absence of stale duplicate entries reviewed
  (23 focused tests passed).
- [x] Release readiness: `release_checklist*` and
  `stable_release_publication_notes*` (stable only) — release repository,
  tag/Zenodo sequence, page scope, data boundary, CI-artifact boundary, and
  English/Japanese links reviewed (30 focused tests passed).
- [x] `testing.md` / `testing_Japanese.md` — CI matrix, development/stable
  wheel scope, no-runtime-network boundary, manual-QA boundary, Page 90 scope,
  and English/Japanese links reviewed (30 focused tests passed).
- [x] `code_guide.md` / `code_guide_Japanese.md` — source/module and page
  inventory, stable/development boundary, Page 80 archive policy, Page 90/91/99
  scope, documentation convention, and English/Japanese links reviewed
  (23 focused tests passed).
- [ ] Remaining review document: this checklist.
- [x] `citation_and_license_plan.md` / `citation_and_license_plan_Japanese.md`
  — current scholarly-use dataset scope, project-versus-third-party ownership,
  staged `CITATION.cff`/Zenodo DOI sequence, Natural Earth attribution policy,
  current GEBCO terms, concise README acknowledgements, and English/Japanese
  links reviewed.
- [x] `provenance_inventory*` / `dataset_redistribution_audit*` — five bundled
  workbooks, zero-value local sample, canonical/stable checksum identity,
  package-data scope, source records, and outstanding non-wheel asset review
  boundary verified in English and Japanese.
- [x] `external_dataset_workbook_notes*` — NASA GISS and CoralHydro2k source
  mapping records, current bundled workbook structure, citations, and the
  non-observational transformation boundary reviewed in English and Japanese.
- [x] `geospatial_assets*` — current wheel asset scope, Natural Earth and
  GEBCO boundaries, canonical/stable asset identity, and the outstanding
  coastline-CSV source-version record reviewed in English and Japanese.
- [x] `sprint3_distribution_design_and_acceptance*` — current 1.3.4,
  Python 3.10/3.12 CI, 12-page development scope, manual-QA boundary, and
  Release-not-approval status reviewed in English and Japanese.
- [x] `distribution_foundation_audit_and_plan*` / `wheel_proof_report*` —
  historical Sprint 2 audit and proof evidence, its distinction from current
  1.3.4 verification, and English/Japanese links reviewed.
- [x] `offline_operation_log*` — historical v1.3.3 implementation notes are
  separated from the v1.3.4 release basis; offline fallback, HTML export, and
  tile-provider recheck boundary reviewed.
- [x] `streamlit_migration*` — historical results and future migration work
  are separated from the current v1.3.4 Python 3.10/3.12 release basis.
- [x] `ts_density_contour_review_and_plan*` — v1.3.4 approximate-σ0
  transparency work is distinguished from future SA–CT/TEOS-10 development.
- [x] `integrated_visualizer_strategy*` — Page 90 migration history, current
  v1.3.4 public scope, and deferred shared-core ideas are explicitly separated.
- [x] `development_notes*`: retained as historical records; current v1.3.4
  release state, completed packaging, Page 90 exclusion, and deferred
  shared-core scope are explicit.
- [x] `paper.md` / `paper.bib` — synchronized canonical manuscript pair with
  the stable repository; reviewed v1.3.4 release-candidate status, repository
  URL, citations, AI disclosure, approximate-σ0 wording, and the explicit
  absence of a DOI claim. The superseded `paper_revised*` files are not used.

## 7. Data, assets, and public-surface audit

- [x] Every `dataset/*.xlsx` — no workbook contents changed; citations,
  DOI/source records, transformation notes, current row/column structures, and
  canonical/stable SHA-256 identity were verified. Historical raw acquisition
  details are retained where available and explicitly recorded as unavailable
  where they were not preserved.
- [x] `coastline/`, including Natural Earth attribution, the bundled CSVs,
  and derived CSV source-version record — Natural Earth v4.1.0 intermediates,
  numerical correspondence, and current output checksums were verified.
- [x] `data/`, `bathymetry/`, and public images/GIFs — confirmed the required
  Home animation, project map GIFs, GEBCO grid/generator record, and package
  references. No personal or unpublished data were identified; unused legacy
  spreadsheets were moved outside the application and public-release trees.
- [x] `local_data/user_data.xlsx` — verified as the intended zero-row,
  22-column public sample with no researcher measurements. It was regenerated
  without a local absolute-path metadata field while retaining its headers and
  visible template formatting.
- [x] Confirmed the stable clone excludes `Claude outputs/`, screenshots,
  diagnostic-only Page 99, internal review logs, and legacy files; `.DS_Store`,
  caches, build products, virtual environments, and temporary files are
  ignored rather than tracked. A 2026-10-03 local wheel inspection contained
  the allowed public assets and no excluded path class. The final GitHub
  Release/Zenodo archive must still be built from the reviewed release commit.

## 8. Completion record

Before marking this checklist complete, record the reviewed commit ID, reviewer,
date, Python version, CI run URLs, wheel SHA-256, and the final release version
in the release checklist or release notes. Do not mark an item complete merely
because it was copied between directories.
