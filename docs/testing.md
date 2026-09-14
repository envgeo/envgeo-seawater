# Testing Guide

This document explains the current pytest suite for EnvGeo-Seawater, what it checks, and what still needs to be expanded as the software matures.

## How To Run Tests

Run the full test suite from the project root:

```bash
pytest
```

For a quick check during development, run the core utility and repository-health tests:

```bash
pytest -q test/test_envgeo_utils.py test/test_repository_health.py
```

## Test Files

### `test/test_basic.py`

Basic import and version checks.

This file confirms that:

- `envgeo_utils` can be imported.
- The app version value is available.

### `test/test_envgeo_utils.py`

Core utility tests for reusable functions in `envgeo_utils.py`.

This file currently checks:

- App version metadata.
- Quality-flag rules and compact quality criteria text.
- Safe figure filename generation.
- Shared map-region presets and Data filtering area-preset bounds.
- Dataset loading for Japan Sea, Around Japan, and Global data choices.
- Required dataset columns and numeric column conversion.
- Gap-row insertion for grouped depth-profile plotting.
- Shared EnvGeo and cmocean colormap helpers.
- Uploaded-data column-name standardization.
- Coastline coordinate loading.
- d-excess calculation and missing-value handling.
- Filtered-data summary statistics and CSV report generation.

### `test/test_data_integrity.py`

Data integrity and broad sanity checks for public datasets.

This file currently checks:

- All public data-source choices load as non-empty dataframes.
- Required columns exist in loaded datasets.
- Placeholder strings such as `**` are removed from numeric columns.
- Geographic and hydrographic values remain within broad physical sanity ranges.
- Known invalid values are converted to `NaN` while original values remain visible through quality information.
- Invalid data-source names fail safely by returning an empty dataframe.

### `test/test_repository_health.py`

Repository-level tests that protect the public app structure and common workflows.

This file currently checks:

- Streamlit page files and support tools compile as valid Python.
- README image links point to existing files.
- Key project documents exist.
- 4D Visualizer selected-data tables include quality information.
- Integrated Visualizer beta keeps upload-overlay support.
- Uploaded-data overlays use WebGL-compatible traces where needed.
- Integrated map views use shared ocean-region presets.
- Filtered-data summary CSV export is available.
- Quality-flag criteria are shown near relevant tables.
- Shared-filter beta tabs use readable compact labels.
- Standard map style avoids CARTO tiles that require API keys.
- Mapping pages use shared colormap helpers.
- The standalone uploader is excluded from Integrated Visualizer full-page workflows.
- Uploaded user files are handled in memory during the Streamlit session.

### `test/test_public_surface.py`

Public-facing surface and source-tree cleanup tests.

This file currently checks:

- README and app update text do not include internal submission-status wording.
- The `pages/` directory contains only stable pages or explicitly named beta/local-development pages.
- Retired pages are not kept in the active public source tree.

## Current Scope

The current pytest suite focuses on:

- Importability and version metadata.
- Dataset loading and numeric conversion.
- Data-quality rules.
- Common utility functions.
- Shared plotting support such as colormaps, filenames, map presets, and coastline loading.
- Repository structure and public-facing page hygiene.
- Integrated beta upload workflow checks.

These tests are intended to catch common breakage during refactoring and release preparation.

## Current Limitations

The current suite does not yet fully automate:

- Streamlit browser interactions.
- Visual regression testing of figures.
- End-to-end uploaded-file workflows through the Streamlit UI.
- Plotly selection workflows such as Box/Lasso selection.
- Scientific validation of every plotted relationship.
- Performance checks for large global selections.

For now, visual inspection and manual smoke tests remain important. The release checklist in `docs/release_checklist.md` records those manual checks.

## Planned Test Expansion

Priority areas for future test expansion:

- User-data upload utilities after they are moved into shared `envgeo_utils.py` functions.
- More detailed quality-flag tests, including exported quality reports.
- d-excess behavior across uploaded and reference datasets.
- Smoke tests for representative page workflows.
- Figure-generation tests for Matplotlib outputs.
- Additional tests for Area filter preset behavior in common sidebar filtering.
- Tests for public/private page inclusion before deployment.
