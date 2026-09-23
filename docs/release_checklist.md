# EnvGeo-Seawater Release Checklist

Use this checklist before uploading a test site, updating GitHub, creating a release, or archiving a version with Zenodo.

## 1. Local Environment

### Verified for the 1.3.3 release candidate (2026-09-23)

- [x] Python 3.12 / Streamlit 1.63 / Plotly 5.24 environment used for the
  final test run.
- [x] `python -m pytest -q test` completed: **302 passed, 1 warning**.
  The warning is Pytest's future deprecation notice for a class-scoped fixture;
  it is not an application test failure and remains a maintenance follow-up.

- [ ] Confirm the intended Python environment is active.
- [ ] Confirm one of the tested baselines is active: Python 3.10.15 / Streamlit 1.42 or Python 3.12.14 / Streamlit 1.63, both with Plotly 5.24.
- [ ] Confirm `requirements.txt` matches the tested environment.
- [ ] Run the local environment checker if needed:

```bash
streamlit run tools/env_check_streamlit.py
```

- [ ] Export the environment report as CSV or PDF when useful for release records.

## 2. Basic App Startup

- [ ] Start the app locally:

```bash
streamlit run home.py
```

- [ ] Confirm `home.py` opens successfully.
- [ ] Confirm the app version is shown as `1.3.3 (2026-09-23)`.
- [ ] Confirm the Home tabs load: Main, About, Data Sources, Manual, Update History, and Japanese information where applicable.

## 3. Sidebar And Filtering

- [ ] Confirm `Data filtering` appears in the sidebar.
- [ ] Confirm the note about clicking `Apply settings` is visible.
- [ ] Confirm `Area filter preset` changes the initial Longitude / Latitude range.
- [ ] Confirm manual Longitude / Latitude slider adjustment still works after choosing an area preset.
- [ ] Confirm `Apply settings` updates figures after changing filters.
- [ ] Confirm `Details and statistics of filtered data` opens and exports CSV correctly.

## 4. Main Visualization Pages

Open each page and perform a light visual check.

- [ ] `03_[Interactive]_2Dplus_Visualizer.py`
- [ ] `04_[Interactive]_3D_4D_Visualizer.py`
- [ ] `31_Salinity-d18O_Relationship.py`
- [ ] `32_Isotope_Hydrographic_Mapping.py`
- [ ] `34_T-S_diagram.py`
- [ ] `35_Custom_Parameter_Plot_beta.py`
- [ ] `37_Depth_Profile.py`
- [ ] `80_Correlation_Overview.py`
- [ ] `53_Vertical_Section_Visualizer.py`
- [ ] `90_Integrated_Visualizer_beta.py`

For each page:

- [ ] Data source selection works.
- [ ] Filtering works after `Apply settings`.
- [ ] Main figures render without errors.
- [ ] Captions, help buttons, and labels are understandable.
- [ ] `Sampling Location Map` renders correctly where present.
- [ ] `Map controls` and `Map style` work where present.
- [ ] No obvious text overlap or layout break appears.

## 5. Figure Export

- [ ] Confirm 2D Matplotlib figures show `Download image` below the figure.
- [ ] Confirm downloaded PNG files open correctly.
- [ ] Confirm figure titles fit within downloaded images, especially Depth Profile.
- [ ] Confirm file names are reasonable and do not contain unsafe characters.
- [ ] Decide whether Plotly figures and maps should rely on the Plotly modebar camera/export behavior for this release.

## 6. User Data Upload

- [ ] Confirm an optional `local_data/user_data.xlsx` is labeled `User Excel data`
      and appended once to each reference source without entering browser-upload
      session state.
- [ ] Confirm a deployment without a local user table loads bundled reference
      data normally.
- [ ] Confirm the integrated beta upload workflow still works.
- [ ] Confirm uploaded data are visually distinguishable from reference data.
- [ ] Confirm uploaded data remain session-only and are not saved to disk or server storage.
- [ ] Confirm common column aliases are standardized where supported.
- [ ] Confirm uploaded-data quality summaries are visible where supported.
- [ ] Confirm the User Data Check & Quick Visualizer accepts CSV/XLSX uploads and keeps them session-only.

## 7. Local-Only And Development Pages

- [ ] Decide whether `pages/99_Environment_Check.py` should be included in the current deployment.
- [x] `pages/05_User_Data_Check_Quick_Visualizer.py` is the public User Data Check & Quick Visualizer for upload-first quality review and 2D/3D/4D exploration.
- [ ] Decide whether beta pages should be shown publicly, hidden, or documented as experimental.
- [ ] Confirm no private notes, restricted data, or unpublished datasets are included in public deployment files.

## 8. Tests

- [ ] Run the core test suite:

```bash
pytest -q test/test_envgeo_utils.py test/test_repository_health.py
```

- [ ] Run the broader test suite when preparing a GitHub release:

```bash
pytest
```

- [ ] Review any skipped or expected-failing tests.
- [ ] If a test fails because of an intentional page-list change, update the test or document the reason.

## 9. Documentation

- [ ] Confirm `README.md` reflects the current public-facing state.
- [ ] Confirm `README_Japanese.md` reflects the current public-facing state.
- [ ] Confirm `data_text/update_log.md` includes the latest unreleased changes.
- [ ] Confirm `data_text/update_log_Japanese.md` includes the latest unreleased changes.
- [ ] Confirm beta and local-development pages are clearly described.
- [ ] Confirm citation and data-source guidance are understandable.

## 10. GitHub Release Preparation

- [ ] Confirm the repository destination and branch.
- [ ] Confirm no temporary files, private files, downloaded reports, or local cache files are staged.
- [ ] Review changed files before committing.
- [ ] Use a clear commit message describing the release-preparation scope.
- [ ] Tag the release only after local checks and public test deployment checks pass.

## 11. Streamlit Deployment

- [ ] Confirm the deployment repository includes required files:
  - `home.py`
  - `envgeo_utils.py`
  - `pages/`
  - `dataset/`
  - `data/`
  - `data_text/`
  - `coastline/`
  - `requirements.txt`
  - `runtime.txt` if used by the deployment platform.
- [ ] Confirm excluded local-only pages are actually excluded if needed.
- [ ] Open the deployed app and repeat a short smoke test:
  - Home opens.
  - T-S page opens.
  - Mapping page opens.
  - Depth Profile opens.
  - Integrated Visualizer beta opens if included.

## 12. Zenodo / DOI Preparation

- [ ] Confirm the GitHub release is final before creating the Zenodo archive.
- [ ] Confirm title, authors, affiliations, license, and description.
- [ ] Confirm the archived version matches the release tag.
- [ ] Record the DOI in the README and citation files after the archive is created.

## 13. JOSS-Oriented Follow-Up

- [ ] Prepare a separate `docs/joss_checklist.md` before resubmission.
- [ ] Confirm pytest coverage is meaningful and not only superficial.
- [ ] Confirm examples and user documentation are sufficient for reviewers.
- [ ] Confirm installation instructions are reproducible on a clean environment.
- [ ] Confirm citation instructions include both EnvGeo-Seawater and original data providers.
