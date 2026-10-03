# Streamlit 1.63 Migration Log (Historical Record)

This document records earlier local compatibility work during migration from the
verified Streamlit 1.42 environment. It is not the current stable-release
support statement; see `release_checklist.md`, `testing.md`, and the stable
repository's `stable_release_publication_notes.md` for the current release
boundary.

For v1.3.4, the package declares Python >=3.10 and CI validates the final wheel
on Python 3.10 and 3.12. The verified application baselines remain Python 3.10 /
Streamlit 1.42 / Plotly 5.24 and Python 3.12 / Streamlit 1.63 / Plotly 5.24.
Plotly 7, Pandas 3, and NumPy 2 results below are forward-looking experiments,
not supported release configurations.

## Environment Matrix

| Purpose | Conda environment | Python | Streamlit | Pandas | NumPy | Plotly |
|---|---|---:|---:|---:|---:|---:|
| Verified baseline | `envgeo_st142_py310_plotly5` | 3.10.15 | 1.42.0 | 2.3.3 | 1.26.4 | 5.24.1 |
| Earlier intermediate test | `envgeo_st155_py312_plotly6` | 3.12.9 | 1.55.0 | 2.3.3 | 2.4.4 | 6.7.0 |
| Future-stack test | `envgeo_st163_py312_plotly7` | 3.12.14 | 1.63.0 | 3.0.6 | 2.5.3 | 7.1.0 |
| Plotly compatibility test | `envgeo_st163_py312_plotly5` | 3.12.14 | 1.63.0 | 3.0.6 | 2.5.3 | 5.24.1 |

Preserve the verified baseline for reproducibility unless it is intentionally
recreated; it is not required as a separate environment for release use.

## Historical Test Results

Date: 2026-09-18

Both Streamlit 1.63 environments passed the automated tests:

- EnvGeo-Seawater: 57 passed.
- `pip check`: no broken requirements.
- The Seawater Home page started successfully and returned HTTP 200.

An AppTest initial-render smoke test was also run in
`envgeo_st163_py312_plotly5`. Home and all 13 page scripts present at that time completed without a
Streamlit exception. This confirms basic page startup, but it does not exercise
sidebar submissions, Plotly selections, downloads, uploads, or every plotting
branch. Errors seen after user interaction must therefore be recorded and
tested separately.

Compatibility helpers now select `use_container_width=True` for the Streamlit
1.42 API and `width="stretch"` for the Streamlit 1.63 API. Pandas future options
are enabled only before Pandas 3, where they are still needed. This removes the
repeated `use_container_width`, `copy_on_write`, and
`future.no_silent_downcasting` migration warnings without dropping Streamlit
1.42 compatibility. The Seawater test suite passed in both environments after
the change, and the 1.63 initial-render smoke test remained clean across all
pages.

## Plotly Finding

Interactive rendering errors were observed with Plotly 7.1. Plotly 7 removed
the Mapbox-based APIs currently used across several EnvGeo pages:

- `px.scatter_mapbox` was replaced by `px.scatter_map`.
- `go.Scattermapbox` was replaced by `go.Scattermap`.
- `layout.mapbox` was replaced by `layout.map`.
- `mapbox_style` was replaced by `map_style`.

The current application uses the former APIs in mapping, profile, integrated,
and interactive pages. Therefore, these errors should not be treated as proof
of a Streamlit 1.63 incompatibility.

`streamlit-plotly-events==0.0.6` must be evaluated separately because it is used
for Box/Lasso-linked selection in the Interactive 2D/2.5D Visualizer.

## Historical Migration Decisions And Deferred Work

- Do not make page-by-page Plotly 7 fixes during the first Streamlit migration
  check.
- Use `envgeo_st163_py312_plotly5` to test Streamlit 1.63 while retaining the
  existing Plotly 5 behavior.
- Keep `envgeo_st163_py312_plotly7` as a future-stack test environment for Plotly 7,
  Pandas 3, and NumPy 2 compatibility.
- For the then-planned 1.3.2 test site, allow Streamlit 1.42-1.63 in `requirements.txt`; a
  fresh deployment resolves to Streamlit 1.63 while the 1.42 baseline remains
  available for local regression checks.
- Keep Plotly 5.24 as the current release baseline while migration tests are in
  progress.
- Use the MapLibre APIs introduced in Plotly 5.24 (`scatter_map`, `Scattermap`,
  `layout.map`, and `map_style`) before changing the runtime to Plotly 7.
- Test the same MapLibre-based code with Plotly 5.24, 6.7, and 7.1. If the
  checks pass, target one implementation across Plotly `>=5.24,<8` instead of
  maintaining version-specific map code.
- Treat the MapLibre conversion as a dedicated migration with visual checks for
  map center, zoom, extent, tiles, attribution, colorbars, overlays, selection,
  and exports.
- Evaluate `streamlit-plotly-events` separately and consider replacing it with
  native `st.plotly_chart` selection events where practical.
- Add a Python 3.10 / Streamlit 1.63 / Plotly 7 cross-environment test before
  claiming that every supported-version combination is verified.

## Superseded Planned Version Sequence

- At the time of this record, EnvGeo-Seawater 1.3.2 was planned for Python 3.10-3.12 and Streamlit 1.42-1.63
  compatibility consolidation, including the Streamlit 1.63 tab-DOM update,
  while retaining Plotly 5.24 as the verified baseline.
- A later minor release: MapLibre-based maps verified across Plotly 5.24, 6.7,
  and 7.1.

## Historical Local Comparison

- Plotly 7.1 future-stack test: `http://localhost:8503`
- Plotly 5.24 compatibility test: `http://localhost:8504`

Start the compatibility environment with:

```bash
conda activate envgeo_st163_py312_plotly5
cd /path/to/envgeo_seawater_v130
python -m streamlit run home.py --server.port 8504
```

Use `python -m streamlit` instead of the bare `streamlit` command so the command
cannot accidentally resolve to the Homebrew installation.

## Historical Remaining Visual Checks

These items were a migration-era checklist, not the current stable-release
acceptance list. Current manual checks are maintained in `release_checklist.md`.

- Home and Environment Check
- Interactive 2D/2.5D: scatter rendering and Box/Lasso selection
- Interactive 3D/4D: 3D figures, map-depth views, and colorbars
- Salinity-d18O, T-S, Mapping, and Depth Profile maps
- Vertical Section Plotly and Folium views
- Integrated Visualizer upload, quality check, and Plotly views
- Image/data downloads and sidebar forms
