# Code Guide

This is a concise map of the EnvGeo-Seawater source tree for contributors and
reviewers. It explains responsibilities and boundaries; it is not an API
reference and does not replace the in-code docstrings or user manuals.

[日本語版](code_guide_Japanese.md)

For the distribution plan, see
[`distribution_foundation_audit_and_plan.md`](distribution_foundation_audit_and_plan.md).

## Core modules

| File | Responsibility | Change with care |
|---|---|---|
| `home.py` | Streamlit entry point and Home tabs: overview, references, manuals, and update history. | It is the current local/Cloud launch target. Keep `streamlit run home.py` working until a planned entry-point migration. |
| `envgeo_assets.py` | Resolves paths to read-only bundled resources from the application root, independently of the current working directory. | Use `asset_path()` for bundled resources. Do not use it for private data, user output, or future writable caches. |
| `envgeo_launcher.py` | Console entry point that starts the installed application with `envgeo-seawater`. | Keep its launch path independent of the caller's working directory. |
| `envgeo_diagnostic_launcher.py` | Console entry point for the local diagnostic tool, `envgeo-seawater-check`. | Keep diagnostics outside normal public Streamlit navigation. |
| `envgeo_utils.py` | Shared dataset loading, normalisation, quality checks, filtering, map styles, coastlines, and common exports. | It is intentionally broad legacy infrastructure. Make focused, tested changes; avoid unrelated refactoring. |
| `envgeo_user_data.py` | Session-only browser-upload workflow, column normalisation, quality feedback, and uploaded-marker helpers. | Browser uploads must not be silently written to disk or merged into public reference data. |

## Page scripts

| Page | Main purpose | Status / special boundary |
|---|---|---|
| `03_[Interactive]_2Dplus_Visualizer.py` | Interactive 2-D comparison views, including the optional approximate σ0 T–S reference-contour pilot. | Preserve Plotly Box/Lasso behaviour and the distinction between data and reference contours. |
| `04_[Interactive]_3D_4D_Visualizer.py` | Interactive 3-D/4-D visualisation. | Main Plotly workflow; HTML export is self-contained except for online map tiles. |
| `05_[Utils]_User_Data_Check_Quick_Visualizer.py` | Public upload-first quality check and simple 2-D–4-D exploration. | Keep `Uploaded marker style` before filtering and retain data-origin distinctions. |
| `31_Salinity-d18O_Relationship.py` | Salinity–δ18O relationships. | Uses shared filtering and optional uploaded overlays. |
| `32_Isotope_Hydrographic_Mapping.py` | Plotly and Cartopy mapping. | Uses bundled Natural Earth 50m land; never reintroduce Cartopy's automatic Natural Earth download. |
| `34_T-S_diagram.py` | T–S diagram and approximate σ0 reference contours. | Present contours are not pointwise density. Future TEOS-10 work is defined in the T–S review plan. |
| `35_Custom_Parameter_Plot.py` | Flexible custom-parameter plot. | Protect common upload/filter behaviour. |
| `37_Depth_Profile.py` | Depth-profile visualisation. | Uses shared filtering and uploaded overlays. |
| `53_Vertical_Section_Visualizer.py` | A–B vertical sections and optional derived GEBCO bathymetry. | Beta. Keep Manual A–B offline fallback and all scientific limitations explicit. |
| `80_Correlation_Overview.py` | Preserved correlation/exploratory workflow. | Historical hand-written archive page; make only necessary bug, compatibility, safety, or distribution fixes. |
| `90_Integrated_Visualizer_beta.py` | Development/pre-release integrated visualizer. | Outside the stable release and JOSS scope; do not expand it by default. |
| `91_EnvGeo_Earthquake.py` | Development/pre-release redirect to the specialised Earthquake application. | Do not start Earthquake packaging or move its scientific logic into Seawater before a separate audit. |
| `99_Environment_Check.py` | Local developer diagnostic-page wrapper. | Exclude it from the stable clone, wheel, and public navigation. |

## Data and resource directories

| Directory | Contents and rule |
|---|---|
| `dataset/` | Public reference workbooks. Do not copy, delete, move, or alter them without a dedicated data/provenance review. |
| `local_data/` | Tracked zero-value public `user_data.xlsx` sample and bilingual instructions. Researcher-owned measurements belong in an external file or must be restored to the zero-value sample before public sync. |
| `coastline/` | Offline coastline CSVs and Natural Earth 50m land shapefile plus provenance. |
| `bathymetry/` | Derived GEBCO grid and its source-only generation helper. Page 53 reads the grid only; the helper is excluded from wheels. |
| `data/` and `data_text/` | Demonstration media, in-app text, references, manuals, and update logs. |
| `docs/` | Development decisions, scientific plans, release records, and bilingual documentation. |
| `test/` | Pytest regression tests. Add focused tests with every behavioural change. |

## Documentation rule of thumb

- Keep a short docstring on reusable modules and public helpers: purpose,
  inputs/outputs, and important safety boundary.
- Use code comments only for non-obvious intent, scientific assumptions, or a
  compatibility constraint. Do not translate every obvious assignment into a
  comment.
- Use bilingual section headers to mark a meaningful boundary between settings,
  data structures, and related helper functions. Prefer a small number of
  role-based groups over a header for every short assignment or every `def`.
- Put longer explanations, data provenance, workflow decisions, and bilingual
  guides in `docs/`, not inside page scripts.
- Keep English and Japanese records aligned when a decision affects users,
  reproducibility, release claims, or scientific interpretation.

## Code comment and formatting convention

Apply this convention to active shared code and active stable pages. The
historical Page 80 archive is exempt except for necessary functional, safety,
or distribution fixes.

```python
# =============================================================================
# Major role-based section / 主要な役割区分
# =============================================================================

# -------------------------------------------------------------------
# Related subgroup / 関連する下位区分
# -------------------------------------------------------------------
```

- Use `=` for a module-level area such as configuration, data loading, map
  styling, or reporting; use `-` for a related subgroup or coherent stage in a
  long function.
- Place English first and Japanese second. State the role or reason, not edit
  history; avoid “fixed”, dates, and temporary-development notes.
- Do not number major sections. Short numbered steps inside one non-trivial
  procedure are acceptable when they make the sequence clearer.
- Retain explanations for scientific assumptions, missing-value/gap-row
  preservation, compatibility constraints, provenance boundaries, and
  non-obvious safety handling. Remove confirmed-unused debug output, obsolete
  commented-out code, and duplicate implementation examples.
- Keep ordinary assignments uncommented. Put lengthy rationale, decisions, and
  user-facing explanations in `docs/`.
- Separate formatting-only from behaviour changes where practical; then run
  syntax checks and focused tests appropriate to the risk.

## Before changing code

1. Read the relevant manual, update log, and plan in `docs/` or `TODO.md`.
2. Preserve `dataset/`, public/private data boundaries, and existing common
   filtering behaviour.
3. Add or update focused tests, then run the appropriate pytest suite from the
   application root.
4. Record material changes in both update logs. Codex does not commit or push;
   public-clone synchronization is reviewed in GitHub Desktop.
