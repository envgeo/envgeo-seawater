# Distribution Foundation Audit and Staged Plan (Sprint 2)

“Sprint 2” is the project's internal name for the foundation-audit and local
wheel-proof phase. The audit findings below describe the state at the time of
the 2026-09-25 audit; they are retained as historical context, not as the
current release configuration.

**Status:** historical Sprint 2A design record. It did not authorize packaging
implementation, directory moves, or dataset changes at the time.  
**Audited:** 2026-09-25  
**Sprint 2A decision confirmed:** 2026-09-25

**Progress:** The local Sprint 2B–2C proof passed on Python 3.12; see
`wheel_proof_report.md`. Its findings informed the implemented Sprint 3
configuration. The current 1.3.4 distribution and CI verification record is
[`sprint3_distribution_design_and_acceptance.md`](sprint3_distribution_design_and_acceptance.md);
formal release approval remains separate.

## Decision summary

Stabilize distribution of EnvGeo-Seawater as a standalone application before
extracting an `envgeo-utils` shared core or integrating infrastructure with the
Earthquake application.

Sprint 1 already introduced `envgeo_assets.py` and migrated the main bundled
asset families to CWD-independent source-checkout paths. Sprint 2 must now
answer a narrower question: **can the current Streamlit layout be installed
from a wheel and launched outside the source checkout without first moving the
application into `src/`?**

The first wheel is a local technical proof only. It is not a release artifact,
does not establish the final package layout or dependency policy, and must not
be advertised as a supported `pip install` workflow.

## Audit findings at the time (2026-09-25)

| Area | Current state | Sprint 2 consequence |
|---|---|---|
| Application shape | `home.py` is the only application entry script. Streamlit discovers the sibling `pages/` directory. | Keep this physical relationship unchanged during the proof. |
| Asset resolver | `envgeo_assets.asset_path()` resolves from the module location, rejects absolute/traversal inputs, and is independent of the launch CWD. | Preserve this API during the wheel proof; do not convert it wholesale to `importlib.resources`. |
| Migrated resources | The five `dataset/*.xlsx` files, coastline CSV, Natural Earth land files, GEBCO grid, local user-data sample, Home text/media, and representative page media use `asset_path()`. | Test these installed paths explicitly instead of assuming packaging compatibility. |
| Remaining file-relative logic | A small amount of `Path(__file__)` logic remains for local module/page relationships and the local-only environment check. | Inventory it during the proof; do not treat every `Path(__file__)` use as an asset-loader defect. |
| Packaging | No `pyproject.toml`, package declaration, package-data manifest, or installed launcher exists. | Sprint 2B may add only the minimum experimental build configuration required for a local wheel. |
| Dependencies and Cloud | `requirements.txt` and `runtime.txt` are the working source-checkout/Streamlit Cloud configuration. | Do not change the Cloud dependency setup or declare a new source of truth in Sprint 2A–2C. |
| Write locations | Streamlit in-memory/data caches are used; there is no EnvGeo OS cache directory. | Confirm that the installed app does not write beside package files. Design a persistent cache only in Sprint 2D. |
| Bundled geospatial assets | Coastline CSVs (~2 MB), Natural Earth 50m land files (~1 MB), and the derived GEBCO grid (~12–13 MB) are included. | Keep GEBCO bundled for the initial proof. Do not add a downloader or describe it as a Python optional extra. |
| Local user data | `local_data/user_data.xlsx` is the tracked, zero-value public sample. Private data may be supplied by `ENVGEO_LOCAL_USER_DATA_PATH`. | Include only the public sample; never copy private user data into a wheel or cache. |
| Public-only boundary | `pages/99_Environment_Check.py` and internal review records are local-development material excluded from the public clone. | They must also be excluded from the wheel proof inventory. |
| Provenance | Natural Earth and GEBCO terms are recorded. Dataset provenance and redistribution records remain active release material. | A local proof may test the current files, but it does not authorize a public full-data wheel or release archive. |

## Fixed boundaries for Sprint 2

### Preserve

- `home.py` and the sibling `pages/` directory.
- Existing UI, uploads, common filters, and page names/order.
- The current `asset_path()` behaviour and source-checkout launch command.
- The zero-value public `local_data/user_data.xlsx` sample.
- The bundled GEBCO grid for the initial proof.
- Current Streamlit Cloud dependency files and deployment configuration.

### Do not change yet

- Do not move code into `src/` or a new application package directory.
- Do not extract or publish an `envgeo-utils` shared package.
- Do not integrate Seawater and Earthquake infrastructure.
- Do not replace `asset_path()` with a resource API in one operation.
- Do not add OS cache, remote asset download, or explicit offline-mode logic.
- Do not edit, delete, transform, or regenerate anything under `dataset/`.
- Do not make GEBCO an external download or claim that an optional dependency
  extra can selectively install package data.
- Do not change which dependency file Streamlit Community Cloud uses.
- Do not publish, commit, push, or document the proof wheel as a release.

## Staged plan

| Stage | Purpose | Allowed change |
|---|---|---|
| **2A** | Correct the design record and define proof acceptance gates. | Documentation only. |
| **2B** | Build a minimal wheel while preserving the current application layout. | Minimum experimental build metadata, package-data rules, launcher/proof support, and focused tests. |
| **2C** | Install and verify the wheel outside the checkout. | Test environments and test/report adjustments; no layout migration. |
| **2D** | Use the proof results to design resource contexts, cache policy, and explicit offline behaviour. | Design review first; implementation requires separate approval. |
| **3** | Introduce the agreed production packaging, launcher, CI, and Cloud arrangement. | Implementation after the Sprint 2 review. |
| **Later, only if needed** | Reconsider a package-directory or `src/` migration. | Separate migration design and regression plan. |

## Sprint 2B–2C wheel proof

### Purpose

Determine whether an installed copy can retain the current Streamlit entry
script/pages relationship and make the required files available without a
large source-tree refactor. The proof should also reveal which consumers need
a stable filesystem `Path` and which could later use a resource stream or
context manager.

### Required checks

1. Build a wheel from a clean copy of the current source layout.
2. Inspect the wheel contents against an explicit allowlist and exclusion list.
3. Install it into a fresh virtual environment outside the repository.
4. Launch the installed Streamlit app while the process CWD is outside the
   checkout and without importing modules from the checkout.
5. Confirm that Streamlit discovers the expected public pages.
6. Read representative installed assets from every required family:
   `dataset/`, coastline CSV, the complete Natural Earth shapefile set,
   `local_data/user_data.xlsx`, Home text/media, and GEBCO.
7. Confirm that normal startup and reads do not write into `site-packages`.
8. Confirm that `pages/99_Environment_Check.py`, caches, private data, local
   outputs, `.DS_Store`, and internal review records are absent.
9. Re-run the relevant asset tests and the full supported test suite in the
   source checkout after the proof configuration is added.
10. Record manual checks separately; do not convert a failed UI observation
    into a weakened automated test.

### Acceptance gates

The proof passes only if all of the following are true:

- The wheel builds and installs reproducibly in a fresh environment.
- The installed launcher starts the intended `home.py` without relying on the
  source checkout or its CWD.
- The expected public `pages/` files are found; no local-only page appears.
- All representative bundled assets above can be read from the installation.
- Source-checkout execution still works with `streamlit run home.py`.
- Existing UI, upload behaviour, common filtering, and scientific data content
  are unchanged.
- No runtime download is introduced for bundled assets, including GEBCO.
- No private user data or generated state is included or written beside the
  installed application.
- The supported automated suite passes; any manual limitations are recorded.

Failure of the proof is useful evidence. It does not authorize an immediate
`src/` migration. Record the failed assumption and return to design review.

## Resource API decision after the proof

`importlib.resources` remains the likely standard interface for installed
resources, but it does not guarantee a permanent filesystem path. An
`as_file()` context can remove temporary materialization after the context
closes. Therefore Sprint 2D must first classify consumers such as Cartopy,
NetCDF, pandas, Streamlit media, and plain text readers.

Possible later interfaces include a resource object/stream API plus a scoped
`asset_file()` context manager, while retaining `asset_path()` as a
source-checkout compatibility layer. Python 3.10 compatibility must also be
considered because standard-library directory materialization was expanded in
Python 3.12. No such API change is part of Sprint 2A–2C.

## Dependencies, cache, and offline mode

- Keep `requirements.txt` and `runtime.txt` unchanged through the proof unless
  a separately reviewed test requirement is unavoidable.
- Do not choose between `pyproject.toml` and `requirements.txt` as the final
  dependency source of truth until the installed proof and Cloud behaviour are
  understood.
- A future persistent cache should use an OS-appropriate user-writable location
  and must remain separate from package resources and private user data.
- A future explicit offline mode should suppress connectivity probes and
  external service/tile attempts, while preserving the current automatic
  fallback. It is not part of the wheel proof.

## Deferred: shared EnvGeo infrastructure

Extraction of a common `envgeo-utils` package and integration with the
Earthquake application remain deferred until the Seawater distribution model
is proven. Later candidates are asset resolution, cache-location policy,
offline capability/fallback messages, and local coastline rendering.
Seawater processing, isotope/GSW/GEBCO semantics, earthquake catalogues, and
plate-boundary interpretation remain application-specific.

## Release work after packaging is stable

Production packaging is followed—not preceded—by the platform test matrix,
CI, `CITATION.cff`, third-party notices, tagged GitHub Release, Zenodo archive
and DOI, and JOSS readiness review. The local wheel proof alone satisfies none
of those release gates.

## Sprint 2D confirmed decisions

The following decisions were confirmed after reviewing the Sprint 2B–2C proof
results. They govern Sprint 3 design scope and any future implementation
proposals. No code, tests, configuration, dataset, or bundled asset was
changed in Sprint 2D.

1. **Distribution mechanism.** Official distribution uses standard pip/wheel
   install with expanded real files on disk. No zip-import or zipapp
   arrangement. The condition for an `importlib.resources` migration is
   non-filesystem resource loaders, zipapp, or an embedded distribution — not
   a standard pip install.

2. **Asset resolver.** `envgeo_assets.asset_path()` is maintained as the
   primary public API for bundled assets. Its
   `Path(__file__).resolve().parent`-based resolution is correct for wheel
   installs and is not replaced.

3. **No `importlib.resources` migration, no `open_asset()`, no OS-level cache
   materialization.** None of these are added until a concrete consumer
   requiring resource streams or temporary materialization exists. The Sprint 2D
   conclusion (maintain `asset_path()`, no resource API) is unchanged; the
   reason is that the current standard wheel install works stably and there is
   no concrete migration requirement. pandas and Streamlit media are consumers
   where future stream/bytes use could be considered, but no migration is needed
   now.

4. **Cartopy shapefile sidecars and GEBCO NetCDF.** Cartopy's Natural Earth
   consumer requires a real filesystem directory (`.shp`, `.shx`, `.dbf`
   siblings in the same directory). The current GEBCO implementation passes a
   `Path` to `scipy.io.netcdf_file`; this works stably with expanded wheel
   files. `scipy.io.netcdf_file` can also accept a file-like object, so real
   `Path` is not strictly required by its API. Both continue to use
   `asset_path()`-based paths. No context manager or stream API is introduced
   for them.

5. **`home.py` `BASE_DIR`.** `BASE_DIR = Path(__file__).resolve().parent` is
   maintained as an auxiliary base for resolving relative image paths in Home's
   Markdown display. It is passed to `render_markdown_file()` and also used
   directly as `render_markdown_streamlit(..., base_dir=BASE_DIR)` for the
   English and Japanese README calls. `resolve_path()` delegates to
   `envgeo_assets.asset_path()` and `BASE_DIR` is not a parallel asset loader.

6. **No EnvGeo-specific persistent OS cache.** The proof confirmed no writes
   into `site-packages`. A persistent OS cache is not needed for this sprint.
   Future design (Sprint 3 or later) uses an OS-appropriate user-writable
   location, separate from package resources and private user data.

7. **Explicit offline launch.** This remains a future feature to suppress
   connectivity probes and external service/tile attempts. It is not a wheel
   requirement and is not used to fix test network isolation. Tests that need
   network isolation use `monkeypatch`; `ENVGEO_OFFLINE` is not a test-fix
   tool. These are two separate problems.

8. **GEBCO remains bundled.** No external download mechanism is added. GEBCO
   is not described as a Python optional extra. Unchanged from Sprint 2A.

9. **`envgeo-utils` separation, Earthquake integration, `src/` move.** All
   remain deferred until Seawater distribution is stable.

10. **Sprint 3 scope.** Design a dependency source of truth (`pyproject.toml`
    vs `requirements.txt`), a clean install into a fresh environment without
    pre-installed runtime packages, a package-data allowlist audit (including
    `bathymetry/make_lightweight_gebco.py` as an identified Sprint 3 audit
    item), and OS/Cloud/CI compatibility testing.
