# Distribution Design and Verification Record (Sprint 3)

“Sprint 3” is the project's internal name for the development phase that
established installable distribution, CI, and pre-release checks. It is kept
in this title solely to preserve the historical connection to contemporaneous
records.

**Status:** Sprint 3 design and implementation-verification record; originally
approved as the Sprint 3B design basis on 2026-09-27 and rechecked 2026-09-30.
**Scope:** EnvGeo-Seawater development/pre-release repository only. This is
not a release approval. The corresponding Japanese record is
[`sprint3_distribution_design_and_acceptance_Japanese.md`](sprint3_distribution_design_and_acceptance_Japanese.md).

## Fixed boundaries

- Preserve `home.py` beside `pages/`, `envgeo_assets.asset_path()`, and real
  on-disk wheel files.
- Do not edit `dataset/`, Cloud deployment, offline-launch behaviour, or the
  current layout. Do not package EnvGeo-Earthquake or extract a shared core.
- Use the 2026-09-27 pre-Sprint-3 archive as the restoration baseline.

## Dependency and Python policy

- `requirements.txt` is the canonical direct-runtime dependency declaration.
  It remains the Community Cloud input and is the source for wheel dependency
  metadata.
- `pyproject.toml` owns build metadata, package data, and console commands; it
  must not become an independently maintained duplicate dependency list.
- A future constraints/lock-style file records fully resolved versions per
  tested environment; it is evidence, not a second dependency declaration.
- Python 3.10 and 3.12 are verified Linux CI targets. The first local
  clean-install baseline is Python 3.12 on Apple Silicon; the Python 3.10
  clean-install proof is maintained through CI.

## Production package-data policy

Include application modules, the 12 current public pages, runtime media/text,
coastline and Natural Earth assets, GEBCO, the zero-value User Excel template,
and the diagnostic tool implementation.

- Keep the technical inclusion of `dataset/*.xlsx` separate from any
  redistribution or public-release decision; do not silently exclude or
  publish data as a consequence of packaging work.
- Current decision (2026-09-28): include all current `dataset/*.xlsx`
  workbooks in the scholarly-use package. Preserve each workbook's source
  citation and provenance/transform record; inclusion does not transfer
  ownership and does not automatically decide the treatment of future data.
- Exclude `pages/99_Environment_Check.py`: a script in `pages/` appears
  automatically in both installed and Cloud navigation.
- Include the diagnostic tool without page 99 and expose it through an explicit
  command. The normal application command must expose only public pages.
- Exclude `bathymetry/make_lightweight_gebco.py`: it has no runtime reference;
  retain it in source as the documented GEBCO-generation procedure.
- Exclude `Claude outputs/`, cache files, build products, `.DS_Store`, internal
  review logs, and local-only diagnostic wrappers.

## Clean-install acceptance criteria

Sprint 3C uses an environment without system-site packages or user-site
packages, with no checkout on `PYTHONPATH`, and a CWD outside the checkout. It
installs the final wheel, not the source tree.

1. Wheel metadata installs declared runtime dependencies.
2. Installed imports resolve under the fresh environment, not the checkout.
3. All 12 public pages are present; page 99 and internal material are absent.
4. Home and representative Pages 32, 34, and 53 start without application
   exceptions and read their required assets.
5. Dataset workbooks, coastline CSVs, Natural Earth sidecars, GEBCO, runtime
   media/text, and the zero-value User Excel template are available.
6. `ENVGEO_LOCAL_USER_DATA_PATH` works without copying private data into
   package files, cache, or logs.
7. EnvGeo does not write beside installed package files. Dependency caches such
   as Matplotlib or Cartopy are observed and recorded separately.
8. The diagnostic command launches without exposing page 99 in normal app
   navigation.

Full visual review of every page, external tiles, and online integrations is a
separate manual QA item.

## Sprint 3C local verification record (2026-09-27)

Python 3.12 on Apple Silicon was used to build a wheel from a clean staging
copy and install that wheel into a new `venv` with neither system-site nor
user-site packages. The working directory was outside both the source and
wheel staging trees. The verified wheel SHA-256 was
`c5a34a2a7356b81e0fa42aaa2b61b0855099c77dedb44089d1b4c70c4de728c5`.

- `pip check` reported no broken requirements. `pyproject.toml` uses the
  portable explicit license table form, and `pyproj==3.6.1` is declared so the
  Python 3.12 Apple Silicon wheel is selected instead of an incompatible
  source-only resolver result.
- The installed package resolved in the new environment, contained 12 public
  pages and the diagnostic tool, and did not contain page 99 or the GEBCO
  generation script. Both console commands accepted their Streamlit launch
  arguments from the external working directory, and the diagnostic tool
  executed without an application exception.
- Home and Pages 32, 34, and 53 executed without application exceptions using
  the installed files. The established launcher-compatible import path remains
  necessary for individual pages because they retain checkout-compatible
  top-level imports.
- An external CSV supplied through `ENVGEO_LOCAL_USER_DATA_PATH` was read as
  one `User Excel data` row. No file was written below the installed package
  during that check. Matplotlib cache activity was directed to a temporary
  directory and is treated as dependency cache, not application output.

This is a local technical verification, not a release artifact or a claim of
Windows, Intel-macOS, Linux, or Python 3.10 support.

## CI, Cloud, and platform order

1. Run the clean-install procedure locally on Python 3.12.
2. Add Linux CI for build, wheel-content checks, isolated install, and tests
   without runtime network dependency.
3. Confirm current Community Cloud configuration after local/Linux stability;
   do not change Cloud configuration in Sprint 3B.
4. Run Windows and Intel-macOS smoke tests when available, then decide the
   supported matrix.

CI may download declared dependencies during setup, but tests and application
checks must not depend on external tiles, downloads, or live network services.

## Sprint 3D implementation record (2026-09-27)

The first Linux/Python 3.12 workflow is defined in
`.github/workflows/ci.yml`. It installs development requirements, runs the
test suite with local-user data disabled, builds a wheel, and validates an
isolated wheel install from outside the checkout. It uploads the wheel as a
workflow artifact for inspection; it does not publish or release it. The
workflow makes no Cloud configuration change. Windows and Intel-macOS remain
manual smoke-test targets before their support status is declared.
The first successful workflow run was #3 on 2026-09-27 (4 minutes 25 seconds),
including the test suite, wheel build, isolated-install verification, and wheel
artifact upload.
The documentation-policy update was independently verified by successful run
#4 on 2026-09-28 (5 minutes 58 seconds), with the same test, wheel, isolated
install, and artifact checks.
Python 3.10 and 3.12 both passed these checks in run #6 on 2026-09-28
(5 minutes 57 seconds). The workflow retained separate wheel artifacts for
each Python version. The initial Python 3.10 collection error was confined to
the test's use of the Python 3.11+ `tomllib` module; its conditional
test-only `tomli` fallback corrected that compatibility issue without changing
runtime dependencies or wheel contents.

### Community Cloud manual QA

This is a browser-based acceptance check, separate from wheel and CI testing.
The current public site identifies `envgeo-seawater-map.streamlit.app` from
`envgeo/seawater_map` as the stable demo and
`envgeo-seawater-pre.streamlit.app` from `envgeo/envgeo-seawater` as the
development/pre-release demo. Confirm the actual repository, branch, and
revision in the Streamlit deployment settings before recording a result; do
not infer them from the wheel configuration.

For the development/pre-release deployment, record the date, URL, deployment
repository and branch/revision, and any visible errors. Confirm that:

- Home loads and its main navigation is usable.
- The sidebar shows the intended public pages and does not show
  `99_Environment_Check.py`.
- Representative pages open and render without application errors: Mapping,
  T-S Diagram, Depth Profile, and, if currently exposed, Integrated Visualizer
  beta.
- Dataset selection and `Apply settings` update a representative view.
- Online map tiles and other live external services are recorded as separate
  online-availability observations, not as CI requirements.
- No local filesystem path, private user data, token, or diagnostic-only
  information is visible in the deployment.

#### Manual QA record (2026-09-28)

The development/pre-release deployment was checked manually. Home opened; the
sidebar did not show `99_Environment_Check.py`; Mapping, T-S Diagram, and
Depth Profile opened without application errors; and dataset selection with
`Apply settings` updated a representative view. Online map tiles rendered.
No application error was observed during this normal-use check. Deliberately
triggering an error to inspect its details is not part of manual QA; private
data and diagnostic-information exclusion remain release-audit and
source/package checks.

## Release boundary

The proof version `1.3.3` is not a release decision. The planned formal release
is `1.3.4`; it records a clean checkout, source revision, Python version,
resolved dependencies, wheel SHA-256, test result, Git tag, GitHub Release,
and Zenodo DOI. JOSS scope centres on stable Seawater workflows. Pages 90 and
91 remain in the development/pre-release repository for now, but are excluded
from the planned stable `seawater_map` release scope.
