# Historical Wheel Proof Report (Sprint 2B–2C)

“Sprint 2B–2C” is the project's internal name for the first local wheel proof.
This report preserves that proof's evidence. It is not the current release
verification record; see
[`sprint3_distribution_design_and_acceptance.md`](sprint3_distribution_design_and_acceptance.md)
for the implemented 1.3.4 distribution and CI checks.

**Date:** 2026-09-25  
**Status:** historical local technical proof passed on the Python 3.12 baseline.  
**Release status:** not a public release artifact or supported installation guide.

## Scope

The proof tested whether the existing repository-root layout can be installed
without moving `home.py`, `pages/`, or bundled assets into `src/`. It did not
change Streamlit Community Cloud configuration, dependency ownership,
`asset_path()`, cache/offline behaviour, any file under `dataset/`, or GEBCO.

## Minimal proof files

- `pyproject.toml`: setuptools root-to-package mapping, explicit package-data
  allowlist/exclusions, and the `envgeo-seawater` proof launcher.
- `__init__.py`: package identity for the installed proof.
- `envgeo_launcher.py`: locates the installed sibling `home.py` independently
  of the process CWD, then starts Streamlit.
- `test/test_packaging_proof.py`: protects the non-moving layout, dependency
  boundary, asset families, exclusions, and launcher declaration.

Runtime dependencies remain owned by `requirements.txt` during the proof.
The `pyproject.toml` deliberately does not duplicate them.

## Environment and artifact

- Python 3.12.14
- Streamlit 1.63 / Plotly 5.24 environment
- setuptools 83.0.0 / wheel 0.47.0
- Build command:
  `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist`
- Artifact: `envgeo_seawater-1.3.3-py3-none-any.whl`
- Size: 25,970,909 bytes
- SHA-256: `225699be477cc35028d1edfce7bc32149ba72203116bd3d5f5439be8d12bdab2`

The artifact under `dist/` is generated and ignored. Do not publish or commit
it as a release artifact from this proof.
The proof reports themselves are intentionally excluded from the wheel, so
recording the artifact checksum does not create a self-referential build.

## Verified results

- Built the wheel without moving the application tree.
- Installed it into a new virtual environment located outside the checkout.
  The environment inherited the already verified runtime dependencies through
  `--system-site-packages`; dependency resolution itself was not tested.
- Imported the installed package from `site-packages`, with the CWD outside the
  checkout.
- Confirmed all 12 intended public page scripts were included.
- Confirmed `pages/99_Environment_Check.py`, `.DS_Store`, and cache files were
  excluded.
- Confirmed representative installed files from `dataset/`, coastline CSV,
  the complete Natural Earth shapefile family, the zero-value user-data
  sample, Home media, and GEBCO.
- Ran the installed Home page with Streamlit AppTest: title rendered and no
  application exception was reported.
- Ran Page 34 from a read-only installed package directory: bundled data loaded
  and no application exception was reported.
- Started the installed `envgeo-seawater` command from outside the checkout;
  the Streamlit health endpoint returned `ok`.
- Confirmed Home and Page 34 still ran after removing write permission from the
  installed package directory. Matplotlib used its own temporary cache when
  its normal user cache was unavailable; no EnvGeo package write was required.

## Automated tests

Focused package/asset/public-surface tests:

```text
35 passed
```

Complete supported Python 3.12 suite:

```text
403 passed, 1 warning
```

The warning is the existing pytest class-scoped fixture deprecation warning in
`test/test_self_contained_html.py`, not a packaging failure.

An initial sandboxed full-suite run produced one failure because the Vertical
Section AppTest entered its valid offline branch after the sandbox blocked its
connectivity probe; the test expected the online-only `A-B input` radio. The
same test and the complete suite passed under normal local connectivity. The
test's network sensitivity is a separate maintenance item and was not weakened
or changed in this sprint.

## What this proof does not establish

- Fresh dependency installation without preinstalled runtime packages.
- Windows, Linux, Intel macOS, or Streamlit Community Cloud compatibility.
- A final dependency source of truth.
- A final `importlib.resources`, cache, or explicit offline-mode API.
- Permission to publish a full-data wheel or archive.
- A final decision against a later package-directory or `src/` migration.

These remain gates for Sprint 2D, Sprint 3, and the later platform/release
review.
