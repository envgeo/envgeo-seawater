# Stable Release Publication Notes

## Purpose

`seawater_map` is the official stable-release repository for EnvGeo-Seawater.
It is the source repository for the GitHub Release, Zenodo archive, and
JOSS-facing software record. This note is a concise publication boundary; use
the [release checklist](release_checklist.md) for the detailed checks.

[日本語版](stable_release_publication_notes_Japanese.md)

## Stable scope

The stable package contains `home.py`, shared application modules, runtime
assets and cited datasets, and these ten Streamlit pages:

- 03 Interactive 2Dplus Visualizer
- 04 Interactive 3D/4D Visualizer
- 05 User Data Check & Quick Visualizer
- 31 Salinity-d18O Relationship
- 32 Isotope Hydrographic Mapping
- 34 T-S Diagram
- 35 Custom Parameter Plot
- 37 Depth Profile
- 53 Vertical Section Visualizer beta
- 80 Correlation Overview archive

Page 80 is retained as an archive of the original hand-written exploratory
workflow. Do not refactor or remove its historical comments merely for style;
limit changes to demonstrated bugs, compatibility, safety, or distribution
requirements. Pages 35 and 53 remain clearly labelled beta workflows.

## Deliberately excluded material

Do not copy the development repository into this repository wholesale.
The stable package and public deployment exclude:

- `pages/90_Integrated_Visualizer_beta.py` and `pages/91_EnvGeo_Earthquake.py`
- `pages/99_Environment_Check.py`; use `envgeo-seawater-check` locally instead
- `Claude outputs/`, internal review notes, private correspondence, and local
  user-data paths
- build directories, wheel files, environments, caches, `.DS_Store`, reports,
  screenshots, and other generated outputs

The CI wheel check asserts that Pages 90, 91, and 99 are absent.

## Data and researcher-owned files

The current scholarly-use package includes the listed `dataset/*.xlsx`
workbooks with their documented citations, provenance, and
source-to-workbook transformations. Inclusion does not transfer ownership or
automatically decide the treatment of future datasets. See
[dataset redistribution audit](dataset_redistribution_audit.md) and
[provenance inventory](provenance_inventory.md).

Do not commit researcher-owned measurements. The bundled
`local_data/user_data.xlsx` is a zero-value public sample. Use
`ENVGEO_LOCAL_USER_DATA_PATH` for private local data, and do not record its
value in public documentation, logs, screenshots, or CI output.

## Before a release or Zenodo archive

1. Start from a clean checkout of the intended commit; review every staged
   change rather than copying files from another working folder.
2. Confirm the latest CI passes on Python 3.10 and 3.12, including tests,
   wheel build, and isolated-wheel installation.
3. Perform the checklist's stable-app smoke test, including Home, Page 05,
   Mapping, T-S Diagram, Depth Profile, and Vertical Section.
4. Confirm the version, Git tag, commit ID, Python version, resolved
   dependency record, test result, and wheel SHA-256 refer to the same source
   revision. A CI wheel artifact is inspection evidence only; rebuild the
   release-record wheel from the clean tagged checkout.
5. Create the GitHub Release and then the Zenodo archive from that tag; add the
   resulting DOI to the citation material only after it is issued.

## Change control

Document each release-relevant decision in a public `docs/` record or the
user-facing update log. Keep internal coordination material outside the public
repository. Do not make a data-policy, page-scope, or release-boundary change
as a side effect of a technical packaging change.
