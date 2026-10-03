# Release and Zenodo Metadata Draft

[日本語版](release_metadata_draft_Japanese.md)

This is a transfer-ready draft for the final GitHub Release and Zenodo record.
It is **not** a release record: the tag date, clean-build wheel checksum, and
Zenodo version DOI must be filled only after the reviewed stable commit has
been tagged and archived.

## Core metadata

| Field | Draft value |
|---|---|
| Title | EnvGeo-Seawater: An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data |
| Release label | EnvGeo-Seawater v1.3.4 |
| Version | 1.3.4 |
| Author | Toyoho Ishimura |
| ORCID | <https://orcid.org/0000-0001-9708-3743> |
| Affiliation | Graduate School of Human and Environmental Studies, Kyoto University, Japan |
| License | MIT |
| Source repository | <https://github.com/envgeo/seawater_map> |
| Release date | Set from the final Git tag; do not reuse the release-candidate date automatically. |
| Version DOI | Not yet assigned. Add only after the Zenodo archive is published. |

## Short description

EnvGeo-Seawater is an interactive Python and Streamlit platform for exploring
seawater isotope and hydrographic data. It integrates approximately 50,000
cited records, including NASA GISS and PAGES CoralHydro2k reference datasets,
with regional reference data and the Kodama et al. (2024) Japan-region core
collection. The application supports mapping, 2D–4D visualization,
salinity–isotope relationships, T–S diagrams, depth profiles, vertical
sections, and session-only comparison with user-supplied data.

## Keywords

`seawater isotopes`; `oceanography`; `hydrography`; `stable isotopes`; `data
visualization`; `Streamlit`; `Python`

## Stable-release scope

The stable public repository contains pages 03, 04, 05, 31, 32, 34, 35, 37,
53, and historical archive page 80. Development pages 90 and 91, and the
local diagnostic Page 99, are excluded from the stable release, its wheel,
GitHub Release, and Zenodo archive.

## Finalization steps

1. Create the final commit and confirm its GitHub Actions CI result.
2. Tag that exact stable commit as `v1.3.4`.
3. Build the wheel from a clean checkout of the tag and record its SHA-256.
4. Create the GitHub Release from the tag, using the title and description
   above.
5. Publish the matching Zenodo archive; then add its version DOI to
   `CITATION.cff`, README citation text, and this record.

The citation instruction remains: cite EnvGeo-Seawater **and** each original
data provider used in the analysis.
