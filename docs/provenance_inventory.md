# Bundled Asset and Data Provenance Inventory

**Status:** release-readiness inventory; source evidence audited 2026-09-25,
bundled-workbook snapshot verified 2026-09-30, and derived-asset records
verified 2026-10-03, and initial cross-dataset overlap screening recorded
2026-10-05.
This record identifies what is bundled and what still needs evidence. “Publicly
available” is not treated as proof that redistribution in a software release is
permitted.

| Resource | Consumer | Source / transformation record | Redistribution status | Required follow-up |
|---|---|---|---|---|
| `coastline/world_coastline_coordinates_50m.csv` and `110m.csv` | Offline Plotly maps; Cartopy coastline line | Derived from Natural Earth coastline v4.1.0 via retained 50m/110m Excel intermediates; numerical correspondence and output checksums verified 2026-10-03. | Natural Earth is public domain; retain source credit. | Preserve the documented intermediate-file relationship and current output checksums. The original raw-download retrieval checksum was not retained. |
| `coastline/natural_earth_50m_land/` | Page 32 land mask | Natural Earth 50m land; detailed file record in `LICENSE_OR_SOURCE.md`. | Suitable for redistribution with retained provenance/credit. | Retain “Made with Natural Earth” and checksums. |
| `bathymetry/GEBCO_2025_6min.nc` | Page 53 optional bathymetry | 6 arc-minute GEBCO_2025 subsample; NetCDF history and retained generator record stride 24. Output checksum verified 2026-10-03. | GEBCO Grid is public domain with attribution, disclaimer, and non-navigation conditions. | Retain the GEBCO 2025 citation, generator, output checksum, and non-navigation notice. The original input-grid retrieval checksum was not retained. |
| `local_data/user_data.xlsx` | Always-loaded local User Excel workflow | Project-created, zero-value public template. Headers, zero-row state, and visible formatting were retained when the workbook was regenerated without a local absolute-path metadata field on 2026-10-03. | Approved for public distribution. | Keep the zero-value, metadata-safe template before public synchronization; private data use an external path or local uncommitted edit. |
| `dataset/01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx` | Japan Sea / global loader | Kodama et al. (2024), with DOI and analytical record in `data_text/main_references.md`; current workbook fingerprint is recorded below. | Included in the current scholarly-use package. | Preserve the canonical source, citation, available version/access record, and transformation record. |
| `dataset/11_AROUND_JAPAN_PUB_20260305.xlsx` | Around Japan / global loader | Regional compilation of Yamamoto, Sakamoto, Kodaira, and Horikawa records; four source labels and current workbook fingerprint verified. | Included in the current scholarly-use package. | Preserve row/source mapping, citations, and transformations for each incorporated source. |
| `dataset/71_GLOBA_NASA_20260226.xlsx` | Global loader | NASA GISS Global Seawater Oxygen-18 Database; `data_text/NASA_references.md` records v1.22, source URLs, citation, and access date 2026-03-01. The source-to-workbook correspondence (25,514 rows) and project `Transect = Nasa_database` convention are documented. | Approved for the current scholarly-use distribution; retain source attribution and do not describe as project-owned. | Preserve the existing version/access record, citation, and transformation record; reassess only for an explicit restriction, rights-holder request, or concrete reviewer concern. |
| `dataset/71_GLOBAL_Atwood_et_al_2026.xlsx` | Global loader | PAGES CoralHydro2k Seawater δ18O Database; `data_text/CoralHydro2_references.md` records study URL, DOI, citation, and access date 2026-03-16. The source-to-workbook correspondence (18,598 rows), citation preservation, short-reference labels, and `Transect` schema mapping are documented. | Approved for the current scholarly-use distribution; retain source attribution and do not describe as project-owned. | Preserve the existing DOI/access record, citation, and transformation record; reassess only for an explicit restriction, rights-holder request, or concrete reviewer concern. |
| `dataset/72_GLOBAL_RECENT_REPORTS_20260302.xlsx` | Global loader | Project compilation currently containing 35 Sakamoto et al. (2022) records; source citation and current workbook fingerprint verified. | Included in the current scholarly-use package. | Preserve source mapping, citations, and transformations for report-derived records. |
| Archived `d18O_upload_data_tmp_seawater.xlsx` | Not used by the application. | Legacy source-only workbook; no code, test, or public-clone reference was found. | Not part of the application or a future package-data manifest. | Retained in the parent workspace archive (`過去のパーツ/`), outside the application tree; review separately only if reuse is proposed. |
| `data/` media and `data_text/` references/manuals | Home and in-app documentation | `d18O_all.mp4` is the Home animation labelled as created with GMT. `sites_20230515.gif` and `year_20230517.gif` are project map visualizations used by the application/documentation. The three assets are packaged; no third-party media asset requiring a separate license record was identified in this review. `data_text/*.md` is likewise bundled. The unused legacy spreadsheets formerly in `data/` were moved to a local historical archive outside the application and public-release trees on 2026-10-03. | Project documentation/media can remain bundled; historical spreadsheets are excluded from public releases. | Retain any future source or reuse record when adding non-project media; do not reintroduce archived spreadsheets without a separate review. |

## Initial cross-dataset overlap screening (2026-10-05)

This is a read-only candidate screen, not a deduplication operation. It does
not alter bundled workbooks, change source attribution, or establish that any
candidate pair is the same physical observation. The current NASA and
CoralHydro2k workbooks have no sufficiently complete shared date field for
sample-identifier matching, and nearby profile observations can produce more
than one candidate pair.

| Comparison | Review threshold | Strict threshold | Result of initial screen |
|---|---|---|---|
| NASA GISS × PAGES CoralHydro2k | horizontal distance ≤30 km; depth difference ≤10 m; salinity difference ≤0.20; δ18O difference ≤0.10‰ | ≤15 km; ≤3 m; ≤0.10; ≤0.05‰ | 4,068 candidate pairs at the review threshold; 2,463 pairs at the strict threshold. Pair counts are not counts of unique duplicate observations. |
| Around Japan × NASA GISS | same review threshold | same strict threshold | 139 strict candidate pairs, involving 131 of 419 Around Japan rows. The Japan records are principally `Yamamoto et al. (2001)` / `PI=KAWAI`; NASA labels include Yamamoto et al. (2001) and (2002). |
| Around Japan × PAGES CoralHydro2k | same review threshold | same strict threshold | No candidate pairs under either threshold in this initial screen. |

Before a future analytical exclusion feature is enabled, the project will
publish a row-level audit table (source row identifiers, coordinates, variable
deltas, and reference metadata), review original source/sample or campaign
identifiers where available, and establish one-to-one, review-confirmed pairs.
The feature will be optional and will suppress only the selected duplicate
representation in combined-dataset statistics and figures; source records and
their required citations will remain available and unchanged.

### Time-aware candidate screen for the v1.3.5 design

The following additional read-only screen requires a valid matching collection
year and month, horizontal distance ≤15 km, salinity difference ≤0.1, and
δ18O difference ≤0.1‰. It excludes withheld/non-numeric month values. NASA
does not provide a usable collection day in this workbook, so this is a
same-year-and-month criterion, not a same-day confirmation. Counts are
candidate pairs; the parenthesized values are distinct source rows on the left
and right, respectively.

| Comparison | Depth difference ≤1 m | ≤3 m | ≤10 m | ≤50 m |
|---|---:|---:|---:|---:|
| NASA GISS × PAGES CoralHydro2k | 1,730 (1,644 / 1,654) | 1,764 (1,670 / 1,680) | 1,897 (1,704 / 1,712) | 2,389 (1,805 / 1,777) |
| Around Japan × NASA GISS | 55 (55 / 55) | 55 (55 / 55) | 62 (55 / 55) | 91 (55 / 55) |

The 55 Around Japan--NASA pairs are a same-year-and-month September 1996
profile associated with `Yamamoto et al. (2001)` / `PI=KAWAI`. The Japan and
NASA coordinates differ by up to approximately 11 km because of coordinate
rounding, while the profile depth sequence and salinity/δ18O values agree to
their recorded precision. This is a particularly strong candidate group, but
it remains an audit finding until source-level assignment is recorded.

At broad depth tolerances, a single profile observation can form several
candidate pairs. Consequently, the ≤10 m and ≤50 m columns are appropriate for
review and candidate marking; they must not be interpreted as counts of rows
to suppress. The planned UI will retain an "exclude none" default and permit
suppression only for review-confirmed, one-to-one pairs.

## Verified bundled-workbook snapshot (2026-09-30)

This read-only check confirmed that the following five files are present in
both the canonical working folder and the stable `seawater_map` clone, with
identical SHA-256 checksums. They are exactly the current files selected by
the `dataset/*.xlsx` package-data rule. No workbook content was modified by
this audit.

| Bundled file | Worksheet | Rows | Columns | SHA-256 |
|---|---|---:|---:|---|
| `01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx` | `Kodam_et_al_2024` | 2,222 | 22 | `8184016fe53fb3b537b5ca60061b2e3d798f69b63534ffea6d6a11d9904d2962` |
| `11_AROUND_JAPAN_PUB_20260305.xlsx` | `for_streamlit_YSKH_20260227` | 419 | 22 | `ac49cf552e88caff1d294bbcbff978b8ea72faa31c7fc1f698456ea978663dd0` |
| `71_GLOBAL_Atwood_et_al_2026.xlsx` | `CoralHydro2k_SW_1_0_0_20260303` | 18,598 | 58 | `27aa53ac15867d571a5efd94aa63403bf60718d6bc6881ef2c8e7a92960b6a15` |
| `71_GLOBA_NASA_20260226.xlsx` | `NASA_20260227` | 25,514 | 22 | `13cccbffa3948a2570fd7c6faa342a888d1e14d3bb076f48d520564a25c85840` |
| `72_GLOBAL_RECENT_REPORTS_20260302.xlsx` | `20260303` | 35 | 23 | `d8027747739247601bfbc9ae8ffc2c9b9036edfaaa83b7f84972e28b9f84a565` |

The tracked `local_data/user_data.xlsx` public sample was also verified as a
zero-row, 22-column template in both locations. A release that changes any
listed workbook must update this snapshot, its source-level provenance record,
and the release checksum record together.

## Verified derived-asset snapshot (2026-10-03)

The following checks are read-only and do not reconstruct or replace the
original source downloads. They record the reproducible relationship retained
inside the project and the exact fingerprints of the distributed outputs.

| Bundled resource | Verified derivation record | SHA-256 |
|---|---|---|
| `coastline/world_coastline_coordinates_50m.csv` | Numerically matches the retained `world_coastline_coordinates_50m.xlsx` intermediate (61,844 rows; same NaN positions; maximum floating-point difference `1.42e-14`). The intermediate was derived from Natural Earth 50m coastline v4.1.0 in the separate coastline working folder. | `c3d7bee4fb696b011fa34bb13bed0c335c5250eeaf37d8739d77d29a27fe385c` |
| `coastline/world_coastline_coordinates_110m.csv` | Numerically matches the retained `world_coastline_coordinates_110m.xlsx` intermediate (5,261 rows; same NaN positions; maximum floating-point difference `1.42e-14`). The intermediate was derived from Natural Earth 110m coastline v4.1.0 in the separate coastline working folder. | `a31df3aeee9dc4195af35a31b0605fdb572c7c7dd7cde17f773c9438f5ec7f3f` |
| `bathymetry/GEBCO_2025_6min.nc` | NetCDF history identifies `GEBCO_2025.nc`, 6.0 arc-minute spacing, and stride 24; the retained `make_lightweight_gebco.py` implements that derivation. Grid dimensions are 1,800 latitude by 3,600 longitude cells. | `0afdf1d0e023b0529c56b69a2684e505c7e2ea28d78a3af8814817af59b09030` |

The original source-download checksums and retrieval dates for these derived
assets were not retained. This limitation is explicit: the records above
identify the distributed artifact, its retained derivation path, and its
upstream public source, but do not claim bit-for-bit reconstruction of a past
raw download.

## Packaging gate

The coastline, Natural Earth, GEBCO, and zero-value User Excel sample now have
release-facing source and fingerprint records. The current wheel includes all
current `dataset/*.xlsx` workbooks as the chosen
scholarly-use package scope. Every workbook remains source-attributed;
inclusion does not transfer ownership or determine the treatment of a future
dataset. Preserve and complete source-level provenance records, and reassess
only when a new explicit restriction, rights-holder request, or concrete
reviewer concern arises. This inventory is also the starting point for the
Stage 2 T–S data provenance audit.
