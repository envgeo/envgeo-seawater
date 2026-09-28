# Bundled Asset and Data Provenance Inventory

**Status:** working release-readiness inventory, audited 2026-09-25.  
This record identifies what is bundled and what still needs evidence. “Publicly
available” is not treated as proof that redistribution in a software release is
permitted.

| Resource | Consumer | Source / transformation record | Redistribution status | Required follow-up |
|---|---|---|---|---|
| `coastline/world_coastline_coordinates_50m.csv` and `110m.csv` | Offline Plotly maps; Cartopy coastline line | Derived from Natural Earth vector data. | Natural Earth is public domain; derived-CSV source version/checksum record is incomplete. | Add a CSV-specific source URL, retrieval date, generator procedure, and checksums. |
| `coastline/natural_earth_50m_land/` | Page 32 land mask | Natural Earth 50m land; detailed file record in `LICENSE_OR_SOURCE.md`. | Suitable for redistribution with retained provenance/credit. | Retain “Made with Natural Earth” and checksums. |
| `data_beta/GEBCO_2025_6min.nc` | Page 53 optional bathymetry | 6 arc-minute subsample of GEBCO_2025; generator: `make_lightweight_gebco.py`. | GEBCO Grid is public domain with attribution, disclaimer, and non-navigation conditions. | Record source retrieval date/checksum for the input grid and retain the GEBCO 2025 citation. |
| `local_data/user_data.xlsx` | Always-loaded local User Excel workflow | Project-created, zero-value public template. | Approved for public distribution. | Keep zero value before public synchronization; private data use an external path or local uncommitted edit. |
| `dataset/01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx` | Japan Sea / global loader | Kodama et al. (2024); source citation in `data_text/main_references.md`. | Included in the current scholarly-use package; file-level provenance is being completed. | Preserve the canonical source, citation, version/access date where available, and transformation record. |
| `dataset/11_AROUND_JAPAN_PUB_20260305.xlsx` | Around Japan / global loader | Regional compilation; citations include Yamamoto, Sakamoto, Kodaira, and Horikawa records. | Included in the current scholarly-use package; file-level provenance is being completed. | Preserve row/source mapping, citations, and transformations for each incorporated source. |
| `dataset/71_GLOBA_NASA_20260226.xlsx` | Global loader | NASA GISS Global Seawater Oxygen-18 Database; `data_text/NASA_references.md` records v1.22, source URLs, citation, and access date 2026-03-01. The source-to-workbook correspondence (25,514 rows) and project `Transect = Nasa_database` convention are documented. | Approved for the current scholarly-use distribution; retain source attribution and do not describe as project-owned. | Preserve the existing version/access record, citation, and transformation record; reassess only for an explicit restriction, rights-holder request, or concrete reviewer concern. |
| `dataset/71_GLOBAL_Atwood_et_al_2026.xlsx` | Global loader | PAGES CoralHydro2k Seawater δ18O Database; `data_text/CoralHydro2_references.md` records study URL, DOI, citation, and access date 2026-03-16. The source-to-workbook correspondence (18,598 rows), citation preservation, short-reference labels, and `Transect` schema mapping are documented. | Approved for the current scholarly-use distribution; retain source attribution and do not describe as project-owned. | Preserve the existing DOI/access record, citation, and transformation record; reassess only for an explicit restriction, rights-holder request, or concrete reviewer concern. |
| `dataset/72_GLOBAL_RECENT_REPORTS_20260302.xlsx` | Global loader | Project compilation of recent reports. | Included in the current scholarly-use package; file-level provenance is being completed. | Preserve source mapping, citations, and transformations for report-derived records. |
| Archived `d18O_upload_data_tmp_seawater.xlsx` | Not used by the application. | Legacy source-only workbook; no code, test, or public-clone reference was found. | Not part of the application or a future package-data manifest. | Retained in the parent workspace archive (`過去のパーツ/`), outside the application tree; review separately only if reuse is proposed. |
| `data/` media and `data_text/` references/manuals | Home and in-app documentation | Project media and documentation, with external reference links where applicable. | Needs item-level review only where third-party media are bundled. | Record creation/source and reuse status for any non-project media. |

## Packaging gate

The coastline, Natural Earth, GEBCO, and zero-value User Excel sample have a
clear enough direction for a future asset-loader proof. The current Sprint 3
wheel includes all current `dataset/*.xlsx` workbooks as the chosen
scholarly-use package scope. Every workbook remains source-attributed;
inclusion does not transfer ownership or determine the treatment of a future
dataset. Preserve and complete source-level provenance records, and reassess
only when a new explicit restriction, rights-holder request, or concrete
reviewer concern arises. This inventory is also the starting point for the
Stage 2 T–S data provenance audit.
