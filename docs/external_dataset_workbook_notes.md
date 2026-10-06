# External Dataset Workbook Notes

**Status:** source-to-workbook comparison, 2026-09-25; current bundled
workbook structure rechecked, 2026-09-30; CoralHydro2k citation-helper
metadata completed, 2026-10-06. This note compares the bundled
workbooks with the locally retained source snapshots supplied for the audit. It
records observed differences only; it does not alter either source or bundled
data. The corresponding Japanese record is
[`external_dataset_workbook_notes_Japanese.md`](external_dataset_workbook_notes_Japanese.md).

## Purpose

The NASA GISS and PAGES CoralHydro2k workbooks are cited third-party reference
data used for comparison and visualization. This record keeps their source
citations separate from project-side fields used by the application. It
contains no private correspondence or contact details.

## Current bundled-workbook recheck (2026-09-30)

This read-only recheck confirms the currently bundled files and their workbook
structure; it does not repeat the historical source-file comparison because the
source snapshots are intentionally outside the application tree. The two
workbook SHA-256 values and their relationship to the stable-release clone are
listed in [`provenance_inventory.md`](provenance_inventory.md).

- NASA GISS: `NASA_20260227` has 25,514 rows and 22 columns. All rows have
  `Transect = Nasa_database`; `Cruise`, `Station`, and `remarks by TI` remain
  empty.
- PAGES CoralHydro2k: `CoralHydro2k_SW_1_0_0_20260303` has 18,598 rows and 58
  columns. `Transect` is populated in all rows with 93 distinct values; the
  project-added short `reference` field is now populated in all rows; the six
  documented common-schema placeholders remain empty.

## NASA GISS workbook

| Item | Current record |
|---|---|
| Bundled file | `dataset/71_GLOBA_NASA_20260226.xlsx` |
| Worksheet | `NASA_20260227` |
| Source / citation | `data_text/NASA_references.md`: Global Seawater Oxygen-18 Database v1.22, GISS URL, accessed 2026-03-01. |
| Source snapshot | `geto18.cgitab.txt` (tab-delimited source download retained outside the application tree) |
| Current data rows | 25,514; equal to the source row count |
| Common app fields present | `Transect`, location/depth/time fields, `Temperature_degC`, `Salinity`, `d18O`, `dD`, `reference`, and notes fields. |
| Known application convention | Every row has `Transect = Nasa_database`. NASA provides no transect metadata usable for the application's sub-dataset selector. This is an application grouping label, not a source-supplied oceanographic transect. |

The 11 source fields are retained with common-schema header names:
`Longitude` → `Longitude_degE`, `Latitude` → `Latitude_degN`, `Depth` →
`Depth_m`, `pTemperature` → `Temperature_degC`, `Reference` → `reference`,
with `Salinity`, `d18O`, `dD`, `Year`, `Month`, and `Notes` retained by name.
All 25,514 source rows match their corresponding bundled values numerically or
as the original `**` missing-value marker. Project-only common-schema fields
are otherwise blank, except `Transect`. No observation-value difference was
found in this comparison.

`Cruise` and `Station` are present as common-schema fields but have no values in
the current NASA workbook. This is a statement about the current workbook, not
an assertion about the original database.

## PAGES CoralHydro2k workbook

| Item | Current record |
|---|---|
| Bundled file | `dataset/71_GLOBAL_Atwood_et_al_2026_v02.xlsx` |
| Worksheet | `CoralHydro2k_SW_1_0_0_20260303` |
| Source / citation | `data_text/CoralHydro2_references.md`: PAGES CoralHydro2k Seawater δ18O Database, NCEI study URL, dataset DOI, Atwood et al. (2026), accessed 2026-03-16. |
| Source snapshot | `CoralHydro2k_Seawater_1_0_0.xlsx` (retained outside the application tree) |
| Current data rows | 18,598; equal to the source row count |
| Common app fields present | `Cruise`, `Station`, `Transect`, location/depth/time fields, `Temperature_degC`, `Salinity`, `d18O`, `dD`, `reference`, and extensive source-provenance fields. |
| Current grouping field | `Transect` is populated in all rows and contains 93 distinct values. It is the common-schema name for the source field `Site name or geographic area`. |

The 50 source columns are retained or mapped to common-schema names. Examples
include `Cruise ID` → `Cruise`, `Station ID` → `Station`, collection year/month/day
→ `Year`/`Month`/`Day`, location/depth/temperature/salinity/isotope fields →
their corresponding common fields, and `Publication citation` →
`reference_full`. The current workbook additionally contains blank
common-schema placeholders (`Date`, `TargetDepth_m`, `Bottle`, `d13C`, `PI`,
and `Vertical`) and a project-added short `reference` field for application
display and selection. It follows `Surname (year)` for one author and
`Surname et al. (year)` for multiple authors; a corporate author remains the
author label. This is a display/classification aid, not a replacement for the
full citation.

On 2026-10-06, the 3,258 previously blank short-reference cells were completed
without changing any observation or source-provenance field. Those rows lacked
`Publication citation`/`reference_full` but retained a non-empty source
`Dataset citation`. Seven source-specific mappings were reviewed explicitly:
`GEOTRACES Intermediate Data Product Group (2021)`, `Lamb et al. (2014)`,
`Dickson (2016)`, `Schmidt et al. (1999)`, `Lo Monaco et al. (2013)`,
`Henley et al. (2020)`, and `Stoll et al. (2013)`. Compound and particle-based
names are not inferred by a general surname parser. The full source citation
remains in `reference_full` or `Dataset citation`; the project-added
`reference` value only supports compact display and grouping.

The edited workbook is versioned as `71_GLOBAL_Atwood_et_al_2026_v02.xlsx`.
Before the next public repository synchronization, update the loader, page
catalogue, provenance records, package checksum table, and release notes in
the same change set.

Observed metadata-format differences are limited to two date-formatted source
fields: 5,969 `Water isotope analysis date` cells and 9 `Station ID` cells are
stored as Excel serial-date values in the bundled workbook. The underlying
source dates were not edited by this audit, and this record does not infer why
the date formatting changed. All other mapped source fields match their
corresponding bundled values, allowing for equivalent numeric formatting.

## Documentation boundary

The workbooks contain project-side common-schema labels and fields. This note
now records the observable source-to-workbook mapping for the two supplied
source snapshots. It must not be extended to other versions or data sources
without a corresponding source snapshot. Preserve the DOI/source, access date,
citation, source snapshots, and this mapping record for future reproducibility.

## Current package decision

Both workbooks remain in the current `dataset/*.xlsx` package-data collection
for scholarly use. The comparison supports describing the observation records
as preserved: documented differences are common-schema or grouping labels,
blank placeholders, a display-oriented short citation, one NASA QC note, and
the noted Excel date-format representation. They are not replacement
observation data. Retain the source citations and this change record; neither
workbook is project-owned.
