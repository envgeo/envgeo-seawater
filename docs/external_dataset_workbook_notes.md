# External Dataset Workbook Notes

**Status:** source-to-workbook comparison, 2026-09-25. This note compares the
bundled workbooks with the locally retained source snapshots supplied for the
audit. It records observed differences only; it does not alter either source or
bundled data.

## Purpose

The NASA GISS and PAGES CoralHydro2k workbooks are cited third-party reference
data used for comparison and visualization. This record keeps their source
citations separate from project-side fields used by the application. It
contains no private correspondence or contact details.

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
are otherwise blank, except `Transect` and one manually entered
`remarks by TI` QC note. No observation-value difference was found in this
comparison.

`Cruise` and `Station` are present as common-schema fields but have no values in
the current NASA workbook. This is a statement about the current workbook, not
an assertion about the original database.

## PAGES CoralHydro2k workbook

| Item | Current record |
|---|---|
| Bundled file | `dataset/71_GLOBAL_Atwood_et_al_2026.xlsx` |
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
and `Vertical`) and an added short `reference` field (15,340 populated rows).
That short field is extracted from the longer source `Publication citation` as
an author-and-year form for application display and selection; the original
long citation is retained as `reference_full`. A project-side Python
transformation extracts a four-digit parenthesized year and the first author's
family name. It writes `Surname (year)` for one author and `Surname et al.
(year)` for multiple authors; its intermediate output field is named
`reference_short`, corresponding to the current workbook's short
`reference` field. This is a display/classification aid, not a replacement for
the full citation.

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
