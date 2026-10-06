# Data Filtering

## What This Section Does

Most EnvGeo-Seawater pages use a shared sidebar filter to select the data shown in figures and tables.

## Basic Workflow

1. Choose datasets, source references, and, where available, transects.
2. Select year and month ranges.
3. Choose an Area filter preset if useful.
4. Fine-tune longitude and latitude.
5. Adjust depth, salinity, d18O, and temperature ranges as needed.
6. Click **Apply settings** to update the figures.

## Uploaded Data

Browser uploads are available on **User Data Check & Quick Visualizer**,
salinity--d18O, mapping, T--S, custom-parameter, depth-profile, and
vertical-section pages. On these pages, uploaded rows appear as **Uploaded
data** in the dataset selector. Select it alone to plot only uploaded rows, or
select it with reference datasets to apply the same filters to both. Uploaded
markers are drawn in the foreground where overlays are supported.

The 2Dplus and 3D/4D pages do not currently provide browser uploads. See the
page-specific manual for any plotting or calculation limitation.

## Always-loaded User Excel Data

The bundled zero-value public sample at `local_data/user_data.xlsx` is the
default optional local table. Because it contains no observations, **User
Excel data** appears in the dataset selector only when the configured local
table contains data rows.

For persistent personal data, set `ENVGEO_LOCAL_USER_DATA_PATH` before
starting the app to an external CSV, XLSX, or XLS file. Do not place personal
data in the repository or application folder. `ENVGEO_DISABLE_LOCAL_USER_DATA`
can disable optional local-table loading when needed.

**User Excel data** and browser **Uploaded data** are independent categories.
Selecting both includes both; the browser upload does not replace or rewrite
the always-loaded workbook.

Numeric cells copied from another spreadsheet may contain invisible whitespace,
including non-breaking or full-width spaces. The shared loader removes these
characters before numeric conversion. Values that still cannot be interpreted
as numbers remain missing and are handled by the normal quality checks.

## Main Controls

- **Area filter preset**  
  Initializes longitude and latitude ranges using common ocean-region presets.

- **Reference / Citation**
  Filters bundled seawater records by their source-reference label.
  It is useful for comparing or plotting one cited source within an integrated
  collection. Some legacy source labels retain cruise or processing context;
  the full citation remains in dataset metadata and source documentation. This
  control does not filter browser-uploaded rows, whose reference field is
  optional.

- **Duplicate-candidate display / 重複候補の表示**
  Leaves all rows visible by default. When selected and applied, it can hide
  only bundled-data overlap candidates from the Data Overlap Check: the
  recommended one-to-one rounding-compatible subset, or the broader Strong
  screen for a sensitivity check. This is reversible and never edits source
  workbooks or uploaded rows.

- **Longitude / Latitude**  
  Manually adjusts the geographic range.

- **Year / Month**  
  Filters sampling time.

- **Water Depth, Salinity, d18O, Temperature**  
  Filters hydrographic and isotope conditions.

## Outputs

- Filtered dataset
- Details and statistics of filtered data
- Filtered-data summary CSV
- Quality information and export controls where the current page provides them

## Notes And Limitations

- Changing filters does not update figures until **Apply settings** is clicked.
- The reference filter is applied before **Area / Transect**, so the latter
  lists only locations available in the selected source records.
- The duplicate-candidate screen is calculated only when it is selected. Use
  **Show all records (default)** to restore all bundled rows immediately.
- Area presets initialize the filter range; users can still fine-tune sliders afterward.
- Some filters retain missing values so that gap rows used to preserve data
  segmentation are not removed unintentionally.
- **Filtered dataset** contains every row that passes the sidebar filters. A plot
  may show fewer rows when its required axes, such as Salinity and d18O, are
  missing; the page reports this excluded count separately.
