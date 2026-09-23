# Data Filtering

## What This Section Does

Most EnvGeo-Seawater pages use a shared sidebar filter to select the data shown in figures and tables.

## Basic Workflow

1. Choose datasets and transects.
2. Select year and month ranges.
3. Choose an Area filter preset if useful.
4. Fine-tune longitude and latitude.
5. Adjust depth, salinity, d18O, and temperature ranges.
6. Click **Apply settings** to update the figures.

## Uploaded Data

On supported pages, browser-uploaded rows appear as **Uploaded data** in
**Select sub-dataset**. Select it alone to plot only the uploaded rows, or
select it with reference datasets to use the same filters for both. The page
reports the uploaded-row count where relevant. Uploaded markers are drawn in
the foreground.

Vertical Section uses the selected uploaded rows together with selected
reference rows for compatible section calculations. Other pages may retain a
dedicated overlay path where a calculation must intentionally exclude uploads;
follow the page-specific note.

## Always-loaded User Excel Data

For repeated local analysis, `local_data/user_data.xlsx` is read at app startup
and assigned the dataset name **User Excel data**. The loader appends it to the
selected Japan Sea, Around Japan, or Global reference source, so it is available
in the dataset selector without a browser upload. The file is ignored by Git.
Use `ENVGEO_LOCAL_USER_DATA_PATH` to select a different CSV/XLSX/XLS file.

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
- Quality flag criteria near exported tables

## Notes And Limitations

- Changing filters does not update figures until **Apply settings** is clicked.
- Area presets initialize the filter range; users can still fine-tune sliders afterward.
- Missing values may be retained in some filters to avoid accidentally removing metadata-only rows.
- **Filtered dataset** contains every row that passes the sidebar filters. A plot
  may show fewer rows when its required axes, such as Salinity and d18O, are
  missing; the page reports this excluded count separately.
