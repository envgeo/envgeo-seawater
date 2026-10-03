# Depth Profile

## What This Page Does

This page plots vertical profiles of δ18O, δD, d-excess, temperature, or salinity
against water depth. Filtered reference data and an optional browser upload can be
shown together.

## Basic Workflow

1. Select reference data and a **Profile parameter**.
2. Optionally upload a CSV, XLSX, or XLS file. The selected parameter and
   `Depth_m` are required for a profile; coordinates are optional for the map.
3. Set the sidebar filters and select **Apply settings** to update the view.
4. Choose whether to show unfiltered background data.
5. Adjust the depth range, horizontal-axis scale, figure size, fonts, and ticks.
6. Inspect the profile and sampling map, then download a PNG figure if needed.

## Main Controls

- **Show background data**
- **Profile parameter**
- **Depth range**
- **Axis scale**
- **Figure width / height**
- **Tick font size / Label font size**
- **X tick count / Y tick count**
- **Map controls / Map style**
- **Show uploaded data without Month information** — controls whether upload rows
  without a valid month remain visible in the profile.
- **Uploaded data quality check** — lists uploaded rows that trigger the current
  quality-flag criteria.

## Outputs

- Depth profile plot
- Sampling-location map when valid longitude and latitude are available
- Downloadable PNG figure
- Expandable table for the filtered data used by the current view

## Notes And Limitations

- The selected parameter and depth must both be valid numeric values. dD and
  d-excess may have many missing values; excluded-row counts are reported.
- Gap rows are inserted so lines do not connect unrelated station/date groups.
- Uploaded profiles are grouped by rounded coordinates when coordinates are
  available. With Month values, they are coloured by four month bands (1–3, 4–6,
  7–9, and 10–12); rows without a valid month use the selected upload style when
  **Show uploaded data without Month information** is enabled.
- Browser uploads exist only for the current browser session. They do not modify
  reference datasets or persistent **User Excel data**.
- Long filter-condition titles are wrapped in downloaded images.
