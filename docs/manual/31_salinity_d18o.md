# Salinity-δ18O Relationship

## What This Page Does

This page plots the relationship between salinity and δ18O for filtered
reference data and, when supplied, browser-uploaded data.

## Basic Workflow

1. Select a reference data source and set the sidebar filters.
2. Optionally upload session-only data in **Uploaded data overlay**. The plot
   requires assigned salinity and d18O columns; valid longitude and latitude
   are also needed to show uploaded locations on the map.
3. Choose whether to show background data and a regression line.
4. Select **Apply settings** after changing filters.
5. Adjust color, color range, axis scales, size, and tick-label settings.
6. Review uploaded-data quality information when an upload is present.
7. Download the PNG figure or filtered table where provided.

## Main Controls

- **Show background data**
- **Regression line**
- **Color parameter**
- **Salinity scale**
- **d18O scale**
- **Figure width / height**
- **Tick font size / Label font size**
- **X tick count / Y tick count**
- **Map controls / Map style**

## Outputs

- Salinity-d18O scatter plot
- Optional regression line
- Sampling location map
- Downloadable PNG figure
- Filtered dataset table
- Uploaded-data quality check and uploaded overlays when applicable

## Notes And Limitations

- Regression is intended for exploratory interpretation.
- Regression requires at least two valid salinity--d18O points; it is skipped
  when the requirement is not met.
- Points missing salinity or d18O are excluded from the relationship plot.
  Points without the selected color parameter can optionally be shown with the
  uploaded-marker fallback style.
- Browser uploads remain in the current Streamlit session and do not alter
  reference data or persistent **User Excel data**.
