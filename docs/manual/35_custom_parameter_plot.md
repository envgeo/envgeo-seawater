# Custom Parameter Plot

## What This Page Does

This page creates a flexible 2D scatter plot from selected numeric
parameters. It can combine filtered reference data with an optional browser upload.

## Basic Workflow

1. Select reference data and set the sidebar filters.
2. Optionally upload a CSV, XLSX, or XLS file. Any numeric uploaded columns can
   become X, Y, colour, or size parameters.
3. Select the X and Y axes, then optional **Color by** and **Size by** parameters.
4. Select **Apply settings** to update the figure.
5. Adjust the ranges, marker appearance, and figure layout as needed.
6. Inspect the plotted rows and download a PNG figure when needed.

## Main Controls

- **X axis**
- **Y axis**
- **Color by / Colormap / Color range**
- **Size by / Base marker size / Size contrast**
- **Marker transparency**
- **Show background data**
- **Show legend**
- **Regression line** — adds a simple least-squares line and reports its equation,
  correlation coefficient, and sample count when at least two distinct valid X values exist.
- **Axis ranges**
- **Figure width / height**
- **Tick font size / Label font size**
- **X tick count / Y tick count**
- **Uploaded data quality check** — lists uploaded rows that trigger the current
  quality-flag criteria.

## Outputs

- Custom 2D scatter plot
- Optional colour bar and size scaling
- Optional least-squares regression line
- Downloadable PNG figure
- Expandable filtered and uploaded plotted-data tables

## Notes And Limitations

- This is an exploration tool. It is useful for relationships such as dD versus
  δ18O, coloured by depth, but does not establish a scientific model.
- A plotted row requires valid numeric values for the selected X and Y parameters.
  If point size is scaled by a parameter, that value must also be valid. Excluded-row
  counts are reported below the controls.
- The regression line is a visual summary only. It has no automatic treatment of
  data-source differences, uncertainty, autocorrelation, or scientific outliers.
- Browser uploads exist only for the current browser session. They do not modify
  reference datasets or persistent **User Excel data**.
