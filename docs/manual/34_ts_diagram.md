# Temperature-Salinity Diagram

## What This Page Does

This page plots temperature-salinity (T–S) relationships for filtered reference
data and optional browser-uploaded data. It includes approximate σ0 reference
contours and a sampling-location map when coordinates are available.

## Basic Workflow

1. Select reference data and set the sidebar filters.
2. Optionally upload a CSV, XLSX, or XLS file, then assign its temperature and
   salinity columns. Longitude and latitude are also needed for the location map.
3. Select **Show background data**, a colour parameter, and other display options.
4. Select **Apply settings** to update the figure.
5. Adjust axes, the colour bar, and the σ0 reference-contour interval as needed.
6. Inspect the plotted data and download a PNG figure when needed.

## Main Controls

- **Show background data**
- **Show legend**
- **Color parameter**
- **Colorbar range**
- **Salinity scale**
- **Temperature scale**
- **Density contour interval (approx. σ0)** — spacing between approximate σ0 reference
  contour lines (0.2, 0.5, or 1.0 kg m⁻³; default 0.5 kg m⁻³)
- **Figure width / height**
- **Tick font size / Label font size**
- **X tick count / Y tick count**
- **Map controls / Map style**
- **Uploaded data quality** — reports usable rows and missing required values for
  the current browser upload.

## Outputs

- Temperature-Salinity diagram
- Density contours
- Sampling location map, when valid longitude and latitude are available
- Downloadable PNG figure
- Filtered dataset table

## Notes And Limitations

- Density contours require valid salinity and temperature values.
- Missing values are excluded from the plotted points and reported in captions.
- Browser uploads exist only for the current browser session. They do not alter a
  reference dataset or persistent **User Excel data**.
- The density contours are **approximate σ0 reference contours**, not pointwise sample density.
  They pass Practical Salinity (≈ Absolute Salinity) and in-situ temperature
  (≈ Conservative Temperature) directly to `gsw.sigma0`; the application does
  not convert each observation to Absolute Salinity and Conservative Temperature
  using its location and pressure. Differences from fully converted TEOS-10 values
  vary by data source, location, depth, and hydrographic conditions. Use the contour
  lines as visual reference guides,
  not as true TEOS-10 density values or a basis for quantitative water-mass classification.
- The contour grid is bounded by the selected display-axis range and clipped to the
  quality-checked input domain (Salinity 0–50; Temperature −5–45 °C), so negative
  salinity values selected on the axis are not passed to the density calculation.
