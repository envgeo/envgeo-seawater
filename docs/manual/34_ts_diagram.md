# Temperature-Salinity Diagram

## What This Page Does

This page plots temperature-salinity relationships with density contours.

## Basic Workflow

1. Select a dataset.
2. Choose whether to show background data.
3. Apply sidebar filters.
4. Choose a color parameter if needed.
5. Adjust figure appearance.
6. Download the figure if needed.

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

## Outputs

- Temperature-Salinity diagram
- Density contours
- Sampling location map
- Downloadable PNG figure
- Filtered dataset table

## Notes And Limitations

- Density contours require valid salinity and temperature values.
- Missing values are excluded from the plotted points and reported in captions.
- The density contours are **approximate σ0 reference contours**, not pointwise sample density.
  They are computed by passing Practical Salinity (≈ Absolute Salinity) and
  in-situ temperature (≈ Conservative Temperature) to `gsw.sigma0`. The contour lines
  serve as visual reference guides only. Differences from true TEOS-10 σ0 vary by
  data source, geographic location, depth, and hydrographic conditions.
- The contour grid is bounded by the selected display-axis range and clipped to the
  quality-checked input domain (Salinity 0–50; Temperature −5–45 °C), so negative
  salinity values selected on the axis are not passed to the density calculation.
