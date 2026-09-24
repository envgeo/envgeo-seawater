# Interactive 2D/2.5D Visualizer

## What This Page Does

This page links interactive Plotly scatter plots with sampling-location maps.

## Basic Workflow

1. Select a dataset.
2. Apply sidebar filters.
3. Choose a plot type.
4. Select a color parameter and colormap.
5. Use Box Select or Lasso Select to highlight points.
6. Inspect the corresponding sampling locations on the map.

## Main Controls

- **Plot type**
- **Color parameter**
- **Colormap**
- **Density contour interval (approx. σ0)** — spacing between approximate σ0
  reference contour lines on the T–S diagram (0.2, 0.5, or 1.0 kg m⁻³;
  default 1.0 kg m⁻³). Available only in the Temperature–Salinity view.
- **Show density contours** — show or hide the approximate σ0 reference lines
- **Regression line** for salinity-d18O view
- **Map controls / Map style**

## Outputs

- Temperature-Salinity interactive plot
- Salinity-d18O interactive plot
- Sampling location map
- Filtered dataset table
- Box/Lasso-selected dataset table

## Notes And Limitations

- The regression line is intended for quick visual reference.
- Box/Lasso selection depends on Plotly interaction behavior.
- Very dense selections may be slower on older computers.
- The T–S diagram density contour lines are **approximate σ0 reference
  contours**, not pointwise sample density. They are computed by passing
  Practical Salinity (≈ Absolute Salinity) and in-situ temperature
  (≈ Conservative Temperature) to `gsw.sigma0`. The contour lines serve as
  visual reference guides only. Differences from true TEOS-10 σ0 vary by
  data source, geographic location, depth, and hydrographic conditions.
- The contour grid is bounded by the displayed axis range and clipped to the
  quality-checked input domain (Salinity 0–50; Temperature −5–45 °C).
