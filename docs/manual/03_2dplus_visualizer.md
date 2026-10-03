# Interactive 2Dplus Visualizer

## What This Page Does

This page links interactive Plotly relationship plots with sampling-location
maps for selected reference datasets.

## Basic Workflow

1. Select a reference data source and set the sidebar filters.
2. Choose a plot type: Temperature--Salinity, δD--δ18O, custom 2D/2.5D, or
   Salinity--δ18O.
3. Select a color parameter and colormap where those controls are available.
4. Select **Apply settings** after changing the filters.
5. Use Box Select or Lasso Select on an interactive scatter plot.
6. Inspect the corresponding sampling locations and selected-data table.

## Main Controls

- **Plot type** — Temperature--Salinity, δD--δ18O, custom 2D/2.5D beta, or
  Salinity--δ18O
- **Color parameter**
- **Colormap**
- **Density contour interval (approx. σ0)** — spacing between approximate σ0
  reference contour lines on the T–S diagram (0.2, 0.5, or 1.0 kg m⁻³;
  default 1.0 kg m⁻³). Available only in the Temperature–Salinity view.
- **Show density contours** — show or hide the approximate σ0 reference lines
- **Regression line** for salinity-d18O view
- **Map controls / Map style**

## Outputs

- Temperature--Salinity interactive plot
- δD--δ18O interactive plot
- Custom 2D/2.5D plot beta
- Salinity--δ18O interactive plot
- Sampling location map
- Filtered dataset table
- Box/Lasso-selected dataset table

## Notes And Limitations

- The regression line is intended for quick visual reference.
- Box/Lasso selection depends on Plotly interaction behavior.
- Very dense selections may be slower on older computers.
- This page does not accept browser uploads. Use **User Data Check & Quick
  Visualizer** or a page with an **Uploaded data overlay** for session-only
  uploaded data.
- The T–S diagram density contour lines are **approximate σ0 reference
  contours**, not pointwise sample density. They are computed by passing
  Practical Salinity (≈ Absolute Salinity) and in-situ temperature
  (≈ Conservative Temperature) to `gsw.sigma0`. The contour lines serve as
  visual reference guides only. Differences from true TEOS-10 σ0 vary by
  data source, geographic location, depth, and hydrographic conditions.
- The contour grid is bounded by the displayed axis range and clipped to the
  quality-checked input domain (Salinity 0–50; Temperature −5–45 °C).
