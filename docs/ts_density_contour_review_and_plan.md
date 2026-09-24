# T-S Density Contours — Review and Implementation Plan

**Status:** design record; no implementation change is made by this document.  
**Recorded:** 2026-09-24  
**Scope:** `pages/34_T-S_diagram.py` and the data fields used by its T-S plot.

## Why this record exists

The T-S diagram is an exploratory visualisation.  A change to the definition
or calculation of its density contours would change a scientific quantity, so
the current behaviour, audit evidence, and proposed migration are recorded
before making code changes.

## Current implementation

The page currently builds a salinity-temperature grid and evaluates:

```python
sigma_theta = gsw.sigma0(Sg, Tg)
```

where `Sg` is derived from the `Salinity` column and `Tg` from
`Temperature_degC`. The page itself labels contour lines numerically and its
dedicated manual uses the general phrase “density contours.” However, the
repository README files have described these contours as `σθ`. Stage 1 must
therefore correct the README wording and add an explicit explanation to the
page and its dedicated manuals.

This is not a TEOS-10-consistent call.  `gsw.sigma0` calculates the potential
density anomaly referenced to 0 dbar (`σ0`) and expects Absolute Salinity
(`SA`) and Conservative Temperature (`CT`), rather than Practical Salinity
(`SP`) and in-situ temperature (`t`).  Therefore the present contours must be
treated as approximate reference contours, not as calculated sample density.

## Audit evidence (2026-09-24)

The five public source workbooks loaded by `envgeo_utils.load_isotope_data()`
were inspected read-only.  For all rows with usable temperature and salinity,
the relevant location and depth fields are available, subject to the existing
quality-cleaning rules:

| Source workbook | Rows with usable salinity / temperature | Latitude, longitude, and depth availability |
| --- | ---: | --- |
| `01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx` | 2,222 / 2,221 | Complete |
| `11_AROUND_JAPAN_PUB_20260305.xlsx` | 419 / 416 | Complete |
| `71_GLOBA_NASA_20260226.xlsx` | 23,246 / 20,605 | Complete |
| `71_GLOBAL_Atwood_et_al_2026.xlsx` | 16,098 / 13,871 | Complete |
| `72_GLOBAL_RECENT_REPORTS_20260302.xlsx` | 35 / 35 | Complete |

For rows in a conservative physical screening range, comparing the current
numeric approximation against a pointwise TEOS-10 calculation gave median
`σ0` differences of approximately +0.11 to +0.13 kg m^-3 by workbook, with
absolute differences as large as about 0.43 kg m^-3 in the global sources.
This comparison is an implementation-priority indicator, not a replacement
for dataset-specific scientific validation.

The source loader already converts values outside its present temperature,
salinity, and depth validity ranges to `NaN` and preserves a quality flag.  It
does not, however, establish that every generic `Salinity` field is confirmed
to be Practical Salinity on PSS-78, nor that every temperature is explicitly
documented as ITS-90 in-situ temperature.  That provenance decision remains
necessary before enabling a TEOS-10 calculation for an arbitrary user table.

## Scientific interpretation

`σ0` and `σθ` must not be used interchangeably in UI text.  The former is the
potential density anomaly referenced to 0 dbar in the TEOS-10 GSW function.
The latter convention should only be used if its exact definition and
calculation are implemented and documented.

There is no single location- and pressure-independent, exact TEOS-10 contour
surface on a plot whose axes are raw SP and in-situ temperature.  Geographic
location and pressure enter the conversion from SP/t to SA/CT.  Consequently,
pointwise TEOS-10 density and a two-dimensional background reference-contour
layer should be presented as distinct concepts.

## Recommended staged work

### Stage 1 — transparent current behaviour

- Change UI/manual/README wording from `σθ` to an explicit label such as
  “Approximate σ0 reference contours (SP≈SA; in-situ temperature≈CT)”.
- Explain that the lines are visual reference guides, not pointwise sample
  densities.
- Generate the contour grid from the displayed axis limits and constrain it
  to the documented valid domain, rather than silently extending observed
  salinity and temperature by a fixed amount.
- Add a regression test for the visible wording and contour-grid domain.

#### Stage 1 acceptance conditions

- The T-S page displays an approximation notice adjacent to the figure.
- The contour grid is bounded by the selected displayed salinity and
  temperature axes, rather than by data extrema plus a fixed extension.
- English and Japanese README/manual text states that the layer is an
  approximate `σ0` reference grid, not pointwise sample density.
- At least one focused test asserts the grid-bound calculation and the
  approximation wording. Existing page/upload behaviour remains covered by
  the full suite.

### Stage 2 — testable TEOS-10 data transformation

- Implement a pure, documented helper that accepts confirmed SP, in-situ
  temperature, depth, latitude, and longitude.
- Calculate pressure with `gsw.p_from_z(-depth_m, latitude_degN)`, then
  `SA = gsw.SA_from_SP(SP, p, lon, lat)` and
  `CT = gsw.CT_from_t(SA, t, p)`.
- Return explicit eligibility/status information; never fabricate missing
  depth or coordinates.
- Add fixed-input reference tests based on GSW/TEOS-10 examples, plus tests
  for missing fields, invalid values, and quality-flag preservation.
- Keep the derived values distinct from source columns in tables/downloads
  until their provenance and presentation policy is reviewed.

### Stage 3 — explicit TEOS-10 visual mode

- Offer an opt-in **SA–CT (TEOS-10)** diagram for eligible rows.  In that
  coordinate system, `gsw.sigma0(SA, CT)` contours have an unambiguous
  meaning.
- Retain the familiar SP–in-situ-temperature view for exploratory comparison
  and incomplete uploads, with its approximation label.
- Require an explicit declaration that a user-data salinity field is Practical
  Salinity (PSS-78) before the TEOS-10 mode is available for it.
- Report counts excluded from TEOS-10 conversion and the reason for exclusion.

## Acceptance conditions before release

- A domain reviewer confirms the source-field interpretation for each bundled
  dataset.
- GSW dependency/version and TEOS-10 citation are recorded in release
  documentation.
- Tests verify numerical reference values and do not merely test that a plot
  renders.
- UI, manuals, downloads, and citations distinguish source measurements from
  derived TEOS-10 quantities.
- The release notes state whether the current approximate layer remains,
  changes label only, or is superseded by the SA–CT mode.

## References

- TEOS-10 / GSW: [gsw_sigma0](https://teos-10.org/pubs/gsw/html/gsw_sigma0.html)
  (SA and CT inputs; reference pressure 0 dbar).
- TEOS-10 / GSW: [gsw_CT_from_t](https://www.teos-10.org/pubs/gsw/html/gsw_CT_from_t.html)
  (CT from SA, in-situ temperature, and pressure).
- IOC, SCOR and IAPSO (2010), *The international thermodynamic equation of
  seawater – 2010*, Manuals and Guides No. 56.
