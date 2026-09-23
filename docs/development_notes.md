# EnvGeo-Seawater Development Notes — 1.3.3

Internal handoff notes. These are not a user manual or a release claim.

## Current Boundary

- Version 1.3.3 is the local release-preparation point as of 2026-09-23.
- Plotly 5.24 remains the verified plotting baseline; Streamlit 1.63 is tested
  through the Python 3.12 compatibility environment.
- Page 05, **User Data Check & Quick Visualizer**, is the public upload-first
  quality-check and simple-visualization page. Page 90 remains transitional.

## AI Disclosure — scope, basis, and pre-JOSS TODO (2026-09-22)

Public-facing disclosure of AI-assisted development was added in this round,
across `data_text/about.md`, `data_text/japanese.md` (rendered on Home's
About/Japanese tabs), `README.md`, `README_Japanese.md`,
`data_text/update_log.md`, `data_text/update_log_Japanese.md`, and
`paper.md`'s existing "AI Usage Disclosure" section (which previously named
only "ChatGPT, OpenAI" and lacked scope/version detail).

**Why version 1.3 is the stated boundary:** `envgeo_utils.APP_VERSION`
history and `home.py`'s `render_update_history()` place 1.3.0 at
2026-09-11. Searching the repository for existing AI-tool mentions before
that point found none; the earliest in-repo evidence of AI-assisted work
is inside `data_text/update_log.md`'s 1.3.2 development log ("Claude
review" entries) and the `Claude outputs/` session logs (dated
2026-09-21), both after the 1.3.0 boundary. The disclosure text says
"from version 1.3 onward... substantial use" without asserting "zero AI
use before 1.3" as a hard, provable claim — this is a deliberate choice to
stay defensible; it is accurate that no earlier disclosure or session
record exists in this repository, but the absence of a record is not
positive proof of absence of any AI use.

**Basis for naming specific tools:** "OpenAI Codex" and "Anthropic Claude
Code" are the tool names already used consistently in-repo — e.g. the
`53_Vertical_Section_Visualizer.py` header ("Developed and improved with
assistance from Codex"), `update_log.md`'s "Claude review" entries, and
the `Claude outputs/` directory. These were treated as confirmed, not
inferred.

**Model/version specificity — what is confirmed vs. not:** grepping the
repository for model-version strings found exactly one: "Claude Sonnet
4.6" in `Claude outputs/handoff_to_codex.md` and
`Claude outputs/worklog_upload_overlay_rollout.md` (dated 2026-09-21). No
Codex model/version string (e.g. a specific GPT/Codex model identifier) is
recorded anywhere in the repository. Per the task instructions for this
round, unconfirmed model/version details were not guessed; `paper.md`
names the one confirmed instance (Claude Sonnet 4.6) and explicitly flags
that per-session model versions were not consistently tracked, leaving a
literal "TODO" in the disclosure text itself for pre-submission
follow-up.

**Pre-JOSS-submission TODO:** before submitting/updating the JOSS paper,
reconfirm and, if possible, enumerate the specific AI tool versions used
across the development history (Codex model identifier(s); each Claude
Code session's model, not only the one already recorded), and update
`paper.md`'s disclosure accordingly. Do not backfill guessed version
numbers into this or any other file in the meantime.

## User-Data Architecture

- Uploads are session-memory only and never written by the app.
- Pages 31, 32, 34, 35, 37, and 53 expose `Uploaded data` in common Data
  filtering. Uploaded markers remain foreground traces.
- Page 53 combines selected valid uploaded rows with selected reference rows
  for its local section calculation. Its interpolation needs a separate
  scientific/algorithmic review before further feature work.
- Pages 03 and 04 are intentionally deferred.

## Always-loaded local user table

For repeated local work, use the Git-ignored `local_data/user_data.xlsx` path
or set `ENVGEO_LOCAL_USER_DATA_PATH`. The common loader assigns these rows the
dataset name `User Excel data` and appends them to every selected reference
source. The same preparation and quality checks as browser uploads are applied.
Browser `Uploaded data` remain a separate session-only category, preventing
the local workbook from being inserted twice.

## Post-v1.3.3 scientific and release-readiness review (recorded 2026-09-23)

This is a follow-up backlog, not a change to the v1.3.3 scope or a claim that
the current exploratory workflows are invalid. Verify each item against data
provenance and intended use before implementation.

- **T-S density contours:** audit the meaning and availability of `Salinity`,
  `Temperature_degC`, pressure, latitude, and longitude in every supported
  dataset. `gsw.sigma0` requires Absolute Salinity and Conservative
  Temperature. Either make and test the TEOS-10 conversions (`SA_from_SP`,
  `CT_from_t`) where justified, or label the existing contours explicitly as
  an approximation. Do not silently change a scientific quantity.
- **Scientific QC and provenance:** extend the current range-based safety
  checks only after documenting dataset-specific missing-value codes, isotope
  precision/standard scale, coordinates, dates, source identifiers, and the
  distinction between duplicate records and repeated observations.
- **Exploratory findings:** describe map interpolation, regressions, water-mass
  attribution, and d-excess patterns as exploratory hypotheses until their
  uncertainty, non-independence by cruise/station/profile, and measurement
  error are evaluated. Candidate methods include grouped bootstrap and
  error-aware regression; select methods with domain review rather than by
  default.
- **Testable scientific core:** move selected numerical transformations and
  scientific rules into pure functions with fixed-input reference tests before
  treating them as reproducible analysis claims.
- **Publication engineering:** after the v1.3.3 compatibility suite is green
  in supported environments, complete packaging, CI, citation metadata,
  tagged release, archival DOI, and machine-readable dataset provenance before
  making a JOSS submission claim.

The JOSS narrative should centre on the stable data-discovery, filtering,
quality-review, user-data comparison, and visualisation workflow. Keep
Vertical Section beta and unvalidated scientific interpretation outside the
central claim until separately validated.

## Vertical Section — algorithm review and fixes (2026-09-22)

Page 53's projection/interpolation core was audited and fixed. Why each
change exists, so the reasoning survives past this session:

- **Corridor gate was measuring the wrong distance.** `project_points_to_polyline`
  filtered on the perpendicular distance to the *infinite* line through A-B,
  not the true distance to the *finite* segment. A point past A or B could
  look "close" to the infinite line while sitting far outside the intended
  corridor, and leak into the section. Fixed to gate on the true
  nearest-point distance to the segment instead. Regression-tested with a
  synthetic point 5 km past B with a 1 km perpendicular offset.
- **Uploaded bathymetry could crash the whole page.** `griddata(method="linear")`
  raises `QhullError` on collinear point layouts (a very plausible upload:
  soundings taken along the transect itself). This call had no fallback,
  unlike the main target-parameter interpolation, which already degrades
  cubic → linear → nearest safely. Added the same fallback.
- **GEBCO land cells were reported as 0 m depth, not "no data".** Any
  transect grazing an island/coastline at built-in-GEBCO resolution got a
  false seafloor at the surface there, masking real observations in that
  column. Land cells (height ≥ 0) now return `NaN`.
- **The 3,000-row safety cap was gating the wrong count.** It compared the
  row count *before* corridor projection, so a wide Data-filtering selection
  with a narrow corridor could be blocked even though the actual
  interpolation input would be tiny — and, in the other direction, a wide
  corridor could still feed griddata far more points than the cap intended
  once the user raised the number input. The cap now gates on the row count
  *after* corridor projection (the number that actually reaches `griddata`);
  the pre-projection count still separately gates the interactive
  "Draw on map" widget, which is a real cost of its own (many map markers).
- **`gaussian_filter` on the seafloor-masked grid corrupted valid data.**
  scipy's Gaussian filter propagates NaN into every cell within the kernel
  radius, silently erasing correct values within a few grid cells of the
  seafloor mask. Replaced with a NaN-aware normalized-convolution smoother
  local to this page.
- **Added lightweight, ODV-adjacent visual cues** without new heavy
  dependencies: a translucent overlay marking cells that only
  nearest-neighbor extrapolation could fill (cubic/linear both failed
  there), and a single combined trace connecting samples from the same
  station/cast. Both reuse arrays the pipeline already computes.
- Also fixed a latent divide-by-zero risk in `densify_section_line` near the
  poles (already guarded elsewhere in the file, just missing here).

Added `test/test_vertical_section_visualizer.py` (17 tests) covering all of
the above at both the function level and as AppTest end-to-end checks. Full
suite: 122 passed, 4 skipped (pre-existing, cartopy-related).

Not done, and worth a future pass if this page gets more attention: this
change deliberately kept the confidence overlay and profile lines simple
(single overlay trace, single combined line trace) to stay cheap; going
further toward ODV (e.g. an actual alpha-shape data-support boundary,
duplicate-coordinate flags surfaced in the UI, negative-depth validation)
is still open and was intentionally left out of this pass.

## Data filtering — Bering Sea dateline bug (2026-09-22)

The "Bering Sea" area-filter preset let North Atlantic points through at
the same latitude. Root cause: its longitude span crosses the antimeridian
(160E..-162W), but `sidebar_filter_and_display` applies longitude via a
single `st.slider` range, which cannot express a wrap-around span —
`area_filter_bounds()` silently fell back to the full data longitude range
for such presets, so only the latitude slider was actually narrowing
anything.

Fix: added `normalize_longitude_deg()` (canonical -180..180, works
regardless of whether the data uses -180..180 or 0..360),
`region_preset_crosses_dateline()`, and `region_preset_longitude_mask()`
(OR condition: east arm OR west arm) to `envgeo_utils.py`. Non-crossing
presets are untouched — they still use the slider values exactly as
before. Only presets with `lon_max > 180` in `MAP_REGION_PRESETS` (Bering
Sea, North Pacific, Tropical/Equatorial/South Pacific, Southern Ocean -
Pacific sector) now bypass the slider and use the OR mask directly; this
generic mechanism means the same bug is fixed for all of them, not just
Bering Sea. Bering Sea's stored bounds were also corrected to match the
requested extent: `(160.0, 198.0, 51.0, 66.0)` (198.0 encodes -162.0).

Tests: `test/test_envgeo_utils.py` (pure-function checks in both longitude
conventions) and one AppTest in `test/test_uploaded_page_overlays.py`
(end-to-end, via page 53) confirming Bering Sea points survive and
North-America/North-Atlantic points at the same latitude are excluded.

## Vertical Section — follow-up fixes (2026-09-22, later same day)

Three remaining items from the review above:

- **Profile lines could join unrelated casts.** Same rounded lon/lat but a
  different `Dataset` (e.g. reference vs. an uploaded point) or a different
  observation date used to be drawn as one continuous line. **Update, same
  day, third pass:** the first fix (folding in any *existing* identifier
  column) was still too loose — a column existing but empty everywhere
  still "counted", and `Dataset`/`Year` alone were treated as sufficient,
  so a same-`Dataset`/same-`Year`/same-coordinate *revisit* was still
  wrongly joined. Replaced with an explicit identifiability test:
  `Station` alone is sufficient; otherwise a real combination is required
  (`Cruise`+`Date`, `Cruise`+`Year`+`Month`, `Transect`+`Date`, or
  `Transect`+`Year`+`Month`); `Dataset`/`Year`/`Month`/`Day` alone are
  never sufficient. Rows failing every rule are dropped before grouping,
  not left to form a "nan"-keyed group. Known residual gap (unchanged,
  documented, not silently patched): two *anonymous* casts with none of
  those columns at the same coordinates still can't be told apart.
- **The 8,000-row interpolation cap is now non-bypassable.** Added
  `MAX_ROWS_FOR_SECTION_INTERPOLATION_HARD_CAP = 8000`; the "Max valid
  rows" number input's own max is set to it, and the actual gate clamps to
  it independently of whatever the widget reports. Reasoning (local
  `griddata` benchmarking vs. Streamlit Cloud's shared, per-interaction
  full-script-rerun cost) is in a code comment and in the review log.
- **Station map now shows the section corridor.** A new "Show section
  corridor" checkbox (A-B mode only, default on) draws a translucent
  capsule-shaped band matching `project_points_to_polyline`'s own
  `best_dist <= corridor_km` test — finite past A/B (not an infinite
  strip), one capsule per polyline segment for drawn multi-vertex lines,
  and safely dropped (not drawn) rather than rendered giant if a segment's
  longitude would wrap across the antimeridian.

Full detail (reproduction, geometry derivation, benchmarks, test list) is
in `vertical_section_review_log_2026-09-22.md`, not repeated here.

**Antimeridian sections are explicitly not scientifically validated.** The
corridor-band seam guard above only hides a giant polygon; it says nothing
about the section calculation itself, which still uses a single
flat-tangent-plane approximation
(`project_points_to_polyline`/`build_section_polyline`, unchanged). Added
`section_crosses_antimeridian()` and a matching `st.warning` in `main()`
that fires whenever the active A-B line directly crosses the antimeridian,
regardless of the corridor checkbox — it says results are "not
scientifically validated ... treat them as a rough visual reference only,"
not merely "corridor hidden." Do not remove this warning without first
implementing the redesign in the next bullet.

Tests for both this and the profile-line fix above:
`test/test_vertical_section_visualizer.py`, 39 passed; full suite 149
passed, 4 skipped (pre-existing).

## Next Safe Work Items

1. **Recorded future task, not started:** a scientifically supportable
   antimeridian-crossing A-B section needs either (a) longitude unwrapping
   carried consistently through `lonlat_to_local_km` /
   `project_points_to_polyline` / `densify_section_line` / the corridor-band
   builder, or (b) a real geodesic (great-circle) projection replacing the
   flat-tangent-plane approximation entirely (which would also improve
   accuracy for long non-dateline sections). Plan as its own task with its
   own review — it touches the section-extraction core that the last two
   rounds deliberately left alone.
2. Continue hardening Vertical Section as needed (duplicate-coordinate and
   negative-depth flags, anonymous-cast profile-line disambiguation,
   further ODV-likeness) — the core algorithmic audit is done; remaining
   items are polish, not correctness fixes.
3. Complete visual checks in the Streamlit 1.63 / Plotly 5.24 environment.
4. Plan the MapLibre migration separately from scientific workflow changes.
5. Decide the eventual public/archive status of page 90 only after page 05 and
   specialist-page behavior is confirmed.
6. Plan an `envgeo-core` coastline component: package the 50m/110m CSV assets,
   path-safe cached loading, resolution validation, and tests. Keep each app's
   local CSV copy until Seawater and Earthquake have migrated and been checked
   independently; only then remove duplicated assets.

## Vertical Section — release and JOSS positioning (2026-09-23)

Keep page 53 as `Vertical Section Visualizer beta`: an advanced, experimental
workflow available in the application, rather than a core JOSS feature. It
must remain visibly labelled as beta and retain its documented scientific and
offline-input limitations. Until its section geometry, validation scope, and
offline mouse-input design mature, exclude it from the paper's central claims,
representative figures, and validation examples. The JOSS narrative should
instead focus on the stable isotope/hydrographic data, filtering, quality
review, upload, and visualization workflows.

## Natural Earth 50m Land Shapefile — Bundling (2026-09-23)

Added `coastline/natural_earth_50m_land/` containing:

- `ne_50m_land.shp` (1.1 MB)
- `ne_50m_land.shx` (12 KB)
- `ne_50m_land.dbf` (72 KB)
- `ne_50m_land.prj` (4 KB)
- `ne_50m_land.cpg` (<1 KB)
- `LICENSE_OR_SOURCE.md` — source URL, SHA-256 hashes, public domain licence

Source: `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/` (retrieved 2026-09-23). Natural Earth data is public domain — "Made with Natural Earth."

**Why:** `cartopy.feature.LAND` (which used `cfeature.NaturalEarthFeature`) triggers
Cartopy's downloader on first use in a fresh cache environment. This caused network
access even in "offline" deployments. `page 32 – Isotope Hydrographic Mapping` now
reads the shapefile with `cartopy.io.shapereader.Reader(local_path)` and calls
`ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor="white", zorder=2)`.
No call to `natural_earth()`, `NaturalEarthFeature`, or Cartopy's downloader is made.

**Degradation path:** If any of `.shp`, `.shx`, `.dbf` is missing, `_load_ne50m_land_geometries()` emits `st.warning` (English) and returns an empty list — coastlines and data points still render; only the land fill is omitted. No crash, no network fallback.

Drawing order maintained: contours (z1) → land mask (z2) → coastlines (z3) → gridlines (z4) → data points (z5+).

New tests: `test/test_natural_earth_land.py` (15 tests — presence, readability without network, helper degradation).

---

## Future Asset Reorganization — Packaging TODO (2026-09-23)

When moving to a Python package (pyproject.toml / package data), reorganize bundled
geographic and scientific assets. Do not reorganize yet — design a backward-compatible
asset loader first.

Candidate future layout:
```
assets/
  geospatial/   ← coastline CSVs + Natural Earth land shapefile
  bathymetry/   ← GEBCO data (currently data_beta/)
  metadata/     ← source records, licences, checksums
```

Current locations to keep until migration is ready:
- `coastline/` — 50m/110m CSV coastlines + `natural_earth_50m_land/` (static map land mask)
- `data_beta/` — GEBCO (bathymetry / section analysis, page 53)

Migration requirements:
- Asset loader must resolve relative to the installed package, not individual page `__file__`.
- Backward-compatible fallback for existing local checkouts without a package install.
- Move `LICENSE_OR_SOURCE.md` files into `assets/metadata/` when reorganizing.
- Verify Streamlit Cloud and local Conda environments find bundled files after migration.

## Page 32 contour spatial support — recorded follow-up (2026-09-23)

The bundled land mask fixes land leakage; it does not validate the scientific
support of the ocean interpolation. Page 32 currently uses linear SciPy
`griddata`. Within the observation convex hull, sparse stations can form very
large Delaunay triangles, so values may visually span poorly constrained areas
such as parts of the tropical Pacific. Treat this as a scientific display issue,
not an offline-map failure. Evaluate masking grid cells whose nearest
observation exceeds a documented distance threshold, showing them as not
interpolated. First establish and test a conservative default with a clear
legend/caption; only then consider a user-selectable interpolation method or
threshold, with explicit support and caveat information for every option.
