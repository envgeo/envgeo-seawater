# Vertical Section Visualizer review log — 2026-09-22

## Context

The Vertical Section Visualizer implementation was delegated to Claude Code after a
read-only audit. This log records review findings to check against its completion
report and diff.

## Confirmed issue: Bering Sea data filtering

`MAP_REGION_PRESETS["Bering Sea"]` spans the international date line. The current
shared area-filter path cannot represent that geography with its single continuous
longitude slider. When the preset is converted to the `-180..180` longitude frame,
the code identifies it as dateline-crossing and returns the full dataset longitude
range. The subsequent filter is therefore effectively latitude-only, allowing North
Atlantic records at the same latitude into the Bering Sea selection.

Required behaviour:

- Apply a date-line-aware OR predicate for the Bering Sea preset.
- Support both `-180..180` and `0..360` source longitude conventions.
- Keep conventional, non-date-line-crossing presets and existing manual slider
  behaviour unchanged.
- Add regression tests: 160E, 179E, -179W, and -162W should remain; -150W and
  -30W at matching latitude should be excluded (with the final chosen boundary and
  any documented display buffer applied consistently).

## Vertical-section implementation checks

- Corridor membership must use finite-polyline distance (`best_dist`) or an
  equivalent end-cap rule; a point on an extension beyond A/B must not be included
  merely because its infinite-line cross-track distance is small.
- Uploaded bathymetry interpolation must safely handle degenerate point geometry.
- GEBCO land elevations must not become a zero-metre water-column bottom.
- The section cost limit needs a post-corridor guard as well as any early guard.
- Handle high-latitude longitude conversion safely.
- Preserve existing filter, upload, and manual/drawn A-B workflows.
- Treat `gaussian_filter` over NaNs deliberately; verify that the seafloor mask
  does not cause unintended NaN propagation.
- For ODV-like usability, distinguish observed samples from nearest-neighbour
  extrapolated regions without adding costly repeated interpolation.

## Review evidence to request on completion

- Changed-file list and concise design rationale.
- pytest / AppTest command output.
- Evidence for date-line regression cases and finite-section corridor endpoint
  cases.
- Any retained limitations (e.g. local planar approximation at high latitude).

## Review of the reported Vertical Section implementation

Verified in the working tree:

- Finite-segment corridor gating uses `best_dist`.
- Uploaded-bathymetry interpolation catches degenerate linear interpolation.
- GEBCO land cells are returned as `NaN`.
- The interpolation limit is checked after corridor projection.
- NaN-aware normalized Gaussian smoothing and an extrapolation overlay are wired
  into both plot modes.
- The new page-level regression suite passes locally: `17 passed`.

Follow-up items before treating the broader work as complete:

- The shared Bering Sea date-line filtering defect remains in `envgeo_utils.py`
  at this review point; the preset still expands to the dataset's full longitude
  span when its endpoints cross the date line.
- Profile lines must not equate rounded longitude/latitude alone with one CTD
  cast. When reference and uploaded datasets are combined, repeated locations
  across cruises/dates can be incorrectly joined. Prefer a stable cast/station
  identity combined with source/cruise/date, and otherwise omit the joining
  line.
- The post-corridor limit is still user-adjustable up to 50,000 rows. Consider
  a non-bypassable internal ceiling appropriate to the selected grid resolution,
  or reduce the permitted maximum, to retain a Cloud-safe upper bound.

## Review of the Bering Sea / antimeridian filtering implementation

Verified in the working tree:

- The Bering Sea bounds are now `160E..198E` (`198E == 162W`) and
  `51N..66N`.
- `normalize_longitude_deg`, `region_preset_crosses_dateline`, and
  `region_preset_longitude_mask` normalize source longitudes and apply an
  east-arm OR west-arm predicate to date-line-crossing presets.
- The shared sidebar filter uses that predicate instead of the unrepresentable
  single longitude-slider interval for such presets; the normal slider path
  remains in use for non-crossing presets.
- The mechanism is intentionally shared by all presets stored with an eastward
  endpoint beyond 180 degrees, including the Pacific-sector presets.
- Focused local verification passed:
  - `pytest -q test/test_envgeo_utils.py -k 'normalize_longitude or region_preset_crosses or bering_sea_longitude'`: `4 passed`.
  - `pytest -q test/test_uploaded_page_overlays.py -k 'bering or dateline or antimeridian'`: `1 passed`.

Documentation policy:

- Keep `docs/development_notes*.md` concise: issue, design decision, affected
  behavior, and test result.
- Keep this review log detailed: reproduction conditions, predicates, filenames,
  verification commands, and residual limitations. This makes later maintenance
  and regression investigation possible without turning user-facing development
  notes into an implementation diary.

## Review of the corridor-band, profile-key, and row-cap implementation

Verified in the working tree:

- The post-corridor interpolation cap is now non-bypassable at 8,000 rows,
  while the default remains 3,000.
- The Station map's optional corridor overlay is drawn behind selected points,
  the centreline, and the endpoint markers; its finite segment capsules match
  the existing `best_dist <= corridor_km` selection rule for ordinary lines.
- The A-B-only checkbox is present and the Axis-based path does not request a
  corridor overlay.
- The page-specific suite was rerun locally: `30 passed`.

Residual review items:

- The anti-meridian corridor overlay deliberately drops a band with a seam.
  It prevents a world-spanning polygon but does not make a date-line-crossing
  section usable; the underlying local planar section geometry still needs an
  eventual longitude-unwrapping/geodesic design before Bering Sea A-B sections
  can be considered scientifically supported.
- `build_station_profile_trace` currently treats the presence of any identifier
  column as sufficient and stringifies missing values. A column such as
  `Dataset` or `Year` alone is not necessarily a cast identity, and rows with
  missing identifiers can collapse to the same string key. Before treating
  profile lines as reliable, require non-missing per-row cast identity (for
  example Station plus source/cruise/date, or an explicit cast ID); otherwise
  skip that group. Add regression tests for an all-missing identifier column
  and for same-dataset, same-year repeat casts at one coordinate.

## Follow-up implementation — cast-ID profile lines, hard interpolation cap,
## Station-map corridor band (2026-09-22, later same day)

Addresses the three remaining follow-up items above. All changes are in
`pages/53_Vertical_Section_Visualizer.py` unless noted; the Bering Sea /
antimeridian fix from the previous section is untouched and still exercised
by its own tests (re-run below, all green).

### 1. Profile-line cast identity

Reproduction of the prior bug: two rows sharing the same rounded
longitude/latitude but differing `Dataset` (e.g. a reference station and an
uploaded point that coincidentally land on the same coordinates) or differing
`Year` were grouped into one `groupby(station_key)` bucket by
`build_station_profile_trace`, because the old `station_key` was built from
`Longitude_degE`/`Latitude_degN` alone. That produced a single connected
line jumping between what are really two separate casts.

Fix: `STATION_PROFILE_ID_COLUMNS = ["Dataset", "Station", "Cruise",
"Transect", "Date", "Year", "Month", "Day"]`. `build_station_profile_trace`
appends `.astype(str)` of every one of these columns that is present in
`df_points` onto the lon/lat key. If none of them are present at all, the
function returns `None` rather than guessing — this is the "safe side" the
task asked for; it only matters for a stripped-down/synthetic frame, since
both `envgeo_utils.prepare_uploaded_data` and the reference loader always
set `Dataset`.

Still an accepted, documented gap: two *anonymous* uploaded casts
(no Station/Cruise/Transect/date at all, both `Dataset == "Uploaded data"`)
that coincidentally share the same rounded coordinates cannot be told apart
by this scheme and would still be joined. Perfect disambiguation would need
a caller-supplied cast/profile id, which no page currently produces; flagged
here rather than silently fixed, since it is out of scope for this pass.

The "single trace, `None`-separated" rendering approach (from the earlier
profile-line addition) is unchanged — only the grouping key changed.

Tests (`test/test_vertical_section_visualizer.py`):
- `test_build_station_profile_trace_returns_none_without_any_identifier_column`
- `test_build_station_profile_trace_does_not_join_different_datasets_at_same_coordinates`
- `test_build_station_profile_trace_does_not_join_different_dates_at_same_coordinates`
- Updated the two pre-existing profile-line tests to include a `Dataset`
  column (now required for a non-`None` result).

### 2. Non-bypassable interpolation row cap

`MAX_ROWS_FOR_SECTION_INTERPOLATION_HARD_CAP = 8000`, defined next to
`DEFAULT_MAX_ROWS_FOR_SECTION_PLOT` (still 3000, unchanged). Rationale
recorded both in a code comment at the constant and here: local
benchmarking (`scipy.interpolate.griddata`, `grid_res=180` → 32,400 target
nodes) showed cubic+linear+nearest combined stayed under ~0.3s even at
50,000 post-corridor input points on a modern laptop —

```
n=  1000 cubic=0.01s linear=0.01s nearest=0.01s
n=  8000 cubic=0.04s linear=0.04s nearest=0.02s
n= 20000 cubic=0.10s linear=0.09s nearest=0.02s
n= 50000 cubic=0.30s linear=0.27s nearest=0.03s
```

— but Streamlit Community Cloud's shared, weaker containers re-run the
*entire* script on every widget interaction (not just the interpolation
call), so the realistic cost is this number multiplied by however many
other computations/renders happen in the same rerun. 8,000 was chosen as a
round number comfortably above any corridor-filtered row count seen in
this app's own test fixtures and normal usage, while an order of magnitude
below the point where local timing itself starts growing noticeably.

Implementation: `resolve_section_row_limit(user_value, hard_cap=...)`
returns `min(user_value, hard_cap)`. The "Max valid rows for section
plotting" `st.number_input`'s own `max_value` is now set to the hard-cap
constant directly (so the widget itself cannot ask for more), and the
actual gate right before the `griddata` call uses
`resolve_section_row_limit(max_rows_for_section_plot)` as a second,
independent check — defense in depth in case the effective value is ever
produced some other way (e.g. `session_state` manipulation in a test).
The post-corridor-only gating design from the previous round (checking
`len(df_section)`, not the pre-projection count) is unchanged.

The warning message states the current row count, the effective limit, and
— only when the user's own setting actually exceeded the hard cap — a note
that it was clamped, plus how to narrow the selection (corridor half-width,
A-B section, or Data filtering).

Tests:
- `test_resolve_section_row_limit_clamps_to_hard_cap`
- `test_vertical_section_row_limit_widget_cannot_exceed_hard_cap` (checks
  the AppTest widget's own `proto.max`)
- Pre-existing `test_vertical_section_wide_filter_narrow_corridor_is_not_blocked`
  and `test_vertical_section_post_projection_cap_blocks_when_corridor_stays_wide`
  from the previous round still pass unchanged (both set the row-limit
  input to 100, well under the new 8,000 ceiling).

### 3. Station-map corridor band

Three-tier design, in order of increasing "Streamlit-ness" (kept separate
for testability without AppTest):

1. `build_corridor_capsule_local_km(p0, p1, corridor_km, cap_points=16)` —
   pure geometry in the same local-km space `project_points_to_polyline`
   already uses (`build_section_polyline`'s `xy_km`). Returns a closed
   capsule/stadium polygon: two straight sides offset by `corridor_km`
   perpendicular to the segment, closed by two semicircular end caps
   centered on `p0`/`p1`. This is exactly the boundary of
   `{point : distance-to-finite-segment(p0, p1) <= corridor_km}`, i.e. the
   same shape implied by `project_points_to_polyline`'s `best_dist <=
   corridor_km` test (see the previous Bering-Sea-round fix to that exact
   line). Verified by hand (worked example in the implementation) and by
   test: perpendicular half-width equals `corridor_km` exactly; the
   along-segment extent is bounded at `corridor_km` past each endpoint
   (not infinite).
2. `build_corridor_band_polygons(section_vertices, corridor_km,
   cap_points=16)` — calls `build_section_polyline` (unchanged, existing
   function; no changes to section-extraction logic anywhere in this
   round) and builds one capsule per polyline segment via (1), then
   converts each polygon's local-km coordinates back to lon/lat using the
   *same* formula and pole guard as `densify_section_line`
   (`cos_lat = cos(radians(lat_ref))`, clamped away from 0), reused
   verbatim rather than reimplemented. Multi-segment (Folium-drawn)
   polylines get one polygon per segment; no boolean union is attempted,
   so overlapping bands at a sharp bend are simply drawn on top of each
   other (self-intersection-safe by construction — it is just more
   independent closed loops, which Plotly's `fill="toself"` renders fine
   individually).
   Dateline safety: after the lon/lat conversion, every polygon's
   longitudes are passed through `envgeo_utils.normalize_longitude_deg`
   (same helper as the Bering Sea fix — this is the "既存の...unwrap方針
   と統一する" reuse the task asked for), then checked with
   `polygon_has_dateline_seam` (any consecutive-vertex — including the
   closing edge — longitude jump over 180° is treated as a wrap artifact).
   A polygon that fails this check is dropped rather than drawn. Verified
   directly: an A-B line from 170E to 170W (55N, a Bering-Sea-like
   dateline crossing) currently returns **zero** polygons — the
   *underlying, pre-existing* local-tangent-plane approximation in
   `lonlat_to_local_km`/`build_section_polyline` (unchanged, out of scope
   per the task's explicit "断面抽出ロジック自体は変更しないこと")
   already misplaces such a segment's own endpoints before this feature
   ever runs, so the seam check's job here is specifically to make sure
   that *pre-existing* distortion cannot manifest as a giant rendered
   polygon — it can only manifest as "no band drawn for that segment",
   which is judged an acceptable degradation for an edge case this task
   was not asked to fix at its root.
3. `build_corridor_band_trace(section_vertices, corridor_km,
   cap_points=16)` — Plotly-specific: flattens the polygon list from (2)
   into one `go.Scattermapbox` trace with `None`-separated closed loops
   (mirrors the existing single-trace pattern used for the vertical
   profile lines), `fill="toself"`, light blue fill
   (`rgba(66,135,245,0.18)`), thin blue outline
   (`rgba(30,90,200,0.65)`), legend name
   `f"Section corridor (±{corridor_km:.0f} km)"`.

Wiring: `create_station_map` gained `show_corridor=False, corridor_km=None`
parameters; the corridor trace is added right after the background-points
trace and before the foreground/in-corridor-points trace, matching the
required draw order (background → corridor band → in-corridor points → red
line → A/B markers — the two trailing steps were already in that relative
order). Only the **A-B section** call site in `main()` passes
`show_corridor=show_section_corridor, corridor_km=corridor_km`; the
Axis-based call site is untouched (no corridor concept applies there). A
new `st.checkbox("Show section corridor", value=True)` was added directly
under the existing "Half-width of section corridor (km)" slider, inside
the same `if section_mode == "A-B section":` block that already guards
A-B-only controls — no new width control was introduced; the checkbox
reuses `corridor_km` as-is.

Tests:
- `test_corridor_capsule_local_km_width_matches_corridor_half_width`
- `test_corridor_capsule_local_km_is_finite_not_extended_past_endpoints`
- `test_build_corridor_band_polygons_handles_multi_vertex_polyline`
- `test_build_corridor_band_polygons_avoids_giant_dateline_polygon`
- `test_build_corridor_band_trace_has_expected_legend_label`
- `test_create_station_map_corridor_trace_only_when_requested`
- `test_create_station_map_draw_order_keeps_corridor_behind_points_and_line`
- `test_vertical_section_corridor_trace_only_in_ab_section_mode` (AppTest;
  reads `st.plotly_chart`'s figure JSON via `chart.proto.spec` — AppTest
  has no first-class `.value` accessor for this element type, unlike
  stateful widgets)

### Verification run (this round)

```
pytest -q test/test_vertical_section_visualizer.py   # 30 passed
pytest -q test/                                       # 140 passed, 4 skipped
```

The 4 skips are pre-existing and unrelated (cartopy-gated tests, confirmed
present before this round's changes too).

### Known limitations carried forward

- Anonymous, identifier-less duplicate-coordinate casts in profile lines
  (see §1) — accepted gap, not silently fixed.
- The corridor band (and, more broadly, all of this page's local-km
  section math) uses a single flat-tangent-plane approximation anchored at
  the polyline's first vertex; for an A-B line that genuinely crosses the
  antimeridian, that pre-existing approximation itself becomes distorted
  before the corridor-band code ever runs. This round only guarantees the
  *symptom* (a giant rendered polygon) cannot occur — it does not fix the
  root cause, since doing so would mean changing
  `project_points_to_polyline`/`lonlat_to_local_km`, which the task
  explicitly scoped out ("断面抽出ロジック自体は変更しないこと").
- Multi-segment corridor bands are drawn as independent per-segment
  capsules, not a true polygon union; at a sharp bend the overlap is
  simply double-painted (slightly more opaque), which is visually correct
  but not geometrically a single simple polygon.

## Post-review follow-up — stricter cast identity, antimeridian
## scientific-support caveat (2026-09-22, third pass)

Codex flagged two residual issues after the previous round's implementation
(see "Residual review items" above). Both addressed now, in
`pages/53_Vertical_Section_Visualizer.py`.

### 1. `build_station_profile_trace` cast identity was still too loose

The previous fix folded in whichever of `Dataset/Station/Cruise/Transect/
Date/Year/Month/Day` merely *existed as a column*, stringifying missing
values (`NaN` → the literal string `"nan"`). Two concrete failure modes
remained, both exactly as Codex described:

- A column existing but every value missing for the rows in question
  (e.g. an uploaded frame with a `Station` column that is `NaN`
  everywhere) still counted as "identified" — all such rows shared the
  same `"nan"` key fragment and could be wrongly joined.
- `Dataset` alone, or `Year` alone, is not real evidence of "same cast":
  a same-`Dataset`, same-`Year`, same-coordinate *revisit* (a genuinely
  different cast) was previously joined into one line.

Fix — replaced the old "any existing column" rule with an explicit
identifiability test, `_station_profile_identifiable_mask`:

- **Sufficient alone:** `Station` present (non-missing) for that row.
- **Sufficient combinations** (all columns in the tuple present for that
  row): `(Cruise, Date)`, `(Cruise, Year, Month)`, `(Transect, Date)`,
  `(Transect, Year, Month)`.
- **Never sufficient alone:** `Dataset`, `Year`, `Month`, `Day` — these are
  only folded into the grouping key *after* a row already passed one of
  the rules above, as extra precision, never as the sole basis for
  identification.
- "Present" is column-type-aware: numeric/datetime columns treat only
  `NaN`/`NaT` as missing; string columns also treat an empty or
  whitespace-only value as missing (`_station_profile_column_present_mask`).

Rows that fail every rule are dropped from consideration *before*
grouping (`df_points = df_points.loc[identifiable]`), not merely left to
form their own (still-connectable) group — so a frame where nothing is
identifiable now returns `None` outright, matching the letter of "判別不能
な行群は線を描かないでください".

Verified this actually changes behavior versus the previous round, using
the exact scenario Codex named: same `Dataset`, same `Year`, same
coordinates, no `Station`/`Cruise`/`Transect`/`Date` — under the old rule
this produced one connected line (bug); under the new rule it returns
`None` (fixed). Cross-checked empirically that `Cruise`+`Date` is not
degenerate for this dataset's actual schema before relying on it: grouping
the Japan Sea reference data by `(Cruise, Date)` gives a mean of ~2.85
distinct `Station` values per group (max 11) — i.e. `Cruise`+`Date` alone
does *not* uniquely pick out one physical station within a multi-station
cruise-day. This is not a problem for the feature as implemented, because
the grouping key always includes rounded `Longitude_degE`/`Latitude_degN`
*first* — genuinely different stations on the same cruise-day sit at
different coordinates and are already separated by that; `Cruise`+`Date`
(or the other combinations) exists specifically to catch the case where
coordinates coincide but the cast does not (a revisit, or reference+upload
overlap), which is exactly what it is used for here.

Tests added (`test/test_vertical_section_visualizer.py`):
- `test_build_station_profile_trace_ignores_weak_identifier_columns_alone`
  — `Dataset`+`Year` alone, no strong identifier → `None`.
- `test_build_station_profile_trace_returns_none_when_identifier_columns_are_all_missing`
  — `Station`/`Cruise`/`Transect`/`Date` columns all present, all `None`
  → `None` (the literal "全値欠損" case).
- `test_build_station_profile_trace_same_dataset_same_year_same_coordinates_revisit_not_joined`
  — the literal "同Dataset・同Year・同座標の再訪" case named in the
  request → `None`.
- `test_build_station_profile_trace_cruise_alone_is_not_a_sufficient_identifier`
  — `Cruise` present but no `Date`/`Year`+`Month` and no `Station` → `None`
  (confirms a combination, not a single weaker column, is required).
- Updated the three pre-existing tests that used to rely on `Dataset`
  alone or `Year` alone as "sufficient" to instead use `Station`, or
  `Cruise`+`Date`, so they still test what they were meant to test (that
  genuinely different casts are not joined) without relying on the
  now-rejected weak-column-alone behavior.

Residual, deliberately accepted gap (documented, not silently patched):
two *anonymous* casts (no `Station`/`Cruise`/`Transect`/`Date` at all,
identical `Dataset`) that also coincide in rounded lon/lat still cannot be
told apart — there is no column left to distinguish them. This is the
same gap noted in the previous round; it cannot be closed without either a
caller-supplied cast/profile id (no page currently produces one) or a
policy decision to treat all-`Dataset`-only-same-coordinate groups as
always-separate (which would make every single-cast multi-depth profile
with no metadata at all also fail to draw — a strictly worse outcome for
the common case). Left as-is and flagged again here.

### 2. Antimeridian-crossing A-B sections: added an explicit non-validation warning

The corridor-band safety guard from the previous round (dropping a
polygon whose longitude sequence jumps across the antimeridian) only
prevents the *visual* symptom of a giant polygon. It does nothing about —
and was never claimed to fix — the section calculation itself
(`project_points_to_polyline`/`build_section_polyline`'s single
flat-tangent-plane approximation), which is still distorted for a line
that genuinely crosses the antimeridian. Nothing in the app's UI
previously told the user this, so a user could reasonably (and wrongly)
read "the corridor band is just hidden" as "the section itself is fine."

Added `section_crosses_antimeridian(section_vertices)`: flags a polyline
segment when two consecutive vertices' *raw* longitude difference exceeds
180° (the standard heuristic for "this pair is meant to be read as
crossing the dateline the short way, not the long way round"). Wired into
`main()` right after the existing "A and B are identical" guard: when the
active A-B section (in A-B mode) trips this check, the app now shows —
unconditionally, regardless of whether the corridor checkbox is on —

> "This A-B line crosses the antimeridian (180°/-180° longitude) directly.
> The section's along-track distance, corridor, and seafloor geometry all
> use a single flat local approximation that is not designed for this
> case, so results below are not scientifically validated for a
> dateline-crossing section — treat them as a rough visual reference only."

This is deliberately worded to avoid the two failure modes Codex warned
against: it does not claim the feature "handles" dateline sections (it
says the opposite), and it does not merely say "corridor hidden" (it says
the whole section calculation, not only the corridor band, is unvalidated
here).

Tests added:
- `test_section_crosses_antimeridian_detects_direct_crossing`,
  `test_section_crosses_antimeridian_false_for_ordinary_section`,
  `test_section_crosses_antimeridian_checks_each_polyline_segment` — unit
  tests on the pure function, including a multi-vertex polyline where only
  the *second* segment crosses.
- `test_vertical_section_warns_when_ab_line_crosses_antimeridian` (AppTest,
  A=170E/55N to B=170W/55.1N via Manual input) — the warning text appears.
- `test_vertical_section_no_antimeridian_warning_for_ordinary_ab_line`
  (AppTest, default app state) — the warning does not appear for a normal
  section, i.e. this is not spuriously shown for everything.

**Future task, recorded rather than attempted here:** a scientifically
supportable dateline-crossing A-B section needs a redesign of the section
geometry itself — either (a) unwrap longitude into a continuous coordinate
before computing local km offsets (choosing, per polyline, whichever unwrap
direction keeps consecutive vertices within 180° of each other, then
carrying that unwrapped longitude through `lonlat_to_local_km`,
`project_points_to_polyline`, `densify_section_line`, and the corridor-band
builder consistently), or (b) replace the flat-tangent-plane approximation
with an actual geodesic (great-circle) projection, which would also improve
accuracy for very long non-dateline sections as a side effect. Either
option touches `project_points_to_polyline`/`build_section_polyline`, which
this round and the previous one were both explicitly scoped to leave
alone; it should be planned as its own task with its own review, not folded
into a future unrelated request. Do not remove the warning added above
until that redesign is actually implemented and verified.

### Verification run (this pass)

```
pytest -q test/test_vertical_section_visualizer.py   # 39 passed
pytest -q test/                                       # 149 passed, 4 skipped
```

The 4 skips are the same pre-existing, cartopy-gated tests as every prior
round in this log.

### Codex verification of the follow-up implementation (2026-09-22)

The follow-up implementation was inspected and its page-level regression
suite was independently rerun by Codex.

- `build_station_profile_trace` now filters rows through
  `_station_profile_identifiable_mask` *before* constructing a grouping key.
  A non-empty `Station`, or a complete `Cruise`/`Transect` plus `Date` (or
  `Year`+`Month`) combination, is required. `Dataset`, `Year`, `Month`, and
  `Day` alone cannot enable a profile line; they only refine an already valid
  key. This addresses the prior same-dataset/same-year revisit misconnection
  risk without turning missing values into a shared string key.
- `section_crosses_antimeridian` detects every adjacent A-B/polyline segment
  whose raw longitude difference exceeds 180 degrees. In A-B mode the UI now
  states that along-track distance, corridor, and seafloor geometry are not
  scientifically validated for that case. The calculation is intentionally
  not presented as supported; a longitude-unwrapping or geodesic redesign
  remains a separate future task.

Independent verification:

```
pytest -q test/test_vertical_section_visualizer.py   # 39 passed
```

Claude Code additionally reported the full repository run as `149 passed,
4 skipped`; that full-suite result is recorded as its report and was not
rerun in this verification pass.

### Shared Section Map layout migration (2026-09-22)

The Station/Section Map now uses the reusable
`envgeo_utils.apply_standard_map_layout()` helper instead of owning its own
margin, height, and legend layout. The helper keeps the Mapbox viewport at
the full figure domain, disables automatic external-margin expansion, and
places the legend at lower left with a translucent white background and thin
border. This preserves the existing shared `apply_map_style()` entry point,
so Standard (OpenStreetMap), Satellite, Bathymetry (Sea), and Contour (GSI)
remain one common map-mode path rather than a Section-Map-specific tile
implementation.

Vertical Section is intentionally the first adopter; no unrelated map pages
were rewritten in this small change. The map height is 480 px, keeping the
full-width Streamlit rendering while making the viewport more landscape-like.

Verification:

```
pytest -q test/test_vertical_section_visualizer.py   # 40 passed
```

### Section Map controls placement (2026-09-22)

`Map mode` was moved out of the sidebar's advanced Display controls and into
the shared-page pattern directly above Section Map: a `Map controls` popover
with a horizontal `Map style` radio and a compact current-style caption. The
single `vertical_section_map_style` session-state value is read before the
A-B draw-on-map UI and written by the lower control. Therefore the Folium
endpoint-selection map and the Plotly Section Map use the same selected
background after each Streamlit rerun, without duplicating settings.

Verification:

```
pytest -q test/test_envgeo_utils.py test/test_vertical_section_visualizer.py
# 88 passed
```
