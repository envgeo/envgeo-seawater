#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for the Vertical Section Visualizer's section-calculation
core: corridor projection, bathymetry interpolation safety, GEBCO fallback,
and the two-stage row-count guard.

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import envgeo_utils


def _vertical_section_module():
    page_path = ROOT / "pages" / "53_Vertical_Section_Visualizer.py"
    spec = importlib.util.spec_from_file_location(
        "test_vertical_section_visualizer_core",
        page_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _app_test():
    return pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def vs():
    return _vertical_section_module()


@pytest.fixture(autouse=True)
def _force_online_connectivity(monkeypatch):
    """Isolate every test from real network state.

    `envgeo_utils.check_online_connectivity` is patched to return True so that
    AppTest-based tests always see an online effective_mode (Standard) and the
    'A-B input' radio is rendered — regardless of whether the test runner's
    host can actually reach tile servers.

    実ネットワーク状態に依存しないようにする autouse fixture。
    AppTest がオンラインモードで動作し 'A-B input' radio が描画されることを保証する。

    Tests that need offline behaviour are exempt:
    - Tests using session_state["vertical_section_map_style"] = "Coastline (offline)"
      bypass check_online_connectivity entirely (resolve_map_mode short-circuits).
    - test_offline_fallback_from_standard_hides_ab_radio and
      test_online_standard_shows_ab_radio call monkeypatch.setattr again inside
      the test body, which overrides this fixture's setting for that test.
    """
    monkeypatch.setattr(
        envgeo_utils,
        "check_online_connectivity",
        lambda mode="Standard": True,
    )


# ---------------------------------------------------------------------------
# project_points_to_polyline: corridor must be gated on the true distance to
# the finite A-B segment, not the perpendicular distance to the infinite
# line through it (otherwise points beyond A/B leak into the corridor).
# ---------------------------------------------------------------------------


def test_project_points_to_polyline_excludes_point_beyond_endpoint_extension(vs):
    # Straight east-west line near the equator, ~55.66 km long.
    section_vertices = vs.section_vertices_from_ab(0.0, 130.0, 0.0, 130.5)

    # ~5 km past B, ~1 km perpendicular offset. The true nearest-point
    # distance to the finite segment is ~5.1 km; the perpendicular distance
    # to the infinite line through A-B is only ~1 km.
    far_point = pd.DataFrame(
        {
            "Longitude_degE": [130.5 + 5.0 / 111.32],
            "Latitude_degN": [1.0 / 111.32],
        }
    )

    projected, _, _ = vs.project_points_to_polyline(far_point, section_vertices, corridor_km=2.0)
    assert projected.empty, (
        "A point ~5 km beyond B with only ~1 km perpendicular offset must be "
        "excluded from a 2 km corridor: it is far outside the finite A-B "
        "segment even though it looks close to the infinite line through it."
    )


def test_project_points_to_polyline_includes_same_point_with_wide_enough_corridor(vs):
    section_vertices = vs.section_vertices_from_ab(0.0, 130.0, 0.0, 130.5)
    far_point = pd.DataFrame(
        {
            "Longitude_degE": [130.5 + 5.0 / 111.32],
            "Latitude_degN": [1.0 / 111.32],
        }
    )

    projected, length_km, _ = vs.project_points_to_polyline(
        far_point, section_vertices, corridor_km=10.0
    )
    assert len(projected) == 1
    # Clipped onto the finite segment, so it lands exactly at B.
    assert projected["SectionDistance_km"].iloc[0] == pytest.approx(length_km, abs=1e-6)


def test_project_points_to_polyline_keeps_interior_point_unaffected(vs):
    # A point directly "above" the segment's interior must behave exactly
    # as before the fix (cross-track equals the perpendicular distance).
    section_vertices = vs.section_vertices_from_ab(0.0, 130.0, 0.0, 130.5)
    interior_point = pd.DataFrame(
        {"Longitude_degE": [130.25], "Latitude_degN": [1.0 / 111.32]}
    )

    projected, _, _ = vs.project_points_to_polyline(
        interior_point, section_vertices, corridor_km=2.0
    )
    assert len(projected) == 1
    assert abs(projected["CrossTrack_km"].iloc[0]) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# sample_bathymetry_along_section: must never raise, even on degenerate
# (collinear) point geometry that breaks Qhull triangulation.
# ---------------------------------------------------------------------------


def test_sample_bathymetry_along_section_collinear_points_does_not_raise(vs):
    section_vertices = vs.section_vertices_from_ab(0.0, 130.0, 0.0, 130.5)
    # All bathymetry samples sit on the same latitude as the section line,
    # i.e. collinear in (SectionDistance_km, CrossTrack_km) space — this is
    # exactly the layout that raises scipy.spatial.QhullError for the
    # "linear" method.
    df_bathy = pd.DataFrame(
        {
            "Longitude_degE": [130.0, 130.1, 130.2, 130.3, 130.4],
            "Latitude_degN": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Bathymetry_m": [100.0, 200.0, 150.0, 250.0, 300.0],
        }
    )
    xi = np.linspace(0.0, 55.66, 10)

    result = vs.sample_bathymetry_along_section(df_bathy, section_vertices, xi, corridor_km=10.0)

    assert result is not None
    assert np.isfinite(result).any()


def test_sample_bathymetry_along_section_too_few_points_returns_none(vs):
    section_vertices = vs.section_vertices_from_ab(0.0, 130.0, 0.0, 130.5)
    df_bathy = pd.DataFrame(
        {
            "Longitude_degE": [130.1, 130.2],
            "Latitude_degN": [0.0, 0.0],
            "Bathymetry_m": [100.0, 200.0],
        }
    )
    xi = np.linspace(0.0, 55.66, 10)
    assert vs.sample_bathymetry_along_section(df_bathy, section_vertices, xi, corridor_km=10.0) is None


# ---------------------------------------------------------------------------
# GEBCO land handling: cells at/above sea level must become NaN, not a
# false 0 m seafloor that would mask real data crossing an island/coast.
# ---------------------------------------------------------------------------


def test_sample_netcdf_bathymetry_land_cells_are_nan_not_zero(vs, monkeypatch):
    lon = np.array([129.0, 129.5, 130.0, 130.5, 131.0, 131.5])
    lat = np.array([-0.5, 0.0, 0.5])
    height = np.full((3, 6), -500.0)
    height[1, 2:4] = 50.0  # a land bump crossing the middle of the section

    monkeypatch.setattr(vs, "load_gebco_grid", lambda nc_path: (lon, lat, height))

    section_vertices = vs.section_vertices_from_ab(0.0, 129.5, 0.0, 131.0)
    xi = np.linspace(0.0, 1.5 * 111.32, 20)

    result = vs.sample_netcdf_bathymetry_along_section("dummy.nc", section_vertices, xi)

    assert np.isnan(result).any(), "Land cells must produce NaN, signalling 'no seafloor data' here."
    assert not np.any(result == 0.0), (
        "Land cells must not collapse to a literal 0 m depth — that would "
        "make mask_below_bottom hide real observations at that position."
    )


# ---------------------------------------------------------------------------
# densify_section_line: must stay finite even for a near-polar reference
# latitude, matching the guard already used in suggest_default_section_vertices.
# ---------------------------------------------------------------------------


def test_densify_section_line_handles_near_polar_reference_latitude(vs):
    section_vertices = vs.section_vertices_from_ab(89.9999, 130.0, 89.9999, 131.0)
    lon, lat = vs.densify_section_line(section_vertices, 5)
    assert np.isfinite(lon).all()
    assert np.isfinite(lat).all()


# ---------------------------------------------------------------------------
# interpolate_section_grid: now returns (z_grid, extrapolated_mask), and
# must still fall back safely on collinear target-parameter geometry.
# ---------------------------------------------------------------------------


def test_interpolate_section_grid_returns_mask_and_survives_collinear_points(vs):
    df_section = pd.DataFrame(
        {
            "SectionDistance_km": [0.0, 10.0, 20.0, 30.0, 40.0],
            "Depth_m": [0.0, 0.0, 0.0, 0.0, 0.0],  # collinear: constant depth
            "d18O": [0.1, 0.2, 0.15, 0.25, 0.3],
        }
    )
    x_grid, y_grid = np.meshgrid(np.linspace(0.0, 40.0, 5), np.linspace(0.0, 10.0, 5))

    z_grid, extrapolated_mask = vs.interpolate_section_grid(df_section, "d18O", x_grid, y_grid)

    assert z_grid.shape == x_grid.shape
    assert extrapolated_mask.shape == x_grid.shape
    assert extrapolated_mask.dtype == bool
    # Every row except y=0 is far from any sample (all samples sit at
    # Depth_m=0), so nearest-neighbor extrapolation must dominate there.
    assert extrapolated_mask[1:, :].any()


# ---------------------------------------------------------------------------
# smooth_with_nan_gaps: must not let NaN (below-seafloor mask) bleed into
# originally-valid cells, unlike a plain scipy.ndimage.gaussian_filter call.
# ---------------------------------------------------------------------------


def test_smooth_with_nan_gaps_does_not_leak_nan_into_valid_cells(vs):
    # A plain scipy.ndimage.gaussian_filter call lets NaN bleed several
    # pixels into the valid region above a below-seafloor mask (verified
    # directly against scipy before writing this fix). The NaN-aware
    # version must keep every originally-valid cell finite.
    z = np.ones((10, 10)) * 5.0
    z[6:, :] = np.nan  # simulate a below-seafloor mask

    result = vs.smooth_with_nan_gaps(z, 1.0)

    assert not np.isnan(result[:6, :]).any(), (
        "Valid cells above the masked seafloor must stay finite after "
        "NaN-aware smoothing."
    )
    # Cells far from any valid sample must still end up NaN (not just the
    # cells right at the boundary, which the app re-masks again anyway via
    # mask_below_bottom immediately after smoothing).
    z_far = np.full((30, 5), np.nan)
    z_far[0, :] = 5.0
    result_far = vs.smooth_with_nan_gaps(z_far, 1.0)
    assert np.isnan(result_far[-1, :]).all(), (
        "A cell far outside the valid region's influence must remain NaN."
    )


def test_smooth_with_nan_gaps_matches_plain_filter_when_no_nan_present(vs):
    from scipy.ndimage import gaussian_filter

    z = np.random.RandomState(0).rand(8, 8)
    expected = gaussian_filter(z, sigma=1.0)
    result = vs.smooth_with_nan_gaps(z, 1.0)
    np.testing.assert_allclose(result, expected)


def test_smooth_with_nan_gaps_noop_for_zero_sigma(vs):
    z = np.array([[1.0, np.nan], [2.0, 3.0]])
    result = vs.smooth_with_nan_gaps(z, 0.0)
    assert result is z


# ---------------------------------------------------------------------------
# build_station_profile_trace: one combined trace per call, not one trace
# per station (keeps rendering cheap even with many stations).
# ---------------------------------------------------------------------------


def test_build_station_profile_trace_groups_by_station_as_single_trace(vs):
    df_points = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 135.0, 136.0, 136.0, 136.0],
            "Latitude_degN": [35.0, 35.0, 36.0, 36.0, 36.0],
            "SectionDistance_km": [0.0, 0.0, 10.0, 10.0, 10.0],
            "Depth_m": [10.0, 20.0, 5.0, 15.0, 25.0],
            "Station": ["St1", "St1", "St2", "St2", "St2"],
        }
    )
    trace = vs.build_station_profile_trace(df_points)
    assert trace is not None
    # Two station groups -> two line segments separated by a single None.
    assert trace.x.count(None) == 2
    assert trace.showlegend is False


def test_build_station_profile_trace_returns_none_without_multi_depth_stations(vs):
    df_points = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 136.0],
            "Latitude_degN": [35.0, 36.0],
            "SectionDistance_km": [0.0, 10.0],
            "Depth_m": [10.0, 20.0],
            "Station": ["St1", "St2"],
        }
    )
    assert vs.build_station_profile_trace(df_points) is None


def test_build_station_profile_trace_returns_none_without_any_identifier_column(vs):
    # No Dataset/Station/Cruise/Transect/date column at all: without one,
    # two coincidentally-colocated but genuinely different casts could be
    # wrongly joined, so the safe behavior is to skip the line entirely.
    df_points = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 135.0],
            "Latitude_degN": [35.0, 35.0],
            "SectionDistance_km": [0.0, 0.0],
            "Depth_m": [10.0, 20.0],
        }
    )
    assert vs.build_station_profile_trace(df_points) is None


def test_build_station_profile_trace_ignores_weak_identifier_columns_alone(vs):
    # Dataset and Year alone (no Station/Cruise/Transect/Date) are not
    # sufficient evidence of "same cast" -- a same-Dataset, same-Year
    # revisit at the same coordinates could still be a genuinely different
    # cast. Rows must be excluded rather than joined on these weak columns.
    df_points = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 135.0],
            "Latitude_degN": [35.0, 35.0],
            "SectionDistance_km": [0.0, 0.0],
            "Depth_m": [10.0, 20.0],
            "Dataset": ["Around Japan", "Around Japan"],
            "Year": [2020, 2020],
        }
    )
    assert vs.build_station_profile_trace(df_points) is None


def test_build_station_profile_trace_returns_none_when_identifier_columns_are_all_missing(vs):
    # Station/Cruise/Transect/Date columns exist in the frame, but every
    # value is missing for these rows -- a column merely being present
    # must not be treated as identification.
    df_points = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 135.0, 136.0, 136.0],
            "Latitude_degN": [35.0, 35.0, 36.0, 36.0],
            "SectionDistance_km": [0.0, 0.0, 10.0, 10.0],
            "Depth_m": [10.0, 20.0, 10.0, 20.0],
            "Station": [None, None, None, None],
            "Cruise": [None, None, None, None],
            "Transect": [None, None, None, None],
            "Date": [None, None, None, None],
        }
    )
    assert vs.build_station_profile_trace(df_points) is None


def test_build_station_profile_trace_same_dataset_same_year_same_coordinates_revisit_not_joined(vs):
    # Same Dataset, same Year, same coordinates, but no Station/Cruise/
    # Transect/Date to actually confirm it is the same physical cast (this
    # is exactly the "different cast that happens to share weak metadata"
    # scenario) -- must not be drawn as one connected profile line.
    df_points = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 135.0, 135.0, 135.0],
            "Latitude_degN": [35.0, 35.0, 35.0, 35.0],
            "SectionDistance_km": [0.0, 0.0, 0.0, 0.0],
            "Depth_m": [10.0, 20.0, 10.0, 20.0],
            "Dataset": ["Around Japan", "Around Japan", "Around Japan", "Around Japan"],
            "Year": [2020, 2020, 2020, 2020],
        }
    )
    assert vs.build_station_profile_trace(df_points) is None


def test_build_station_profile_trace_does_not_join_different_datasets_at_same_coordinates(vs):
    # Same lon/lat (e.g. a reference station and an uploaded point that
    # happen to coincide), each properly identified by its own Station,
    # but from a different Dataset -> must NOT be treated as one cast.
    df_points = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 135.0, 135.0, 135.0],
            "Latitude_degN": [35.0, 35.0, 35.0, 35.0],
            "SectionDistance_km": [0.0, 0.0, 0.0, 0.0],
            "Depth_m": [10.0, 20.0, 10.0, 20.0],
            "Dataset": ["Around Japan", "Around Japan", "Uploaded data", "Uploaded data"],
            "Station": ["St1", "St1", "St1", "St1"],
        }
    )
    trace = vs.build_station_profile_trace(df_points)
    assert trace is not None
    # Two distinct casts (by Dataset, even with the same Station label)
    # at the same coordinates -> two separate line segments, not one
    # continuous line jumping between them.
    assert trace.x.count(None) == 2


def test_build_station_profile_trace_does_not_join_different_dates_at_same_coordinates(vs):
    # Same lon/lat, same Dataset, same Cruise, but a different observation
    # date -> a separate re-visit cast, must not be joined either. Cruise
    # alone would not be enough to identify a row; Cruise+Date together is.
    df_points = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 135.0, 135.0, 135.0],
            "Latitude_degN": [35.0, 35.0, 35.0, 35.0],
            "SectionDistance_km": [0.0, 0.0, 0.0, 0.0],
            "Depth_m": [10.0, 20.0, 10.0, 20.0],
            "Dataset": ["Around Japan", "Around Japan", "Around Japan", "Around Japan"],
            "Cruise": ["KT-01", "KT-01", "KT-01", "KT-01"],
            "Date": ["2020-05-01", "2020-05-01", "2021-05-01", "2021-05-01"],
        }
    )
    trace = vs.build_station_profile_trace(df_points)
    assert trace is not None
    assert trace.x.count(None) == 2


def test_build_station_profile_trace_cruise_alone_is_not_a_sufficient_identifier(vs):
    # Cruise by itself (no Date/Year+Month, no Station) is not one of the
    # accepted combinations, so these rows must be excluded even though a
    # "real" identifier column is present in the frame.
    df_points = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 135.0],
            "Latitude_degN": [35.0, 35.0],
            "SectionDistance_km": [0.0, 0.0],
            "Depth_m": [10.0, 20.0],
            "Cruise": ["KT-01", "KT-01"],
        }
    )
    assert vs.build_station_profile_trace(df_points) is None


# ---------------------------------------------------------------------------
# create_section_plot: the extrapolation overlay must be additive and must
# not disturb the existing "uploaded trace drawn last" contract.
# ---------------------------------------------------------------------------


def test_create_section_plot_extrapolation_overlay_keeps_uploaded_trace_last(vs):
    points = pd.DataFrame(
        {
            "SectionDistance_km": [0.0, 1.0],
            "Depth_m": [10.0, 20.0],
            "d18O": [0.1, 0.2],
            "Longitude_degE": [135.0, 136.0],
            "Latitude_degN": [35.0, 36.0],
            "CrossTrack_km": [0.0, 0.0],
        }
    )
    style = {
        "size": 36,
        "marker": "Diamond",
        "alpha": 1.0,
        "outline_color": "#000000",
        "outline_width": 1.0,
        "color_mode": "Single color",
        "color": "#ff0000",
    }
    extrapolated_mask = np.array([[True, False], [False, True]])

    figure = vs.create_section_plot(
        z_grid=[[0.1, 0.2], [0.1, 0.2]],
        xi=[0.0, 1.0],
        yi=[0.0, 20.0],
        df_points=points,
        target_col="d18O",
        z_min=-1.0,
        z_max=1.0,
        uploaded_points=points,
        uploaded_style=style,
        extrapolated_mask=extrapolated_mask,
    )

    assert figure.data[-1].name == "Uploaded data"
    assert any(trace.name == "Extrapolated (low confidence)" for trace in figure.data)


def test_create_section_plot_without_extrapolation_mask_is_unchanged(vs):
    points = pd.DataFrame(
        {
            "SectionDistance_km": [0.0, 1.0],
            "Depth_m": [10.0, 20.0],
            "d18O": [0.1, 0.2],
            "Longitude_degE": [135.0, 136.0],
            "Latitude_degN": [35.0, 36.0],
            "CrossTrack_km": [0.0, 0.0],
        }
    )
    figure = vs.create_section_plot(
        z_grid=[[0.1, 0.2], [0.1, 0.2]],
        xi=[0.0, 1.0],
        yi=[0.0, 20.0],
        df_points=points,
        target_col="d18O",
        z_min=-1.0,
        z_max=1.0,
    )
    assert not any(trace.name == "Extrapolated (low confidence)" for trace in figure.data)


# ---------------------------------------------------------------------------
# End-to-end: a wide Data-filtering selection combined with a narrow A-B
# corridor must no longer be blocked outright (the old pre-projection-only
# guard would have rejected this before allowing the cheap corridor
# projection to shrink it down).
# ---------------------------------------------------------------------------


def test_vertical_section_wide_filter_narrow_corridor_is_not_blocked():
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )

    n_far = 150
    far_lons = np.linspace(120.0, 160.0, n_far)
    far_lats = np.linspace(-30.0, 30.0, n_far) + 20.0  # always >=10 deg off the line
    near_lons = [130.1, 130.15, 130.2, 130.25, 130.3]
    near_lats = [0.001, 0.002, 0.0015, 0.0025, 0.0018]

    data = pd.DataFrame(
        {
            "Longitude_degE": list(far_lons) + near_lons,
            "Latitude_degN": list(far_lats) + near_lats,
            "Depth_m": [10.0] * (n_far + 5),
            "d18O": [0.1] * (n_far + 5),
        }
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_DATA_KEY] = (
        envgeo_utils.prepare_uploaded_data(data)
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_FILENAME_KEY] = "apptest.csv"
    app.run(timeout=60)

    # Apply the dataset-selection change first and let default A-B
    # endpoints recompute, then set explicit endpoints in a second step —
    # un-keyed number_input widgets are recreated when their computed
    # default changes, so staging both changes in one run is unreliable.
    next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    ).set_value(["Uploaded data"])
    next(item for item in app.radio if item.label == "A-B input").set_value("Manual")
    app.run(timeout=60)

    for label, value in {
        "A lat": 0.0,
        "A lon": 130.0,
        "B lat": 0.0,
        "B lon": 130.5,
    }.items():
        next(item for item in app.number_input if item.label == label).set_value(value)
    next(
        item for item in app.sidebar.slider
        if item.label == "Half-width of section corridor (km)"
    ).set_value(5.0)
    next(
        item for item in app.sidebar.number_input
        if item.label == "Max valid rows for section plotting"
    ).set_value(100)
    app.run(timeout=60)

    assert not app.exception
    visible = [str(item.value) for item in app.warning]
    assert not any("row limit used for interpolation" in text for text in visible)
    assert any(
        "5 in corridor" in text
        for text in [str(item.value) for item in app.caption] + visible
    )
    assert len(app.get("plotly_chart")) >= 1


def test_vertical_section_post_projection_cap_blocks_when_corridor_stays_wide():
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )

    n_rows = 120
    lons = np.linspace(130.0, 130.5, n_rows)
    lats = np.linspace(0.0, 0.001, n_rows)  # essentially on the line, but not degenerate
    data = pd.DataFrame(
        {
            "Longitude_degE": lons,
            "Latitude_degN": lats,
            "Depth_m": [10.0] * n_rows,
            "d18O": [0.1] * n_rows,
        }
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_DATA_KEY] = (
        envgeo_utils.prepare_uploaded_data(data)
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_FILENAME_KEY] = "apptest.csv"
    app.run(timeout=60)

    next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    ).set_value(["Uploaded data"])
    next(item for item in app.radio if item.label == "A-B input").set_value("Manual")
    app.run(timeout=60)

    for label, value in {
        "A lat": 0.0,
        "A lon": 130.0,
        "B lat": 0.0,
        "B lon": 130.5,
    }.items():
        next(item for item in app.number_input if item.label == label).set_value(value)
    next(
        item for item in app.sidebar.slider
        if item.label == "Half-width of section corridor (km)"
    ).set_value(150.0)
    next(
        item for item in app.sidebar.number_input
        if item.label == "Max valid rows for section plotting"
    ).set_value(100)
    app.run(timeout=60)

    assert not app.exception
    visible = [str(item.value) for item in app.warning]
    assert any("row limit used for interpolation" in text for text in visible)


# ---------------------------------------------------------------------------
# resolve_section_row_limit: an internal hard cap the "Max valid rows for
# section plotting" number input cannot override.
# ---------------------------------------------------------------------------


def test_resolve_section_row_limit_clamps_to_hard_cap(vs):
    hard_cap = vs.MAX_ROWS_FOR_SECTION_INTERPOLATION_HARD_CAP
    assert vs.resolve_section_row_limit(3000) == 3000
    assert vs.resolve_section_row_limit(hard_cap + 5000) == hard_cap
    assert vs.resolve_section_row_limit(hard_cap) == hard_cap


def test_vertical_section_row_limit_widget_cannot_exceed_hard_cap():
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.run(timeout=60)

    row_limit_widget = next(
        item for item in app.sidebar.number_input
        if item.label == "Max valid rows for section plotting"
    )
    vertical_section = _vertical_section_module()
    assert row_limit_widget.proto.max == (
        vertical_section.MAX_ROWS_FOR_SECTION_INTERPOLATION_HARD_CAP
    )


# ---------------------------------------------------------------------------
# Corridor band geometry (Station map): must match the same best_dist <=
# corridor_km test used by project_points_to_polyline, stay finite (not an
# infinite strip past A/B), handle multi-vertex polylines, and avoid a
# giant polygon when the section crosses the antimeridian.
# ---------------------------------------------------------------------------


def test_corridor_capsule_local_km_width_matches_corridor_half_width(vs):
    polygon = vs.build_corridor_capsule_local_km((0.0, 0.0), (10.0, 0.0), 2.0, cap_points=32)
    assert polygon is not None
    # Perpendicular to the segment (y-axis here), the band must extend
    # exactly +/- corridor_km, matching the best_dist <= corridor_km test.
    assert float(polygon[:, 1].max()) == pytest.approx(2.0, abs=1e-6)
    assert float(polygon[:, 1].min()) == pytest.approx(-2.0, abs=1e-6)


def test_corridor_capsule_local_km_is_finite_not_extended_past_endpoints(vs):
    polygon = vs.build_corridor_capsule_local_km((0.0, 0.0), (10.0, 0.0), 2.0, cap_points=32)
    assert polygon is not None
    # Along the segment direction (x-axis here), the band must stay within
    # corridor_km of each endpoint (a finite capsule), not extend forever.
    assert float(polygon[:, 0].min()) >= -2.0 - 1e-6
    assert float(polygon[:, 0].max()) <= 12.0 + 1e-6
    assert np.isfinite(polygon).all()


def test_build_corridor_band_polygons_handles_multi_vertex_polyline(vs):
    section_vertices = [[0.0, 130.0], [1.0, 131.0], [0.0, 132.0]]
    polygons = vs.build_corridor_band_polygons(section_vertices, 20.0)
    # One capsule per segment.
    assert len(polygons) == 2
    for polygon in polygons:
        lons = [pt[0] for pt in polygon]
        lats = [pt[1] for pt in polygon]
        assert np.isfinite(lons).all()
        assert np.isfinite(lats).all()
        assert not vs.polygon_has_dateline_seam(lons)


def test_build_corridor_band_polygons_avoids_giant_dateline_polygon(vs):
    # A-B crossing the antimeridian near the Bering Sea. Whatever polygons
    # (if any) come back must never contain a longitude jump signalling a
    # dateline-wrap artifact — no giant polygon spanning most of the globe.
    section_vertices = vs.section_vertices_from_ab(55.0, 170.0, 55.0, -170.0)
    polygons = vs.build_corridor_band_polygons(section_vertices, 50.0)
    for polygon in polygons:
        lons = [pt[0] for pt in polygon]
        assert not vs.polygon_has_dateline_seam(lons)


def test_build_corridor_band_trace_has_expected_legend_label(vs):
    trace = vs.build_corridor_band_trace(
        vs.section_vertices_from_ab(0.0, 130.0, 0.0, 130.5), 30.0
    )
    assert trace is not None
    assert trace.name == "Section corridor (±30 km)"
    assert trace.fill == "toself"


def test_create_station_map_corridor_trace_only_when_requested(vs):
    section_vertices = vs.section_vertices_from_ab(0.0, 130.0, 0.0, 130.5)
    df_points = pd.DataFrame(
        {"Longitude_degE": [130.2], "Latitude_degN": [0.001]}
    )

    fig_with_corridor = vs.create_station_map(
        df_points, section_vertices, show_corridor=True, corridor_km=30.0
    )
    fig_without_corridor = vs.create_station_map(
        df_points, section_vertices, show_corridor=False, corridor_km=30.0
    )

    assert any(
        "Section corridor" in str(trace.name) for trace in fig_with_corridor.data
    )
    assert not any(
        "Section corridor" in str(trace.name) for trace in fig_without_corridor.data
    )


def test_create_station_map_draw_order_keeps_corridor_behind_points_and_line(vs):
    section_vertices = vs.section_vertices_from_ab(0.0, 130.0, 0.0, 130.5)
    df_points = pd.DataFrame(
        {"Longitude_degE": [130.2], "Latitude_degN": [0.001]}
    )
    df_background = pd.DataFrame(
        {"Longitude_degE": [125.0], "Latitude_degN": [10.0]}
    )

    fig = vs.create_station_map(
        df_points,
        section_vertices,
        df_background=df_background,
        show_corridor=True,
        corridor_km=30.0,
    )
    names = [str(trace.name) for trace in fig.data]
    # background -> corridor band -> in-corridor points -> red line -> A/B markers
    assert names.index("Filtered stations") < names.index("Section corridor (±30 km)")
    assert names.index("Section corridor (±30 km)") < names.index("Samples in corridor")
    assert names.index("Samples in corridor") < names.index("A-B Section")
    assert names.index("A-B Section") < names.index("Section endpoints")


def test_create_station_map_uses_shared_full_width_overlay_layout(vs):
    section_vertices = vs.section_vertices_from_ab(35.0, 135.0, 35.5, 136.0)
    df_points = pd.DataFrame(
        {"Longitude_degE": [135.2], "Latitude_degN": [35.1]}
    )

    fig = vs.create_station_map(df_points, section_vertices)

    assert fig.layout.height == 480
    assert tuple(fig.layout.mapbox.domain.x) == (0.0, 1.0)
    assert tuple(fig.layout.mapbox.domain.y) == (0.0, 1.0)
    assert fig.layout.margin.autoexpand is False
    assert fig.layout.legend.x == 0.01
    assert fig.layout.legend.y == 0.01
    assert fig.layout.legend.bgcolor == "rgba(255,255,255,0.85)"


# ---------------------------------------------------------------------------
# End-to-end: the corridor band must appear on the Station map only for
# A-B section mode, never for Axis-based mode.
# ---------------------------------------------------------------------------


def _plotly_chart_trace_names(chart_element):
    # AppTest exposes st.plotly_chart as a generic element whose figure is
    # only available as a JSON string on the proto (no stateful `.value`).
    spec = json.loads(chart_element.proto.spec)
    return [trace.get("name", "") for trace in spec.get("data", [])]


def test_vertical_section_corridor_trace_only_in_ab_section_mode():
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_DATA_KEY] = (
        envgeo_utils.prepare_uploaded_data(
            pd.DataFrame(
                {
                    "Longitude_degE": [130.0, 130.2, 130.4],
                    "Latitude_degN": [0.0, 0.001, 0.002],
                    "Depth_m": [10.0, 20.0, 30.0],
                    "d18O": [0.1, 0.2, 0.15],
                }
            )
        )
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_FILENAME_KEY] = "apptest.csv"
    app.run(timeout=60)

    next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    ).set_value(["Uploaded data"])
    next(item for item in app.radio if item.label == "A-B input").set_value("Manual")
    app.run(timeout=60)

    assert not app.exception
    ab_charts = app.get("plotly_chart")
    assert any(
        any("Section corridor" in name for name in _plotly_chart_trace_names(chart))
        for chart in ab_charts
    )

    next(item for item in app.radio if item.label == "Section mode").set_value(
        "Axis-based"
    )
    app.run(timeout=60)

    assert not app.exception
    axis_charts = app.get("plotly_chart")
    assert not any(
        "Section corridor" in name
        for chart in axis_charts
        for name in _plotly_chart_trace_names(chart)
    )


# ---------------------------------------------------------------------------
# section_crosses_antimeridian: flags A-B lines that directly cross the
# antimeridian, so the app can warn that such sections are not
# scientifically validated (the underlying flat-plane approximation is
# unchanged and out of scope for this round).
# ---------------------------------------------------------------------------


def test_section_crosses_antimeridian_detects_direct_crossing(vs):
    assert vs.section_crosses_antimeridian(
        vs.section_vertices_from_ab(55.0, 170.0, 55.0, -170.0)
    )


def test_section_crosses_antimeridian_false_for_ordinary_section(vs):
    assert not vs.section_crosses_antimeridian(
        vs.section_vertices_from_ab(35.0, 130.0, 36.0, 140.0)
    )


def test_section_crosses_antimeridian_checks_each_polyline_segment(vs):
    # Only the second segment crosses; the helper must still catch it.
    vertices = [[55.0, 160.0], [55.0, 175.0], [55.0, -170.0]]
    assert vs.section_crosses_antimeridian(vertices)


def test_vertical_section_warns_when_ab_line_crosses_antimeridian():
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_DATA_KEY] = (
        envgeo_utils.prepare_uploaded_data(
            pd.DataFrame(
                {
                    "Longitude_degE": [170.0, -170.0],
                    "Latitude_degN": [55.0, 55.1],
                    "Depth_m": [10.0, 20.0],
                    "d18O": [0.1, 0.2],
                }
            )
        )
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_FILENAME_KEY] = "apptest.csv"
    app.run(timeout=60)

    next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    ).set_value(["Uploaded data"])
    next(item for item in app.radio if item.label == "A-B input").set_value("Manual")
    app.run(timeout=60)

    for label, value in {
        "A lat": 55.0,
        "A lon": 170.0,
        "B lat": 55.0,
        "B lon": -170.0,
    }.items():
        next(item for item in app.number_input if item.label == label).set_value(value)
    app.run(timeout=60)

    assert not app.exception
    visible = [str(item.value) for item in app.warning]
    assert any("crosses the antimeridian" in text for text in visible)


def test_vertical_section_no_antimeridian_warning_for_ordinary_ab_line():
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.run(timeout=60)

    assert not app.exception
    visible = [str(item.value) for item in app.warning]
    assert not any("crosses the antimeridian" in text for text in visible)


# ────────────────────────────────────────────────────────────────────────────
# Task-14: offline / online Map controls branch tests
# ────────────────────────────────────────────────────────────────────────────


def test_offline_coastline_no_ab_radio():
    """When map_style is 'Coastline (offline)', the 'A-B input' radio must not be rendered.

    Folium/Draw は使えないため endpoint_mode は強制 Manual となり、ラジオは表示しない。
    """
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state["vertical_section_map_style"] = "Coastline (offline)"
    app.run(timeout=60)

    assert not app.exception
    ab_radios = [item for item in app.radio if item.label == "A-B input"]
    assert len(ab_radios) == 0, (
        "effective_mode=='Coastline (offline)' → 'A-B input' radio must be absent "
        "(Draw on map is disabled offline; endpoint_mode is forced to Manual)."
    )


def test_offline_coastline_shows_english_message():
    """Offline mode must display the English-only notice in st.info.

    The bilingual (Japanese+English) message was replaced by an English-only
    message. This test verifies that the expected English text is present and
    that no residual Japanese text remains.
    """
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state["vertical_section_map_style"] = "Coastline (offline)"
    app.run(timeout=60)

    assert not app.exception
    all_info = [str(item.value) for item in app.info]
    assert any("Drawing an A–B line on the map is unavailable offline" in text
               for text in all_info), (
        "English offline notice not found in st.info messages. "
        f"Found info messages: {all_info}"
    )
    assert any("Enter A and B coordinates manually" in text for text in all_info), (
        "English offline notice not found in st.info messages. "
        f"Found info messages: {all_info}"
    )


def test_offline_coastline_shows_plotly_preview():
    """Offline mode must render a Plotly preview map (no Folium/tile calls)."""
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state["vertical_section_map_style"] = "Coastline (offline)"
    app.session_state[envgeo_utils.UPLOAD_SESSION_DATA_KEY] = (
        envgeo_utils.prepare_uploaded_data(
            pd.DataFrame(
                {
                    "Longitude_degE": [135.0, 136.0, 137.0],
                    "Latitude_degN": [35.0, 35.5, 36.0],
                    "Depth_m": [10.0, 20.0, 30.0],
                    "d18O": [0.1, 0.2, 0.3],
                }
            )
        )
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_FILENAME_KEY] = "apptest_offline.csv"
    app.run(timeout=60)

    assert not app.exception
    charts = app.get("plotly_chart")
    assert len(charts) >= 1, (
        "Offline mode must render at least one Plotly chart (the A-B preview map). "
        f"Found {len(charts)} plotly_chart elements."
    )


def test_offline_fallback_from_standard_hides_ab_radio(monkeypatch):
    """When connectivity fails, Standard→offline fallback also hides the 'A-B input' radio.

    resolve_map_mode は接続失敗時に ("Coastline (offline)", True) を返す。
    ページはその effective_mode で分岐するため、ユーザーが Standard を選択していても
    'A-B input' ラジオを表示しない。
    """
    # Patch check_online_connectivity so that resolve_map_mode falls back offline
    # without making a real network call.
    monkeypatch.setattr(
        envgeo_utils, "check_online_connectivity", lambda mode="Standard": False
    )

    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    # User selected Standard, but connectivity fails → effective mode is offline.
    app.session_state["vertical_section_map_style"] = "Standard"
    app.run(timeout=60)

    assert not app.exception
    ab_radios = [item for item in app.radio if item.label == "A-B input"]
    assert len(ab_radios) == 0, (
        "Even when the user selected 'Standard' but connectivity failed "
        "(fell back to Coastline offline), 'A-B input' radio must be hidden."
    )
    # The fallback warning must be visible.
    all_warnings = [str(item.value) for item in app.warning]
    assert any(
        "Online map tiles are unavailable" in text
        for text in all_warnings
    ), (
        "OFFLINE_FALLBACK_WARNING must appear when falling back from Standard to offline. "
        f"Found warnings: {all_warnings}"
    )


def test_online_standard_shows_ab_radio(monkeypatch):
    """In online Standard mode, the 'A-B input' radio ('Manual'/'Draw on map') is shown.

    resolve_map_mode をパッチして実通信なしに Standard が到達可能な状態をシミュレート。
    """
    # Ensure connectivity check passes without a real network call.
    monkeypatch.setattr(
        envgeo_utils, "check_online_connectivity", lambda mode="Standard": True
    )

    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state["vertical_section_map_style"] = "Standard"
    app.run(timeout=60)

    assert not app.exception
    ab_radios = [item for item in app.radio if item.label == "A-B input"]
    assert len(ab_radios) == 1, (
        "In online Standard mode, exactly one 'A-B input' radio "
        "('Manual' / 'Draw on map') must be rendered. "
        f"Found {len(ab_radios)} such radio(s)."
    )


# ---------------------------------------------------------------------------
# create_station_map center/zoom for offline A-B preview
# The offline preview now delegates to create_station_map(), so the mapbox
# centre must be the A-B midpoint and the zoom must follow the span thresholds.
# ---------------------------------------------------------------------------


def test_offline_preview_map_centers_on_ab_midpoint(vs):
    """Offline A-B preview figure must be centred on the A-B midpoint with correct zoom.

    create_station_map() calculates:
        lats, lons  from section_vertices
        center_lat  = mean(lats)
        center_lon  = mean(lons)
        span        = max(lon_span, lat_span, 0.2)
        zoom        = 7 if span < 0.5 else 6 if span < 1.5 else 5 if span < 4 else 4

    For A=(35.0, 135.0) B=(36.0, 137.0):
        center = (35.5, 136.0), lon_span=2.0, lat_span=1.0, span=2.0 → zoom=5
    """
    a_lat, a_lon = 35.0, 135.0
    b_lat, b_lon = 36.0, 137.0
    section_vertices = vs.section_vertices_from_ab(a_lat, a_lon, b_lat, b_lon)

    df_stations = pd.DataFrame(
        {
            "Longitude_degE": [135.5, 136.0, 136.5],
            "Latitude_degN": [35.2, 35.5, 35.8],
        }
    )

    fig = vs.create_station_map(
        df_stations,
        section_vertices,
        sample_name="Filtered stations (preview)",
        line_name="A-B line",
        map_mode="Coastline (offline)",
    )

    mb = fig.layout.mapbox
    assert mb.center.lat == pytest.approx(35.5, abs=1e-6), (
        f"Preview map centre lat must be A-B midpoint 35.5, got {mb.center.lat}"
    )
    assert mb.center.lon == pytest.approx(136.0, abs=1e-6), (
        f"Preview map centre lon must be A-B midpoint 136.0, got {mb.center.lon}"
    )
    # span = max(2.0, 1.0, 0.2) = 2.0 → zoom threshold: 1.5 <= 2.0 < 4 → zoom 5
    assert mb.zoom == 5, (
        f"With A-B span=2.0°, zoom must be 5, got {mb.zoom}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Task-15b: UI layout reorganisation tests
# ────────────────────────────────────────────────────────────────────────────


def test_target_parameter_selectbox_in_main_area_not_sidebar():
    """Target parameter must be a selectbox in the main area, not a radio or in the sidebar.

    After Task-15b+ the 'Target parameter' widget was changed from a radio to a
    st.selectbox, placed in the main area after status displays and before the
    section-plot tabs — not in the sidebar.
    AppTest exposes sidebar widgets via app.sidebar; main-area widgets appear
    in app.selectbox (with no sidebar qualifier).
    """
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.run(timeout=60)
    assert not app.exception

    # Exactly one 'Target parameter' selectbox in the whole page.
    all_target_selectboxes = [s for s in app.selectbox if s.label == "Target parameter"]
    assert len(all_target_selectboxes) == 1, (
        f"Expected exactly 1 'Target parameter' selectbox, found {len(all_target_selectboxes)}"
    )
    assert "d-excess" in all_target_selectboxes[0].options, (
        "Target parameter selectbox must include the derived d-excess variable."
    )

    # No 'Target parameter' radio anywhere.
    all_target_radios = [r for r in app.radio if r.label == "Target parameter"]
    assert len(all_target_radios) == 0, (
        f"'Target parameter' must be a selectbox, not a radio; "
        f"found {len(all_target_radios)} radio(s) with that label."
    )

    # The selectbox must not appear in the sidebar.
    sidebar_target = [s for s in app.sidebar.selectbox if s.label == "Target parameter"]
    assert len(sidebar_target) == 0, (
        "'Target parameter' selectbox must not be in the sidebar; "
        f"found {len(sidebar_target)} sidebar selectbox(es) with that label."
    )


def test_ab_endpoints_present_when_ab_section_mode():
    """When section_mode is 'A-B section', the A-B endpoint inputs must be available.

    The A-B endpoints expander contains number_input widgets labelled
    'A lat', 'A lon', 'B lat', 'B lon'.  They must be rendered when
    section_mode == 'A-B section' (the default).
    """
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    # Default is A-B section; ensure session state agrees.
    app.session_state["vertical_section_section_mode"] = "A-B section"
    app.run(timeout=60)
    assert not app.exception

    all_labels = [n.label for n in app.number_input]
    for expected in ("A lat", "A lon", "B lat", "B lon"):
        assert expected in all_labels, (
            f"'{expected}' number_input not found when section_mode=='A-B section'. "
            f"Found labels: {all_labels}"
        )


def test_ab_endpoints_absent_when_axis_based_mode():
    """When section_mode is 'Axis-based', the A-B endpoint inputs must not appear.

    The A-B endpoints sidebar expander is only shown when section_mode == 'A-B section'.
    Switching to 'Axis-based' must hide the expander and its number_input widgets.
    """
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state["vertical_section_section_mode"] = "Axis-based"
    app.run(timeout=60)
    assert not app.exception

    all_labels = [n.label for n in app.number_input]
    for absent in ("A lat", "A lon", "B lat", "B lon"):
        assert absent not in all_labels, (
            f"'{absent}' number_input must not appear when section_mode=='Axis-based'. "
            f"Found labels: {all_labels}"
        )


def test_offline_manual_forced_no_folium_english_message():
    """Coastline (offline) → Manual forced, English message, no Folium call.

    This consolidates the Task-14 offline invariants and confirms they still
    hold after the Task-15b layout reorganisation:
      - No 'A-B input' radio (Manual is forced, no Draw on map option).
      - English-only info message present.
      - No exception (i.e. Folium/render_ab_selector_map is never called).
    """
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state["vertical_section_map_style"] = "Coastline (offline)"
    app.session_state["vertical_section_section_mode"] = "A-B section"
    app.run(timeout=60)

    assert not app.exception

    # No 'A-B input' radio (Draw on map must be hidden).
    ab_radios = [r for r in app.radio if r.label == "A-B input"]
    assert len(ab_radios) == 0, (
        "Offline: 'A-B input' radio must be absent (endpoint_mode forced Manual)."
    )

    # English offline notice must appear.
    all_info = [str(item.value) for item in app.info]
    assert any(
        "Drawing an A–B line on the map is unavailable offline" in text
        for text in all_info
    ), f"English offline notice not found. info messages: {all_info}"


def test_online_standard_ab_input_radio_selectable(monkeypatch):
    """In online Standard mode, 'A-B input' radio must offer Manual and Draw on map.

    After Task-15b the radio is still inside the sidebar A-B endpoints expander;
    its options and presence must be unchanged for the online path.
    """
    monkeypatch.setattr(
        envgeo_utils, "check_online_connectivity", lambda mode="Standard": True
    )

    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state["vertical_section_map_style"] = "Standard"
    app.session_state["vertical_section_section_mode"] = "A-B section"
    app.run(timeout=60)

    assert not app.exception
    ab_radios = [r for r in app.radio if r.label == "A-B input"]
    assert len(ab_radios) == 1, (
        f"Online Standard: expected exactly 1 'A-B input' radio, found {len(ab_radios)}"
    )
    options = list(ab_radios[0].options)
    assert "Manual" in options, f"'Manual' not in A-B input options: {options}"
    assert "Draw on map" in options, f"'Draw on map' not in A-B input options: {options}"
