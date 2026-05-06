#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USGS earthquake hypocenter 4D visualizer for EnvGeo.
Created on Sun May 1 2026
Created from 04_4D_Visualizer.py and simplified as an earthquake-only page.
@author: Toyoho Ishimura @Kyoto-U
"""

import math
from datetime import datetime, time, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import envgeo_utils


version = "0.2.3" #2026/05/06


st.set_page_config(
    page_title="EnvGeo-Earthquake",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://envgeo.h.kyoto-u.ac.jp/simple-earthquake-hypocenter-visualization/",
        "Report a bug": "https://www.h.kyoto-u.ac.jp/en_f/faculty_f/ishimura_toyoho_4dea/#mailform",
        "About": (
            "EnvGeo-Earthquake: a simple research/education earthquake "
            "visualization app based on EnvGeo-Seawater."
            " / EnvGeo-Seawater をもとにした、"
            "研究・教育向けの簡易地震可視化アプリです。/ "
            "https://envgeo.h.kyoto-u.ac.jp"
            " / (Toyoho Ishimura@Kyoto-Univ. 2026) "
        ),
    },
)

JAPAN_REGION_LABEL = "Japan and surrounding area"
GLOBAL_REGION_LABEL = "Global"
INDONESIA_REGION_LABEL = "Indonesia (Sunda Arc)"
NEW_ZEALAND_REGION_LABEL = "New Zealand (Kermadec-Tonga edge)"
CHILE_REGION_LABEL = "Offshore Chile (Peru-Chile Trench)"
ALEUTIAN_REGION_LABEL = "Aleutian (North Pacific)"
ANTARCTIC_REGION_LABEL = "Antarctic margin (Scotia/South Sandwich)"
CALIFORNIA_MEXICO_REGION_LABEL = "California to W. Mexico"
KAMCHATKA_REGION_LABEL = "Kamchatka / Kuril"
NORTH_PACIFIC_WIDE_REGION_LABEL = "North Pacific Rim (Japan-Kuril-Aleutian-California)"
MEDITERRANEAN_REGION_LABEL = "Mediterranean / Anatolia"
HIMALAYA_REGION_LABEL = "Himalaya / Hindu Kush"
PHILIPPINES_REGION_LABEL = "Philippines / Taiwan"
HOTSPOT_NONE_LABEL = "(none)"
HOTSPOT_REGION_LABELS = [
    INDONESIA_REGION_LABEL,
    NEW_ZEALAND_REGION_LABEL,
    CHILE_REGION_LABEL,
    ALEUTIAN_REGION_LABEL,
    ANTARCTIC_REGION_LABEL,
    CALIFORNIA_MEXICO_REGION_LABEL,
    KAMCHATKA_REGION_LABEL,
    NORTH_PACIFIC_WIDE_REGION_LABEL,
    MEDITERRANEAN_REGION_LABEL,
    HIMALAYA_REGION_LABEL,
    PHILIPPINES_REGION_LABEL,
]
REGION_BOUNDS = {
    JAPAN_REGION_LABEL: (120.0, 155.0, 20.0, 50.0),
    GLOBAL_REGION_LABEL: (-180.0, 180.0, -90.0, 90.0),
    INDONESIA_REGION_LABEL: (92.0, 142.0, -14.0, 11.0),
    NEW_ZEALAND_REGION_LABEL: (165.0, 185.5, -52.0, -10.0),
    CHILE_REGION_LABEL: (-80.0, -65.0, -48.0, -12.0),
    ALEUTIAN_REGION_LABEL: (-180.0, -140.0, 45.0, 62.0),
    ANTARCTIC_REGION_LABEL: (-75.0, -10.0, -70.0, -48.0),
    CALIFORNIA_MEXICO_REGION_LABEL: (-130.0, -102.0, 14.0, 42.0),
    KAMCHATKA_REGION_LABEL: (145.0, 170.0, 42.0, 62.0),
    NORTH_PACIFIC_WIDE_REGION_LABEL: (130.0, 255.0, 20.0, 65.0),
    MEDITERRANEAN_REGION_LABEL: (15.0, 45.0, 30.0, 43.0),
    HIMALAYA_REGION_LABEL: (65.0, 100.0, 20.0, 40.0),
    PHILIPPINES_REGION_LABEL: (117.0, 132.0, 5.0, 28.0),
}


def expanded_float_bounds(series, default_min, default_max, pad=1.0):
    """
    Return stable slider bounds even when the data are empty or have one value.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float(default_min), float(default_max)

    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        min_value -= pad
        max_value += pad

    return min_value, max_value


def earthquake_color_scale(color_column):
    """
    Color scales for hypocenter depth and magnitude.
    """
    if color_column == "Depth_km":
        return [
            [0.0, "red"],
            [0.08, "orange"],
            [0.18, "yellow"],
            [0.35, "lightgreen"],
            [0.55, "lightblue"],
            [0.75, "blue"],
            [1.0, "darkblue"],
        ]

    return ["green", "yellow", "orange", "red", "darkred"]


def build_datetime_range(date_range, start_clock, end_clock):
    """
    Build UTC datetime objects from Streamlit date/time widgets.
    """
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        return None, None

    start_date, end_date = date_range
    start_dt = datetime.combine(start_date, start_clock, tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, end_clock, tzinfo=timezone.utc)
    return start_dt, end_dt


def auto_map_view(df, lon_center_hint=None):
    """
    Compute a map center and zoom from the selected earthquake distribution.
    """
    lat_values = pd.to_numeric(df["Latitude_degN"], errors="coerce").dropna()
    lon_values = pd.to_numeric(df["Longitude_degE"], errors="coerce").dropna()
    if lat_values.empty or lon_values.empty:
        return 0.0, 0.0, 1.0

    lat_min, lat_max = lat_values.min(), lat_values.max()
    if lon_center_hint is not None:
        lon_values = wrap_longitudes_to_central_meridian(lon_values, float(lon_center_hint))
    lon_min, lon_max = lon_values.min(), lon_values.max()

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    lat_diff = max(lat_max - lat_min, 0.1)
    lon_diff = max(lon_max - lon_min, 0.1)

    map_width_px, map_height_px = 1200, 700
    zoom_lon = math.log2((map_width_px * 360) / (lon_diff * 256))
    zoom_lat = math.log2((map_height_px * 180) / (lat_diff * 256))
    auto_zoom = max(1, min(15, min(zoom_lon, zoom_lat) - 2.0))

    if lon_diff > 220:
        center_lat, center_lon, auto_zoom = 0.0, 0.0, 1.0

    center_lon = ((float(center_lon) + 180.0) % 360.0) - 180.0
    return center_lat, center_lon, auto_zoom


def wrap_longitudes_to_central_meridian(longitudes, central_meridian):
    """
    Wrap longitudes to [central_meridian-180, central_meridian+180).
    """
    lon_values = pd.to_numeric(pd.Series(longitudes), errors="coerce")
    return ((lon_values - central_meridian + 180.0) % 360.0) - 180.0 + central_meridian


def wrap_line_with_breaks(longitudes, latitudes, central_meridian, jump_threshold=180.0):
    """
    Wrap a line to the selected central meridian and insert NaN breaks at big jumps.
    """
    lon_wrapped = wrap_longitudes_to_central_meridian(longitudes, central_meridian).tolist()
    lat_values = pd.to_numeric(pd.Series(latitudes), errors="coerce").tolist()
    if not lon_wrapped:
        return [], []

    line_lon = [lon_wrapped[0]]
    line_lat = [lat_values[0]]
    for i in range(1, len(lon_wrapped)):
        prev_lon, curr_lon = lon_wrapped[i - 1], lon_wrapped[i]
        prev_lat, curr_lat = lat_values[i - 1], lat_values[i]
        if (
            pd.isna(prev_lon)
            or pd.isna(curr_lon)
            or pd.isna(prev_lat)
            or pd.isna(curr_lat)
            or abs(curr_lon - prev_lon) > jump_threshold
        ):
            line_lon.append(float("nan"))
            line_lat.append(float("nan"))
        line_lon.append(curr_lon)
        line_lat.append(curr_lat)

    return line_lon, line_lat


def lonlat_to_local_km(longitudes, latitudes, center_lon, center_lat, central_meridian=None):
    """
    Convert lon/lat coordinates to local equirectangular kilometers.
    """
    lon_values = pd.to_numeric(pd.Series(longitudes), errors="coerce")
    lat_values = pd.to_numeric(pd.Series(latitudes), errors="coerce")
    if central_meridian is not None:
        lon_values = wrap_longitudes_to_central_meridian(lon_values, central_meridian)

    km_per_lat_degree = 110.574
    km_per_lon_degree = 111.320 * math.cos(math.radians(center_lat))
    if abs(km_per_lon_degree) < 0.001:
        km_per_lon_degree = 0.001

    east_km = (lon_values - center_lon) * km_per_lon_degree
    north_km = (lat_values - center_lat) * km_per_lat_degree
    return east_km, north_km


def add_local_km_coordinates(df, query, pacific_centered=False):
    """
    Add local kilometer coordinates for 3D plots with correct horizontal scale.
    """
    df_km = df.copy()
    use_pacific_center = pacific_centered and query.get("region_preset") == GLOBAL_REGION_LABEL
    crosses_dateline = query["lon_min"] < -180.0 or query["lon_max"] > 180.0
    center_lon = 180.0 if use_pacific_center else (query["lon_min"] + query["lon_max"]) / 2
    center_lat = (query["lat_min"] + query["lat_max"]) / 2
    central_meridian = center_lon if (use_pacific_center or crosses_dateline) else None

    east_km, north_km = lonlat_to_local_km(
        df_km["Longitude_degE"],
        df_km["Latitude_degN"],
        center_lon,
        center_lat,
        central_meridian=central_meridian,
    )
    df_km["East_km"] = east_km
    df_km["North_km"] = north_km
    return df_km, center_lon, center_lat, use_pacific_center


def selected_area_km_ranges(
    query,
    center_lon,
    center_lat,
    z_min,
    z_max,
    pacific_centered=False,
    z_aspect_scale=0.5,
):
    """
    Convert the selected lon/lat/depth box into km ranges and aspect ratios.
    """
    is_global = query.get("region_preset") == GLOBAL_REGION_LABEL
    crosses_dateline = query["lon_min"] < -180.0 or query["lon_max"] > 180.0
    if pacific_centered and is_global:
        km_per_lat_degree = 110.574
        km_per_lon_degree = 111.320 * math.cos(math.radians(center_lat))
        km_per_lon_degree = max(abs(km_per_lon_degree), 0.001)
        x_half_span = 180.0 * km_per_lon_degree
        y_min = (query["lat_min"] - center_lat) * km_per_lat_degree
        y_max = (query["lat_max"] - center_lat) * km_per_lat_degree
        x_range = [-x_half_span, x_half_span]
        y_range = [min(float(y_min), float(y_max)), max(float(y_min), float(y_max))]
    else:
        central_meridian = center_lon if crosses_dateline else None
        x_bounds, _ = lonlat_to_local_km(
            [query["lon_min"], query["lon_max"]],
            [center_lat, center_lat],
            center_lon,
            center_lat,
            central_meridian=central_meridian,
        )
        _, y_bounds = lonlat_to_local_km(
            [center_lon, center_lon],
            [query["lat_min"], query["lat_max"]],
            center_lon,
            center_lat,
        )
        x_range = sorted([float(x_bounds.iloc[0]), float(x_bounds.iloc[1])])
        y_range = sorted([float(y_bounds.iloc[0]), float(y_bounds.iloc[1])])

    z_range = [float(z_max), float(z_min)]

    x_span = max(abs(x_range[1] - x_range[0]), 1.0)
    y_span = max(abs(y_range[1] - y_range[0]), 1.0)
    z_span = max(abs(z_max - z_min), 1.0)
    max_span = max(x_span, y_span, z_span)
    horizontal_aspect = max(x_span, y_span) / max_span

    return (
        x_range,
        y_range,
        z_range,
        dict(
            x=x_span / max_span,
            y=y_span / max_span,
            z=horizontal_aspect * max(float(z_aspect_scale), 0.05),
        ),
    )


def apply_region_bounds_to_session(region_label):
    """
    Reset lon/lat slider state to the selected preset bounds.
    """
    if region_label not in REGION_BOUNDS:
        return
    lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region_label]
    st.session_state[f"eq_lon_range_{region_label}"] = (float(lon_min), float(lon_max))
    st.session_state[f"eq_lat_range_{region_label}"] = (float(lat_min), float(lat_max))


def set_region_japan():
    """
    Keep the main-page region checkboxes mutually exclusive.
    """
    if st.session_state.eq_region_japan:
        st.session_state.eq_region_global = False
        st.session_state.eq_region_hotspot = HOTSPOT_NONE_LABEL
        st.session_state.eq_region_choice = JAPAN_REGION_LABEL
        apply_region_bounds_to_session(JAPAN_REGION_LABEL)
    elif not st.session_state.eq_region_global:
        st.session_state.eq_region_japan = True
        st.session_state.eq_region_hotspot = HOTSPOT_NONE_LABEL
        st.session_state.eq_region_choice = JAPAN_REGION_LABEL
        apply_region_bounds_to_session(JAPAN_REGION_LABEL)


def set_region_global():
    """
    Keep the main-page region checkboxes mutually exclusive.
    """
    if st.session_state.eq_region_global:
        st.session_state.eq_region_japan = False
        st.session_state.eq_region_hotspot = HOTSPOT_NONE_LABEL
        st.session_state.eq_region_choice = GLOBAL_REGION_LABEL
        apply_region_bounds_to_session(GLOBAL_REGION_LABEL)
    elif not st.session_state.eq_region_japan:
        st.session_state.eq_region_global = True
        st.session_state.eq_region_hotspot = HOTSPOT_NONE_LABEL
        st.session_state.eq_region_choice = GLOBAL_REGION_LABEL
        apply_region_bounds_to_session(GLOBAL_REGION_LABEL)


def set_region_hotspot():
    """
    Apply additional hotspot presets from the main page.
    """
    hotspot_choice = st.session_state.get("eq_region_hotspot", HOTSPOT_NONE_LABEL)
    if hotspot_choice in HOTSPOT_REGION_LABELS:
        st.session_state.eq_region_japan = False
        st.session_state.eq_region_global = False
        st.session_state.eq_region_choice = hotspot_choice
        apply_region_bounds_to_session(hotspot_choice)
        return

    if st.session_state.get("eq_region_global", False):
        st.session_state.eq_region_choice = GLOBAL_REGION_LABEL
    elif st.session_state.get("eq_region_japan", False):
        st.session_state.eq_region_choice = JAPAN_REGION_LABEL
    else:
        st.session_state.eq_region_japan = True
        st.session_state.eq_region_choice = JAPAN_REGION_LABEL
    if st.session_state.eq_region_choice in REGION_BOUNDS:
        apply_region_bounds_to_session(st.session_state.eq_region_choice)
    st.session_state.eq_region_hotspot = HOTSPOT_NONE_LABEL


def main_region_selector():
    """
    Select Japan-area or global API bounds from the main page.
    """
    if "eq_region_choice" not in st.session_state:
        st.session_state.eq_region_choice = JAPAN_REGION_LABEL
    if "eq_region_japan" not in st.session_state:
        st.session_state.eq_region_japan = True
    if "eq_region_global" not in st.session_state:
        st.session_state.eq_region_global = False
    if "eq_region_hotspot" not in st.session_state:
        if st.session_state.eq_region_choice in HOTSPOT_REGION_LABELS:
            st.session_state.eq_region_hotspot = st.session_state.eq_region_choice
        else:
            st.session_state.eq_region_hotspot = HOTSPOT_NONE_LABEL

    current_choice = st.session_state.eq_region_choice
    if current_choice == JAPAN_REGION_LABEL:
        st.session_state.eq_region_japan = True
        st.session_state.eq_region_global = False
        st.session_state.eq_region_hotspot = HOTSPOT_NONE_LABEL
    elif current_choice == GLOBAL_REGION_LABEL:
        st.session_state.eq_region_japan = False
        st.session_state.eq_region_global = True
        st.session_state.eq_region_hotspot = HOTSPOT_NONE_LABEL
    elif current_choice in HOTSPOT_REGION_LABELS:
        st.session_state.eq_region_japan = False
        st.session_state.eq_region_global = False
        st.session_state.eq_region_hotspot = current_choice

    st.subheader("Region")
    col_japan, col_global = st.columns(2)
    with col_japan:
        st.checkbox(
            JAPAN_REGION_LABEL,
            key="eq_region_japan",
            on_change=set_region_japan,
        )
    with col_global:
        st.checkbox(
            GLOBAL_REGION_LABEL,
            key="eq_region_global",
            on_change=set_region_global,
        )

    st.selectbox(
        "Additional seismic hotspot presets",
        [HOTSPOT_NONE_LABEL] + HOTSPOT_REGION_LABELS,
        key="eq_region_hotspot",
        on_change=set_region_hotspot,
    )

    hotspot_choice = st.session_state.get("eq_region_hotspot", HOTSPOT_NONE_LABEL)
    if hotspot_choice in HOTSPOT_REGION_LABELS:
        st.caption(f"Selected hotspot: {hotspot_choice}")
        st.session_state.eq_region_choice = hotspot_choice
        return hotspot_choice

    if st.session_state.eq_region_global:
        return GLOBAL_REGION_LABEL
    return JAPAN_REGION_LABEL


def sidebar_controls(region_preset):
    """
    Sidebar controls for USGS API query parameters.
    """
    now_utc = datetime.now(timezone.utc)
    default_end_date = now_utc.date()
    default_start_date = default_end_date - timedelta(days=30)

    default_lon_min, default_lon_max, default_lat_min, default_lat_max = REGION_BOUNDS[region_preset]

    with st.sidebar.form("earthquake_api_parameter", clear_on_submit=False):
        st.header(":blue[--- USGS Earthquake API ---]")
        date_range = st.date_input(
            "Date range (UTC)",
            value=(default_start_date, default_end_date),
            key="eq_date_range",
        )

        col_start, col_end = st.columns(2)
        with col_start:
            start_clock = st.time_input(
                "Start time",
                value=time(0, 0),
                step=3600,
                key="eq_start_clock",
            )
        with col_end:
            end_clock = st.time_input(
                "End time",
                value=time(23, 59),
                step=3600,
                key="eq_end_clock",
            )

        mag_min, mag_max = st.slider(
            "Magnitude",
            min_value=0.0,
            max_value=10.0,
            value=(4.5, 10.0),
            step=0.1,
            key="eq_magnitude_range",
        )

        depth_min, depth_max = st.slider(
            "Hypocenter depth (km)",
            min_value=-100.0,
            max_value=1000.0,
            value=(0.0, 700.0),
            step=10.0,
            key="eq_depth_range",
        )

        with st.expander("Latitude / Longitude", expanded=True):
            lon_min, lon_max = st.slider(
                "Longitude",
                min_value=-180.0,
                max_value=360.0,
                value=(default_lon_min, default_lon_max),
                step=0.5,
                key=f"eq_lon_range_{region_preset}",
            )
            lat_min, lat_max = st.slider(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=(default_lat_min, default_lat_max),
                step=0.5,
                key=f"eq_lat_range_{region_preset}",
            )

        orderby = st.selectbox(
            "Order by",
            ["time", "time-asc", "magnitude", "magnitude-asc"],
            index=0,
            key="eq_orderby",
            help=(
                "**Select data priority / データの優先順位を選択:**\n\n"
                "- **time**: Newest first / 新しい順 (Default)\n"
                "- **time-asc**: Oldest first / 古い順\n"
                "- **magnitude**: Largest first / マグニチュードが大きい順\n"
                "- **magnitude-asc**: Smallest first / マグニチュードが小さい順\n\n"
                "**Usage Tip / 使い方:**\n\n"
                "If you want to quickly see recent events, keep it set to **'time'**. "
                "However, if you want to ensure large historical earthquakes in a specific area are displayed without being missed due to the API limit, switching to **'magnitude'** is more effective.\n\n"
                "最近の地震を見たい場合は **'time'** 、巨大地震を優先して表示したい場合は、**'magnitude'** で。"
            )
        )

        limit = st.number_input(
            "Max events",
            min_value=1,
            max_value=20000,
            value=2000,
            step=100,
            key="eq_limit",
        )

        st.form_submit_button(":red[Fetch / update]")

    start_dt, end_dt = build_datetime_range(date_range, start_clock, end_clock)
    if start_dt is None or end_dt is None:
        st.warning("Please select both start and end dates.")
        st.stop()
    if start_dt > end_dt:
        st.warning("Start datetime must be earlier than end datetime.")
        st.stop()
    if lon_max <= lon_min:
        st.warning("Longitude min must be smaller than longitude max.")
        st.stop()
    if (lon_max - lon_min) > 360.0:
        st.warning(
            "Longitude range is too wide for USGS rectangle query. "
            "Please use 360° or less (e.g., -20 to 340, not -20 to 360)."
        )
        st.stop()

    return {
        "region_preset": region_preset,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "mag_min": mag_min,
        "mag_max": mag_max,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "limit": int(limit),
        "orderby": orderby,
    }


def fetch_earthquake_dataframe(query):
    """
    Fetch earthquake records from USGS with the sidebar query.
    """
    with st.spinner("Fetching earthquake hypocenters from USGS..."):
        try:
            return envgeo_utils.load_usgs_earthquake_data(
                query["start_dt"],
                query["end_dt"],
                minmagnitude=query["mag_min"],
                maxmagnitude=query["mag_max"],
                mindepth=query["depth_min"],
                maxdepth=query["depth_max"],
                minlatitude=query["lat_min"],
                maxlatitude=query["lat_max"],
                minlongitude=query["lon_min"],
                maxlongitude=query["lon_max"],
                limit=query["limit"],
                orderby=query["orderby"],
            )
        except RuntimeError as e:
            st.error(str(e))
            st.stop()


def prepare_plot_dataframe(df_eq):
    """
    Keep plottable hypocenter rows and define marker sizes from magnitude.
    """
    df_plot = df_eq.dropna(subset=["Longitude_degE", "Latitude_degN", "Depth_km"]).copy()
    if df_plot.empty:
        return df_plot

    magnitude = pd.to_numeric(df_plot["Magnitude"], errors="coerce").fillna(0).clip(lower=0)
    df_plot["MarkerSize"] = (magnitude + 1.0) ** 2
    df_plot["MagnitudeMarkerSize"] = 2.0 + magnitude * 3.0
    return df_plot


def visualization_controls(df_plot, query):
    """
    Sidebar and main-panel controls for 4D rendering.
    """
    with st.sidebar.container(border=True):
        st.subheader(":blue[--- Visualization ---]")

        depth_min_actual, depth_max_actual = expanded_float_bounds(
            df_plot["Depth_km"], query["depth_min"], query["depth_max"], pad=10.0
        )
        depth_slider_min = float(math.floor(depth_min_actual / 10.0) * 10.0)
        depth_slider_max = float(math.ceil(depth_max_actual / 10.0) * 10.0)
        if depth_slider_min == depth_slider_max:
            depth_slider_max += 10.0

        fig_depth_min, fig_depth_max = st.slider(
            "Depth scale (km)",
            min_value=depth_slider_min,
            max_value=1000.0,
            value=(depth_slider_min, depth_slider_max),
            step=10.0,
            key="eq_fig_depth_scale",
        )

        marker_size_scale_3d = st.slider(
            "3D marker size scale",
            min_value=0.2,
            max_value=3.0,
            value=0.7,
            step=0.1,
            key="eq_marker_size_scale_3d",
        )

        marker_size_scale_2d = st.slider(
            "2D map marker size scale",
            min_value=0.2,
            max_value=3.0,
            value=0.6,
            step=0.1,
            key="eq_marker_size_scale_2d",
        )

        z_aspect_scale_3d = st.slider(
            "3D Z-axis display scale",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            key="eq_z_aspect_scale_3d",
            help="Lower values flatten depth; higher values emphasize depth.",
        )

        pacific_center_3d = st.checkbox(
            "3D Pacific-centered view (180°)",
            value=(query.get("region_preset") == GLOBAL_REGION_LABEL),
            disabled=(query.get("region_preset") != GLOBAL_REGION_LABEL),
            key="eq_pacific_center_3d",
        )

        color_option = st.radio(
            "Colorbar variable",
            ["Magnitude", "Hypocenter depth"],
            horizontal=False,
            key="eq_color_option",
        )

    color_column = "Depth_km" if color_option == "Hypocenter depth" else "Magnitude"
    color_label = "Depth (km)" if color_column == "Depth_km" else "Magnitude"

    c_min_actual, c_max_actual = expanded_float_bounds(
        df_plot[color_column], 0.0, 1.0, pad=1.0
    )
    c_step = 10.0 if color_column == "Depth_km" else 0.1
    c_slider_min = float(math.floor(c_min_actual / c_step) * c_step)
    c_slider_max = float(math.ceil(c_max_actual / c_step) * c_step)
    if c_slider_min == c_slider_max:
        c_slider_max += c_step

    color_range = st.slider(
        f"Colorbar scale adjustment: {color_label}",
        min_value=0.0,
        max_value=c_slider_max,
        value=(c_slider_min, c_slider_max),
        step=c_step,
        key=f"eq_colorbar_{color_column}",
    )

    return {
        "fig_depth_min": fig_depth_min,
        "fig_depth_max": fig_depth_max,
        "marker_size_scale_3d": marker_size_scale_3d,
        "marker_size_scale_2d": marker_size_scale_2d,
        "z_aspect_scale_3d": z_aspect_scale_3d,
        "pacific_center_3d": pacific_center_3d and query.get("region_preset") == GLOBAL_REGION_LABEL,
        "color_column": color_column,
        "color_label": color_label,
        "color_range": color_range,
    }


def render_4d_hypocenter_map(df_plot, query, viz):
    """
    Render the EnvGeo-style 4D hypocenter map.
    """
    df_plot, center_lon, center_lat, using_pacific_center = add_local_km_coordinates(
        df_plot,
        query,
        pacific_centered=viz.get("pacific_center_3d", False),
    )
    crosses_dateline = query["lon_min"] < -180.0 or query["lon_max"] > 180.0
    line_central_meridian = center_lon if (using_pacific_center or crosses_dateline) else None

    # Plotly 3D/WebGL marker sizes can appear much larger than 2D markers,
    # and the apparent size can differ by browser.  Use a separate, conservative
    # 3D marker-size column instead of Plotly Express size normalization.
    magnitude_3d = pd.to_numeric(df_plot["Magnitude"], errors="coerce").fillna(0).clip(lower=0)
    df_plot["MarkerSize3D"] = (1.8 + magnitude_3d * 0.8) * viz["marker_size_scale_3d"]
    df_plot["MarkerSize3D"] = df_plot["MarkerSize3D"].clip(lower=1.5, upper=16.0)

    df_plot = df_plot.sort_values(by=["Depth_km", "Magnitude"], ascending=[False, True])

    x_range, y_range, z_range, aspectratio = selected_area_km_ranges(
        query,
        center_lon,
        center_lat,
        viz["fig_depth_min"],
        viz["fig_depth_max"],
        pacific_centered=using_pacific_center,
        z_aspect_scale=viz["z_aspect_scale_3d"],
    )

    fig_eq = px.scatter_3d(
        df_plot,
        x="East_km",
        y="North_km",
        z="Depth_km",
        color=viz["color_column"],
        width=700,
        height=620,
        color_continuous_scale=earthquake_color_scale(viz["color_column"]),
        hover_data={
            "DateTime_UTC": True,
            "Place": True,
            "Magnitude": True,
            "MagnitudeType": True,
            "Depth_km": True,
            "Longitude_degE": True,
            "Latitude_degN": True,
            "EventID": True,
            "East_km": False,
            "North_km": False,
            "MarkerSize": False,
            "MarkerSize3D": False,
        },
    )

    fig_eq.update_traces(
        mode="markers",
        marker=dict(
            size=df_plot["MarkerSize3D"].tolist(),
            opacity=0.72,
            line=dict(color="rgba(255,255,255,0.0)", width=0.0),
        ),
        name="USGS earthquakes",
    )

    fig_eq.update_layout(
        scene=dict(
            xaxis_title="East-West distance (km)",
            yaxis_title="North-South distance (km)",
            zaxis_title="Hypocenter depth (km)",
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(
                range=z_range,
                autorange=False,
            ),
            aspectmode="manual",
            aspectratio=aspectratio,
            camera=dict(
                eye=dict(x=-0.8, y=-0.9, z=1.8),
                center=dict(x=0, y=0, z=-0.1),
            ),
        ),
        coloraxis_colorbar=dict(
            title=viz["color_label"],
            orientation="h",
            yanchor="top",
            y=-0.15,
            x=0.5,
            xanchor="center",
            thickness=15,
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.55)",
            bordercolor="rgba(80,80,80,0.35)",
            borderwidth=1,
            tracegroupgap=2,
        ),
        margin=dict(r=20, l=10, b=110, t=10),
    )
    fig_eq.update_coloraxes(
        cmin=viz["color_range"][0],
        cmax=viz["color_range"][1],
    )

    coastline_x, coastline_y = envgeo_utils.load_coastline_data(envgeo_utils.data_source_GLOBAL)
    if coastline_x and coastline_y:
        if line_central_meridian is not None:
            coastline_lon_plot, coastline_lat_plot = wrap_line_with_breaks(
                coastline_x,
                coastline_y,
                line_central_meridian,
            )
        else:
            coastline_lon_plot, coastline_lat_plot = coastline_x, coastline_y

        coastline_east, coastline_north = lonlat_to_local_km(
            coastline_lon_plot,
            coastline_lat_plot,
            center_lon,
            center_lat,
            central_meridian=line_central_meridian,
        )
        fig_eq.add_trace(
            go.Scatter3d(
                x=coastline_east,
                y=coastline_north,
                z=[viz["fig_depth_min"]] * len(coastline_east),
                mode="lines",
                name="coastline (top)",
                line=dict(color="blue", width=0.8),
                hoverinfo="none",
            )
        )
        fig_eq.add_trace(
            go.Scatter3d(
                x=coastline_east,
                y=coastline_north,
                z=[viz["fig_depth_max"]] * len(coastline_east),
                mode="lines",
                name="coastline (bottom)",
                line=dict(color="gray", width=0.5),
                hoverinfo="none",
            )
        )

    st.plotly_chart(
        fig_eq,
        key="earthquake_4d_hypocenter_map",
        config={"scrollZoom": True, "displayModeBar": True},
    )


def render_2d_distribution_map(df_plot, query, viz):
    """
    Render the selected hypocenters on an interactive map.
    """
    st.divider()
    st.subheader("Geographical Distribution Map (Auto-Zoom)")

    map_mode = st.radio(
        "Map Style:",
        ["Standard", "Satellite", "Bathymetry (Sea)", "Contour (GSI)"],
        horizontal=True,
        key="eq_map_style",
    )

    lon_center_hint = None
    if query["lon_min"] < -180.0 or query["lon_max"] > 180.0:
        lon_center_hint = (query["lon_min"] + query["lon_max"]) / 2
    center_lat, center_lon, auto_zoom = auto_map_view(df_plot, lon_center_hint=lon_center_hint)
    df_map = df_plot.copy()
    df_map["MagnitudeMarkerSize"] = df_map["MagnitudeMarkerSize"] * viz["marker_size_scale_2d"]

    fig_map = px.scatter_mapbox(
        df_map,
        lat="Latitude_degN",
        lon="Longitude_degE",
        color=viz["color_column"],
        color_continuous_scale=earthquake_color_scale(viz["color_column"]),
        hover_data={
            "DateTime_UTC": True,
            "Place": True,
            "Magnitude": True,
            "Depth_km": True,
            "EventID": True,
            "MagnitudeMarkerSize": False,
            "MarkerSize": False,
        },
        opacity=0.65,
        height=520,
    )
    fig_map.update_traces(
        marker=dict(size=df_map["MagnitudeMarkerSize"].tolist())
    )
    fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)
    fig_map.update_layout(
        coloraxis_colorbar=dict(
            title=viz["color_label"],
            orientation="h",
            yanchor="top",
            y=-0.15,
            x=0.5,
            xanchor="center",
            thickness=15,
        ),
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
        margin=dict(l=0, r=0, t=0, b=100),
        autosize=True,
    )
    fig_map.update_coloraxes(
        cmin=viz["color_range"][0],
        cmax=viz["color_range"][1],
    )

    st.plotly_chart(
        fig_map,
        key="earthquake_distribution_map",
        config={"scrollZoom": True, "displayModeBar": True},
    )


def display_earthquake_table(df_eq):
    """
    Display the USGS dataframe and offer a CSV download.
    """
    with st.expander("selected earthquake dataset (CSV)", expanded=False):
        table_cols = [
            "EventID",
            "DateTime_UTC",
            "Magnitude",
            "MagnitudeType",
            "Depth_km",
            "Longitude_degE",
            "Latitude_degN",
            "Place",
            "URL",
        ]
        df_table = df_eq[[col for col in table_cols if col in df_eq.columns]].copy()
        df_table = df_table.astype(str).replace(["<NA>", "nan", "NaT", "None"], "")
        st.dataframe(df_table)
        st.download_button(
            "Download CSV",
            data=df_table.to_csv(index=False).encode("utf-8"),
            file_name="usgs_earthquake_catalog.csv",
            mime="text/csv",
        )


def main():
    st.title(f"EnvGeo-Earthquake")
    st.header(f"4D Visualizer Earthquake ({version})")
    st.caption("Source: USGS Earthquake Catalog. Data may be preliminary and updated.")
    st.caption("震源データ: USGS Earthquake Catalog。速報値を含み、更新される場合があります。")

    with st.expander("Data use note / データ利用上の注意", expanded=False):
        st.write(
            "This visualization is for research and educational use only. "
            "For emergency response or official earthquake information, refer to official agencies."
        )
        st.write(
            "本ページは研究・教育・可視化を目的としたものです。"
            "防災判断や緊急対応には、必ず気象庁などの公式情報を確認してください。"
        )
        st.write("Data source: USGS Earthquake Catalog API (GeoJSON, eventtype=earthquake).")
        st.write(
            "Earthquake data are accessed from the USGS Earthquake Catalog. "
            "USGS data may be revised after publication."
        )

    region_preset = main_region_selector()
    query = sidebar_controls(region_preset)
    df_eq = fetch_earthquake_dataframe(query)

    query_url = df_eq.attrs.get("query_url", "")
    st.write(f"{len(df_eq)} earthquake events found")
    if query_url:
        st.markdown(f"[USGS API query]({query_url})")
    if len(df_eq) >= query["limit"]:
        st.caption(
            "⚠️ "
            f"The retrieval limit of {query['limit']} events was reached, so you may be seeing only a subset of all matching events. "
            "To view recent earthquakes, choose 'time' in 'Order by' on the sidebar. "
            "To prioritize larger earthquakes, choose 'magnitude'. "
            "You can change the limit in 'Max events'. / "
            f"{query['limit']}件の取得上限に達しました。条件に一致する全件ではなく一部のみ表示している可能性があります。"
            "最近の地震を見たい場合はサイドバーの「Order by」で 'time' 、巨大地震を優先して表示したい場合は、"
            "'magnitude' を選択してください。上限値の変更は「Max events」です。"
        )

    if df_eq.empty:
        st.warning("No earthquake data available for the selected conditions.")
        return

    df_plot = prepare_plot_dataframe(df_eq)
    if df_plot.empty:
        st.warning("No plottable hypocenter data were returned.")
        return

    st.subheader("4D Hypocenter Map")
    viz = visualization_controls(df_plot, query)
    render_4d_hypocenter_map(df_plot, query, viz)
    render_2d_distribution_map(df_plot, query, viz)
    display_earthquake_table(df_eq)

    if st.sidebar.button("Reload / clear API cache"):
        envgeo_utils.clear_app_cache()
        st.rerun()

if __name__ == "__main__":
    main()
