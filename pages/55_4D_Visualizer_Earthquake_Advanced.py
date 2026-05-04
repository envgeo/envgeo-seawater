#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USGS earthquake hypocenter 4D visualizer for EnvGeo.
Created on Sun May 1 2026
Created from 04_4D_Visualizer.py and simplified as an earthquake-only page.
"""

import math
import json
from datetime import datetime, time, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import envgeo_utils


version = "0.2.0"


JAPAN_REGION_LABEL = "Japan and surrounding area"
GLOBAL_REGION_LABEL = "Global"
USGS_PLATE_BOUNDARY_SERVICE = (
    "https://earthquake.usgs.gov/arcgis/rest/services/eq/map_plateboundaries/MapServer"
)
USGS_EVENT_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/"
USGS_COMCAT_FDSN_URL = "https://www.fdsn.org/datacenters/detail/USGS/"
USGS_CREDIT_URL = "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"
PLATE_LAYER_IDS = {
    1: "Plates",
    0: "Microplates",
}
PLATE_BOUNDARY_COLORS = {
    "Convergent Boundary": "#b30000",
    "Divergent Boundary": "#006d9c",
    "Transform Boundary": "#6a3d9a",
    "Other": "#4d4d4d",
    "Approximate": "#7f7f7f",
}
JMA_BULLETIN_URL = "https://www.data.jma.go.jp/eqev/data/bulletin/index_e.html"
JMA_EARTHQUAKE_INFO_URL = "https://www.data.jma.go.jp/eqev/data/en/guide/earthinfo.html"
NIED_HINET_DATA_URL = "https://www.hinet.bosai.go.jp/about_data/?LANG=en"
PLATE_BOUNDARY_NOTE_EN = (
    "Plate boundaries are from the USGS Tectonic Plate Boundaries service. "
    "Sources cited in the USGS service metadata include Bird (2003) and DeMets et al. (2010). "
    "Boundary locations are approximate and are intended for educational/research visualization, "
    "not for official hazard assessment or disaster-response decisions."
)
PLATE_BOUNDARY_NOTE_JA = (
    "プレート境界は USGS Tectonic Plate Boundaries service を使用しています。"
    "USGSサービスのメタデータに基づき、主な出典には Bird (2003) および DeMets et al. (2010) が含まれます。"
    "境界位置は概略であり、教育・研究用の可視化を目的としたもので、"
    "公式なハザード評価や防災判断には使用しないでください。"
)
PLATE_BOUNDARY_FALLBACK_NOTE_EN = (
    "If the USGS plate-boundary service cannot be reached, the Japan-area fallback lines are "
    "rough schematic guides only."
)
PLATE_BOUNDARY_FALLBACK_NOTE_JA = (
    "USGSのプレート境界サービスに接続できない場合に表示される日本周辺のフォールバック線は、"
    "概略的な模式線です。"
)


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


def fallback_japan_plate_boundary_features():
    """
    Minimal Japan-area fallback lines used only when the USGS plate service is unavailable.
    """
    fallback_lines = [
        (
            "Japan Trench (approx.)",
            "Approximate",
            [(143.7, 34.0), (143.5, 36.0), (144.2, 38.5), (145.2, 40.5), (146.5, 42.8)],
        ),
        (
            "Kuril Trench (approx.)",
            "Approximate",
            [(146.5, 42.8), (148.5, 44.0), (151.0, 45.5), (154.0, 47.0)],
        ),
        (
            "Izu-Bonin Trench (approx.)",
            "Approximate",
            [(142.0, 24.0), (142.5, 27.0), (143.0, 30.0), (143.5, 33.5)],
        ),
        (
            "Nankai Trough (approx.)",
            "Approximate",
            [(131.0, 31.0), (133.0, 32.0), (136.0, 33.0), (139.0, 34.5)],
        ),
        (
            "Ryukyu Trench (approx.)",
            "Approximate",
            [(122.0, 23.5), (125.0, 25.0), (128.0, 27.0), (131.0, 29.0)],
        ),
        (
            "Sagami Trough (approx.)",
            "Approximate",
            [(138.7, 34.0), (139.5, 34.6), (140.5, 35.0), (141.2, 35.2)],
        ),
    ]

    features = []
    for name, label, coordinates in fallback_lines:
        features.append(
            {
                "type": "Feature",
                "properties": {"NAME": name, "LABEL": label, "Layer": "Approximate Japan"},
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
    return features


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_usgs_plate_boundary_features(include_microplates):
    """
    Fetch USGS tectonic plate boundary GeoJSON features from the ArcGIS REST service.
    """
    layer_ids = [1]
    if include_microplates:
        layer_ids.append(0)

    features = []
    errors = []
    for layer_id in layer_ids:
        offset = 0
        page_size = 1000
        while True:
            params = {
                "where": "1=1",
                "outFields": "NAME,LABEL",
                "returnGeometry": "true",
                "outSR": 4326,
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }
            query_url = f"{USGS_PLATE_BOUNDARY_SERVICE}/{layer_id}/query?{urlencode(params)}"
            request = Request(
                query_url,
                headers={"User-Agent": "EnvGeo-Earthquake visualizer"},
            )
            try:
                with urlopen(request, timeout=30) as response:
                    payload = json.load(response)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
                errors.append(f"{PLATE_LAYER_IDS[layer_id]}: {e}")
                break

            page_features = payload.get("features", []) if isinstance(payload, dict) else []
            for feature in page_features:
                feature.setdefault("properties", {})
                feature["properties"]["Layer"] = PLATE_LAYER_IDS[layer_id]
            features.extend(page_features)

            if len(page_features) < page_size:
                break
            offset += page_size
            if offset >= 10000:
                break

    return features, errors


def iter_line_coordinates(geometry):
    """
    Yield LineString coordinate arrays from GeoJSON LineString/MultiLineString geometry.
    """
    if not geometry:
        return

    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geom_type == "LineString":
        yield coordinates
    elif geom_type == "MultiLineString":
        for line in coordinates:
            yield line


def coordinate_in_query_window(lon, lat, query, buffer_degrees=4.0):
    """
    Keep plate boundary vertices close to the selected plotting window.
    """
    lon_span = query["lon_max"] - query["lon_min"]
    lat_span = query["lat_max"] - query["lat_min"]
    if lon_span >= 350 and lat_span >= 170:
        return True

    return (
        query["lon_min"] - buffer_degrees <= lon <= query["lon_max"] + buffer_degrees
        and query["lat_min"] - buffer_degrees <= lat <= query["lat_max"] + buffer_degrees
    )


def plate_features_to_dataframe(features, query):
    """
    Convert plate boundary GeoJSON features into a Plotly-friendly dataframe.
    """
    rows = []
    segment_id = 0
    for feature in features:
        properties = feature.get("properties") or {}
        name = properties.get("NAME") or "Plate boundary"
        label = properties.get("LABEL") or "Other"
        layer = properties.get("Layer") or "Plates"

        for line in iter_line_coordinates(feature.get("geometry") or {}):
            previous_lon = None
            has_points = False
            for coord in line:
                if len(coord) < 2:
                    continue
                lon, lat = float(coord[0]), float(coord[1])
                crosses_dateline = previous_lon is not None and abs(lon - previous_lon) > 180
                in_window = coordinate_in_query_window(lon, lat, query)

                if crosses_dateline or not in_window:
                    if has_points:
                        rows.append(
                            {
                                "Longitude_degE": None,
                                "Latitude_degN": None,
                                "Name": name,
                                "Label": label,
                                "Layer": layer,
                                "SegmentID": segment_id,
                            }
                        )
                        segment_id += 1
                        has_points = False

                if in_window:
                    rows.append(
                        {
                            "Longitude_degE": lon,
                            "Latitude_degN": lat,
                            "Name": name,
                            "Label": label,
                            "Layer": layer,
                            "SegmentID": segment_id,
                        }
                    )
                    has_points = True
                previous_lon = lon

            if has_points:
                rows.append(
                    {
                        "Longitude_degE": None,
                        "Latitude_degN": None,
                        "Name": name,
                        "Label": label,
                        "Layer": layer,
                        "SegmentID": segment_id,
                    }
                )
                segment_id += 1

    return pd.DataFrame(rows)


def load_plate_boundary_dataframe(query, include_microplates):
    """
    Load plate boundaries, filtered to the current map extent.
    """
    features, errors = fetch_usgs_plate_boundary_features(include_microplates)
    source_label = "USGS Tectonic Plate Boundaries"
    if errors and not features and query["region_preset"] == JAPAN_REGION_LABEL:
        features = fallback_japan_plate_boundary_features()
        source_label = "Approximate Japan plate-boundary fallback"

    if not features:
        return pd.DataFrame(), source_label, errors

    return plate_features_to_dataframe(features, query), source_label, errors


def plate_boundary_trace_style(label):
    """
    Return the map color/width for a plate boundary class.
    """
    color = PLATE_BOUNDARY_COLORS.get(label, PLATE_BOUNDARY_COLORS["Other"])
    width = 2.8 if label in ["Convergent Boundary", "Approximate"] else 2.0
    return color, width


def render_plate_boundary_note():
    """
    Display concise plate-boundary source and use notes.
    """
    st.write(PLATE_BOUNDARY_NOTE_EN)
    st.write(PLATE_BOUNDARY_NOTE_JA)
    st.write(PLATE_BOUNDARY_FALLBACK_NOTE_EN)
    st.write(PLATE_BOUNDARY_FALLBACK_NOTE_JA)
    st.markdown(f"[USGS Tectonic Plate Boundaries service]({USGS_PLATE_BOUNDARY_SERVICE})")
    st.markdown("- Bird (2003): https://doi.org/10.1029/2001GC000252")
    st.markdown("- DeMets et al. (2010): https://doi.org/10.1111/j.1365-246X.2009.04491.x")


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


def auto_map_view(df):
    """
    Compute a map center and zoom from the selected earthquake distribution.
    """
    lat_min, lat_max = df["Latitude_degN"].min(), df["Latitude_degN"].max()
    lon_min, lon_max = df["Longitude_degE"].min(), df["Longitude_degE"].max()

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

    return center_lat, center_lon, auto_zoom


def lonlat_to_local_km(longitudes, latitudes, center_lon, center_lat):
    """
    Convert lon/lat coordinates to local equirectangular kilometers.
    """
    lon_values = pd.to_numeric(pd.Series(longitudes), errors="coerce")
    lat_values = pd.to_numeric(pd.Series(latitudes), errors="coerce")

    km_per_lat_degree = 110.574
    km_per_lon_degree = 111.320 * math.cos(math.radians(center_lat))
    if abs(km_per_lon_degree) < 0.001:
        km_per_lon_degree = 0.001

    east_km = (lon_values - center_lon) * km_per_lon_degree
    north_km = (lat_values - center_lat) * km_per_lat_degree
    return east_km, north_km


def add_local_km_coordinates(df, query):
    """
    Add local kilometer coordinates for 3D plots with correct horizontal scale.
    """
    df_km = df.copy()
    center_lon = (query["lon_min"] + query["lon_max"]) / 2
    center_lat = (query["lat_min"] + query["lat_max"]) / 2

    east_km, north_km = lonlat_to_local_km(
        df_km["Longitude_degE"],
        df_km["Latitude_degN"],
        center_lon,
        center_lat,
    )
    df_km["East_km"] = east_km
    df_km["North_km"] = north_km
    return df_km, center_lon, center_lat


def selected_area_km_ranges(query, center_lon, center_lat, z_min, z_max):
    """
    Convert the selected lon/lat/depth box into km ranges and aspect ratios.
    """
    x_bounds, _ = lonlat_to_local_km(
        [query["lon_min"], query["lon_max"]],
        [center_lat, center_lat],
        center_lon,
        center_lat,
    )
    _, y_bounds = lonlat_to_local_km(
        [center_lon, center_lon],
        [query["lat_min"], query["lat_max"]],
        center_lon,
        center_lat,
    )

    x_range = [float(x_bounds.iloc[0]), float(x_bounds.iloc[1])]
    y_range = [float(y_bounds.iloc[0]), float(y_bounds.iloc[1])]
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
            z=horizontal_aspect * 0.5,
        ),
    )


def default_cross_section_points(query):
    """
    Provide reasonable default section endpoints for Japan/global views.
    """
    if query["region_preset"] == JAPAN_REGION_LABEL:
        return 130.0, 32.0, 146.0, 41.5

    center_lat = (query["lat_min"] + query["lat_max"]) / 2
    return query["lon_min"], center_lat, query["lon_max"], center_lat


def add_cross_section_coordinates(df_plot, start_lon, start_lat, end_lon, end_lat):
    """
    Project hypocenters onto an arbitrary section line in local km coordinates.
    """
    center_lon = (start_lon + end_lon) / 2
    center_lat = (start_lat + end_lat) / 2
    df_section = df_plot.copy()
    east_km, north_km = lonlat_to_local_km(
        df_section["Longitude_degE"],
        df_section["Latitude_degN"],
        center_lon,
        center_lat,
    )

    endpoint_east, endpoint_north = lonlat_to_local_km(
        [start_lon, end_lon],
        [start_lat, end_lat],
        center_lon,
        center_lat,
    )
    start_x, end_x = float(endpoint_east.iloc[0]), float(endpoint_east.iloc[1])
    start_y, end_y = float(endpoint_north.iloc[0]), float(endpoint_north.iloc[1])

    dx = end_x - start_x
    dy = end_y - start_y
    length_km = math.hypot(dx, dy)
    if length_km < 1.0:
        return df_section.iloc[0:0].copy(), 0.0

    df_section["SectionDistance_km"] = ((east_km - start_x) * dx + (north_km - start_y) * dy) / length_km
    df_section["SectionOffset_km"] = ((east_km - start_x) * (-dy) + (north_km - start_y) * dx) / length_km
    return df_section, length_km


def local_km_to_lonlat(east_km, north_km, center_lon, center_lat):
    """
    Convert local equirectangular kilometers back to lon/lat coordinates.
    """
    east_values = pd.to_numeric(pd.Series(east_km), errors="coerce")
    north_values = pd.to_numeric(pd.Series(north_km), errors="coerce")

    km_per_lat_degree = 110.574
    km_per_lon_degree = 111.320 * math.cos(math.radians(center_lat))
    if abs(km_per_lon_degree) < 0.001:
        km_per_lon_degree = 0.001

    lon_values = center_lon + east_values / km_per_lon_degree
    lat_values = center_lat + north_values / km_per_lat_degree
    return lon_values, lat_values


def cross_section_corridor_dataframe(start_lon, start_lat, end_lon, end_lat, half_width_km):
    """
    Build a lon/lat polygon showing the selected cross-section swath.
    """
    center_lon = (start_lon + end_lon) / 2
    center_lat = (start_lat + end_lat) / 2
    endpoint_east, endpoint_north = lonlat_to_local_km(
        [start_lon, end_lon],
        [start_lat, end_lat],
        center_lon,
        center_lat,
    )
    start_x, end_x = float(endpoint_east.iloc[0]), float(endpoint_east.iloc[1])
    start_y, end_y = float(endpoint_north.iloc[0]), float(endpoint_north.iloc[1])
    dx = end_x - start_x
    dy = end_y - start_y
    length_km = math.hypot(dx, dy)
    if length_km < 1.0:
        return pd.DataFrame()

    normal_x = -dy / length_km
    normal_y = dx / length_km
    polygon_east = [
        start_x + normal_x * half_width_km,
        end_x + normal_x * half_width_km,
        end_x - normal_x * half_width_km,
        start_x - normal_x * half_width_km,
        start_x + normal_x * half_width_km,
    ]
    polygon_north = [
        start_y + normal_y * half_width_km,
        end_y + normal_y * half_width_km,
        end_y - normal_y * half_width_km,
        start_y - normal_y * half_width_km,
        start_y + normal_y * half_width_km,
    ]
    polygon_lon, polygon_lat = local_km_to_lonlat(
        polygon_east,
        polygon_north,
        center_lon,
        center_lat,
    )
    return pd.DataFrame(
        {
            "Longitude_degE": polygon_lon,
            "Latitude_degN": polygon_lat,
        }
    )


def depth_profile_dataframe(df_plot, bin_size_km):
    """
    Build a simple depth-frequency profile.
    """
    df_depth = df_plot.dropna(subset=["Depth_km"]).copy()
    if df_depth.empty:
        return pd.DataFrame()

    df_depth["DepthBinTop_km"] = (df_depth["Depth_km"] // bin_size_km) * bin_size_km
    grouped = (
        df_depth.groupby("DepthBinTop_km", as_index=False)
        .agg(
            Count=("Depth_km", "size"),
            MeanMagnitude=("Magnitude", "mean"),
            MaxMagnitude=("Magnitude", "max"),
        )
        .sort_values("DepthBinTop_km")
    )
    grouped["DepthMid_km"] = grouped["DepthBinTop_km"] + bin_size_km / 2
    return grouped


def set_region_japan():
    """
    Keep the main-page region checkboxes mutually exclusive.
    """
    if st.session_state.eq_region_japan:
        st.session_state.eq_region_global = False
        st.session_state.eq_region_choice = JAPAN_REGION_LABEL
    elif not st.session_state.eq_region_global:
        st.session_state.eq_region_japan = True
        st.session_state.eq_region_choice = JAPAN_REGION_LABEL


def set_region_global():
    """
    Keep the main-page region checkboxes mutually exclusive.
    """
    if st.session_state.eq_region_global:
        st.session_state.eq_region_japan = False
        st.session_state.eq_region_choice = GLOBAL_REGION_LABEL
    elif not st.session_state.eq_region_japan:
        st.session_state.eq_region_global = True
        st.session_state.eq_region_choice = GLOBAL_REGION_LABEL


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

    region_defaults = {
        JAPAN_REGION_LABEL: (120.0, 155.0, 20.0, 50.0),
        GLOBAL_REGION_LABEL: (-180.0, 180.0, -90.0, 90.0),
    }

    default_lon_min, default_lon_max, default_lat_min, default_lat_max = region_defaults[region_preset]

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
                max_value=180.0,
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

        marker_size_scale = st.slider(
            "Marker size scale",
            min_value=0.2,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="eq_marker_size_scale",
        )

        color_option = st.radio(
            "Colorbar variable",
            ["Magnitude", "Hypocenter depth"],
            horizontal=False,
            key="eq_color_option",
        )

        show_plate_boundaries = st.checkbox(
            "Overlay plate boundaries",
            value=True,
            key="eq_show_plate_boundaries",
        )
        include_microplates = st.checkbox(
            "Include microplates",
            value=(query["region_preset"] == JAPAN_REGION_LABEL),
            disabled=not show_plate_boundaries,
            key="eq_include_microplates",
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

    c_slider_floor = min(0.0, c_slider_min)
    color_range = st.slider(
        f"Colorbar scale adjustment: {color_label}",
        min_value=c_slider_floor,
        max_value=c_slider_max,
        value=(c_slider_min, c_slider_max),
        step=c_step,
        key=f"eq_colorbar_{color_column}",
    )

    return {
        "fig_depth_min": fig_depth_min,
        "fig_depth_max": fig_depth_max,
        "marker_size_scale": marker_size_scale,
        "color_column": color_column,
        "color_label": color_label,
        "color_range": color_range,
        "show_plate_boundaries": show_plate_boundaries,
        "include_microplates": include_microplates and show_plate_boundaries,
    }


def add_plate_boundaries_to_3d(fig_eq, plate_boundary_df, center_lon, center_lat, z_level):
    """
    Add plate boundary lines to a 3D hypocenter figure.
    """
    if plate_boundary_df is None or plate_boundary_df.empty:
        return fig_eq

    for label, df_label in plate_boundary_df.groupby("Label", dropna=False):
        color, width = plate_boundary_trace_style(label)
        boundary_east, boundary_north = lonlat_to_local_km(
            df_label["Longitude_degE"],
            df_label["Latitude_degN"],
            center_lon,
            center_lat,
        )
        fig_eq.add_trace(
            go.Scatter3d(
                x=boundary_east,
                y=boundary_north,
                z=[z_level] * len(df_label),
                mode="lines",
                name=f"plate boundary: {label}",
                line=dict(color=color, width=width),
                text=df_label["Name"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    return fig_eq


def render_4d_hypocenter_map(df_plot, query, viz, plate_boundary_df=None):
    """
    Render the EnvGeo-style 4D hypocenter map.
    """
    df_plot, center_lon, center_lat = add_local_km_coordinates(df_plot, query)
    df_plot["MarkerSize"] = df_plot["MarkerSize"] * viz["marker_size_scale"]
    df_plot = df_plot.sort_values(by=["Depth_km", "Magnitude"], ascending=[False, True])

    x_range, y_range, z_range, aspectratio = selected_area_km_ranges(
        query,
        center_lon,
        center_lat,
        viz["fig_depth_min"],
        viz["fig_depth_max"],
    )

    fig_eq = px.scatter_3d(
        df_plot,
        x="East_km",
        y="North_km",
        z="Depth_km",
        color=viz["color_column"],
        size="MarkerSize",
        size_max=18,
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
        },
    )

    fig_eq.update_traces(
        mode="markers",
        marker=dict(opacity=0.78, line=dict(color="white", width=0.5)),
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
        margin=dict(r=20, l=10, b=110, t=10),
    )
    fig_eq.update_coloraxes(
        cmin=viz["color_range"][0],
        cmax=viz["color_range"][1],
    )

    coastline_x, coastline_y = envgeo_utils.load_coastline_data(envgeo_utils.data_source_GLOBAL)
    if coastline_x and coastline_y:
        coastline_east, coastline_north = lonlat_to_local_km(
            coastline_x,
            coastline_y,
            center_lon,
            center_lat,
        )
        fig_eq.add_trace(
            go.Scatter3d(
                x=coastline_east,
                y=coastline_north,
                z=[viz["fig_depth_min"]] * len(coastline_x),
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
                z=[viz["fig_depth_max"]] * len(coastline_x),
                mode="lines",
                name="coastline (bottom)",
                line=dict(color="gray", width=0.5),
                hoverinfo="none",
            )
        )

    fig_eq = add_plate_boundaries_to_3d(
        fig_eq,
        plate_boundary_df,
        center_lon,
        center_lat,
        viz["fig_depth_min"],
    )

    st.plotly_chart(
        fig_eq,
        key="earthquake_4d_hypocenter_map",
        config={"scrollZoom": True, "displayModeBar": True},
        use_container_width=True,
    )


def add_plate_boundaries_to_2d(fig_map, plate_boundary_df):
    """
    Add plate boundary lines to a 2D mapbox figure.
    """
    if plate_boundary_df is None or plate_boundary_df.empty:
        return fig_map

    for label, df_label in plate_boundary_df.groupby("Label", dropna=False):
        color, width = plate_boundary_trace_style(label)
        fig_map.add_trace(
            go.Scattermapbox(
                lat=df_label["Latitude_degN"],
                lon=df_label["Longitude_degE"],
                mode="lines",
                line=dict(color=color, width=width),
                name=f"plate boundary: {label}",
                text=df_label["Name"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    return fig_map


def render_2d_distribution_map(df_plot, viz, plate_boundary_df=None):
    """
    Render the selected hypocenters on an interactive map.
    """
    st.subheader("Geographical Distribution Map (Auto-Zoom)")

    map_mode = st.radio(
        "Map Style:",
        ["Standard", "Satellite", "Bathymetry (Sea)", "Contour (GSI)"],
        horizontal=True,
        key="eq_map_style",
    )

    center_lat, center_lon, auto_zoom = auto_map_view(df_plot)
    df_map = df_plot.copy()
    df_map["MagnitudeMarkerSize"] = df_map["MagnitudeMarkerSize"] * viz["marker_size_scale"]

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
    fig_map = add_plate_boundaries_to_2d(fig_map, plate_boundary_df)

    st.plotly_chart(
        fig_map,
        key="earthquake_distribution_map",
        config={"scrollZoom": True, "displayModeBar": True},
        use_container_width=True,
    )


def render_time_histogram(df_plot):
    """
    Render earthquake occurrence counts through time.
    """
    st.subheader("Time-series Histogram")
    df_time = df_plot.dropna(subset=["Time_UTC"]).copy()
    if df_time.empty:
        st.warning("No valid event times are available for the histogram.")
        return

    nbins = st.slider(
        "Number of time bins",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        key="eq_time_histogram_bins",
    )
    fig_time = px.histogram(
        df_time,
        x="Time_UTC",
        nbins=nbins,
        hover_data={"Magnitude": True, "Depth_km": True},
        height=420,
    )
    fig_time.update_traces(marker_color="#3b6ea8", marker_line_color="white", marker_line_width=0.5)
    fig_time.update_layout(
        xaxis_title="Origin time (UTC)",
        yaxis_title="Number of earthquakes",
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(fig_time, key="earthquake_time_histogram", use_container_width=True)


def render_cross_section_location_map(
    df_plot,
    df_section,
    query,
    viz,
    start_lon,
    start_lat,
    end_lon,
    end_lat,
    half_width_km,
    plate_boundary_df=None,
):
    """
    Show the cross-section A-B line and selected swath on a 2D map.
    """
    st.caption("Cross-section location map")

    endpoint_df = pd.DataFrame(
        {
            "Label": ["A", "B"],
            "Longitude_degE": [start_lon, end_lon],
            "Latitude_degN": [start_lat, end_lat],
        }
    )
    context_df = pd.concat(
        [
            df_plot[["Longitude_degE", "Latitude_degN"]].copy(),
            endpoint_df[["Longitude_degE", "Latitude_degN"]],
        ],
        ignore_index=True,
    )
    center_lat, center_lon, auto_zoom = auto_map_view(context_df)

    fig_location = go.Figure()
    fig_location.add_trace(
        go.Scattermapbox(
            lat=df_plot["Latitude_degN"],
            lon=df_plot["Longitude_degE"],
            mode="markers",
            name="all events",
            marker=dict(size=4, color="rgba(70, 70, 70, 0.28)"),
            hoverinfo="skip",
        )
    )

    corridor_df = cross_section_corridor_dataframe(
        start_lon,
        start_lat,
        end_lon,
        end_lat,
        half_width_km,
    )
    if not corridor_df.empty:
        fig_location.add_trace(
            go.Scattermapbox(
                lat=corridor_df["Latitude_degN"],
                lon=corridor_df["Longitude_degE"],
                mode="lines",
                fill="toself",
                name="section width",
                line=dict(color="rgba(30, 120, 180, 0.35)", width=1),
                fillcolor="rgba(30, 120, 180, 0.16)",
                hoverinfo="skip",
            )
        )

    if not df_section.empty:
        selected_size = (2.0 + pd.to_numeric(df_section["Magnitude"], errors="coerce").fillna(0).clip(lower=0) * 2.6)
        fig_location.add_trace(
            go.Scattermapbox(
                lat=df_section["Latitude_degN"],
                lon=df_section["Longitude_degE"],
                mode="markers",
                name="events in section",
                marker=dict(
                    size=(selected_size * viz["marker_size_scale"]).tolist(),
                    color=pd.to_numeric(df_section[viz["color_column"]], errors="coerce"),
                    colorscale=earthquake_color_scale(viz["color_column"]),
                    cmin=viz["color_range"][0],
                    cmax=viz["color_range"][1],
                    opacity=0.78,
                    colorbar=dict(title=viz["color_label"]),
                ),
                text=df_section["Place"],
                customdata=df_section[["DateTime_UTC", "Magnitude", "Depth_km", "SectionOffset_km"]],
                hovertemplate=(
                    "%{text}<br>"
                    "%{customdata[0]}<br>"
                    "M %{customdata[1]}<br>"
                    "Depth %{customdata[2]} km<br>"
                    "Offset %{customdata[3]:.1f} km<extra></extra>"
                ),
            )
        )

    fig_location.add_trace(
        go.Scattermapbox(
            lat=[start_lat, end_lat],
            lon=[start_lon, end_lon],
            mode="lines",
            name="section A-B",
            line=dict(color="#d7301f", width=4),
            hovertemplate="Section A-B<extra></extra>",
        )
    )
    fig_location.add_trace(
        go.Scattermapbox(
            lat=endpoint_df["Latitude_degN"],
            lon=endpoint_df["Longitude_degE"],
            mode="markers+text",
            name="endpoints",
            marker=dict(size=13, color="#d7301f"),
            text=endpoint_df["Label"],
            textfont=dict(color="white", size=12),
            textposition="middle center",
            hovertemplate="%{text}: %{lon:.2f}, %{lat:.2f}<extra></extra>",
        )
    )

    fig_location = envgeo_utils.apply_map_style(fig_location, "Standard")
    fig_location = add_plate_boundaries_to_2d(fig_location, plate_boundary_df)
    fig_location.update_layout(
        height=420,
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    st.plotly_chart(
        fig_location,
        key="earthquake_cross_section_location_map",
        config={"scrollZoom": True, "displayModeBar": True},
        use_container_width=True,
    )


def render_cross_section_and_depth_profile(df_plot, query, viz, plate_boundary_df=None):
    """
    Render an arbitrary cross-section and a depth-frequency profile.
    """
    st.subheader("Arbitrary Cross-section")
    default_start_lon, default_start_lat, default_end_lon, default_end_lat = default_cross_section_points(query)

    with st.container(border=True):
        col_start_lon, col_start_lat, col_end_lon, col_end_lat = st.columns(4)
        with col_start_lon:
            start_lon = st.number_input(
                "Start lon",
                min_value=-180.0,
                max_value=180.0,
                value=float(default_start_lon),
                step=0.5,
                key="eq_section_start_lon",
            )
        with col_start_lat:
            start_lat = st.number_input(
                "Start lat",
                min_value=-90.0,
                max_value=90.0,
                value=float(default_start_lat),
                step=0.5,
                key="eq_section_start_lat",
            )
        with col_end_lon:
            end_lon = st.number_input(
                "End lon",
                min_value=-180.0,
                max_value=180.0,
                value=float(default_end_lon),
                step=0.5,
                key="eq_section_end_lon",
            )
        with col_end_lat:
            end_lat = st.number_input(
                "End lat",
                min_value=-90.0,
                max_value=90.0,
                value=float(default_end_lat),
                step=0.5,
                key="eq_section_end_lat",
            )

        half_width_km = st.slider(
            "Section half-width (km)",
            min_value=10.0,
            max_value=1000.0,
            value=100.0 if query["region_preset"] == JAPAN_REGION_LABEL else 300.0,
            step=10.0,
            key="eq_section_half_width",
        )
        limit_to_segment = st.checkbox(
            "Limit to selected segment",
            value=True,
            key="eq_section_limit_to_segment",
        )

    df_section, section_length_km = add_cross_section_coordinates(
        df_plot,
        start_lon,
        start_lat,
        end_lon,
        end_lat,
    )
    if "MarkerSize" in df_section.columns:
        df_section["MarkerSize"] = df_section["MarkerSize"] * viz["marker_size_scale"]
    if section_length_km <= 0:
        st.warning("Please select two different cross-section endpoints.")
    else:
        section_mask = df_section["SectionOffset_km"].abs() <= half_width_km
        if limit_to_segment:
            section_mask &= df_section["SectionDistance_km"].between(0, section_length_km)
        df_section = df_section[section_mask].copy()

        st.write(f"{len(df_section)} events within the selected cross-section window")
        if df_section.empty:
            st.warning("No earthquakes are inside this cross-section window.")
        else:
            fig_section = px.scatter(
                df_section,
                x="SectionDistance_km",
                y="Depth_km",
                color=viz["color_column"],
                size="MarkerSize",
                size_max=16,
                color_continuous_scale=earthquake_color_scale(viz["color_column"]),
                range_color=viz["color_range"],
                hover_data={
                    "DateTime_UTC": True,
                    "Place": True,
                    "Magnitude": True,
                    "Depth_km": True,
                    "Longitude_degE": True,
                    "Latitude_degN": True,
                    "SectionOffset_km": ":.1f",
                    "MarkerSize": False,
                },
                height=480,
            )
            fig_section.update_layout(
                xaxis_title="Distance along section (km)",
                yaxis_title="Hypocenter depth (km)",
                yaxis=dict(range=[viz["fig_depth_max"], viz["fig_depth_min"]]),
                margin=dict(l=10, r=10, t=20, b=20),
                coloraxis_colorbar=dict(title=viz["color_label"]),
            )
            st.plotly_chart(
                fig_section,
                key="earthquake_cross_section",
                use_container_width=True,
            )
        render_cross_section_location_map(
            df_plot,
            df_section,
            query,
            viz,
            start_lon,
            start_lat,
            end_lon,
            end_lat,
            half_width_km,
            plate_boundary_df,
        )

    st.subheader("Depth Profile")
    bin_size_km = st.slider(
        "Depth bin size (km)",
        min_value=5.0,
        max_value=100.0,
        value=25.0,
        step=5.0,
        key="eq_depth_profile_bin",
    )
    df_depth_profile = depth_profile_dataframe(df_plot, bin_size_km)
    if df_depth_profile.empty:
        st.warning("No depth values are available for the depth profile.")
        return

    fig_depth = go.Figure(
        go.Bar(
            x=df_depth_profile["Count"],
            y=df_depth_profile["DepthMid_km"],
            orientation="h",
            marker=dict(
                color=df_depth_profile["MeanMagnitude"],
                colorscale=earthquake_color_scale("Magnitude"),
                colorbar=dict(title="Mean M"),
                line=dict(color="white", width=0.5),
            ),
            customdata=df_depth_profile[["MeanMagnitude", "MaxMagnitude"]],
            hovertemplate=(
                "Depth: %{y:.1f} km<br>"
                "Count: %{x}<br>"
                "Mean M: %{customdata[0]:.2f}<br>"
                "Max M: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig_depth.update_layout(
        height=430,
        xaxis_title="Number of earthquakes",
        yaxis_title="Hypocenter depth (km)",
        yaxis=dict(range=[viz["fig_depth_max"], viz["fig_depth_min"]]),
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(fig_depth, key="earthquake_depth_profile", use_container_width=True)


def normalize_column_name(name):
    """
    Normalize a dataframe column name for flexible user-upload matching.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_catalog_column(df_catalog, candidates):
    """
    Find the first column matching common JMA/NIED/USGS names.
    """
    normalized_lookup = {normalize_column_name(col): col for col in df_catalog.columns}
    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)
        for normalized_col, original_col in normalized_lookup.items():
            if normalized_candidate and normalized_candidate in normalized_col:
                return original_col

    return None


def read_uploaded_catalog(uploaded_file):
    """
    Read a user-supplied JMA/NIED catalog table.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, sep=r"\s+", engine="python")
        except Exception as e:
            st.error(f"Failed to read uploaded catalog: {e}")
            return pd.DataFrame()


def normalize_external_catalog(df_catalog, catalog_name):
    """
    Normalize common JMA/NIED exported columns into the USGS-like schema.
    """
    if df_catalog.empty:
        return pd.DataFrame()

    lon_col = find_catalog_column(df_catalog, ["Longitude_degE", "Longitude", "Long", "Lon", "経度"])
    lat_col = find_catalog_column(df_catalog, ["Latitude_degN", "Latitude", "Lat", "緯度"])
    depth_col = find_catalog_column(df_catalog, ["Depth_km", "Depth", "Dep", "震源深さ", "深さ"])
    mag_col = find_catalog_column(df_catalog, ["Magnitude", "Mag", "M", "Mj", "マグニチュード"])
    place_col = find_catalog_column(df_catalog, ["Place", "Region", "Name", "震央地名", "震源地域"])
    time_col = find_catalog_column(
        df_catalog,
        ["DateTime_UTC", "Origin Time", "OriginTime", "Time_UTC", "Datetime", "DateTime", "Date"],
    )

    missing = [
        label
        for label, column in [
            ("longitude", lon_col),
            ("latitude", lat_col),
            ("depth", depth_col),
            ("magnitude", mag_col),
        ]
        if column is None
    ]
    if missing:
        st.warning(f"Uploaded catalog is missing required columns: {', '.join(missing)}")
        return pd.DataFrame()

    df_norm = pd.DataFrame()
    df_norm["Longitude_degE"] = pd.to_numeric(df_catalog[lon_col], errors="coerce")
    df_norm["Latitude_degN"] = pd.to_numeric(df_catalog[lat_col], errors="coerce")
    df_norm["Depth_km"] = pd.to_numeric(df_catalog[depth_col], errors="coerce")
    df_norm["Magnitude"] = pd.to_numeric(df_catalog[mag_col], errors="coerce")
    if time_col:
        df_norm["Time_UTC"] = pd.to_datetime(df_catalog[time_col], errors="coerce", utc=True)
    else:
        df_norm["Time_UTC"] = pd.to_datetime(
            pd.Series([pd.NaT] * len(df_norm), index=df_norm.index),
            errors="coerce",
            utc=True,
        )

    df_norm["DateTime_UTC"] = df_norm["Time_UTC"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    df_norm["Place"] = df_catalog[place_col].astype(str) if place_col else catalog_name
    df_norm["Catalog"] = catalog_name
    df_norm["MagnitudeMarkerSize"] = 2.0 + df_norm["Magnitude"].fillna(0).clip(lower=0) * 3.0
    return df_norm.dropna(subset=["Longitude_degE", "Latitude_degN", "Depth_km"])


def catalog_summary_dataframe(df_usgs, df_external):
    """
    Summarize the USGS and uploaded comparison catalogs.
    """
    catalogs = []
    for catalog_name, df_catalog in [
        ("USGS", df_usgs),
        (df_external["Catalog"].iloc[0] if not df_external.empty else "JMA/NIED upload", df_external),
    ]:
        if df_catalog.empty:
            catalogs.append(
                {
                    "Catalog": catalog_name,
                    "Events": 0,
                    "Max magnitude": None,
                    "Median depth (km)": None,
                    "Start": None,
                    "End": None,
                }
            )
            continue
        time_values = pd.to_datetime(df_catalog.get("Time_UTC"), errors="coerce", utc=True)
        catalogs.append(
            {
                "Catalog": catalog_name,
                "Events": len(df_catalog),
                "Max magnitude": pd.to_numeric(df_catalog["Magnitude"], errors="coerce").max(),
                "Median depth (km)": pd.to_numeric(df_catalog["Depth_km"], errors="coerce").median(),
                "Start": time_values.min(),
                "End": time_values.max(),
            }
        )
    return pd.DataFrame(catalogs)


def render_jma_nied_comparison_page(df_plot, query, plate_boundary_df=None):
    """
    Provide a Japan-focused comparison panel for JMA/NIED catalogs.
    """
    st.subheader("JMA / NIED Comparison")
    if query["region_preset"] != JAPAN_REGION_LABEL:
        st.info("JMA/NIED comparison is most useful for Japan and surrounding areas.")

    st.caption(
        "JMA/NIED catalogs are not automatically scraped here. Upload an exported CSV/XLSX catalog "
        "to compare it with the current USGS query on the same map and summary panels."
    )
    with st.expander("JMA/NIED data note", expanded=False):
        st.write(
            "JMA provides official earthquake information and bulletin datasets; NIED Hi-net provides "
            "automatic hypocenter data and JMA unified catalog access under its own guidance. "
            "For research use, follow each provider's terms and acknowledgement requirements."
        )
        st.markdown(f"- [JMA earthquake information]({JMA_EARTHQUAKE_INFO_URL})")
        st.markdown(f"- [JMA Seismological Bulletin of Japan]({JMA_BULLETIN_URL})")
        st.markdown(f"- [NIED Hi-net data guidance]({NIED_HINET_DATA_URL})")

    catalog_name = st.selectbox(
        "Uploaded catalog label",
        ["JMA", "NIED Hi-net", "JMA unified catalog", "Other Japan catalog"],
        key="eq_external_catalog_label",
    )
    uploaded_file = st.file_uploader(
        "Upload JMA/NIED catalog table",
        type=["csv", "tsv", "txt", "xlsx", "xls"],
        key="eq_jma_nied_upload",
    )
    df_external_raw = read_uploaded_catalog(uploaded_file)
    df_external = normalize_external_catalog(df_external_raw, catalog_name)

    if df_external.empty:
        st.info("Upload a catalog with longitude, latitude, depth, and magnitude columns to enable comparison.")
        return

    df_usgs = df_plot.copy()
    df_usgs["Catalog"] = "USGS"
    summary = catalog_summary_dataframe(df_usgs, df_external)
    st.dataframe(summary)

    df_compare = pd.concat(
        [
            df_usgs[
                [
                    "Longitude_degE",
                    "Latitude_degN",
                    "Depth_km",
                    "Magnitude",
                    "DateTime_UTC",
                    "Place",
                    "Catalog",
                    "MagnitudeMarkerSize",
                ]
            ],
            df_external[
                [
                    "Longitude_degE",
                    "Latitude_degN",
                    "Depth_km",
                    "Magnitude",
                    "DateTime_UTC",
                    "Place",
                    "Catalog",
                    "MagnitudeMarkerSize",
                ]
            ],
        ],
        ignore_index=True,
    )
    center_lat, center_lon, auto_zoom = auto_map_view(df_compare)

    fig_compare = px.scatter_mapbox(
        df_compare,
        lat="Latitude_degN",
        lon="Longitude_degE",
        color="Catalog",
        size="MagnitudeMarkerSize",
        size_max=18,
        hover_data={
            "DateTime_UTC": True,
            "Place": True,
            "Magnitude": True,
            "Depth_km": True,
            "MagnitudeMarkerSize": False,
        },
        opacity=0.68,
        height=520,
    )
    fig_compare = envgeo_utils.apply_map_style(fig_compare, "Standard")
    fig_compare.update_layout(
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig_compare = add_plate_boundaries_to_2d(fig_compare, plate_boundary_df)
    st.plotly_chart(fig_compare, key="earthquake_jma_nied_compare_map", use_container_width=True)

    fig_depth_compare = px.histogram(
        df_compare,
        y="Depth_km",
        color="Catalog",
        barmode="overlay",
        nbins=40,
        opacity=0.65,
        height=420,
    )
    fig_depth_compare.update_layout(
        xaxis_title="Number of earthquakes",
        yaxis_title="Hypocenter depth (km)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(
        fig_depth_compare,
        key="earthquake_jma_nied_depth_compare",
        use_container_width=True,
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
    st.header(f"EnvGeo-Earthquake")
    st.header(f"4D Visualizer Earthquake Advanced ({version})")
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
        st.write(
            "Recommended catalog citation: U.S. Geological Survey (2017), "
            "Advanced National Seismic System (ANSS) Comprehensive Catalog, "
            "U.S. Geological Survey, https://doi.org/10.5066/F7MS3QZH."
        )
        st.write(
            "USGS-authored or USGS-produced information is generally public domain, "
            "but USGS requests credit. Because catalogs can include contributions "
            "from multiple networks or agencies, publication or redistribution "
            "should also follow any contributor-specific guidance."
        )
        st.markdown(f"- [USGS FDSN Event Web Service]({USGS_EVENT_API_URL})")
        st.markdown(f"- [ANSS / USGS FDSN data-center record]({USGS_COMCAT_FDSN_URL})")
        st.markdown(f"- [USGS Copyrights and Credits]({USGS_CREDIT_URL})")
        st.caption(
            "Local coastline Excel files used in 3D reference overlays do not contain "
            "source/license metadata in this repository; treat them as visual guides only."
        )
        render_plate_boundary_note()

    region_preset = main_region_selector()
    query = sidebar_controls(region_preset)
    df_eq = fetch_earthquake_dataframe(query)

    query_url = df_eq.attrs.get("query_url", "")
    st.write(f"{len(df_eq)} earthquake events found")
    if query_url:
        st.markdown(f"[USGS API query]({query_url})")
    if len(df_eq) >= query["limit"]:
        if query["limit"] >= 20000:
            st.warning("20000件上限に達したため一部のみ表示している可能性があります。条件を絞るか、Order by を変更してください。")
        else:
            st.warning(
                f"{query['limit']}件の取得上限に達しました。条件に一致する全件ではなく一部のみ表示している可能性があります。"
            )

    if df_eq.empty:
        st.warning("No earthquake data available for the selected conditions.")
        return

    df_plot = prepare_plot_dataframe(df_eq)
    if df_plot.empty:
        st.warning("No plottable hypocenter data were returned.")
        return

    viz = visualization_controls(df_plot, query)
    plate_boundary_df = pd.DataFrame()
    plate_source = ""
    plate_errors = []
    if viz["show_plate_boundaries"]:
        with st.spinner("Loading plate boundaries..."):
            plate_boundary_df, plate_source, plate_errors = load_plate_boundary_dataframe(
                query,
                viz["include_microplates"],
            )
        if plate_errors and plate_boundary_df.empty:
            st.warning("Plate boundary data could not be loaded from USGS.")
        elif plate_errors:
            st.warning("USGS plate boundary service could not be reached; fallback Japan lines are shown.")
        if not plate_boundary_df.empty:
            st.caption(
                f"Plate boundaries: {plate_source}. Boundary locations are approximate; "
                "for educational/research visualization only."
            )
            st.caption("プレート境界位置は概略です。教育・研究用の可視化として利用してください。")

    tab_3d, tab_2d, tab_profiles, tab_time, tab_compare, tab_data = st.tabs(
        [
            "4D / 3D map",
            "2D map",
            "Cross-section / depth",
            "Time histogram",
            "JMA/NIED comparison",
            "Data",
        ]
    )

    with tab_3d:
        st.subheader("4D Hypocenter Map")
        st.caption("PC recommended for 3D interaction; use the 2D tab on smartphones and tablets.")
        render_4d_hypocenter_map(df_plot, query, viz, plate_boundary_df)

    with tab_2d:
        render_2d_distribution_map(df_plot, viz, plate_boundary_df)

    with tab_profiles:
        render_cross_section_and_depth_profile(df_plot, query, viz, plate_boundary_df)

    with tab_time:
        render_time_histogram(df_plot)

    with tab_compare:
        render_jma_nied_comparison_page(df_plot, query, plate_boundary_df)

    with tab_data:
        display_earthquake_table(df_eq)

    if st.sidebar.button("Reload / clear API cache"):
        envgeo_utils.clear_app_cache()
        st.rerun()

    st.caption("3D display is recommended for PC. On smartphones and tablets, the 2D map is recommended.")
    st.caption("3D表示はPC推奨です。スマホ・タブレットでは2Dマップの利用を推奨します。")

if __name__ == "__main__":
    main()
