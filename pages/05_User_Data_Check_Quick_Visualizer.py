#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Data Check & Quick Visualizer for EnvGeo-Seawater.

Upload-first 2D, 3D, and 4D exploratory plots for arbitrary user data.
CSV and Excel files remain in memory only for the current Streamlit session.
任意のユーザーアップロード表を、2D・3D・4Dで簡易可視化するページです。
CSV / Excel は現在の Streamlit セッション内だけで扱い、保存しません。

Created: 2023-05-21
Author: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

import html
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import envgeo_user_data
import envgeo_utils


version = "1.3.2"
NO_COLOR = "No color"
NO_COMPARISON_DATA = "None"
PLOTLY_MARKERS = {"D": "diamond", "o": "circle", "s": "square", "^": "triangle-up", "*": "star", "X": "x"}
MAX_HOVER_COLUMNS = 30
MAX_HOVER_VALUE_LENGTH = 120


def render_tab_style():
    """Apply legacy tab styling / 旧Streamlit用の青いカード型タブ表示を適用する。"""
    st.markdown(
        """
        <style>
        div[data-baseweb="tab-list"] {
            gap: 0.25rem;
            flex-wrap: wrap;
        }
        div[data-baseweb="tab-list"] button[role="tab"] {
            background: rgba(248, 249, 250, 0.95);
            color: #1f2937;
            border: 1px solid rgba(49, 51, 63, 0.22);
            border-radius: 6px 6px 0 0;
            padding: 0.35rem 0.65rem;
            min-height: 2.1rem;
            white-space: nowrap;
            font-weight: 600;
        }
        div[data-baseweb="tab-list"] button[role="tab"] p {
            margin: 0;
            color: inherit;
        }
        div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #e8f2ff 0%, #ddeaff 100%);
            border-color: #4a90e2;
            color: #0b3e75;
            box-shadow: inset 0 0 0 1px rgba(74, 144, 226, 0.35);
        }
        html[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"],
        body[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"] {
            background: rgba(44, 49, 61, 0.96);
            color: rgba(245, 247, 250, 0.95);
            border-color: rgba(240, 244, 250, 0.26);
        }
        html[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"],
        body[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #204061 0%, #1a314a 100%);
            color: #e9f2ff;
            border-color: #76adff;
            box-shadow: inset 0 0 0 1px rgba(118, 173, 255, 0.42);
        }
        @media (prefers-color-scheme: dark) {
            div[data-baseweb="tab-list"] button[role="tab"] {
                background: rgba(44, 49, 61, 0.96);
                color: rgba(245, 247, 250, 0.95);
                border-color: rgba(240, 244, 250, 0.26);
            }
            div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
                background: linear-gradient(180deg, #204061 0%, #1a314a 100%);
                color: #e9f2ff;
                border-color: #76adff;
                box-shadow: inset 0 0 0 1px rgba(118, 173, 255, 0.42);
            }
        }
        @media (max-width: 900px) {
            div[data-baseweb="tab-list"] button[role="tab"] {
                font-size: 0.86rem;
                padding: 0.30rem 0.52rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def numeric_columns(dataframe):
    """Return numeric-capable uploaded columns / 数値を一つ以上含む列を返す。"""
    excluded = {envgeo_utils.QUALITY_FLAG_COLUMN, envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN}
    return [
        column for column in dataframe.columns if column not in excluded
        and pd.to_numeric(dataframe[column], errors="coerce").notna().any()
    ]


def plot_rows(dataframe, columns):
    """Coerce plotting columns / 描画対象列を数値化し有効な行だけ返す。"""
    result = dataframe.loc[:, columns].copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=columns)


def sample_rows(dataframe, maximum_rows):
    """Cap rows reproducibly / 操作性のため同じ乱数条件で表示行を制限する。"""
    if len(dataframe) <= maximum_rows:
        return dataframe, False
    return dataframe.sample(n=maximum_rows, random_state=42), True


def marker_settings(style, color=None, color_column=None, palette_label=None):
    """Convert marker controls / 共通のマーカー設定をPlotly形式へ変換する。"""
    marker = {
        "size": max(4.0, min(32.0, math.sqrt(float(style["size"])))),
        "symbol": PLOTLY_MARKERS.get(style["marker"], "diamond"),
        "opacity": style["alpha"],
        "line": {"color": style["outline_color"], "width": style["outline_width"]},
    }
    if color is None:
        marker["color"] = style["color"]
    else:
        marker.update({
            "color": color,
            "colorscale": envgeo_utils.get_plotly_colormap(color_column, palette_label),
            "colorbar": {"title": color_column},
        })
    return marker


def rich_hover_text(dataframe):
    """Build rich but bounded hover text / 多くの列を安全な長さでhoverへ表示する。"""
    excluded = {
        envgeo_utils.QUALITY_FLAG_COLUMN,
        envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN,
        "_Longitude_plot",
    }
    # Keep the familiar oceanographic identifiers first / 海洋観測でよく使う項目を先に表示する。
    priority = [
        "Dataset", "reference", "Cruise", "Station", "Date", "Year", "Month", "Day",
        "Longitude_degE", "Latitude_degN", "Depth_m", "Temperature_degC", "Salinity",
        "d18O", "dD", "d-excess",
    ]
    available = [column for column in dataframe.columns if column not in excluded]
    ordered_columns = [column for column in priority if column in available]
    ordered_columns += [column for column in available if column not in ordered_columns]
    shown_columns = ordered_columns[:MAX_HOVER_COLUMNS]
    omitted_count = len(ordered_columns) - len(shown_columns)
    text = []
    for _, row in dataframe.iterrows():
        lines = ["<b>Uploaded data</b>"]
        for column in shown_columns:
            value = row[column]
            if pd.isna(value):
                continue
            if isinstance(value, (float, np.floating)):
                value_text = f"{value:.6g}"
            else:
                value_text = str(value)
            if len(value_text) > MAX_HOVER_VALUE_LENGTH:
                value_text = value_text[: MAX_HOVER_VALUE_LENGTH - 3] + "..."
            lines.append(f"{html.escape(str(column))}: {html.escape(value_text)}")
        if omitted_count:
            lines.append(f"({omitted_count} additional columns not shown)")
        text.append("<br>".join(lines))
    return text


def add_regression(figure, dataframe, x_column, y_column):
    """Add a least-squares line / 描画可能な場合に最小二乗の近似直線を追加する。"""
    rows = plot_rows(dataframe, [x_column, y_column])
    if len(rows) < 2 or rows[x_column].nunique() < 2:
        st.caption(":gray[Regression line was skipped because fewer than two valid x-y points are available.]")
        return
    coefficient = np.polyfit(rows[x_column], rows[y_column], 1)
    x_line = np.linspace(rows[x_column].min(), rows[x_column].max(), 100)
    correlation = np.corrcoef(rows[x_column], rows[y_column])[0, 1]
    figure.add_trace(go.Scatter(
        x=x_line, y=np.poly1d(coefficient)(x_line), mode="lines",
        line={"color": "#222222", "width": 2}, name="Linear regression",
    ))
    figure.add_annotation(
        x=0.99, y=0.01, xref="paper", yref="paper", xanchor="right", yanchor="bottom",
        text=f"y = {coefficient[0]:.3g}x + {coefficient[1]:.3g}; R = {correlation:.2f}; N = {len(rows):,}",
        showarrow=False, bgcolor="rgba(255,255,255,0.8)",
    )


def create_2d(dataframe, x_column, y_column, color_column, style, regression, palette_label=None):
    """Create a 2-D scatter plot / ユーザー指定軸で2次元散布図を作成する。"""
    rows = plot_rows(dataframe, [x_column, y_column])
    color = None if color_column == NO_COLOR else pd.to_numeric(dataframe.loc[rows.index, color_column], errors="coerce")
    figure = go.Figure(go.Scatter(
        x=rows[x_column], y=rows[y_column], mode="markers", name="Uploaded data",
        marker=marker_settings(style, color, color_column, palette_label),
        text=rich_hover_text(dataframe.loc[rows.index]),
        hovertemplate="%{text}<extra></extra>",
    ))
    if regression:
        add_regression(figure, dataframe, x_column, y_column)
    figure.update_layout(title="2D scatter plot", xaxis_title=x_column, yaxis_title=y_column, height=620, margin={"r": 30, "l": 30, "b": 30, "t": 45})
    return figure, len(rows)


def create_3d(dataframe, x_column, y_column, z_column, color_column, style, reverse_z, palette_label=None):
    """Create 3-D/4-D scatter / 任意の3軸と任意の色変数で3D・4D図を作成する。"""
    rows = plot_rows(dataframe, [x_column, y_column, z_column])
    color = None if color_column == NO_COLOR else pd.to_numeric(dataframe.loc[rows.index, color_column], errors="coerce")
    figure = go.Figure(go.Scatter3d(
        x=rows[x_column], y=rows[y_column], z=rows[z_column], mode="markers", name="Uploaded data",
        marker=marker_settings(style, color, color_column, palette_label),
        text=rich_hover_text(dataframe.loc[rows.index]),
        hovertemplate="%{text}<extra></extra>",
    ))
    figure.update_layout(
        title="3D / 4D scatter plot",
        scene={"xaxis_title": x_column, "yaxis_title": y_column, "zaxis_title": z_column, "zaxis": {"autorange": "reversed"} if reverse_z else {}},
        height=680, margin={"r": 20, "l": 10, "b": 10, "t": 45},
    )
    return figure, len(rows)


def _padded_range(values, minimum_span):
    """Return a stable geographic range / 短い観測線でも安定した地理範囲を返す。"""
    lower, upper = float(values.min()), float(values.max())
    centre = (lower + upper) / 2.0
    span = max(upper - lower, minimum_span)
    half_span = span * 0.56  # Add 12 % margin / データ周囲に12 %の余白を加える。
    return [centre - half_span, centre + half_span]


def _normalize_longitude(lon, center):
    """Place longitudes in a selected 360-degree frame / 選択中心の経度系へ変換する。"""
    return ((np.asarray(lon, dtype=float) - (center - 180.0)) % 360.0) + (center - 180.0)


def _coastline_with_breaks(lon, lat, center, jump_threshold=180.0):
    """Re-centre coastlines and break wrap jumps / 海岸線を中心化し折返し境界で切る。"""
    lon_wrapped = _normalize_longitude(lon, center)
    lat_array = np.asarray(lat, dtype=float)
    lon_out, lat_out = [], []
    for index, (current_lon, current_lat) in enumerate(zip(lon_wrapped, lat_array)):
        if index and (
            not np.isfinite(current_lon) or not np.isfinite(current_lat)
            or not np.isfinite(lon_wrapped[index - 1]) or not np.isfinite(lat_array[index - 1])
            or abs(current_lon - lon_wrapped[index - 1]) > jump_threshold
        ):
            lon_out.append(np.nan)
            lat_out.append(np.nan)
        lon_out.append(current_lon)
        lat_out.append(current_lat)
    return lon_out, lat_out


def _region_bounds_for_center(region_bounds, center):
    """Convert a shared preset to the selected frame / 共通海域を選択経度系へ変換する。"""
    if region_bounds is None:
        return None
    lon_min, lon_max, lat_min, lat_max = region_bounds
    if abs(lon_max - lon_min) >= 300:
        converted_lon = [center - 180.0, center + 180.0]
    else:
        converted_lon = _normalize_longitude([lon_min, lon_max], center).tolist()
        if converted_lon[0] > converted_lon[1]:
            # Keep the full frame at a wrap boundary / 折返し境界では軸を反転させず全経度系を使う。
            converted_lon = [center - 180.0, center + 180.0]
    return [converted_lon[0], converted_lon[1], float(lat_min), float(lat_max)]


def _geographic_scene(rows, region_bounds=None, longitude_center=0):
    """Build the page-04-style scene / 04ページ準拠の地理3Dシーンを作成する。"""
    # Keep short cruise tracks readable / 短い観測線が細長い立体にならないようにする。
    plot_longitude = _normalize_longitude(rows["Longitude_degE"], longitude_center)
    region_bounds = _region_bounds_for_center(region_bounds, longitude_center)
    if region_bounds is None:
        lon_range = _padded_range(plot_longitude, minimum_span=1.8)
        lat_range = _padded_range(rows["Latitude_degN"], minimum_span=1.4)
    else:
        lon_min, lon_max, lat_min, lat_max = region_bounds
        lon_range = [float(lon_min), float(lon_max)]
        lat_range = [float(lat_min), float(lat_max)]
    # Keep the coast at sea level and reserve a bottom projection plane.
    # 海岸線は海面に置き、最深部下には灰色投影用の余白を確保する。
    depth_top = min(0.0, float(rows["Depth_m"].min()))
    depth_bottom = max(float(rows["Depth_m"].max()), depth_top + 1.0)
    depth_padding = max((depth_bottom - depth_top) * 0.06, 1.0)
    depth_bottom += depth_padding

    lon_span = max(lon_range[1] - lon_range[0], 0.1)
    lat_span = max(lat_range[1] - lat_range[0], 0.1)
    mean_lat = (lat_range[0] + lat_range[1]) / 2.0
    # Use page-04 longitude correction with safe bounds for narrow uploads.
    # 04ページと同じ緯度補正を使い、狭いアップロード範囲でも見やすくする。
    x_ratio = float(np.clip((lon_span * np.cos(np.deg2rad(mean_lat))) / lat_span, 0.35, 3.0))
    global_view = lon_span >= 120.0 or lat_span >= 70.0
    # Compress depth visually for map readability / 地図として読みやすい深度比にする。
    aspectratio = {"x": x_ratio, "y": 1.0, "z": 0.5 if global_view else 0.58}
    camera = (
        {"eye": {"x": 1.2, "y": -0.8, "z": 2.2}, "center": {"x": 0, "y": 0, "z": -0.1}}
        if global_view
        else {"eye": {"x": -1.35, "y": -1.45, "z": 1.35}, "center": {"x": 0, "y": 0, "z": -0.08}}
    )
    scene = {
        "xaxis": {"title": "Longitude (degE)", "range": lon_range, "backgroundcolor": "#edf4f8", "gridcolor": "#cbd8df"},
        "yaxis": {"title": "Latitude (degN)", "range": lat_range, "backgroundcolor": "#edf4f8", "gridcolor": "#cbd8df"},
        "zaxis": {"title": "Depth (m)", "range": [depth_bottom, depth_top], "autorange": False, "backgroundcolor": "#f5f7f8", "gridcolor": "#d5dde2"},
        "aspectmode": "manual",
        "aspectratio": aspectratio,
        "camera": camera,
        "bgcolor": "#f8fbfd",
    }
    return scene, depth_top, depth_bottom


def create_geographic(
    dataframe, color_column, style, coastline, reference_source, region_bounds=None,
    palette_label=None, longitude_center=0,
):
    """Create the page-04-style map-depth view / 04ページ準拠の地理3D図を作成する。"""
    columns = ["Longitude_degE", "Latitude_degN", "Depth_m"]
    rows = plot_rows(dataframe, columns).query("-90 <= Latitude_degN <= 90")
    figure = go.Figure()
    rows = rows.copy()
    rows["_Longitude_plot"] = _normalize_longitude(rows["Longitude_degE"], longitude_center)
    scene, depth_top, depth_bottom = _geographic_scene(rows, region_bounds, longitude_center)
    lon_range = scene["xaxis"]["range"]
    lat_range = scene["yaxis"]["range"]
    if reference_source != "None":
        reference = envgeo_utils.load_isotope_data(reference_source)
        if not reference.empty:
            reference = plot_rows(reference, columns)
            figure.add_trace(go.Scatter3d(x=_normalize_longitude(reference[columns[0]], longitude_center), y=reference[columns[1]], z=reference[columns[2]], mode="markers", name="Reference context", marker={"size": 2, "color": "#999999", "opacity": 0.28}, hoverinfo="skip", showlegend=False))
    if coastline:
        coastline_lon, coastline_lat = envgeo_utils.load_coastline_data(envgeo_utils.data_source_GLOBAL, resolution="50m")
        coast_lon = _normalize_longitude(coastline_lon, longitude_center)
        coast_lat = np.asarray(coastline_lat, dtype=float)
        in_window = (
            np.isfinite(coast_lon) & np.isfinite(coast_lat)
            & (coast_lon >= lon_range[0]) & (coast_lon <= lon_range[1])
            & (coast_lat >= lat_range[0]) & (coast_lat <= lat_range[1])
        )
        if in_window.any():
            # Draw surface and bottom coastlines without cluttering the legend.
            # 海面の青線と底面の灰色投影を描き、凡例は煩雑にしない。
            coast_lon, coast_lat = _coastline_with_breaks(coastline_lon, coastline_lat, longitude_center)
            figure.add_trace(go.Scatter3d(x=coast_lon, y=coast_lat, z=[depth_top] * len(coast_lon), mode="lines", name="Coastline (surface)", line={"color": "#2d6fa3", "width": 1.2}, hoverinfo="skip", showlegend=False))
            figure.add_trace(go.Scatter3d(x=coast_lon, y=coast_lat, z=[depth_bottom] * len(coast_lon), mode="lines", name="Coastline projection", line={"color": "#8c959b", "width": 0.7}, hoverinfo="skip", showlegend=False))
    color = None if color_column == NO_COLOR else pd.to_numeric(dataframe.loc[rows.index, color_column], errors="coerce")
    figure.add_trace(go.Scatter3d(
        x=rows["_Longitude_plot"], y=rows[columns[1]], z=rows[columns[2]], mode="markers", name="Uploaded data",
        marker=marker_settings(style, color, color_column, palette_label),
        text=rich_hover_text(dataframe.loc[rows.index]),
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))
    figure.update_layout(
        title="Geographic 3D view",
        scene=scene,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        height=640, margin={"r": 20, "l": 10, "b": 10, "t": 45},
    )
    return figure, len(rows)


def _auto_map_view(dataframe):
    """Return a practical map view / データ範囲に合わせた地図中心とズームを返す。"""
    lat = pd.to_numeric(dataframe["Latitude_degN"], errors="coerce").dropna()
    lon = pd.to_numeric(dataframe["Longitude_degE"], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return 36.0, 138.0, 3.5
    lat_span, lon_span = max(float(lat.max() - lat.min()), 0.1), max(float(lon.max() - lon.min()), 0.1)
    if lon_span > 100:
        return 0.0, 180.0, 1.4
    zoom = max(1.0, min(12.0, min(np.log2(360 / lon_span), np.log2(180 / lat_span)) + 0.4))
    return float(lat.mean()), float(lon.mean()), float(zoom)


def create_2d_map(dataframe, color_column, palette_label, region_label, map_mode):
    """Create a filtered 2-D map / 統合フィルタ後データの2次元地図を作成する。"""
    required = ["Longitude_degE", "Latitude_degN"]
    rows = plot_rows(dataframe, required).query("-90 <= Latitude_degN <= 90")
    if rows.empty:
        return None, 0
    # Retain all metadata for hover while coordinates are numeric / 座標を数値化しつつhover用の全列を保つ。
    display = dataframe.loc[rows.index].copy()
    display[required] = rows[required]
    arguments = {
        "data_frame": display,
        "lat": "Latitude_degN",
        "lon": "Longitude_degE",
        # Use customdata rather than Plotly's text field: Mapbox renders text
        # permanently on the map, whereas customdata is hover-only.
        # text列は地図上へ常時描画されるため、hover専用customdataを使う。
        "custom_data": ["_Quick_hover"],
        "height": 580,
    }
    display["_Quick_hover"] = rich_hover_text(display)
    if color_column != NO_COLOR and color_column in display.columns:
        arguments["color"] = color_column
        arguments["color_continuous_scale"] = envgeo_utils.get_plotly_colormap(color_column, palette_label)
    figure = px.scatter_mapbox(**arguments)
    figure.update_traces(hovertemplate="%{customdata[0]}<extra></extra>", marker={"size": 8, "opacity": 0.78})
    figure = envgeo_utils.apply_map_style(figure, map_mode)
    if region_label in envgeo_utils.MAP_REGION_PRESETS:
        center_lat, center_lon, zoom = envgeo_utils.map_region_view(region_label)
    else:
        center_lat, center_lon, zoom = _auto_map_view(display)
    layout_updates = {
        "mapbox": {"center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom},
        "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
    }
    if color_column != NO_COLOR and color_column in display.columns:
        # Overlay the colorbar inside the figure / カラーバーを地図内へ重ねて横幅を確保する。
        layout_updates["coloraxis_colorbar"] = {
            "title": color_column,
            "x": 0.985,
            "xanchor": "right",
            "y": 0.5,
            "yanchor": "middle",
            "len": 0.74,
            "thickness": 15,
            "bgcolor": "rgba(255,255,255,0.65)",
        }
    figure.update_layout(**layout_updates)
    return figure, len(display)


def render_overview_and_quality(dataframe, uploaded_count):
    """Show integrated counts and quality checks / 統合データ数・欠損・品質を表示する。"""
    quality_rows = envgeo_utils.get_quality_rows(dataframe)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Filtered rows", f"{len(dataframe):,}")
    metric_columns[1].metric("Uploaded rows", f"{uploaded_count:,}")
    metric_columns[2].metric("Quality-flagged rows", f"{len(quality_rows):,}")
    metric_columns[3].metric("Datasets", f"{dataframe['Dataset'].nunique() if 'Dataset' in dataframe else 0:,}")
    st.subheader("Data availability and missing values")
    availability = pd.DataFrame({
        "Available": dataframe.notna().sum(),
        "Missing": dataframe.isna().sum(),
    })
    availability["Missing (%)"] = (availability["Missing"] / max(len(dataframe), 1) * 100).round(1)
    st.dataframe(envgeo_utils.arrow_display_dataframe(availability), **envgeo_utils.stretch_width_kwargs(st.dataframe))
    envgeo_utils.render_quality_flag_criteria_note()
    if quality_rows.empty:
        st.success("No quality-flagged rows are present in the current filtered data.")
    else:
        st.warning(f"{len(quality_rows):,} filtered rows have quality flags.")
        st.dataframe(envgeo_utils.arrow_display_dataframe(quality_rows), **envgeo_utils.stretch_width_kwargs(st.dataframe))


def ensure_common_filter_columns(dataframe):
    """Add optional filter columns / 任意アップロード表にも共通フィルタ列を安全に補う。"""
    result = dataframe.copy()
    filter_columns = [
        "Dataset", "Transect", "Year", "Month", "Longitude_degE", "Latitude_degN",
        "Depth_m", "Salinity", "d18O", "Temperature_degC",
    ]
    for column in filter_columns:
        if column not in result.columns:
            result[column] = np.nan
    result["Dataset"] = result["Dataset"].fillna(envgeo_utils.UPLOADED_DATA_LABEL)
    result["Transect"] = result["Transect"].fillna("no_name")
    return result


def download_figure(figure, filename, key):
    """Provide a portable interactive Plotly HTML file."""
    st.download_button("Download interactive HTML", figure.to_html(include_plotlyjs="cdn").encode("utf-8"), envgeo_utils.build_figure_filename(filename, extension="html"), "text/html", key=key)


def main():
    """Render the upload-first visualizer / アップロード優先の可視化画面を表示する。"""
    st.header(f"User Data Check & Quick Visualizer ({version})")
    st.caption("Combine reference and uploaded data, check quality and missing values, then explore the filtered data as 2D--4D figures.")
    # Shared session upload and editable role assignment / 共通セッションの読込と列の役割設定。
    uploaded_df = envgeo_user_data.render_upload_panel("quick_visualizer", "Any numeric columns can be used for 2D/3D axes. Assign longitude, latitude, and depth below to enable the geographic 3D view.")
    uploaded_df = envgeo_user_data.render_column_controls(uploaded_df, {}, "quick_visualizer", optional_roles={"Longitude column (optional)": "Longitude_degE", "Latitude column (optional)": "Latitude_degN", "Depth column (optional)": "Depth_m"})

    # Optional comparison data / 比較用の参照データは必要な場合だけ選択する。
    ref_data = st.radio(
        "Comparison data source (optional):",
        (NO_COMPARISON_DATA, envgeo_utils.data_source_JAPAN_SEA, envgeo_utils.data_source_AROUND_JAPAN, envgeo_utils.data_source_GLOBAL),
        horizontal=True,
        key="quick_visualizer_data_source",
    )
    reference_df = (
        pd.DataFrame()
        if ref_data == NO_COMPARISON_DATA
        else envgeo_utils.load_isotope_data(ref_data)
    )
    if ref_data != NO_COMPARISON_DATA and reference_df.empty:
        st.warning("No reference data are available for the selected source.")
        return
    combined_df = envgeo_utils.combine_reference_and_uploaded_for_filtering(reference_df, uploaded_df)
    combined_df = ensure_common_filter_columns(combined_df)
    if combined_df.empty:
        st.info("Upload a CSV/XLSX file, or select a comparison data source to begin.")
        st.download_button("Download CSV template", envgeo_utils.build_upload_template_csv(), "envgeo_seawater_upload_template.csv", "text/csv")
        return
    filter_result = envgeo_utils.sidebar_filter_and_display(
        combined_df,
        # The filter helper needs one reference profile for default ranges.
        # Upload-only mode uses the global profile but does not load its rows.
        envgeo_utils.data_source_GLOBAL if ref_data == NO_COMPARISON_DATA else ref_data,
        envgeo_utils.data_source_JAPAN_SEA,
        envgeo_utils.data_source_AROUND_JAPAN,
    )
    filtered_df = filter_result[0]
    _, selected_uploaded_df = envgeo_utils.split_uploaded_rows(filtered_df)

    # Marker controls and variables apply to every quick-look figure.
    # マーカー設定と変数選択は、すべての簡易可視化で共通に使う。
    style = envgeo_user_data.render_marker_style_controls(filtered_df, "quick_visualizer", marker_size_default=64, marker_size_min=4, marker_size_step=4)
    numeric_options = numeric_columns(filtered_df)
    if len(numeric_options) < 2:
        st.warning("At least two numeric columns are required after filtering.")
        return
    # Figure controls apply consistently to all three tabs / 図の設定は3つのタブで共通に使う。
    with st.sidebar.container(border=True):
        st.subheader("Visualization settings")
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        x_default = "Salinity" if "Salinity" in numeric_options else numeric_options[0]
        y_default = "d18O" if "d18O" in numeric_options else numeric_options[min(1, len(numeric_options) - 1)]
        color_default = "Temperature_degC" if "Temperature_degC" in numeric_options else NO_COLOR
        x_column = st.selectbox("X axis", numeric_options, index=numeric_options.index(x_default))
        y_column = st.selectbox("Y axis", numeric_options, index=numeric_options.index(y_default))
        z_default = "Depth_m" if "Depth_m" in numeric_options else numeric_options[min(2, len(numeric_options) - 1)]
        z_column = st.selectbox("Z axis", numeric_options, index=numeric_options.index(z_default))
        color_options = [NO_COLOR] + numeric_options
        color_column = st.selectbox(
            "Color / fourth dimension (optional)",
            color_options,
            index=color_options.index(color_default),
        )
        palette_options = envgeo_utils.get_plotly_colormap_options(
            None if color_column == NO_COLOR else color_column
        )
        palette_label = st.selectbox(
            "Color palette",
            list(palette_options),
            index=list(palette_options).index(
                envgeo_utils.recommended_plotly_colormap_label(color_column)
            ),
            disabled=color_column == NO_COLOR,
            help="Applied consistently to the 2D, 3D/4D, and Geographic 3D figures.",
        )
        maximum_rows = st.number_input(
            "Maximum plotted rows",
            min_value=100,
            max_value=50000,
            value=5000,
            step=100,
            help="Each interactive figure uses at most this many filtered rows. "
                 "Increase up to 50,000 when your browser can comfortably render the figure.",
        )
        show_regression = st.checkbox("Show 2D linear regression", value=False)
        reverse_z = st.checkbox("Reverse Z axis", value="Depth_m" in z_column)
        show_coastline = st.checkbox("Show coastline in geographic 3D view", value=True)
        # Map extent and longitude frame change only the view, not uploaded rows.
        # 海域範囲と経度中心は表示だけを変え、アップロード行は絞り込まない。
        geographic_region = st.selectbox(
            "Geographic 3D region",
            ["Auto from uploaded data"] + list(envgeo_utils.MAP_REGION_PRESETS),
            help=(
                "Choose the displayed longitude-latitude extent for Geographic 3D. "
                "This changes only the view and does not filter uploaded rows."
            ),
        )
        center_option = st.radio(
            "Map longitude center",
            ["Atlantic (0°)", "Pacific (180°)"],
            horizontal=True,
            help="Pacific-centred mode places the dateline at the map edge for continuous Pacific views.",
        )
        reference_source = st.selectbox("Reference context in geographic 3D view", ["None", envgeo_utils.data_source_JAPAN_SEA, envgeo_utils.data_source_AROUND_JAPAN, envgeo_utils.data_source_GLOBAL], help="Optional light-gray context points. They are not merged with uploaded data.")

    # Keep interactive figures responsive / インタラクティブ図の応答性を保つため表示行を制限する。
    displayed_df, sampled = sample_rows(filtered_df, int(maximum_rows))
    st.subheader("Filtered integrated data")
    st.write(f"{len(filtered_df):,} rows available (Uploaded data: {len(selected_uploaded_df):,})")
    if sampled:
        st.info(
            f"Figures currently show {len(displayed_df):,} of {len(filtered_df):,} filtered rows "
            f"because Maximum plotted rows is set to {int(maximum_rows):,}. "
            "Increase the setting (up to 50,000) to plot more rows."
        )
    else:
        st.caption(
            f"All {len(displayed_df):,} filtered rows are plotted "
            f"(Maximum plotted rows: {int(maximum_rows):,}; adjustable up to 50,000)."
        )
    with st.expander("Preview, data quality, and download", expanded=False):
        st.dataframe(
            envgeo_utils.arrow_display_dataframe(displayed_df),
            **envgeo_utils.stretch_width_kwargs(st.dataframe),
        )
        quality_rows = envgeo_utils.get_quality_rows(filtered_df)
        st.write(f"Quality-flagged rows: {len(quality_rows):,} / {len(filtered_df):,}")
        envgeo_utils.render_quality_flag_criteria_note()
        st.download_button("Download filtered integrated data (CSV)", filtered_df.to_csv(index=False).encode("utf-8-sig"), "envgeo_seawater_filtered_integrated_data.csv", "text/csv")

    # Use shared Streamlit-version-compatible tabs / 対応Streamlit版共通のタブ表示を使う。
    render_tab_style()
    envgeo_utils.render_earthquake_tab_style()
    tab_overview, tab_explore, tab_3d, tab_map_2d, tab_map_3d, tab_data = st.tabs([
        "✅ Overview & Quality", "📈 2D Explore", "🧊 3D / 4D", "🗺️ 2D Map", "🌍 3D Map", "🗂️ Data & Export"
    ])
    with tab_overview:
        render_overview_and_quality(filtered_df, len(selected_uploaded_df))
    with tab_explore:
        plot_mode = st.segmented_control(
            "2D quick plot", ["Custom 2D", "Salinity-d18O", "Temperature-Salinity"], default="Custom 2D", key="quick_visualizer_2d_mode"
        )
        if plot_mode == "Salinity-d18O":
            x_plot, y_plot = "Salinity", "d18O"
        elif plot_mode == "Temperature-Salinity":
            x_plot, y_plot = "Salinity", "Temperature_degC"
        else:
            x_plot, y_plot = x_column, y_column
        if x_plot not in displayed_df.columns or y_plot not in displayed_df.columns:
            st.info(f"{plot_mode} requires {x_plot} and {y_plot}.")
        else:
            figure, count = create_2d(displayed_df, x_plot, y_plot, color_column, style, show_regression, palette_label)
            st.caption(f"{count:,} valid rows plotted using {x_plot} and {y_plot}.")
            st.plotly_chart(figure, **envgeo_utils.stretch_width_kwargs(st.plotly_chart))
            download_figure(figure, "integrated_data_2d", "quick_visualizer_download_2d")
    with tab_3d:
        figure, count = create_3d(displayed_df, x_column, y_column, z_column, color_column, style, reverse_z, palette_label)
        st.caption(f"{count:,} valid rows plotted using {x_column}, {y_column}, and {z_column}.")
        st.plotly_chart(figure, **envgeo_utils.stretch_width_kwargs(st.plotly_chart))
        download_figure(figure, "integrated_data_3d_4d", "quick_visualizer_download_3d")
    with tab_map_2d:
        map_mode = st.selectbox("Map style", envgeo_utils.MAP_MODE_OPTIONS, key="quick_visualizer_map_style")
        map_figure, map_count = create_2d_map(displayed_df, color_column, palette_label, geographic_region, map_mode)
        if map_figure is None:
            st.info("Assign longitude and latitude columns to enable the 2D map.")
        else:
            st.caption(f"{map_count:,} valid longitude-latitude rows plotted.")
            st.plotly_chart(map_figure, **envgeo_utils.stretch_width_kwargs(st.plotly_chart))
            download_figure(map_figure, "integrated_data_map", "quick_visualizer_download_map")
    with tab_map_3d:
        st.subheader("Geographic 3D")
        # Geographic 3-D needs assigned longitude, latitude, and depth columns.
        # 地理3Dには経度・緯度・深度の列割当が必要である。
        required = {"Longitude_degE", "Latitude_degN", "Depth_m"}
        if not required.issubset(displayed_df.columns):
            st.info("Assign longitude, latitude, and depth columns in ‘Uploaded data columns’ to enable this view.")
        else:
            try:
                region_bounds = (
                    envgeo_utils.MAP_REGION_PRESETS[geographic_region]["bounds"]
                    if geographic_region in envgeo_utils.MAP_REGION_PRESETS
                    else None
                )
                figure, count = create_geographic(
                    displayed_df,
                    color_column,
                    style,
                    show_coastline,
                    reference_source,
                    region_bounds,
                    palette_label,
                    180 if center_option.startswith("Pacific") else 0,
                )
            except Exception as exc:
                st.warning(f"Geographic 3D view could not be created: {exc}")
            else:
                st.caption(f"{count:,} valid longitude-latitude-depth rows plotted.")
                st.plotly_chart(figure, **envgeo_utils.stretch_width_kwargs(st.plotly_chart))
                download_figure(figure, "integrated_data_geographic_3d", "quick_visualizer_download_geographic")
    with tab_data:
        st.subheader("Filtered integrated dataset")
        st.dataframe(envgeo_utils.arrow_display_dataframe(filtered_df), **envgeo_utils.stretch_width_kwargs(st.dataframe))
        st.download_button("Download filtered integrated data (CSV)", filtered_df.to_csv(index=False).encode("utf-8-sig"), "envgeo_seawater_filtered_integrated_data.csv", "text/csv", key="quick_visualizer_download_filtered")


if __name__ == "__main__":
    main()
