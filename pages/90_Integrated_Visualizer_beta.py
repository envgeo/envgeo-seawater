#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimental integrated visualizer for EnvGeo-Seawater.

This page tests an Advanced-style workflow: one shared filtered dataset,
then multiple plot/table views switched by tabs.

EnvGeo-Seawater 統合可視化ページの試作版です。
共通フィルタで抽出した同じデータを、タブで複数の表示に切り替えます。

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

import math
import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import envgeo_utils


version = "1.3.2"

BASE_DIR = Path(__file__).resolve().parent

FULL_PAGE_WORKFLOWS = {
    "Interactive 2D/2.5D Visualizer": "03_[Interactive]_2Dplus_Visualizer.py",
    "Interactive 3D/4D Visualizer": "04_[Interactive]_3D_4D_Visualizer.py",
    "Salinity-d18O Relationship": "31_Salinity-d18O_Relationship.py",
    "Isotope & Hydrographic Mapping": "32_Isotope_Hydrographic_Mapping.py",
    "T-S Diagram": "34_T-S_diagram.py",
    "Custom Parameter Plot beta": "35_Custom_Parameter_Plot_beta.py",
    "Depth Profile": "37_Depth_Profile.py",
    "Correlation Overview": "51_Correlation_Overview.py",
    "Vertical Section beta": "53_Vertical_Section_Visualizer.py",
}

# Pages that render their own upload panel, column mapping, marker style, and
# frontmost overlay via envgeo_user_data / INTEGRATED_EMBEDDED_PAGE_KEY. Any
# other page in FULL_PAGE_WORKFLOWS still uses the temporary
# load_isotope_data() merge fallback below until its own overlay is added.
# 独自のアップロードパネル・列対応・マーカー設定・最前面オーバーレイを持つページ一覧。
# ここに無いページは、そのページ固有の重ね描画ができるまで一時的な
# load_isotope_data() 結合フォールバックを使う。
NATIVE_UPLOAD_OVERLAY_PAGES = {
    "31_Salinity-d18O_Relationship.py",
    "32_Isotope_Hydrographic_Mapping.py",
    "34_T-S_diagram.py",
    "35_Custom_Parameter_Plot_beta.py",
    "37_Depth_Profile.py",
    "53_Vertical_Section_Visualizer.py",
}


def render_tab_style():
    """
    Render compact, visible tab styling for the shared-filter workflow.

    earthquake Advancedページと同じ考え方で、タブの境界と選択状態を見やすくします。
    """
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


def _numeric_series(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def _quality_rows(df):
    return envgeo_utils.get_quality_rows(df)


def _read_uploaded_table(uploaded_file):
    return envgeo_utils.read_uploaded_table(uploaded_file)


def _prepare_uploaded_data(df):
    """
    Standardize uploaded data enough for the integrated beta plots.

    アップロードデータを統合ページで扱いやすい形へ軽く標準化します。
    標準列が存在する場合は数値化、品質チェック、d-excess計算を行います。
    """
    return envgeo_utils.prepare_uploaded_data(df)


def _upload_template_csv():
    return envgeo_utils.build_upload_template_csv()


def render_upload_panel():
    uploaded_df = envgeo_utils.get_uploaded_data()
    with st.sidebar.expander("Uploaded data overlay", expanded=False):
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        st.caption(
            "Uploaded files are used only in the current Streamlit session and "
            "are not saved to local or server storage."
        )
        st.download_button(
            "Download CSV template",
            data=_upload_template_csv(),
            file_name="envgeo_seawater_upload_template.csv",
            mime="text/csv",
        )

        if not uploaded_df.empty:
            filename = envgeo_utils.get_uploaded_filename() or "current session"
            st.success(f"Using {len(uploaded_df):,} shared rows from {filename}.")
            if st.button(
                "Clear shared uploaded data",
                key="integrated_clear_uploaded_data",
            ):
                envgeo_utils.clear_uploaded_data()
                st.session_state["integrated_upload_generation"] = (
                    st.session_state.get("integrated_upload_generation", 0) + 1
                )
                st.rerun()

        upload_generation = st.session_state.get("integrated_upload_generation", 0)
        uploaded_file = st.file_uploader(
            "Upload user data",
            type=["xlsx", "xls", "csv"],
            key=f"integrated_uploaded_file_{upload_generation}",
        )
        if uploaded_file is None:
            return uploaded_df

        try:
            uploaded_df = _prepare_uploaded_data(_read_uploaded_table(uploaded_file))
            envgeo_utils.store_uploaded_data(uploaded_df, uploaded_file.name)
        except Exception as exc:
            st.error(f"Could not read uploaded file: {exc}")
            return pd.DataFrame()

        renamed_columns = uploaded_df.attrs.get("standardized_column_renames", {})
        if renamed_columns:
            rename_text = ", ".join(
                f"{source} -> {target}" for source, target in renamed_columns.items()
            )
            st.caption(f"Standardized column names: {rename_text}")
        st.success(f"Loaded {len(uploaded_df):,} uploaded rows.")
        return uploaded_df


def render_uploaded_marker_style_controls():
    """
    Sidebar controls for emphasizing uploaded data overlays.

    アップロードデータを参照データより目立たせるための表示設定です。
    """
    with st.sidebar.expander("Uploaded marker style", expanded=False):
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        st.caption(
            "Applied to Shared-filter beta overlays and Upload-tab plots. "
            "Full existing page mode follows each original page's plotting settings."
        )
        size_multiplier = st.slider(
            "Marker size multiplier",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key="uploaded_marker_size_multiplier",
        )
        color = st.selectbox(
            "Marker color",
            ["crimson", "black", "magenta", "cyan", "yellow", "white"],
            index=0,
            key="uploaded_marker_color",
        )
        color_mode = st.selectbox(
            "Marker color mode",
            ["Use current colorbar when possible", "Fixed marker color"],
            index=0,
            key="uploaded_marker_color_mode",
        )
        symbol = st.selectbox(
            "Marker shape",
            ["diamond", "x", "circle", "square", "cross", "triangle-up", "star"],
            index=0,
            key="uploaded_marker_symbol",
        )
        opacity = st.slider(
            "Marker opacity",
            min_value=0.2,
            max_value=1.0,
            value=0.95,
            step=0.05,
            key="uploaded_marker_opacity",
        )
        outline_width = st.slider(
            "Marker outline width",
            min_value=0,
            max_value=8,
            value=2,
            step=1,
            key="uploaded_marker_outline_width",
            help=(
                "Controls the black outline around uploaded data markers. "
                "For maps, this is drawn as a slightly larger black marker behind the uploaded point."
            ),
        )
        map_offset = st.slider(
            "Map marker offset",
            min_value=0.0,
            max_value=0.2,
            value=0.0,
            step=0.01,
            key="uploaded_map_marker_offset",
            help=(
                "Moves only the displayed uploaded markers by this many degrees in latitude and longitude. "
                "Use this when uploaded points overlap the reference dataset and are hard to see. "
                "The uploaded data values themselves are not changed."
            ),
        )

    map_size = int(6 * size_multiplier)
    plot_size = int(6 * size_multiplier)
    return {
        "size_multiplier": size_multiplier,
        "color": color,
        "color_mode": color_mode,
        "symbol": symbol,
        "opacity": opacity,
        "outline_width": outline_width,
        "map_size": map_size,
        "plot_size": plot_size,
        "outline_map_size": map_size + (outline_width * 2),
        "outline_plot_size": plot_size + (outline_width * 2),
        "map_offset": map_offset,
    }


def _merge_reference_and_uploaded(reference_df, uploaded_df):
    """
    Return an in-memory merged dataframe for plotting only.

    参照データとアップロードデータを、描画用にメモリ上で一時結合します。
    アップロードファイルや結合結果はローカル/サーバーへ保存しません。
    """
    if uploaded_df.empty:
        return reference_df

    merged = pd.concat([reference_df, uploaded_df], ignore_index=True, sort=False)
    for col in ["Dataset", "Transect", "reference", "Cruise", "Station", "Date"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("Uploaded data")
    return merged


def render_full_existing_page(uploaded_df):
    """
    Run one original Streamlit page inside this integrated beta page.

    既存ページを1つ選び、そのまま統合ページ内で実行します。
    Streamlitのタブは非表示タブも実行するため、重い既存ページを同時に
    全部読み込まず、選択された1ページだけを描画します。
    """
    st.subheader("Full Existing Page Workflow")
    workflow_name = st.selectbox(
        "Select workflow",
        list(FULL_PAGE_WORKFLOWS.keys()),
        key="integrated_full_page_workflow",
    )
    page_path = BASE_DIR / FULL_PAGE_WORKFLOWS[workflow_name]
    uses_native_upload_overlay = page_path.name in NATIVE_UPLOAD_OVERLAY_PAGES

    if uploaded_df.empty:
        st.caption(
            "This mode keeps the original page behavior, including its own filters "
            "and figure controls."
        )
    elif uses_native_upload_overlay:
        st.caption(
            "Uploaded rows are handled by this page's native overlay. "
            "Reference-data filters remain separate from uploaded data."
        )
    else:
        st.caption(
            "Uploaded rows are temporarily merged into the selected workflow. "
            "They should appear as `Uploaded data` in the original page filters."
        )
    st.divider()

    module_name = f"envgeo_integrated_{page_path.stem}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, page_path)
    if spec is None or spec.loader is None:
        st.error(f"Could not load page: {page_path.name}")
        return

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        st.error(f"Could not load {page_path.name}: missing Python package `{exc.name}`.")
        return
    except Exception as exc:
        st.error(f"Could not load {page_path.name}: {exc}")
        return

    if hasattr(module, "main") and uses_native_upload_overlay:
        previous_embedded_page = st.session_state.get(
            envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY
        )
        try:
            st.session_state[envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY] = page_path.name
            module.main()
        finally:
            if previous_embedded_page is None:
                st.session_state.pop(envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY, None)
            else:
                st.session_state[envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY] = (
                    previous_embedded_page
                )
    elif hasattr(module, "main"):
        original_loader = envgeo_utils.load_isotope_data

        def load_isotope_data_with_upload(ref_data, sheet_num=0):
            reference_df = original_loader(ref_data, sheet_num=sheet_num)
            return _merge_reference_and_uploaded(reference_df, uploaded_df)

        try:
            envgeo_utils.load_isotope_data = load_isotope_data_with_upload
            module.envgeo_utils.load_isotope_data = load_isotope_data_with_upload
            module.main()
        finally:
            envgeo_utils.load_isotope_data = original_loader
            module.envgeo_utils.load_isotope_data = original_loader


def _auto_map_view(df):
    default_lat, default_lon, default_zoom = 36.0, 138.0, 4.0
    lat = _numeric_series(df, "Latitude_degN").dropna()
    lon = _numeric_series(df, "Longitude_degE").dropna()

    if lat.empty or lon.empty:
        return default_lat, default_lon, default_zoom

    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    lat_diff = max(lat_max - lat_min, 0.1)
    lon_diff = max(lon_max - lon_min, 0.1)
    zoom_lon = math.log2((1200 * 360) / (lon_diff * 256))
    zoom_lat = math.log2((700 * 180) / (lat_diff * 256))
    auto_zoom = max(1, min(15, min(zoom_lon, zoom_lat) - 2.0))

    if lon_diff > 100:
        return default_lat, default_lon, 1.5

    return center_lat, center_lon, auto_zoom


def _available_hover_columns(df):
    candidates = [
        "Dataset",
        "reference",
        "Cruise",
        "Station",
        "Year",
        "Month",
        "Day",
        "Depth_m",
        "Temperature_degC",
        "Salinity",
        "d18O",
        "dD",
        "d-excess",
        envgeo_utils.QUALITY_FLAG_COLUMN,
    ]
    return [column for column in candidates if column in df.columns]


def _can_use_current_colorbar(uploaded_style, uploaded_df, color_column):
    """
    Return True when uploaded points can share the active numeric Plotly colorbar.

    アップロードデータの点を、現在の数値カラーバーに接続できるか確認します。
    """
    return (
        uploaded_style.get("color_mode") == "Use current colorbar when possible"
        and color_column in uploaded_df.columns
        and pd.api.types.is_numeric_dtype(uploaded_df[color_column])
    )


def _apply_shared_coloraxis_range(fig, reference_df, uploaded_df, color_column):
    """
    Expand the active Plotly colorbar range to include uploaded values.

    参照データとアップロードデータを同じカラーバーで比較できるよう、
    カラーバーの範囲を両方の値から決めます。
    """
    values = pd.concat(
        [
            _numeric_series(reference_df, color_column),
            _numeric_series(uploaded_df, color_column),
        ],
        ignore_index=True,
    ).dropna()
    if values.empty:
        return

    fig.update_coloraxes(cmin=float(values.min()), cmax=float(values.max()))


def _numeric_first_color_options(df, candidates):
    """
    Return available color columns with numeric columns first.

    数値カラーバーで比較しやすいよう、数値列を先に並べます。
    """
    available = [column for column in candidates if column in df.columns]
    numeric = [
        column
        for column in available
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical = [column for column in available if column not in numeric]
    return numeric + categorical


def _select_plotly_colormap(color_column, key):
    """
    Render a shared Plotly colormap selector.

    共通のPlotlyカラーマップ選択UIを表示します。
    """
    colormap_options = envgeo_utils.get_plotly_colormap_options(color_column)
    option_labels = list(colormap_options.keys())
    recommended_label = envgeo_utils.recommended_plotly_colormap_label(color_column)
    default_index = option_labels.index(recommended_label)
    return st.selectbox(
        "Colormap",
        option_labels,
        index=default_index,
        key=key,
        help=(
            "cmocean palettes are oceanography-oriented Plotly colorscales. "
            "Use EnvGeo variable default to keep the previous EnvGeo behavior."
        ),
    )


def _plotly_color_scale_args(df, color_column, colormap_label=None):
    """
    Return shared Plotly color-scale settings for numeric color columns.

    数値列の色付けでは、個別ページと同じ共通カラースケールを使います。
    """
    if color_column in df.columns and pd.api.types.is_numeric_dtype(df[color_column]):
        return {
            "color_continuous_scale": envgeo_utils.get_plotly_colormap(
                color_column,
                colormap_label,
            )
        }
    return {}


def render_summary(df, quality_df, uploaded_quality_df=None):
    st.subheader("Filtered Dataset")
    if uploaded_quality_df is None:
        uploaded_quality_df = pd.DataFrame()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{len(df):,}")
    metric_cols[1].metric("Reference quality flags", f"{len(quality_df):,}")
    metric_cols[2].metric("Datasets", df["Dataset"].nunique() if "Dataset" in df.columns else 0)
    metric_cols[3].metric("Uploaded quality flags", f"{len(uploaded_quality_df):,}")

    stats_cols = ["d18O", "dD", "d-excess", "Salinity", "Temperature_degC", "Depth_m"]
    available_stats = [column for column in stats_cols if column in df.columns]
    if available_stats:
        summary = df[available_stats].describe().T[["count", "mean", "std", "min", "max"]]
        st.dataframe(
            summary.round(3), **envgeo_utils.stretch_width_kwargs(st.dataframe)
        )
    envgeo_utils.render_filtered_report_download(
        df,
        {
            "Workflow": "Integrated Visualizer beta / Shared-filter beta",
            "Reference quality flags": len(quality_df),
            "Uploaded quality flags": len(uploaded_quality_df),
        },
        key_prefix="integrated_summary",
    )


@st.fragment
def render_map(df, uploaded_df=None, uploaded_style=None):
    clean = df.dropna(subset=["Latitude_degN", "Longitude_degE"]).copy()
    if clean.empty:
        st.warning("No mappable data for the selected filters.")
        return

    color_options = [
        column
        for column in ["d18O", "dD", "d-excess", "Salinity", "Temperature_degC", "Depth_m"]
        if column in clean.columns
    ]
    with st.popover(
        "Map controls", **envgeo_utils.stretch_width_kwargs(st.popover)
    ):
        left_controls, right_controls = st.columns([2, 1])
        with left_controls:
            color_controls = st.columns(2)
            with color_controls[0]:
                color_column = st.selectbox("Color", color_options, index=0)
            with color_controls[1]:
                colormap_label = _select_plotly_colormap(
                    color_column,
                    key="integrated_map_colormap",
                )

            region_options = [envgeo_utils.MAP_REGION_AUTO] + list(envgeo_utils.MAP_REGION_PRESETS)
            region_label = st.selectbox(
                "Region preset",
                region_options,
                index=0,
                key="integrated_map_region_preset",
            )

        with right_controls:
            map_mode = st.radio(
                "Map Style",
                envgeo_utils.MAP_MODE_OPTIONS,
                horizontal=True,
                key="integrated_map_style",
            )

    # Keep the map close to the tab top after Streamlit reruns.
    # Streamlitの再実行後も地図がすぐ見えるよう、設定UIはポップオーバーに集約する。
    st.caption(
        f"Color: {color_column} | Colormap: {colormap_label} | "
        f"Region: {region_label} | Map Style: {map_mode}"
    )

    if region_label == envgeo_utils.MAP_REGION_AUTO:
        center_lat, center_lon, auto_zoom = _auto_map_view(clean)
    else:
        center_lat, center_lon, auto_zoom = envgeo_utils.map_region_view(region_label)

    fig = px.scatter_mapbox(
        clean,
        lat="Latitude_degN",
        lon="Longitude_degE",
        color=color_column,
        hover_data=_available_hover_columns(clean),
        opacity=0.65,
        height=560,
        **_plotly_color_scale_args(clean, color_column, colormap_label),
    )
    fig = envgeo_utils.apply_map_style(fig, map_mode)
    fig.update_layout(
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
        autosize=True,
        coloraxis_colorbar=dict(title=color_column, x=1.0, xanchor="right"),
    )

    if uploaded_df is not None and not uploaded_df.empty:
        uploaded_style = uploaded_style or render_uploaded_marker_style_controls()
        uploaded_clean = uploaded_df.dropna(subset=["Latitude_degN", "Longitude_degE"]).copy()
        if not uploaded_clean.empty:
            use_colorbar = _can_use_current_colorbar(
                uploaded_style, uploaded_clean, color_column
            )
            if use_colorbar:
                _apply_shared_coloraxis_range(fig, clean, uploaded_clean, color_column)
            display_lat = uploaded_clean["Latitude_degN"] + uploaded_style["map_offset"]
            display_lon = uploaded_clean["Longitude_degE"] + uploaded_style["map_offset"]
            fig.add_trace(
                go.Scattermapbox(
                    lat=display_lat,
                    lon=display_lon,
                    mode="markers",
                    name="Uploaded data outline",
                    below="",
                    marker=dict(
                        size=uploaded_style["outline_map_size"],
                        color="black",
                        opacity=min(1.0, uploaded_style["opacity"] + 0.05),
                    ),
                    text=uploaded_clean.get("Station"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scattermapbox(
                    lat=display_lat,
                    lon=display_lon,
                    mode="markers",
                    name="Uploaded data",
                    below="",
                    marker=dict(
                        size=uploaded_style["map_size"],
                        color=uploaded_clean[color_column] if use_colorbar else uploaded_style["color"],
                        coloraxis="coloraxis" if use_colorbar else None,
                        symbol=uploaded_style["symbol"],
                        opacity=uploaded_style["opacity"],
                    ),
                    text=uploaded_clean.get("Station"),
                    hovertemplate=(
                        "Uploaded data<br>"
                        "Lon=%{lon}<br>Lat=%{lat}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

    st.plotly_chart(
        fig,
        key="integrated_map",
        **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
    )

def render_ts_diagram(df, uploaded_df=None, uploaded_style=None):
    st.subheader("T-S Diagram")
    clean = df.dropna(subset=["Salinity", "Temperature_degC"]).copy()
    if clean.empty:
        st.warning("No T-S data for the selected filters.")
        return

    color_options = [
        column
        for column in ["d18O", "dD", "d-excess", "Depth_m", "Dataset"]
        if column in clean.columns
    ]
    controls = st.columns(2)
    with controls[0]:
        color_column = st.selectbox(
            "Color",
            color_options,
            index=0,
            key="integrated_ts_color",
        )
    with controls[1]:
        colormap_label = _select_plotly_colormap(
            color_column,
            key="integrated_ts_colormap",
        )
    fig = px.scatter(
        clean,
        x="Salinity",
        y="Temperature_degC",
        color=color_column,
        hover_data=_available_hover_columns(clean),
        opacity=0.7,
        height=560,
        **_plotly_color_scale_args(clean, color_column, colormap_label),
    )
    fig.update_layout(
        xaxis_title="Salinity",
        yaxis_title="Temperature (degC)",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    if uploaded_df is not None and not uploaded_df.empty:
        uploaded_style = uploaded_style or render_uploaded_marker_style_controls()
        uploaded_clean = uploaded_df.dropna(subset=["Salinity", "Temperature_degC"]).copy()
        if not uploaded_clean.empty:
            use_colorbar = _can_use_current_colorbar(
                uploaded_style, uploaded_clean, color_column
            )
            if use_colorbar:
                _apply_shared_coloraxis_range(fig, clean, uploaded_clean, color_column)
            fig.add_trace(
                go.Scattergl(
                    x=uploaded_clean["Salinity"],
                    y=uploaded_clean["Temperature_degC"],
                    mode="markers",
                    name="Uploaded data outline",
                    marker=dict(
                        color="black",
                        size=uploaded_style["outline_plot_size"],
                        symbol=uploaded_style["symbol"],
                        opacity=min(1.0, uploaded_style["opacity"] + 0.05),
                    ),
                    text=uploaded_clean.get("Station"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
            )
            fig.add_trace(
                go.Scattergl(
                    x=uploaded_clean["Salinity"],
                    y=uploaded_clean["Temperature_degC"],
                    mode="markers",
                    name="Uploaded data",
                    marker=dict(
                        color=(
                            uploaded_clean[color_column]
                            if use_colorbar
                            else uploaded_style["color"]
                        ),
                        size=uploaded_style["plot_size"],
                        symbol=uploaded_style["symbol"],
                        opacity=uploaded_style["opacity"],
                        line=dict(color="black", width=uploaded_style["outline_width"]),
                        **({"coloraxis": "coloraxis"} if use_colorbar else {}),
                    ),
                    text=uploaded_clean.get("Station"),
                ),
            )

    st.plotly_chart(
        fig,
        key="integrated_ts",
        **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
    )


def render_salinity_d18o(df, uploaded_df=None, uploaded_style=None):
    st.subheader("Salinity-d18O")
    clean = df.dropna(subset=["Salinity", "d18O"]).copy()
    if clean.empty:
        st.warning("No salinity-d18O data for the selected filters.")
        return

    color_options = _numeric_first_color_options(
        clean,
        ["Depth_m", "Temperature_degC", "d-excess", "dD", "Dataset"],
    )
    controls = st.columns(2)
    with controls[0]:
        color_column = st.selectbox(
            "Color",
            color_options,
            index=0,
            key="integrated_salinity_d18o_color",
        )
    with controls[1]:
        colormap_label = _select_plotly_colormap(
            color_column,
            key="integrated_salinity_d18o_colormap",
        )
    fig = px.scatter(
        clean,
        x="Salinity",
        y="d18O",
        color=color_column,
        hover_data=_available_hover_columns(clean),
        opacity=0.7,
        height=560,
        **_plotly_color_scale_args(clean, color_column, colormap_label),
    )
    fig.update_layout(
        xaxis_title="Salinity",
        yaxis_title="d18O (VSMOW)",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    if uploaded_df is not None and not uploaded_df.empty:
        uploaded_style = uploaded_style or render_uploaded_marker_style_controls()
        uploaded_clean = uploaded_df.dropna(subset=["Salinity", "d18O"]).copy()
        if not uploaded_clean.empty:
            use_colorbar = _can_use_current_colorbar(
                uploaded_style, uploaded_clean, color_column
            )
            if use_colorbar:
                _apply_shared_coloraxis_range(fig, clean, uploaded_clean, color_column)
            fig.add_trace(
                go.Scattergl(
                    x=uploaded_clean["Salinity"],
                    y=uploaded_clean["d18O"],
                    mode="markers",
                    name="Uploaded data outline",
                    marker=dict(
                        color="black",
                        size=uploaded_style["outline_plot_size"],
                        symbol=uploaded_style["symbol"],
                        opacity=min(1.0, uploaded_style["opacity"] + 0.05),
                    ),
                    text=uploaded_clean.get("Station"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
            )
            fig.add_trace(
                go.Scattergl(
                    x=uploaded_clean["Salinity"],
                    y=uploaded_clean["d18O"],
                    mode="markers",
                    name="Uploaded data",
                    marker=dict(
                        color=(
                            uploaded_clean[color_column]
                            if use_colorbar
                            else uploaded_style["color"]
                        ),
                        size=uploaded_style["plot_size"],
                        symbol=uploaded_style["symbol"],
                        opacity=uploaded_style["opacity"],
                        line=dict(color="black", width=uploaded_style["outline_width"]),
                        **({"coloraxis": "coloraxis"} if use_colorbar else {}),
                    ),
                    text=uploaded_clean.get("Station"),
                ),
            )

    st.plotly_chart(
        fig,
        key="integrated_salinity_d18o",
        **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
    )


def render_uploaded_quality_summary(uploaded_quality_df, download_key):
    st.subheader("Uploaded Data Quality Check")
    if uploaded_quality_df.empty:
        st.success("No quality-flagged rows in the uploaded dataset.")
        return

    st.warning(f"{len(uploaded_quality_df):,} uploaded rows have quality flags.")
    envgeo_utils.display_isotope_table(
        uploaded_quality_df,
        title="Quality-flagged uploaded dataset (CSV)",
    )
    st.download_button(
        "Download uploaded quality-flagged rows",
        data=uploaded_quality_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="envgeo_seawater_uploaded_quality_flags.csv",
        mime="text/csv",
        key=download_key,
    )


def render_uploaded_plots(uploaded_df, uploaded_style=None, uploaded_quality_df=None):
    st.subheader("Uploaded Data Plots")
    if uploaded_df.empty:
        st.info("Upload an Excel or CSV file to plot user data.")
        return
    uploaded_style = uploaded_style or {
        "plot_size": 12,
        "symbol": "diamond",
        "opacity": 0.95,
        "outline_width": 2,
    }

    envgeo_utils.display_isotope_table(uploaded_df, title="Uploaded dataset (CSV)")
    if uploaded_quality_df is None:
        uploaded_quality_df = _quality_rows(uploaded_df)
    render_uploaded_quality_summary(
        uploaded_quality_df,
        download_key="upload_tab_download_uploaded_quality_flags",
    )

    columns = list(uploaded_df.columns)
    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(uploaded_df[column])
    ]
    if len(numeric_columns) < 2:
        st.warning("At least two numeric columns are required for plotting.")
        return

    plot_mode = st.radio(
        "Plot type",
        ["2D scatter", "3D scatter"],
        horizontal=True,
        key="integrated_uploaded_plot_type",
    )

    if plot_mode == "2D scatter":
        controls = st.columns(5)
        with controls[0]:
            x_col = st.selectbox("X", numeric_columns, index=0, key="uploaded_2d_x")
        with controls[1]:
            y_col = st.selectbox("Y", numeric_columns, index=1, key="uploaded_2d_y")
        with controls[2]:
            color_col = st.selectbox("Color", numeric_columns, index=0, key="uploaded_2d_color")
        with controls[3]:
            colormap_label = _select_plotly_colormap(
                color_col,
                key="uploaded_2d_colormap",
            )
        with controls[4]:
            fixed_color = st.checkbox("Fixed marker color", value=False, key="uploaded_2d_fixed_color")

        scatter_args = dict(
            data_frame=uploaded_df,
            x=x_col,
            y=y_col,
            hover_data=_available_hover_columns(uploaded_df),
            height=560,
        )
        if fixed_color:
            fig = px.scatter(**scatter_args)
        else:
            fig = px.scatter(
                **scatter_args,
                color=color_col,
                color_continuous_scale=envgeo_utils.get_plotly_colormap(
                    color_col,
                    colormap_label,
                ),
            )
        marker_style = dict(
            size=uploaded_style["plot_size"],
            symbol=uploaded_style["symbol"],
            opacity=uploaded_style["opacity"],
            line=dict(color="black", width=uploaded_style["outline_width"]),
        )
        if fixed_color:
            marker_style["color"] = uploaded_style["color"]
        fig.update_traces(marker=marker_style)
        st.plotly_chart(
            fig,
            key="integrated_uploaded_2d",
            **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
        )
        return

    if len(numeric_columns) < 3:
        st.warning("At least three numeric columns are required for 3D plotting.")
        return

    controls = st.columns(6)
    with controls[0]:
        x_col = st.selectbox("X", numeric_columns, index=0, key="uploaded_3d_x")
    with controls[1]:
        y_col = st.selectbox("Y", numeric_columns, index=1, key="uploaded_3d_y")
    with controls[2]:
        z_col = st.selectbox("Z", numeric_columns, index=2, key="uploaded_3d_z")
    with controls[3]:
        color_col = st.selectbox("Color", numeric_columns, index=0, key="uploaded_3d_color")
    with controls[4]:
        colormap_label = _select_plotly_colormap(
            color_col,
            key="uploaded_3d_colormap",
        )
    with controls[5]:
        fixed_color = st.checkbox("Fixed marker color", value=False, key="uploaded_3d_fixed_color")

    scatter_3d_args = dict(
        data_frame=uploaded_df,
        x=x_col,
        y=y_col,
        z=z_col,
        hover_data=_available_hover_columns(uploaded_df),
        height=640,
    )
    if fixed_color:
        fig = px.scatter_3d(**scatter_3d_args)
    else:
        fig = px.scatter_3d(
            **scatter_3d_args,
            color=color_col,
            color_continuous_scale=envgeo_utils.get_plotly_colormap(
                color_col,
                colormap_label,
            ),
        )

    reverse_z = st.checkbox("Reverse Z axis", value=z_col == "Depth_m")
    marker_style = dict(
        size=uploaded_style["plot_size"],
        opacity=uploaded_style["opacity"],
        line=dict(color="black", width=uploaded_style["outline_width"]),
    )
    if fixed_color:
        marker_style["color"] = uploaded_style["color"]
    fig.update_traces(marker=marker_style)
    if reverse_z:
        fig.update_layout(scene=dict(zaxis=dict(autorange="reversed")))
    st.plotly_chart(
        fig,
        key="integrated_uploaded_3d",
        **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
    )


def render_quality_table(quality_df, uploaded_quality_df=None):
    st.subheader("Quality Flags")
    envgeo_utils.render_quality_flag_criteria_note()
    if uploaded_quality_df is None:
        uploaded_quality_df = pd.DataFrame()
    if quality_df.empty and uploaded_quality_df.empty:
        st.success("No quality-flagged rows in the selected filters or uploaded dataset.")
        return

    if quality_df.empty:
        st.success("No quality-flagged rows in the selected reference filters.")
    else:
        envgeo_utils.display_isotope_table(
            quality_df,
            title="Quality-flagged reference dataset (CSV)",
        )
        st.download_button(
            "Download reference quality-flagged rows",
            data=quality_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="envgeo_seawater_reference_quality_flags.csv",
            mime="text/csv",
            key="quality_tab_download_reference_quality_flags",
        )

    render_uploaded_quality_summary(
        uploaded_quality_df,
        download_key="quality_tab_download_uploaded_quality_flags",
    )


def main():
    st.header(f"Integrated Visualizer beta ({version})")
    st.caption(f"Shared app version: {envgeo_utils.APP_VERSION_LABEL}")

    workflow_mode = st.radio(
        "Workflow mode",
        ["Full existing page", "Shared-filter beta"],
        horizontal=True,
        key="integrated_workflow_mode",
    )

    uploaded_df = render_upload_panel()

    if workflow_mode == "Full existing page":
        render_full_existing_page(uploaded_df)
        return

    uploaded_style = render_uploaded_marker_style_controls()

    st.button("Reload")

    data_source_JAPAN_SEA = envgeo_utils.data_source_JAPAN_SEA
    data_source_AROUND_JAPAN = envgeo_utils.data_source_AROUND_JAPAN
    data_source_GLOBAL = envgeo_utils.data_source_GLOBAL

    ref_data = st.radio(
        "Data source (see Home > About):",
        (data_source_JAPAN_SEA, data_source_AROUND_JAPAN, data_source_GLOBAL),
        horizontal=True,
        key="integrated_data_source",
    )

    if ref_data == data_source_JAPAN_SEA:
        st.write(envgeo_utils.refs_JAPAN_SEA)
    elif ref_data == data_source_AROUND_JAPAN:
        st.write(envgeo_utils.refs_AROUND_JAPAN)
    elif ref_data == data_source_GLOBAL:
        st.write(envgeo_utils.refs_GLOBAL)

    df = envgeo_utils.load_isotope_data(ref_data)
    if df.empty:
        st.warning("No data available for the selected source.")
        return

    (
        filtered_df,
        sld_year_min,
        sld_year_max,
        selected_months,
        sld_lon_min,
        sld_lon_max,
        sld_lat_min,
        sld_lat_max,
        sld_depth_min,
        sld_depth_max,
        sld_sal_min,
        sld_sal_max,
        sld_d18O_min,
        sld_d18O_max,
        sld_temp_min,
        sld_temp_max,
        selected_cruise,
        submitted,
    ) = envgeo_utils.sidebar_filter_and_display(
        df,
        ref_data,
        data_source_JAPAN_SEA,
        data_source_AROUND_JAPAN,
    )

    quality_df = _quality_rows(filtered_df)
    uploaded_quality_df = _quality_rows(uploaded_df)

    render_tab_style()
    envgeo_utils.render_earthquake_tab_style()
    st.caption("Select a tab to switch visualization views.")
    tab_summary, tab_map, tab_ts, tab_sal_d18o, tab_upload, tab_data, tab_quality = st.tabs(
        [
            "📌 Summary",
            "🗺️ Map",
            "🌡️ T-S",
            "💧 Salinity-d18O",
            "📤 Upload",
            "🗂️ Data(CSV)",
            "✅ Quality Flags",
        ]
    )

    with tab_summary:
        render_summary(filtered_df, quality_df, uploaded_quality_df)

    with tab_map:
        render_map(filtered_df, uploaded_df, uploaded_style)

    with tab_ts:
        render_ts_diagram(filtered_df, uploaded_df, uploaded_style)

    with tab_sal_d18o:
        render_salinity_d18o(filtered_df, uploaded_df, uploaded_style)

    with tab_upload:
        render_uploaded_plots(uploaded_df, uploaded_style, uploaded_quality_df)

    with tab_data:
        envgeo_utils.display_isotope_table(filtered_df)

    with tab_quality:
        render_quality_table(quality_df, uploaded_quality_df)


if __name__ == "__main__":
    main()
