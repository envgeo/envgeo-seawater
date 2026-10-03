#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive 2D/2.5D visualizer for EnvGeo-Seawater data.

EnvGeo-Seawater データの対話型2D/2.5D可視化ページです。

Created: 2023-05-21
Author: Toyoho Ishimura, Kyoto University
Last reviewed: 2026-09-30
"""

import math

import gsw  # Gibbs SeaWater / TEOS-10 for approximate σ0 reference contours only. / 近似σ0参照等値線だけに用いる。
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

import envgeo_utils

# =============================================================================
# Page configuration and plot parameters / ページ設定と描画パラメータ
# =============================================================================
version = "1.3.4"

COLOR_FILTERED_OPTIONS = {
    "d18O": "d18O",
    "dD": "dD",
    "d-excess": "d-excess",
    "Temperature": "Temperature_degC",
    "Salinity": "Salinity",
    "Water Depth": "Depth_m",
    "Latitude": "Latitude_degN",
    "Longitude": "Longitude_degE",
}

PLOT_PARAMETER_LABELS = {
    "d18O": "δ18O (‰)",
    "dD": "δD (‰)",
    "d-excess": "d-excess (‰)",
    "Temperature_degC": "Temperature (C)",
    "Salinity": "Salinity",
    "Depth_m": "Depth (m)",
    "Latitude_degN": "Latitude (degN)",
    "Longitude_degE": "Longitude (degE)",
    "Year": "Year",
    "Month": "Month",
}

CUSTOM_PLOT_PARAMETER_ORDER = [
    "d18O",
    "dD",
    "d-excess",
    "Salinity",
    "Temperature_degC",
    "Depth_m",
    "Latitude_degN",
    "Longitude_degE",
    "Year",
    "Month",
]

def available_color_filtered_options(df):
    """Return color options available in the current dataframe.

    現在のデータに存在する列だけを、色分け候補として表示します。
    """
    return {
        label: column
        for label, column in COLOR_FILTERED_OPTIONS.items()
        if column in df.columns
    }

def render_color_controls(df, key_prefix):
    """Render color-column and colormap controls side by side.

    色分けする列とカラーマップを横並びで選択します。
    """
    col_map = available_color_filtered_options(df)
    control_col, colormap_col = st.columns([1, 1])
    with control_col:
        selected_label = st.selectbox(
            "Color parameter",
            list(col_map.keys()),
            key=f"{key_prefix}_color_filtered",
            help=getattr(
                envgeo_utils,
                "COLOR_PARAMETER_HELP_TEXT",
                "Choose the variable used to color the plotted points or map markers.",
            ),
        )
    target_column = col_map[selected_label]
    colormap_options = envgeo_utils.get_plotly_colormap_options(target_column)
    colormap_labels = list(colormap_options.keys())
    default_index = (
        colormap_labels.index("EnvGeo variable default")
        if "EnvGeo variable default" in colormap_labels
        else 0
    )
    with colormap_col:
        selected_colormap_label = st.selectbox(
            "Colormap",
            colormap_labels,
            index=default_index,
            key=f"{key_prefix}_colormap",
            help=(
                "Choose the color palette used for the filtered plot and matching map. "
                "cmocean palettes are designed for oceanographic data."
            ),
        )
    colorscale = envgeo_utils.get_plotly_colormap(target_column, selected_colormap_label)
    return selected_label, target_column, colorscale

def available_numeric_plot_columns(df):
    """Return numeric columns suitable for custom Plotly scatter plots.

    カスタムPlotly散布図に使える数値列を返します。
    """
    options = []
    for column in CUSTOM_PLOT_PARAMETER_ORDER:
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any():
            options.append(column)
    return options

def plot_label(column):
    """Return a compact display label for a plot column.

    描画列に対応する簡潔な表示名を返します。
    """
    return PLOT_PARAMETER_LABELS.get(column, column)

def add_regression_line(fig, df, x_col, y_col, line_name="Regression line"):
    """Add a simple least-squares regression line to a Plotly scatter figure.

    試験的な近似直線を追加します。点数不足や同じX値のみの場合は何もしません。
    """
    regression_df = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(regression_df) < 2 or regression_df[x_col].nunique() < 2:
        return None

    slope, intercept = np.polyfit(regression_df[x_col], regression_df[y_col], 1)
    x_min = float(regression_df[x_col].min())
    x_max = float(regression_df[x_col].max())
    x_vals = np.array([x_min, x_max])
    y_vals = slope * x_vals + intercept
    r_value = regression_df[x_col].corr(regression_df[y_col])
    stats_text = f"y = {slope:.3g}x + {intercept:.3g} | R = {r_value:.2f}"
    fig.add_scatter(
        x=x_vals,
        y=y_vals,
        mode="lines",
        line=dict(color="black", width=2),
        name=f"{line_name}: {stats_text}",
        hoverinfo="name",
    )
    return stats_text

def render_regression_stats(stats_text):
    """Render compact regression statistics beside the control.

    操作欄の横に簡潔な回帰統計量を表示します。
    """
    if not stats_text:
        st.caption("Regression unavailable")
        return

    st.markdown(
        f"""
        <div style="
            margin-top: 0.18rem;
            padding: 0.38rem 0.55rem;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            background: #f9fafb;
            color: #374151;
            font-size: 0.82rem;
            line-height: 1.35;
            white-space: nowrap;
        ">
            <strong>Regression</strong>&nbsp;&nbsp;{stats_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def selected_point_indices(selected_points, max_len, scatter_curve_number=0):
    """Return selected indices from the main scatter trace only.

    近似直線などの追加traceが混ざっても、元の散布点だけを地図連動に使います。
    ``scatter_curve_number`` is the Plotly curve number of the scatter trace;
    use 1 when a background trace such as a σ0 contour is prepended.

    ``scatter_curve_number`` は散布図traceのPlotly curve番号です。σ0等値線
    のような背景traceを散布図の前に追加する場合は1を渡します。
    """
    indices = []
    for point in selected_points or []:
        if point.get("curveNumber", 0) != scatter_curve_number:
            continue
        point_index = point.get("pointIndex")
        if point_index is None:
            continue
        if 0 <= point_index < max_len:
            indices.append(point_index)
    return indices

def show_selection_tip():
    """Show a compact guide below Plotly selection figures.

    Box/Lasso 選択の説明を、図の直下に見やすく表示します。
    """
    st.markdown(
        """
        <div style="
            margin: 0.25rem 0 0.8rem 0;
            padding: 0.55rem 0.75rem;
            border-left: 4px solid #3b82f6;
            background: #eef6ff;
            color: #1f2937;
            font-size: 0.92rem;
            line-height: 1.45;
        ">
            <strong>Tip:</strong> Use <strong>Box Select</strong> or <strong>Lasso Select</strong>
            to highlight matching sampling locations on the map.
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    # -------------------------------------------------------------------------
    # Page introduction / ページの概要
    # -------------------------------------------------------------------------
    st.header(f'Interactive 2Dplus Visualizer ({version})')
    st.caption(
        "Use this Plotly page for interactive data exploration. "
        "For publication- or presentation-ready static figures, use the corresponding individual pages."
    )
    

    # Reload control / 再読込操作
    st.button('Reload')
    
 

    # Shared data-source labels / 共通データソース表示名
    data_source_JAPAN_SEA = envgeo_utils.data_source_JAPAN_SEA
    data_source_AROUND_JAPAN = envgeo_utils.data_source_AROUND_JAPAN
    data_source_GLOBAL = envgeo_utils.data_source_GLOBAL
    

    # Reference-data selection / 参照データ選択
    ref_data = st.radio("Data source (see Home > About)", (data_source_JAPAN_SEA, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True)

    # Citation display / 引用表示
    
    if ref_data == data_source_JAPAN_SEA:
        st.write(envgeo_utils.refs_JAPAN_SEA)
        
    elif ref_data == data_source_AROUND_JAPAN:
        st.write(envgeo_utils.refs_AROUND_JAPAN)

        
    elif ref_data == data_source_GLOBAL:
        st.write(envgeo_utils.refs_GLOBAL)
        
    else:
        st.warning("Invalid data source selection.")

    # -------------------------------------------------------------------------
    # Reference-data loading / 参照データの読込
    # -------------------------------------------------------------------------
    df1 = envgeo_utils.load_isotope_data(ref_data)
   
    if df1.empty:
        st.warning("No data available for the selected conditions.")
        return

    

    # -------------------------------------------------------------------------
    # Shared sidebar filtering / 共通sidebarによる絞り込み
    # -------------------------------------------------------------------------
    
    # Apply the shared controls and receive the filtered data and selections.
    # 共通コントロールを適用し、絞り込み後のデータと選択値を受け取る。
    (df1,
     sld_year_min, sld_year_max,
     selected_months,
     sld_lon_min, sld_lon_max,
     sld_lat_min, sld_lat_max,
     sld_depth_min, sld_depth_max,
     sld_sal_min, sld_sal_max,
     sld_d18O_min, sld_d18O_max,
     sld_temp_min, sld_temp_max,
     selected_cruise,
     submitted) = envgeo_utils.sidebar_filter_and_display(df1, ref_data, data_source_JAPAN_SEA, data_source_AROUND_JAPAN)

    # -------------------------------------------------------------------------
    # Plotly coordinate aliases / Plotly用の位置列alias
    # -------------------------------------------------------------------------
        
    df1['lat'] = df1['Latitude_degN']
    df1['lon'] = df1['Longitude_degE']

    # =============================================================================
    # Plot-layout selection / 描画レイアウトの選択
    # =============================================================================

    fig_type_d18Osal = "d18O-Salinity relationship"
    fig_type_dD_d18O = "dD-δ18O relationship"
    fig_type_TS = "Temperature–Salinity (T–S) diagram"
    fig_type_custom_xy = "Custom 2D/2.5D plot beta"

    plot_figure = st.radio(
        "Plot type",
        (fig_type_d18Osal, fig_type_dD_d18O, fig_type_TS, fig_type_custom_xy),
        horizontal=True,
        help="Choose the interactive Plotly view to display.",
    )
    

    # -------------------------------------------------------------------------
    # Cache control / キャッシュ制御
    # -------------------------------------------------------------------------
    if st.sidebar.button("🔄 Clear cache"):
        envgeo_utils.clear_app_cache()
        st.rerun()
    
    
    
    
        
    

    # -------------------------------------------------------------------------
    # Shared figure styling / 共通図スタイル
    # -------------------------------------------------------------------------
    def unify_plot_layout(fig, x_label, y_label, color_title):
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            
            # Keep the desktop height while allowing the width to follow its container.
            # PCの高さは維持し、横幅は表示領域に合わせる。
            height=600,
            autosize=True,
            
            xaxis=dict(
                title=x_label,
                showline=True, linewidth=1, linecolor='grey', mirror=True,
                showgrid=True, gridcolor='rgba(220, 220, 220, 0.4)', gridwidth=0.5,
                ticks="inside", ticklen=5
            ),
            yaxis=dict(
                title=y_label,
                showline=True, linewidth=1, linecolor='grey', mirror=True,
                showgrid=True, gridcolor='rgba(220, 220, 220, 0.4)', gridwidth=0.5,
                ticks="inside", ticklen=5
            ),
            
            # Keep the colorbar just outside the plotting area.
            # カラーバーを描画領域のすぐ外側に固定する。
            coloraxis_colorbar=dict(
                title=color_title,
                thickness=15,
                len=0.8,
                x=1.02,           # グラフ枠のすぐ右外側に固定（1.0が枠の右端）
                xanchor='left',   # 左端を基準に配置
                y=0.5,
                yanchor='middle'
            ),
            
            # Keep room for labels and the colorbar on narrow screens.
            # 狭い画面でもラベルとカラーバーの余白を確保する。
            margin=dict(l=60, r=90, t=50, b=70),
            
            font=dict(size=12)
        )
        
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        
        return fig
    

    # -------------------------------------------------------------------------
    # Linked scatter-and-map helper / 連動散布図・地図の補助関数
    # -------------------------------------------------------------------------
    st.divider()

    def render_linked_xy_plot(
        x_col,
        y_col,
        plot_title,
        key_prefix,
        allow_regression=True,
        allow_single_color=False,
    ):
        """Render a selectable 2D Plotly scatter plot and linked sampling map.

        選択可能な2D Plotly散布図と連動する採水地点地図を描画します。
        """
        if x_col == y_col:
            st.warning("Please choose different X and Y parameters.")
            return

        # ---------------------------------------------------------------------
        # Plot controls / 描画コントロール
        # ---------------------------------------------------------------------
        color_col = None
        colorscale = None
        selected_label = "Single color"
        if allow_single_color:
            available_colors = available_color_filtered_options(df1)
            color_options = ["Single color"] + list(available_colors.keys())
            default_color_choice = (
                "Water Depth"
                if "Water Depth" in available_colors
                else next(iter(available_colors), "Single color")
            )
            color_control_col, colormap_control_col = st.columns([1, 1])
            with color_control_col:
                color_choice = st.selectbox(
                    "Color parameter",
                    color_options,
                    index=color_options.index(default_color_choice),
                    key=f"{key_prefix}_color_mode",
                    help=(
                        "Use a fixed marker color for a simple 2D plot, or choose "
                        "a numeric parameter for a 2.5D colorbar."
                    ),
                )
            if color_choice != "Single color":
                color_col = available_colors[color_choice]
                selected_label = color_choice
                colormap_options = envgeo_utils.get_plotly_colormap_options(color_col)
                colormap_labels = list(colormap_options.keys())
                default_index = (
                    colormap_labels.index("EnvGeo variable default")
                    if "EnvGeo variable default" in colormap_labels
                    else 0
                )
                with colormap_control_col:
                    selected_colormap_label = st.selectbox(
                        "Colormap",
                        colormap_labels,
                        index=default_index,
                        key=f"{key_prefix}_colormap",
                        help=(
                            "Choose the color palette used for the filtered plot "
                            "and matching map."
                        ),
                    )
                colorscale = envgeo_utils.get_plotly_colormap(color_col, selected_colormap_label)
            else:
                with colormap_control_col:
                    st.caption("Colormap is used when a color parameter is selected.")
        else:
            selected_label, color_col, colorscale = render_color_controls(df1, key_prefix)

        regression_stats_text = ""
        if allow_regression:
            regression_control_col, regression_stats_col = st.columns([0.9, 2.1])
            with regression_control_col:
                show_regression = st.checkbox(
                    "Regression line",
                    value=False,
                    key=f"{key_prefix}_regression_line",
                    help=getattr(
                        envgeo_utils,
                        "REGRESSION_HELP_TEXT",
                        "Add a simple least-squares regression line for quick visual reference.",
                    ),
                )
        else:
            show_regression = False
            regression_stats_col = None

        # ---------------------------------------------------------------------
        # Plot-data preparation / 描画データの準備
        # ---------------------------------------------------------------------
        required_columns = [x_col, y_col]
        if color_col is not None:
            required_columns.append(color_col)
        df_plot = df1.dropna(subset=required_columns).reset_index(drop=True)
        excluded_count = len(df1) - len(df_plot)
        if excluded_count > 0:
            st.caption(
                f":red[{excluded_count:,} rows excluded because selected X, Y, "
                "or color values were missing.]"
            )
        if df_plot.empty:
            st.warning("No valid rows remain for the selected plot settings.")
            return

        # ---------------------------------------------------------------------
        # Scatter plot / 散布図
        # ---------------------------------------------------------------------
        scatter_kwargs = {
            "data_frame": df_plot,
            "x": x_col,
            "y": y_col,
            "hover_data": {
                column: True
                for column in [
                    "d18O",
                    "dD",
                    "d-excess",
                    "Salinity",
                    "Temperature_degC",
                    "Depth_m",
                    "Latitude_degN",
                    "Longitude_degE",
                    "Year",
                    "Month",
                    "Day",
                    "Cruise",
                    "Station",
                    "reference",
                ]
                if column in df_plot.columns
            },
        }
        if color_col is not None:
            scatter_kwargs.update(color=color_col, color_continuous_scale=colorscale)
        else:
            scatter_kwargs.update(color_discrete_sequence=["#2563eb"])
        fig_xy = px.scatter(**scatter_kwargs)

        if show_regression:
            regression_stats_text = add_regression_line(fig_xy, df_plot, x_col, y_col)
            with regression_stats_col:
                render_regression_stats(regression_stats_text)

        fig_xy = unify_plot_layout(
            fig_xy,
            plot_label(x_col),
            plot_label(y_col),
            selected_label,
        )
        fig_xy.update_layout(
            hovermode="closest",
            hoverdistance=5,
            margin=dict(l=60, r=90, t=50, b=70, autoexpand=False),
            coloraxis_colorbar=dict(x=1.02, xanchor="left", len=0.8),
            xaxis=dict(
                zeroline=False,
                zerolinewidth=1,
                zerolinecolor="grey",
                showline=True,
                linewidth=1,
                linecolor="grey",
                mirror=True,
            ),
            yaxis=dict(
                zeroline=False,
                zerolinewidth=1,
                zerolinecolor="grey",
                showline=True,
                linewidth=1,
                linecolor="grey",
                mirror=True,
            ),
        )

        with envgeo_utils.bounded_container(850):
            selected_points = plotly_events(
                fig_xy,
                select_event=True,
                key=f"{key_prefix}_event",
                override_height=600,
                override_width="100%",
            )
        st.download_button(
            "Download interactive HTML",
            envgeo_utils.figure_to_self_contained_html(fig_xy),
            envgeo_utils.build_figure_filename(f"p03_{key_prefix}_xy", extension="html"),
            "text/html",
            key=f"{key_prefix}_xy_html_dl",
        )
        show_selection_tip()

        # ---------------------------------------------------------------------
        # Linked map / 連動地図
        # ---------------------------------------------------------------------
        selected_indices_key = f"{key_prefix}_selected_indices"
        selected_indices = selected_point_indices(selected_points, len(df_plot))
        if selected_indices:
            st.session_state[selected_indices_key] = selected_indices
            st.write(f"**Selected points: {len(selected_indices)}**")
        else:
            st.session_state[selected_indices_key] = []

        is_selected = len(st.session_state[selected_indices_key]) > 0
        df_map = df_plot.iloc[st.session_state[selected_indices_key]] if is_selected else df_plot

        lat_min, lat_max = df_map["Latitude_degN"].min(), df_map["Latitude_degN"].max()
        lon_min, lon_max = df_map["Longitude_degE"].min(), df_map["Longitude_degE"].max()
        center_lat = df_map["Latitude_degN"].mean()
        center_lon = df_map["Longitude_degE"].mean()
        lat_diff = max(lat_max - lat_min, 0.1)
        lon_diff = max(lon_max - lon_min, 0.1)
        zoom_lon = math.log2((850 * 360) / (lon_diff * 256))
        zoom_lat = math.log2((600 * 180) / (lat_diff * 256))
        auto_zoom = min(zoom_lon, zoom_lat) - (0.8 if is_selected else 1.5)
        auto_zoom = max(1, min(15, auto_zoom))

        map_kwargs = {
            "data_frame": df_map,
            "lat": "Latitude_degN",
            "lon": "Longitude_degE",
            "mapbox_style": "open-street-map",
            "hover_data": [
                column for column in [
                    "d18O",
                    "dD",
                    "d-excess",
                    "Salinity",
                    "Temperature_degC",
                    "Year",
                    "Month",
                    "Day",
                    "Cruise",
                    "Station",
                    "Depth_m",
                    "reference",
                ] if column in df_map.columns
            ],
        }
        if color_col is not None:
            map_kwargs.update(color=color_col, color_continuous_scale=colorscale)
        else:
            map_kwargs.update(color_discrete_sequence=["#2563eb"])
        fig_map = px.scatter_mapbox(**map_kwargs)
        fig_map = unify_plot_layout(fig_map, "Lon", "Lat", selected_label)
        fig_map.update_layout(
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            height=500,
            coloraxis_colorbar=dict(
                title=selected_label,
                x=0.98,
                xanchor="right",
                y=0.5,
                yanchor="middle",
                len=0.8,
                thickness=15,
            ),
        )

        with st.popover(
            "Map controls", **envgeo_utils.stretch_width_kwargs(st.popover)
        ):
            map_mode = st.radio(
                "Map style",
                envgeo_utils.MAP_MODE_OPTIONS,
                index=envgeo_utils.MAP_MODE_DEFAULT_INDEX,
                horizontal=True,
                key=f"{key_prefix}_map_style",
                help=getattr(
                    envgeo_utils,
                    "MAP_STYLE_HELP_TEXT",
                    "Choose the background map style for the sampling-location map.",
                ),
            )
        st.caption(f"Map style: {map_mode}")
        fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)

        st.plotly_chart(
            fig_map,
            key=f"{key_prefix}_map",
            config={"scrollZoom": True, "displayModeBar": True},
            **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
        )
        st.download_button(
            "Download interactive HTML",
            envgeo_utils.figure_to_self_contained_html(fig_map),
            envgeo_utils.build_figure_filename("p03_map", extension="html"),
            "text/html",
            key=f"{key_prefix}_map_html_dl",
        )

        envgeo_utils.display_isotope_table(df1)
        envgeo_utils.display_isotope_table(df_map, title="Box/Lasso-selected dataset (CSV)")
    
    
    # =============================================================================
    # Temperature–Salinity layout / 水温–塩分レイアウト（T–S図）
    # =============================================================================
    
    if plot_figure == fig_type_TS:
    

        # ---------------------------------------------------------------------
        # Selection state / 選択状態
        # ---------------------------------------------------------------------
        if 'ts_selected_indices' not in st.session_state:
            st.session_state.ts_selected_indices = []
        
        st.subheader('Temperature-Salinity Diagram')

        sel_col, target_item, c_scale_final = render_color_controls(df1, "fig_TS_zoom")

        # Density-contour controls / 密度等値線の操作
        ts_contour_interval = st.selectbox(
            "Density contour interval (approx. σ0)",
            options=[0.2, 0.5, 1.0],
            index=2,  # default 1.0 kg m⁻³
            format_func=lambda v: f"{v} kg m⁻³",
            help=(
                "Spacing between approximate σ0 reference contour lines. "
                "These are visual reference guides, not pointwise sample density."
            ),
            key="p03_ts_contour_interval",
        )
        show_ts_density_contours = st.checkbox(
            "Show density contours",
            value=True,
            help=(
                "Show or hide the approximate σ0 reference contours. "
                "They are visual reference guides, not pointwise sample density."
            ),
            key="p03_ts_show_density_contours",
        )

        # ---------------------------------------------------------------------
        # T–S scatter plot / T–S散布図
        # ---------------------------------------------------------------------
        # Keep rows that contain the selected color variable and both T–S axes.
        # 選択した色変数とT–S両軸を持つ行だけを描画対象にする。
        df_plot_ts = df1.dropna(subset=[target_item, "Salinity", "Temperature_degC"]).reset_index(drop=True)
    
    
        # Report rows excluded from this view / この表示から除外した行数を示す。
        excluded_count = len(df1) - len(df_plot_ts)
        if excluded_count > 0:
            st.caption(f":red[{excluded_count:,} rows excluded because {sel_col}, temperature, or salinity was missing.]")
    
        fig_fixed_TS = px.scatter(
            df_plot_ts,
            x="Salinity", y="Temperature_degC", 
            color=target_item, color_continuous_scale=c_scale_final,
            hover_data={
                "Salinity": True,
                "Temperature_degC": True,
                "lat": True,
                "lon": True,
                "d18O": True,
                "dD": True,
                "Year": True,
                "Month": True,
                "Day": True,
                "Cruise": True,
                "Station": True,
                "Depth_m": True,
                "reference": True
            }
        )
        
        
        
        # Apply shared figure styling / 共通図スタイルを適用する。
        fig_fixed_TS = unify_plot_layout(fig_fixed_TS, "Salinity", "Temperature (C)", sel_col)
        
    
        
        # Fix the interactive viewing area / 操作時の表示領域を固定する。
        fig_fixed_TS.update_layout(
            hovermode='closest',
            hoverdistance=5,
            height=600, 
            margin=dict(l=60, r=90, t=50, b=70, autoexpand=False),
            coloraxis_colorbar=dict(
                x=1.02, 
                xanchor='left',
                len=0.8,
                thickness=15,
            ),
            xaxis=dict(
                zeroline=False, zerolinewidth=1, zerolinecolor='grey',
                range=[df_plot_ts["Salinity"].min()*0.95, df_plot_ts["Salinity"].max()*1.05]
            ),
            yaxis=dict(
                zeroline=False, zerolinewidth=1, zerolinecolor='grey',
                range=[df_plot_ts["Temperature_degC"].min()-1, df_plot_ts["Temperature_degC"].max()+1]
            )
        )

        # ---------------------------------------------------------------------
        # Approximate σ0 reference contours / 近似σ0参照等値線
        # ---------------------------------------------------------------------
        # Clip the displayed grid to the GSW input domain; negative salinity
        # must never be passed to gsw.sigma0.
        # 表示範囲をGSWの入力範囲に制限し、負の塩分をgsw.sigma0へ渡さない。
        _T_GSW_MIN_03, _T_GSW_MAX_03 = -5.0, 45.0
        _S_GSW_MIN_03, _S_GSW_MAX_03 = 0.0, 50.0

        # Displayed axis limits / 表示軸の範囲
        _sal_ax_lo  = float(df_plot_ts["Salinity"].min()) * 0.95
        _sal_ax_hi  = float(df_plot_ts["Salinity"].max()) * 1.05
        _temp_ax_lo = float(df_plot_ts["Temperature_degC"].min()) - 1.0
        _temp_ax_hi = float(df_plot_ts["Temperature_degC"].max()) + 1.0

        # Clip to the GSW input domain / GSW入力範囲に制限する。
        _sal_lo  = max(_sal_ax_lo,  _S_GSW_MIN_03)
        _sal_hi  = min(_sal_ax_hi,  _S_GSW_MAX_03)
        _temp_lo = max(_temp_ax_lo, _T_GSW_MIN_03)
        _temp_hi = min(_temp_ax_hi, _T_GSW_MAX_03)

        _ts_scatter_curve = 0  # fallback: no contour prepended
        if show_ts_density_contours and _sal_lo < _sal_hi and _temp_lo < _temp_hi:
            _sal_grid  = np.linspace(_sal_lo,  _sal_hi,  100)
            _temp_grid = np.linspace(_temp_lo, _temp_hi, 100)
            _Sg, _Tg = np.meshgrid(_sal_grid, _temp_grid)
            # This reference uses Practical Salinity ≈ Absolute Salinity and
            # in-situ temperature ≈ Conservative Temperature; it is approximate only.
            # ここでは実用塩分≒絶対塩分、現場水温≒保存温度として扱う近似参照である。
            _sigma0_approx = gsw.sigma0(_Sg, _Tg)
            _s0_min = float(np.nanmin(_sigma0_approx))
            _s0_max = float(np.nanmax(_sigma0_approx))
            if np.isfinite(_s0_min) and np.isfinite(_s0_max) and _s0_max > _s0_min:
                _contour_start = float(
                    np.ceil(_s0_min / ts_contour_interval) * ts_contour_interval
                )
                _contour_end = float(
                    np.floor(_s0_max / ts_contour_interval) * ts_contour_interval
                )
                if _contour_start <= _contour_end:
                    _contour_trace = go.Contour(
                        x=_sal_grid,
                        y=_temp_grid,
                        z=_sigma0_approx,
                        contours=dict(
                            coloring="lines",
                            start=_contour_start,
                            end=_contour_end,
                            size=ts_contour_interval,
                            showlabels=True,
                            labelfont=dict(size=9, color="rgba(100, 100, 100, 0.70)"),
                        ),
                        # Plotly derives line colors from colorscale in line-only mode.
                        # Plotlyの線のみモードでは、線色はcolorscaleから決まる。
                        colorscale=[
                            [0.0, "rgba(165, 165, 165, 0.22)"],
                            [1.0, "rgba(165, 165, 165, 0.22)"],
                        ],
                        autocolorscale=False,
                        line=dict(width=1),
                        showscale=False,
                        hoverinfo="none",
                        showlegend=False,
                        name="",
                    )
                    # Add the contour, then reorder the complete trace set so it
                    # remains behind points. The scatter becomes curve number 1.
                    # 等値線を追加後、点の背面に来るよう全traceを並べ替える。散布図はcurve 1となる。
                    fig_fixed_TS.add_trace(_contour_trace)
                    fig_fixed_TS.data = (fig_fixed_TS.data[-1],) + fig_fixed_TS.data[:-1]
                    _ts_scatter_curve = 1
        # ---------------------------------------------------------------------
        # Interactive selection and linked map / 選択操作と連動地図
        # ---------------------------------------------------------------------
        with envgeo_utils.bounded_container(850):
            selected_points = plotly_events(
                fig_fixed_TS,
                select_event=True,
                key="ts_zoom_event",
                override_height=600,
                override_width="100%",
            )
        st.caption(
            "Density contour lines are approximate σ0 reference contours "
            "(Practical Salinity ≈ Absolute Salinity; "
            "in-situ temperature ≈ Conservative Temperature). "
            "Not pointwise sample density. Visual reference only."
        )
        st.download_button(
            "Download interactive HTML",
            envgeo_utils.figure_to_self_contained_html(fig_fixed_TS),
            envgeo_utils.build_figure_filename("p03_TS", extension="html"),
            "text/html",
            key="p03_TS_html_dl",
        )
        show_selection_tip()

        # Use the current scatter curve so Box/Lasso selection addresses the
        # scatter trace whether or not a contour was prepended.
        # 等値線の有無にかかわらず、Box/Lasso選択が散布図traceを参照するようにする。
        selected_indices = selected_point_indices(
            selected_points, len(df_plot_ts), scatter_curve_number=_ts_scatter_curve
        )
        if selected_indices:
            st.session_state.ts_selected_indices = selected_indices
            num_selected = len(selected_indices)
            st.write(f"**Selected points: {num_selected}**")
        else:
            st.session_state.ts_selected_indices = []

        # Map data and view extent / 地図データと表示範囲
        is_selected = len(st.session_state.ts_selected_indices) > 0
    
        if is_selected:
            df_ts_map_display = df_plot_ts.iloc[st.session_state.ts_selected_indices]
        else:
            df_ts_map_display = df_plot_ts
        lat_min, lat_max = df_ts_map_display["Latitude_degN"].min(), df_ts_map_display["Latitude_degN"].max()
        lon_min, lon_max = df_ts_map_display["Longitude_degE"].min(), df_ts_map_display["Longitude_degE"].max()
        
        
        lat_diff = lat_max - lat_min if lat_max != lat_min else 0.5
        lon_diff = lon_max - lon_min if lon_max != lon_min else 0.5
        
        # Map center / 地図中心
        center_lat = df_ts_map_display['Latitude_degN'].mean()
        center_lon = df_ts_map_display['Longitude_degE'].mean()
        
        
        # Calculate zoom from data extent / データ範囲からズームを計算する。
        map_width_px = 850
        map_height_px = 600
        zoom_lon = math.log2((map_width_px * 360) / (lon_diff * 256))
        zoom_lat = math.log2((map_height_px * 180) / (lat_diff * 256))
        
        # Leave a wider view before selection / 未選択時は少し広く表示する。
        auto_zoom = min(zoom_lon, zoom_lat) - (0.8 if is_selected else 1.5)
        auto_zoom = max(1, min(15, auto_zoom))
    
        # Render the linked map / 連動地図を描画する。
        fig_ts_map = px.scatter_mapbox(
            df_ts_map_display, 
            lat="Latitude_degN", lon="Longitude_degE", 
            color=target_item, color_continuous_scale=c_scale_final,
            mapbox_style="open-street-map",
            hover_data=["d18O",'dD',"Salinity",'Temperature_degC','Year','Month','Day','Cruise','Station','Depth_m','reference'],
        )
        
        
    
        fig_ts_map = unify_plot_layout(fig_ts_map, "Lon", "Lat", sel_col)
        
        fig_ts_map.update_layout(
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=auto_zoom
                
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            height=500,
            coloraxis_colorbar=dict(
                title=sel_col,
                x=0.98,
                xanchor="right",
                y=0.5,
                yanchor="middle",
                len=0.8,
                thickness=15,
            ),
        )
        

        
        # Keep map controls compact so the map remains visible after Streamlit reruns.
        # Streamlitの再実行後も地図が見つけやすいよう、地図設定をポップオーバーに集約する。
        with st.popover(
            "Map controls", **envgeo_utils.stretch_width_kwargs(st.popover)
        ):
            map_mode_ts = st.radio(
                "Map style",
                envgeo_utils.MAP_MODE_OPTIONS,
                index=envgeo_utils.MAP_MODE_DEFAULT_INDEX,
                horizontal=True,
                key="ms_ts",
                help=getattr(
                    envgeo_utils,
                    "MAP_STYLE_HELP_TEXT",
                    "Choose the background map style for the sampling-location map.",
                ),
            )
        st.caption(f"Map style: {map_mode_ts}")
    
        
        # Apply the selected background style / 選択した背景スタイルを適用する。
        fig_ts_map = envgeo_utils.apply_map_style(fig_ts_map, map_mode_ts)
        

        
        # Use a unique widget key and enable wheel zoom.
        # 一意のwidget keyを用い、マウスホイールズームを有効にする。
        st.plotly_chart(
            fig_ts_map,
            key="3d_visualizer_map_TS",
            config={'scrollZoom': True, 'displayModeBar': True},
            **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
        )
        st.download_button(
            "Download interactive HTML",
            envgeo_utils.figure_to_self_contained_html(fig_ts_map),
            envgeo_utils.build_figure_filename("p03_TS_map", extension="html"),
            "text/html",
            key="p03_TS_map_html_dl",
        )

        # Export tables / 表を出力する。
        envgeo_utils.display_isotope_table(df1)
        envgeo_utils.display_isotope_table(df_ts_map_display,  title="Box/Lasso-selected dataset (CSV)")
        
    

    # =============================================================================
    # δD–δ18O layout / δD–δ18Oレイアウト
    # =============================================================================
    elif plot_figure == fig_type_dD_d18O:
        st.subheader("δD-δ18O Relationship")
        render_linked_xy_plot(
            "d18O",
            "dD",
            "δD-δ18O Relationship",
            "fig_dD_d18O",
            allow_regression=True,
        )

    # =============================================================================
    # Custom 2D/2.5D layout / カスタム2D/2.5Dレイアウト
    # =============================================================================
    elif plot_figure == fig_type_custom_xy:
        st.subheader("Custom 2D/2.5D Plot beta")
        numeric_options = available_numeric_plot_columns(df1)
        if len(numeric_options) < 2:
            st.warning("At least two numeric parameters are required for a custom 2D plot.")
            return
        custom_cols = st.columns(2)
        with custom_cols[0]:
            custom_x = st.selectbox(
                "X axis",
                numeric_options,
                index=numeric_options.index("d18O") if "d18O" in numeric_options else 0,
                key="3d_custom_xy_x",
            )
        with custom_cols[1]:
            y_default = "dD" if "dD" in numeric_options else numeric_options[min(1, len(numeric_options) - 1)]
            custom_y = st.selectbox(
                "Y axis",
                numeric_options,
                index=numeric_options.index(y_default),
                key="3d_custom_xy_y",
            )
        render_linked_xy_plot(
            custom_x,
            custom_y,
            "Custom 2D/2.5D Plot beta",
            f"fig_custom_xy_{custom_x}_{custom_y}",
            allow_regression=True,
            allow_single_color=True,
        )

    
    # =============================================================================
    # Salinity–δ18O layout / 塩分–δ18Oレイアウト
    # =============================================================================
    elif  plot_figure == fig_type_d18Osal:
    
    
        # ---------------------------------------------------------------------
        # Selection state / 選択状態
        # ---------------------------------------------------------------------
        if 'd18o_selected_indices' not in st.session_state:
            st.session_state.d18o_selected_indices = []
    
        st.subheader('Salinity-δ18O Relationship')
    
        sel_col_d18o, target_item, c_scale_final = render_color_controls(df1, "fig_d18O_zoom")
        regression_control_col, regression_stats_col = st.columns([0.9, 2.1])
        with regression_control_col:
            show_regression_d18o = st.checkbox(
                "Regression line",
                value=False,
                key="fig_d18O_zoom_regression_line",
                help=getattr(
                    envgeo_utils,
                    "REGRESSION_HELP_TEXT",
                    "Add a simple least-squares regression line for quick visual reference.",
                ),
            )
        regression_stats_text_d18o = ""

        
        
        # ---------------------------------------------------------------------
        # Salinity–δ18O scatter plot / 塩分–δ18O散布図
        # ---------------------------------------------------------------------
        # Keep rows that contain the selected color variable and both axes.
        # 選択した色変数と両軸を持つ行だけを描画対象にする。
        df_plot_d18o = df1.dropna(subset=[target_item, "Salinity", "d18O"]).reset_index(drop=True)

    
        # Report rows excluded from this view / この表示から除外した行数を示す。
        excluded_count2 = len(df1) - len(df_plot_d18o)
        if excluded_count2 > 0:
            st.caption(f":red[{excluded_count2:,} rows excluded because {sel_col_d18o}, salinity, or d18O was missing.]")
    
    
        fig_d18O = px.scatter(
            df_plot_d18o,
            x="Salinity", y="d18O", 
            color=target_item, color_continuous_scale=c_scale_final,
            hover_data={
                "Salinity": True,
                "d18O": True,
                "lat": True,
                "lon": True,
                "dD": True,
                "Temperature_degC": True,
                "Year": True,
                "Month": True,
                "Day": True,
                "Cruise": True,
                "Station": True,
                "Depth_m": True,
                "reference": True
            }
        )
        if show_regression_d18o:
            regression_stats_text_d18o = add_regression_line(
                fig_d18O,
                df_plot_d18o,
                "Salinity",
                "d18O",
            )
        with regression_stats_col:
            if show_regression_d18o:
                render_regression_stats(regression_stats_text_d18o)
        
        fig_d18O = unify_plot_layout(fig_d18O, "Salinity", "δ18O (‰)", sel_col_d18o)
        
    
        # Fix the interactive viewing area / 操作時の表示領域を固定する。
        fig_d18O.update_layout(
            hovermode='closest',
            hoverdistance=5,
            margin=dict(l=60, r=90, t=50, b=70, autoexpand=False),
            coloraxis_colorbar=dict(x=1.02, xanchor='left', len=0.8),
            xaxis=dict(
                zeroline=False, zerolinewidth=1, zerolinecolor='grey',
                showline=True, linewidth=1, linecolor='grey', mirror=True,
                range=[df_plot_d18o["Salinity"].min()*0.95, df_plot_d18o["Salinity"].max()*1.05]
            ),
            yaxis=dict(
                zeroline=False, zerolinewidth=1, zerolinecolor='grey',
                showline=True, linewidth=1, linecolor='grey', mirror=True,
                range=[df_plot_d18o["d18O"].min()-0.5, df_plot_d18o["d18O"].max()+0.5]
            )
        )
    
    
    
        with envgeo_utils.bounded_container(850):
            selected_points_d18o = plotly_events(
                fig_d18O,
                select_event=True,
                key="d18o_zoom_event",
                override_height=600,
                override_width="100%",
            )
        st.download_button(
            "Download interactive HTML",
            envgeo_utils.figure_to_self_contained_html(fig_d18O),
            envgeo_utils.build_figure_filename("p03_d18O", extension="html"),
            "text/html",
            key="p03_d18O_html_dl",
        )
        show_selection_tip()
    
            
        # Store the current Box/Lasso selection / 現在のBox/Lasso選択を保存する。
        selected_indices_d18o = selected_point_indices(selected_points_d18o, len(df_plot_d18o))
        if selected_indices_d18o:
            st.session_state.d18o_selected_indices = selected_indices_d18o
            num_selected_d18o = len(selected_indices_d18o)
            st.write(f"**Selected points: {num_selected_d18o}**")
        else:
            st.session_state.d18o_selected_indices = []
            
    
        # Map data and view extent / 地図データと表示範囲
        is_selected = len(st.session_state.d18o_selected_indices) > 0
        df_map_d18o = df_plot_d18o.iloc[st.session_state.d18o_selected_indices] if is_selected else df_plot_d18o
    
        lat_min, lat_max = df_map_d18o["Latitude_degN"].min(), df_map_d18o["Latitude_degN"].max()
        lon_min, lon_max = df_map_d18o["Longitude_degE"].min(), df_map_d18o["Longitude_degE"].max()
        
        
        # Map center / 地図中心
        center_lat = df_map_d18o['Latitude_degN'].mean()
        center_lon = df_map_d18o['Longitude_degE'].mean()
        
        
        lat_diff = max(lat_max - lat_min, 0.1)
        lon_diff = max(lon_max - lon_min, 0.1)
        
        # Calculate zoom from data extent / データ範囲からズームを計算する。
        zoom_lon = math.log2((850 * 360) / (lon_diff * 256))
        zoom_lat = math.log2((600 * 180) / (lat_diff * 256))
        auto_zoom = min(zoom_lon, zoom_lat) - (0.8 if is_selected else 1.5)
        auto_zoom = max(1, min(15, auto_zoom))
    
        # Render the linked map / 連動地図を描画する。
        fig_map_d18o = px.scatter_mapbox(
            df_map_d18o, lat="Latitude_degN", lon="Longitude_degE", 
            color=target_item, color_continuous_scale=c_scale_final,
            mapbox_style="open-street-map",
            hover_data=["d18O",'dD',"Salinity",'Temperature_degC','Year','Month','Day','Cruise','Station','Depth_m','reference'],
        )
        fig_map_d18o = unify_plot_layout(fig_map_d18o, "Lon", "Lat", sel_col_d18o)
        fig_map_d18o.update_layout(
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            height=500,
            coloraxis_colorbar=dict(
                title=sel_col_d18o,
                x=0.98,
                xanchor="right",
                y=0.5,
                yanchor="middle",
                len=0.8,
                thickness=15,
            ),
        )
        
        
        
        # Keep map controls compact so the map remains visible after Streamlit reruns.
        # Streamlitの再実行後も地図が見つけやすいよう、地図設定をポップオーバーに集約する。
        with st.popover(
            "Map controls", **envgeo_utils.stretch_width_kwargs(st.popover)
        ):
            map_mode_d18o = st.radio(
                "Map style",
                envgeo_utils.MAP_MODE_OPTIONS,
                index=envgeo_utils.MAP_MODE_DEFAULT_INDEX,
                horizontal=True,
                key="ms_d18o",
                help=getattr(
                    envgeo_utils,
                    "MAP_STYLE_HELP_TEXT",
                    "Choose the background map style for the sampling-location map.",
                ),
            )
        st.caption(f"Map style: {map_mode_d18o}")
    
        
        # Apply the selected background style / 選択した背景スタイルを適用する。
        fig_map_d18o = envgeo_utils.apply_map_style(fig_map_d18o, map_mode_d18o)
        
        
        
        # Use a unique widget key and enable wheel zoom.
        # 一意のwidget keyを用い、マウスホイールズームを有効にする。
        st.plotly_chart(
            fig_map_d18o,
            key="3d_visualizer_map_d18O",
            config={'scrollZoom': True, 'displayModeBar': True},
            **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
        )
        st.download_button(
            "Download interactive HTML",
            envgeo_utils.figure_to_self_contained_html(fig_map_d18o),
            envgeo_utils.build_figure_filename("p03_d18O_map", extension="html"),
            "text/html",
            key="p03_d18O_map_html_dl",
        )

        # Export tables / 表を出力する。
        envgeo_utils.display_isotope_table(df1)
        envgeo_utils.display_isotope_table(df_map_d18o,  title="Box/Lasso-selected dataset (CSV)")
    

    
        
if __name__ == '__main__':
    main()
    
