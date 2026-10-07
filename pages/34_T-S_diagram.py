#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature–salinity diagram visualizer for EnvGeo-Seawater data.

EnvGeo-Seawater データの水温–塩分関係を表示するページです。

Author: Toyoho Ishimura, Kyoto University
Last reviewed: 2026-09-30
"""

# =============================================================================
# Page configuration / ページ設定
# =============================================================================
version = "1.3.4"
fig_title = "envgeo-seawater-database"

import io
import math

import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib.ticker import FormatStrFormatter

import envgeo_user_data
import envgeo_utils

def main():
    # =============================================================================
    # Page header / ページ見出し
    # =============================================================================
    st.header(f'Temperature-Salinity Diagram ({version})')
    

    st.button('Reload')

    # =============================================================================
    # Data-source selection / データソースの選択
    # =============================================================================
    data_source_ENVGEO = envgeo_utils.data_source_ENVGEO
    data_source_AROUND_JAPAN = envgeo_utils.data_source_AROUND_JAPAN
    data_source_GLOBAL = envgeo_utils.data_source_GLOBAL
    ref_data = st.radio("Data source (see Home > About):", (data_source_ENVGEO, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True)

    # -----------------------------------------------------------------------------
    # Figure options / 図の表示オプション
    # -----------------------------------------------------------------------------
    plot_option_col1, plot_option_col2 = st.columns([1, 1])
    with plot_option_col1:
        plot_all_data = st.radio(
            "Show background data",
            ("Yes", "No"),
            index=1,
            horizontal=True,
            help=getattr(envgeo_utils, "BACKGROUND_DATA_HELP_TEXT", "Show the unfiltered dataset behind the currently filtered data for context."),
        )
    with plot_option_col2:
        show_legend = st.radio(
            "Show legend",
            ("Yes", "No"),
            index=0,
            horizontal=True,
            help="Show or hide the legend for plotted data groups.",
        )
    
    # -----------------------------------------------------------------------------
    # Attribution / 出典表示
    # -----------------------------------------------------------------------------
    if ref_data == data_source_ENVGEO:
        st.write(envgeo_utils.refs_ENVGEO)
        
    elif ref_data == data_source_AROUND_JAPAN:
        st.write(envgeo_utils.refs_AROUND_JAPAN)
      
    elif ref_data == data_source_GLOBAL:
        st.write(envgeo_utils.refs_GLOBAL)
        
    else:
        st.warning("Invalid data source selection.")

    # =============================================================================
    # Data loading and uploaded overlay / データ読込とアップロード重ね表示
    # =============================================================================
    df_original = envgeo_utils.load_isotope_data(ref_data)
    df1 = df_original

    if df_original.empty:
        st.warning("No data available for the selected conditions.")
        return

    embedded_in_integrated = (
        st.session_state.get(envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY)
        == "34_T-S_diagram.py"
    )
    if embedded_in_integrated:
        uploaded_df = envgeo_utils.get_uploaded_data()
    else:
        uploaded_df = envgeo_user_data.render_upload_panel(
            "ts",
            "The T-S overlay requires salinity and temperature columns.",
        )
    uploaded_df = envgeo_user_data.render_column_controls(
        uploaded_df,
        {
            "Temperature column": "Temperature_degC",
            "Salinity column": "Salinity",
        },
        "ts",
    )
    uploaded_style = envgeo_user_data.render_marker_style_controls(
        uploaded_df,
        "ts",
    )

    # =============================================================================
    # Shared sidebar filtering / 共通サイドバーによるデータ絞り込み
    # =============================================================================
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
     submitted) = envgeo_utils.sidebar_filter_and_display(
         envgeo_utils.combine_reference_and_uploaded_for_filtering(
             df1, uploaded_df
         ),
         ref_data, data_source_ENVGEO, data_source_AROUND_JAPAN,
         uploaded_df=uploaded_df, uploaded_filter_key="ts_diagram",
         uploaded_dataset_label=envgeo_utils.UPLOADED_DATA_LABEL,
     )
    df1, uploaded_df = envgeo_utils.split_uploaded_rows(
        df1, envgeo_utils.UPLOADED_DATA_LABEL
    )

    # A meaningful T-S diagram requires at least two selected points.
    # 意味のあるT-S図には、選択された2点以上のデータが必要です。
    data_found = len(df1["d18O"])
    if data_found == 1:
        st.warning('Only one data point was found. A T–S diagram could not be meaningfully generated.')
        st.stop()
    
    # -----------------------------------------------------------------------------
    # Coordinate aliases / 座標列の別名
    # -----------------------------------------------------------------------------
    df1['lat'] = df1['Latitude_degN']
    df1['lon'] = df1['Longitude_degE']
    # =============================================================================
    # Figure controls / 図の表示設定
    # =============================================================================
   

    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "FIGURE_CONTROLS_LABEL", "Figure controls"))
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        
        # Marker transparency / マーカーの透明度
        alpha_selected = st.slider(label='Transparency (Filtered Plot)',
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=(0.9),
                                    step=0.05
                                    )

        ts_color_candidates = [
            "Single color",
            "Depth_m",
            "Latitude_degN",
            "Longitude_degE",
            "Year",
            "Month",
            "d18O",
            "dD",
            "d-excess",
        ]
        ts_color_options = [
            item for item in ts_color_candidates
            if item == "Single color" or item in df1.columns
        ]
        ts_color_by = st.selectbox(
            "Color parameter",
            ts_color_options,
            index=0,
            help=(
                "Use a single marker color, or color the filtered T-S data by "
                "a numeric column such as depth, latitude, longitude, year, "
                "month, d18O, dD, or d-excess."
            ),
        )

        ts_color_range = None
        if ts_color_by != "Single color":
            ts_matplotlib_colormap = envgeo_utils.get_matplotlib_colormap(ts_color_by)
            ts_color_sources = [pd.to_numeric(df1[ts_color_by], errors="coerce")]
            if (
                uploaded_style["color_mode"] == "Use current colorbar when possible"
                and ts_color_by in uploaded_df.columns
            ):
                ts_color_sources.append(
                    pd.to_numeric(uploaded_df[ts_color_by], errors="coerce")
                )
            ts_color_source = pd.concat(ts_color_sources, ignore_index=True).dropna()
            if not ts_color_source.empty:
                ts_color_min = float(ts_color_source.min())
                ts_color_max = float(ts_color_source.max())

                if ts_color_min == ts_color_max:
                    ts_color_range = (ts_color_min, ts_color_max)
                    st.caption(f"Colorbar range: {ts_color_min:g}")
                elif ts_color_by in ["Year", "Month"]:
                    ts_color_range = st.slider(
                        "Colorbar range",
                        min_value=int(math.floor(ts_color_min)),
                        max_value=int(math.ceil(ts_color_max)),
                        value=(int(math.floor(ts_color_min)), int(math.ceil(ts_color_max))),
                        step=1,
                        help=(
                            "Set the colorbar range for the selected T-S color "
                            "parameter. Points outside this range are plotted "
                            "using the end colors."
                        ),
                    )
                else:
                    color_step = 10.0 if ts_color_by == "Depth_m" else 0.1
                    ts_color_range = st.slider(
                        "Colorbar range",
                        min_value=float(math.floor(ts_color_min)),
                        max_value=float(math.ceil(ts_color_max)),
                        value=(
                            float(math.floor(ts_color_min)),
                            float(math.ceil(ts_color_max)),
                        ),
                        step=color_step,
                        help=(
                            "Set the colorbar range for the selected T-S color "
                            "parameter. Points outside this range are plotted "
                            "using the end colors."
                        ),
                    )
            else:
                st.caption(f"No valid {ts_color_by} values are available for the colorbar.")
        else:
            ts_matplotlib_colormap = None
        
        # Axis ranges / 軸の表示範囲
        if ref_data == data_source_ENVGEO:
            sal_min, sal_max = st.slider(label='Salinity scale selected',
                                        min_value=20,
                                        max_value=36,
                                        value=(20, 36),
                                        )

            temp_min, temp_max = st.slider(label='Temperature scale selected',
                                        min_value=-1,
                                        max_value=30,
                                        value=(-1, 30),
                                        )
            
            
        elif ref_data == data_source_AROUND_JAPAN:
            sal_min, sal_max = st.slider(label='Salinity scale selected',
                                        min_value=20,
                                        max_value=36,
                                        value=(20, 36),
                                        )

            temp_min, temp_max = st.slider(label='Temperature scale selected',
                                        min_value=-1,
                                        max_value=30,
                                        value=(-1, 30),
                                        )

            
        else:
            sal_min, sal_max = st.slider(label='Salinity scale selected',
                                        min_value=-1,
                                        max_value=40,
                                        value=(-1, 40),
                                        )

            temp_min, temp_max = st.slider(label='Temperature scale selected',
                                        min_value=-5,
                                        max_value=40,
                                        value=(-5,40),
                                        )

        # Matplotlib appearance / Matplotlib図の表示設定
        fig_size_col1, fig_size_col2 = st.columns(2)
        with fig_size_col1:
            sld_fig_size_x = st.number_input(
                "Fig width (x)",
                min_value=4,
                max_value=24,
                value=12,
                step=1,
                key="ts_diagram_fig_width_x",
                help="Adjust the width of the Matplotlib T-S diagram.",
            )
        with fig_size_col2:
            sld_fig_size_y = st.number_input(
                "Fig height (y)",
                min_value=4,
                max_value=24,
                value=9,
                step=1,
                key="ts_diagram_fig_height_y",
                help="Adjust the height of the Matplotlib T-S diagram.",
            )
        font_col1, font_col2 = st.columns(2)
        with font_col1:
            sld_font_size_tick = st.number_input(
                "Tick font size",
                min_value=6,
                max_value=32,
                value=15,
                step=1,
                key="ts_diagram_tick_font_size",
                help="Adjust the tick-label font size for the T-S diagram.",
            )
        with font_col2:
            sld_font_size_label = st.number_input(
                "Label font size",
                min_value=6,
                max_value=32,
                value=16,
                step=1,
                key="ts_diagram_label_font_size",
                help="Adjust the axis-label and colorbar-label font size for the T-S diagram.",
            )

        tick_col1, tick_col2 = st.columns(2)
        with tick_col1:
            tick_count_x = st.number_input(
                "X tick count",
                min_value=3,
                max_value=30,
                value=9,
                step=1,
                key="ts_diagram_x_tick_count",
                help="Adjust the number of major tick marks on the salinity axis.",
            )
        with tick_col2:
            tick_count_y = st.number_input(
                "Y tick count",
                min_value=3,
                max_value=30,
                value=9,
                step=1,
                key="ts_diagram_y_tick_count",
                help="Adjust the number of major tick marks on the temperature axis.",
            )

        # Uploaded rows missing the color parameter / 色分け値がないアップロード行
        show_nodata_uploaded = st.checkbox(
            f"Show uploaded points without {ts_color_by} values",
            value=True,
            key="ts_diagram_show_nodata_uploaded",
            help="Show or hide uploaded data points that have no value for the T-S color parameter.",
        )

        # Approximate σ0 contour interval / 近似σ0等値線の間隔
        contour_interval = st.selectbox(
            "Density contour interval (approx. σ0)",
            options=[0.2, 0.5, 1.0],
            index=1,
            format_func=lambda v: f"{v} kg m⁻³",
            help=(
                "Spacing between approximate σ0 reference contour lines. "
                "These are visual reference guides, not pointwise sample density."
            ),
            key="ts_diagram_contour_interval",
        )

    # =============================================================================
    # Cache control / キャッシュ制御
    # =============================================================================
    if st.sidebar.button("🔄 Clear cache"):
        envgeo_utils.clear_app_cache()
        st.rerun()

    # =============================================================================
    # Figure preparation and drawing / 図の準備と描画
    # =============================================================================
    st.caption(getattr(envgeo_utils, "MAP_AREA_HELP_TEXT", "Map center, extent, colormap, and figure settings can be adjusted in the sidebar."))

    uploaded_ts = pd.DataFrame()
    if not uploaded_df.empty and {
        "Salinity",
        "Temperature_degC",
    }.issubset(uploaded_df.columns):
        uploaded_ts = uploaded_df.dropna(
            subset=["Salinity", "Temperature_degC"]
        ).reset_index(drop=True)
        uploaded_excluded_count = len(uploaded_df) - len(uploaded_ts)
        st.caption(
            f":blue[Uploaded overlay: {len(uploaded_ts):,} / {len(uploaded_df):,} "
            f"plotted ({uploaded_excluded_count:,} excluded due to missing or invalid "
            "salinity/temperature).]"
        )

    if not uploaded_df.empty:
        uploaded_quality_df = envgeo_utils.get_quality_rows(uploaded_df)
        with st.expander("Uploaded data quality check", expanded=False):
            envgeo_utils.render_quality_flag_criteria_note()
            st.write(
                f"Quality-flagged rows: {len(uploaded_quality_df):,} / "
                f"{len(uploaded_df):,}"
            )
            if uploaded_quality_df.empty:
                st.success("No uploaded rows triggered the current quality rules.")
            else:
                st.dataframe(
                    uploaded_quality_df,
                    **envgeo_utils.stretch_width_kwargs(st.dataframe),
                )
    
    
    # -----------------------------------------------------------------------------
    # Matplotlib constants / Matplotlibの共通設定
    # -----------------------------------------------------------------------------
    plt.rcParams["font.size"] = sld_font_size_tick
    fig_size = [sld_fig_size_x, sld_fig_size_y]
    fig_dpi = 150
    ax_length = 15
    
    # -----------------------------------------------------------------------------
    # T-S diagram definition / T-S図の定義
    # -----------------------------------------------------------------------------
    X_data = "Salinity"
    Y_data = "Temperature_degC"
    
    X_label = "salinity"
    Y_label = "Temperature"
    
    iso_scale_X = ""
    iso_scale_Y = "(C)"
    

    # Existing plot switch / 既存の描画切替
    X_Y = 1
    
    X_Y_C = "red"
    X_Y_M = "."
    X_Y_S = 100
    

  
    
    fig_title_X_Y = X_label + " - " + Y_label
    sheet_names_add2 = "filtered data"
    alpha_all = 0.2
    X_Y_C_add = "blue"

    # Axis limits / 軸範囲

    lim_min_X = sal_min
    lim_max_X = sal_max
    lim_min_Y = temp_min
    lim_max_Y = temp_max
    
    
    # -----------------------------------------------------------------------------
    # Background data / 背景データ
    # -----------------------------------------------------------------------------
    if X_Y == 1:

        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
        ax = plt.subplot(111)
    
        ax.set_xlabel(X_label + iso_scale_X, fontsize=sld_font_size_label)
        ax.set_ylabel(Y_label + iso_scale_Y, fontsize=sld_font_size_label)

        if plot_all_data == "Yes":
            # Invisible point supplies the background-series legend label.
            # 非表示点により、背景系列の凡例ラベルを用意する。
            ax.scatter(-1000, -1000, s=X_Y_S, c=X_Y_C, marker=X_Y_M, alpha=alpha_all, label='ALL')

        # Keep rows that can be positioned on both axes. / 両軸に表示できる行だけを用いる。
        df_fig_ALL = df_original.dropna(subset=["Salinity", "Temperature_degC"]).reset_index(drop=True)
        
        excluded_count = len(df_original) - len(df_fig_ALL)
        if excluded_count > 0:
            st.caption(f":red[Background plot: {len(df_fig_ALL):,} / {len(df_original):,} plotted ({excluded_count:,} excluded due to missing salinity/temperature).]")
                

        Ya = df_fig_ALL[Y_data]
        Xa = df_fig_ALL[X_data]

        ax.set_xlim(lim_min_X, lim_max_X)
        ax.set_ylim(lim_min_Y, lim_max_Y)
        ax.set_xticks(np.linspace(lim_min_X, lim_max_X, tick_count_x))
        ax.set_yticks(np.linspace(lim_min_Y, lim_max_Y, tick_count_y))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(labelsize=sld_font_size_tick)
        
        ax.tick_params(length=ax_length)

        if plot_all_data == "Yes":
            ax.scatter(Xa, Ya, s=X_Y_S,c=X_Y_C,marker=X_Y_M,lw=0.5, ec="black", alpha=alpha_all)
            if show_legend == "Yes":
                plt.legend(fontsize=sld_font_size_tick)

        plt.title(fig_title_X_Y)

        # -------------------------------------------------------------------------
        # Filtered data / 絞り込みデータ
        # -------------------------------------------------------------------------
        X_Y_C_add = 'blue'

        # Keep rows that can be positioned on both axes. / 両軸に表示できる行だけを用いる。
        df_fig_add = df1.dropna(subset=["Salinity", "Temperature_degC"]).reset_index(drop=True)

        excluded_count_add = len(df1) - len(df_fig_add)
        if excluded_count_add > 0:
            st.caption(f":blue[Filtered plot: {len(df_fig_add):,} / {len(df1):,} plotted ({excluded_count_add:,} excluded due to missing salinity/temperature).]")
                

            
        Y_add = df_fig_add[Y_data]
        X_add = df_fig_add[X_data]

        
        colorbar_drawn = False
        if ts_color_by == "Single color":
            ax.scatter(
                X_add,
                Y_add,
                s=X_Y_S,
                c=X_Y_C_add,
                marker=X_Y_M,
                alpha=alpha_selected,
                lw=0.5,
                ec="black",
                label=sheet_names_add2,
            )
            if show_legend == "Yes":
                plt.legend(fontsize=sld_font_size_tick)
        else:
            color_values = pd.to_numeric(df_fig_add[ts_color_by], errors="coerce")
            color_valid = color_values.notna()

            if color_valid.any():
                st.caption(
                    f":blue[T-S color: {color_valid.sum():,} / {len(df_fig_add):,} "
                    f"filtered samples have {ts_color_by} values.]"
                )
                ax_color = ax.scatter(
                    X_add[color_valid],
                    Y_add[color_valid],
                    s=X_Y_S,
                    c=color_values[color_valid],
                    cmap=ts_matplotlib_colormap,
                    vmin=ts_color_range[0] if ts_color_range is not None else None,
                    vmax=ts_color_range[1] if ts_color_range is not None else None,
                    marker=X_Y_M,
                    alpha=alpha_selected,
                    lw=0.5,
                    ec="black",
                    label=sheet_names_add2,
                )
                cbar_ts = fig.colorbar(
                    ax_color,
                    ax=ax,
                    orientation="vertical",
                    pad=0.02,
                    fraction=0.04,
                    extend="neither",
                )
                cbar_ts.set_label(ts_color_by, fontsize=sld_font_size_label)
                cbar_ts.ax.tick_params(labelsize=sld_font_size_tick)
                colorbar_drawn = True

            if (~color_valid).any():
                st.caption(
                    f":gray[T-S color: {(~color_valid).sum():,} filtered samples "
                    f"without {ts_color_by} values are shown in gray.]"
                )
                ax.scatter(
                    X_add[~color_valid],
                    Y_add[~color_valid],
                    s=X_Y_S,
                    c="lightgray",
                    marker=X_Y_M,
                    alpha=0.6,
                    lw=0.5,
                    ec="black",
                    label=f"{sheet_names_add2} (no {ts_color_by})",
                )

            if show_legend == "Yes":
                plt.legend(fontsize=sld_font_size_tick)

        # -------------------------------------------------------------------------
        # Approximate σ0 reference contours / 近似σ0参照等値線
        # -------------------------------------------------------------------------
        # The grid is clipped to the valid GSW input domain.
        # グリッドはGSWの有効な入力範囲に制限する。
        _T_GSW_MIN, _T_GSW_MAX = -5.0, 45.0
        _S_GSW_MIN, _S_GSW_MAX = 0.0, 50.0
        _T_lo = max(float(lim_min_Y), _T_GSW_MIN)
        _T_hi = min(float(lim_max_Y), _T_GSW_MAX)
        _S_lo = max(float(lim_min_X), _S_GSW_MIN)
        _S_hi = min(float(lim_max_X), _S_GSW_MAX)
        if _T_lo < _T_hi and _S_lo < _S_hi:
            tempL = np.linspace(_T_lo, _T_hi)
            salL  = np.linspace(_S_lo, _S_hi)
            Tg, Sg = np.meshgrid(tempL, salL)
            # Approximation: Practical Salinity ≈ Absolute Salinity and in-situ
            # temperature ≈ Conservative Temperature; contours are reference only.
            # 近似として実用塩分≒絶対塩分、現場水温≒保存温度を用いる参照等値線。
            sigma0_approx = gsw.sigma0(Sg, Tg)
            _s0_min = float(np.nanmin(sigma0_approx))
            _s0_max = float(np.nanmax(sigma0_approx))
            if np.isfinite(_s0_min) and np.isfinite(_s0_max) and _s0_max > _s0_min:
                _first = np.ceil(_s0_min / contour_interval) * contour_interval
                contour_levels = np.arange(
                    _first, _s0_max + contour_interval * 0.5, contour_interval
                )
                contour_levels = contour_levels[contour_levels <= _s0_max]
                if len(contour_levels) >= 1:
                    cs = ax.contour(
                        Sg, Tg, sigma0_approx,
                        colors='lightgrey', linestyles='dashed', zorder=0,
                        levels=contour_levels,
                    )
                    plt.clabel(
                        cs, fontsize=sld_font_size_tick,
                        inline=True, fmt='%.1f', zorder=0,
                    )

        # -------------------------------------------------------------------------
        # Uploaded overlay / アップロードデータの重ね表示
        # -------------------------------------------------------------------------

        if not uploaded_ts.empty:
            use_shared_colorbar = (
                uploaded_style["color_mode"] == "Use current colorbar when possible"
                and ts_color_by != "Single color"
                and ts_color_by in uploaded_ts.columns
            )

            if use_shared_colorbar:
                uploaded_color_values = pd.to_numeric(
                    uploaded_ts[ts_color_by], errors="coerce"
                )
                uploaded_color_valid = uploaded_color_values.notna()

                if uploaded_color_valid.any():
                    uploaded_color_plot = ax.scatter(
                        uploaded_ts.loc[uploaded_color_valid, "Salinity"],
                        uploaded_ts.loc[uploaded_color_valid, "Temperature_degC"],
                        s=uploaded_style["size"],
                        c=uploaded_color_values[uploaded_color_valid],
                        cmap=ts_matplotlib_colormap,
                        vmin=ts_color_range[0] if ts_color_range is not None else None,
                        vmax=ts_color_range[1] if ts_color_range is not None else None,
                        marker=uploaded_style["marker"],
                        alpha=uploaded_style["alpha"],
                        linewidths=uploaded_style["outline_width"],
                        edgecolors=uploaded_style["outline_color"],
                        label="Uploaded data",
                        zorder=10,
                    )
                    if not colorbar_drawn:
                        cbar_ts = fig.colorbar(
                            uploaded_color_plot,
                            ax=ax,
                            orientation="vertical",
                            pad=0.02,
                            fraction=0.04,
                            extend="neither",
                        )
                        cbar_ts.set_label(ts_color_by, fontsize=sld_font_size_label)
                        cbar_ts.ax.tick_params(labelsize=sld_font_size_tick)
                        colorbar_drawn = True

                if (~uploaded_color_valid).any() and (not use_shared_colorbar or show_nodata_uploaded):
                    ax.scatter(
                        uploaded_ts.loc[~uploaded_color_valid, "Salinity"],
                        uploaded_ts.loc[~uploaded_color_valid, "Temperature_degC"],
                        s=uploaded_style["size"],
                        c=uploaded_style["color"],
                        marker=uploaded_style["marker"],
                        alpha=uploaded_style["alpha"],
                        linewidths=uploaded_style["outline_width"],
                        edgecolors=uploaded_style["outline_color"],
                        label=f"Uploaded data (no {ts_color_by})",
                        zorder=10,
                    )
                    st.caption(
                        f":gray[Uploaded overlay: {(~uploaded_color_valid).sum():,} "
                        f"samples without {ts_color_by} values use the fixed color.]"
                    )
            else:
                ax.scatter(
                    uploaded_ts["Salinity"],
                    uploaded_ts["Temperature_degC"],
                    s=uploaded_style["size"],
                    c=uploaded_style["color"],
                    marker=uploaded_style["marker"],
                    alpha=uploaded_style["alpha"],
                    linewidths=uploaded_style["outline_width"],
                    edgecolors=uploaded_style["outline_color"],
                    label="Uploaded data",
                    zorder=10,
                )

            if show_legend == "Yes":
                ax.legend(fontsize=sld_font_size_tick)

        # -------------------------------------------------------------------------
        # Figure title and file name / 図題とファイル名
        # -------------------------------------------------------------------------
        main_title = fig_title

        # Compact month-range text for the figure title. / 月範囲を図題用に短縮する。
        if len(selected_months) == 12:
            month_display = "All"
        elif len(selected_months) == 0:
            month_display = "None"
        else:
            sorted_m = sorted(list(set(selected_months)))
            ranges = []
            if sorted_m:
                start = sorted_m[0]
                for i in range(len(sorted_m)):
                    if i + 1 == len(sorted_m) or sorted_m[i+1] != sorted_m[i] + 1:
                        end = sorted_m[i]
                        ranges.append(f"{start}-{end}" if start != end else str(start))
                        if i + 1 < len(sorted_m):
                            start = sorted_m[i+1]
            month_display = ", ".join(ranges)
            
            

        sub_title = f"Lon:{sld_lon_min}-{sld_lon_max}, Lat:{sld_lat_min}-{sld_lat_max}, Y:{sld_year_min}-{sld_year_max}, M:{month_display}, S:{sld_sal_min}-{sld_sal_max}, D:{sld_depth_min}-{sld_depth_max}m"
        
        main_title2 = sub_title
        title_head = f"{main_title}\n{main_title2}"
        title_head2 = title_head.replace('_', ' ')
        fig.suptitle(title_head2,fontsize=sld_font_size_label + 4)
        
 
    else:
        pass
            
    # =============================================================================
    # Image download / 図のダウンロード
    # =============================================================================
    # Keep the PNG in memory; no local file is written. / PNGはメモリ上だけで生成する。
    fn = envgeo_utils.build_figure_filename("Fig_T-S_SW", main_title2)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
     
    st.pyplot(fig)
    st.caption(
        "Density contours: approximate σ0 reference grid "
        "(Practical Salinity ≈ Absolute Salinity; "
        "in-situ temperature ≈ Conservative Temperature). "
        "Not pointwise sample density."
    )

    btn = st.download_button(
       label="Download image",
       data=img,
       file_name=fn,
       mime="image/png")
   

    # =============================================================================
    # Sampling-location map / 採取地点の地図
    # =============================================================================
    # -----------------------------------------------------------------------------
    # Coordinate validation / 座標の検証
    # -----------------------------------------------------------------------------
    if {"Latitude_degN", "Longitude_degE"}.issubset(df_fig_add.columns):
        _lat_num = pd.to_numeric(df_fig_add["Latitude_degN"], errors="coerce")
        _lon_num = pd.to_numeric(df_fig_add["Longitude_degE"], errors="coerce")
        _valid_mask = (
            _lat_num.notna() & _lon_num.notna()
            & _lat_num.between(-90, 90) & _lon_num.between(-180, 180)
        )
        _valid_coords_df = df_fig_add.loc[_valid_mask].copy()
        _valid_coords_df["Latitude_degN"] = _lat_num[_valid_mask].values
        _valid_coords_df["Longitude_degE"] = _lon_num[_valid_mask].values
    else:
        _valid_coords_df = df_fig_add.iloc[0:0].copy()
    _has_valid_map_coords = len(_valid_coords_df) > 0

    st.divider()
    st.subheader('Sampling Location Map')
    

    # Keep map controls compact so the map remains visible after Streamlit reruns.
    # Streamlitの再実行後も地図が見つけやすいよう、地図設定をポップオーバーに集約する。
    with st.popover(
        "Map controls", **envgeo_utils.stretch_width_kwargs(st.popover)
    ):
            map_mode = st.radio(
                "Map style",
                envgeo_utils.MAP_MODE_OPTIONS,
                index=envgeo_utils.MAP_MODE_DEFAULT_INDEX,
                horizontal=True,
                key="map_style_34_auto",
                help=getattr(envgeo_utils, "MAP_STYLE_HELP_TEXT", "Choose the background map style for the sampling-location map."),
            )
    st.caption(f"Map style: {map_mode}")

    uploaded_map_df = pd.DataFrame(columns=["Longitude_degE", "Latitude_degN"])
    if not uploaded_df.empty and {
        "Longitude_degE",
        "Latitude_degN",
    }.issubset(uploaded_df.columns):
        uploaded_map_df = uploaded_df.copy()
        uploaded_map_df["Longitude_degE"] = pd.to_numeric(
            uploaded_map_df["Longitude_degE"], errors="coerce"
        )
        uploaded_map_df["Latitude_degN"] = pd.to_numeric(
            uploaded_map_df["Latitude_degN"], errors="coerce"
        )
        uploaded_map_df = uploaded_map_df.dropna(
            subset=["Longitude_degE", "Latitude_degN"]
        )
        uploaded_map_df = uploaded_map_df.loc[
            uploaded_map_df["Latitude_degN"].between(-90, 90)
        ]

    # Uploaded rows may provide the only valid coordinates.
    # アップロード行だけが有効座標を持つ場合も地図を表示する。
    _has_valid_map_coords = _has_valid_map_coords or not uploaded_map_df.empty

    # -----------------------------------------------------------------------------
    # Automatic map extent / 地図範囲の自動計算
    # -----------------------------------------------------------------------------
    map_extent_sources = []
    if _has_valid_map_coords:
        map_extent_sources.append(_valid_coords_df[["Longitude_degE", "Latitude_degN"]])
    if not uploaded_map_df.empty:
        map_extent_sources.append(
            uploaded_map_df[["Longitude_degE", "Latitude_degN"]]
        )
    # Japan-centered fallback when no valid coordinates exist. / 有効座標がない場合の日本中心設定。
    default_lat, default_lon, default_zoom = 36.0, 138.0, 4.0

    if not map_extent_sources:
        center_lat, center_lon, auto_zoom = default_lat, default_lon, default_zoom
    else:
        map_extent_df = pd.concat(map_extent_sources, ignore_index=True)
        lat_min, lat_max = map_extent_df["Latitude_degN"].min(), map_extent_df["Latitude_degN"].max()
        lon_min, lon_max = map_extent_df["Longitude_degE"].min(), map_extent_df["Longitude_degE"].max()

        if pd.isna(lat_min) or pd.isna(lon_min):
            center_lat, center_lon, auto_zoom = default_lat, default_lon, default_zoom
        else:
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2

            lat_diff = max(lat_max - lat_min, 0.1)
            lon_diff = max(lon_max - lon_min, 0.1)

            # Pixel dimensions estimate a zoom level that includes the data extent.
            # ピクセル寸法を用いて、データ範囲を収めるズームを見積もる。
            map_width_px, map_height_px = 1200, 700
            zoom_lon = math.log2((map_width_px * 360) / (lon_diff * 256))
            zoom_lat = math.log2((map_height_px * 180) / (lat_diff * 256))

            # Leave margin around the selected extent. / 選択範囲の周囲に余白を確保する。
            auto_zoom = min(zoom_lon, zoom_lat) - 2.0
            auto_zoom = max(1, min(15, auto_zoom))

            # Use a wide Japan-centered view for near-global longitude spans.
            # 経度範囲がほぼ全球の場合は、日本中心の広域表示にする。
            if lon_diff > 100:
                center_lat, center_lon, auto_zoom = default_lat, default_lon, 1.5
    
 

    # -----------------------------------------------------------------------------
    # Plotly location map / Plotly採取地点地図
    # -----------------------------------------------------------------------------
    if not _has_valid_map_coords:
        st.info(
            "Map view is unavailable because the selected data contain no valid "
            "latitude/longitude coordinates."
        )
    else:
        c_scale_d18o = envgeo_utils.get_custom_colorscale("d18O")

        # Use uploaded coordinates when no reference coordinates are available.
        # 参照データに有効座標がない場合はアップロード座標を基準にする。
        _map_plot_df = (
            _valid_coords_df if not _valid_coords_df.empty else uploaded_map_df
        )
        _d18o_color = "d18O" if "d18O" in _map_plot_df.columns else None
        _hover_cols = [
            "Latitude_degN", "Longitude_degE", "d18O", "dD",
            "Salinity", "Temperature_degC", "Year", "Month", "Day",
            "Cruise", "Station", "Depth_m", "reference",
        ]
        fig_map = px.scatter_mapbox(
            _map_plot_df,
            lat="Latitude_degN",
            lon="Longitude_degE",
            color=_d18o_color,
            color_continuous_scale=c_scale_d18o,
            hover_data={c: True for c in _hover_cols if c in _map_plot_df.columns},
            opacity=0.6,
            height=500,
        )

        map_d18o_sources = [pd.to_numeric(df_fig_add["d18O"], errors="coerce")]
        if (
            uploaded_style["color_mode"] == "Use current colorbar when possible"
            and "d18O" in uploaded_map_df.columns
        ):
            map_d18o_sources.append(
                pd.to_numeric(uploaded_map_df["d18O"], errors="coerce")
            )
        map_d18o_values = pd.concat(map_d18o_sources, ignore_index=True).dropna()
        map_d18o_range = None
        if not map_d18o_values.empty:
            map_d18o_range = (
                float(map_d18o_values.min()),
                float(map_d18o_values.max()),
            )
            fig_map.update_coloraxes(
                cmin=map_d18o_range[0],
                cmax=map_d18o_range[1],
            )

        fig_map, uploaded_map_count = envgeo_user_data.add_uploaded_map_overlay(
            fig_map,
            uploaded_map_df,
            uploaded_style,
            color_column="d18O",
            colorscale=c_scale_d18o,
            color_range=map_d18o_range,
            show_nodata=show_nodata_uploaded,
        )
        if not uploaded_df.empty:
            if uploaded_map_count:
                st.caption(
                    f":blue[Uploaded locations: {uploaded_map_count:,} / "
                    f"{len(uploaded_df):,} plotted on the map.]"
                )
            else:
                st.caption(
                    ":gray[Uploaded locations are not shown because valid longitude "
                    "and latitude columns are unavailable.]"
                )

        # Apply the selected background style. / 選択した背景スタイルを適用する。
        fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)

        # Keep the colorbar and legend inside the map, preserving map width.
        # カラーバーと凡例を地図内に置き、外側余白による圧縮を避ける。
        fig_map.update_layout(
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=auto_zoom,
                domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            ),
            margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
            autosize=True,
            coloraxis_colorbar=dict(
                title="δ18O (‰)",
                x=0.98,
                xanchor='right',
                bgcolor='rgba(255,255,255,0.75)',
                bordercolor='rgba(150,150,150,0.5)',
                borderwidth=1,
            ),
            legend=dict(
                x=0.01,
                y=0.01,
                xanchor='left',
                yanchor='bottom',
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(150,150,150,0.5)',
                borderwidth=1,
            ),
        )

        # The unique key prevents Streamlit element-ID collisions; wheel zoom is enabled.
        # 一意のkeyでStreamlit要素IDの重複を避け、ホイールズームを有効にする。
        st.plotly_chart(
            fig_map,
            key="TS_plot",
            config={'scrollZoom': True, 'displayModeBar': True},
            **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
        )

    # =============================================================================
    # Filtered-data table / 絞り込みデータ表
    # =============================================================================
    envgeo_utils.display_isotope_table(df_fig_add)
    

if __name__ == '__main__':
    main()
    
