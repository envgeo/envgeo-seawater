#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-profile visualizer for EnvGeo-Seawater data.

EnvGeo-Seawater データの深度プロファイルを表示するページです。

Author: Toyoho Ishimura, Kyoto University
Last reviewed: 2026-09-30
"""

import io
import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib.ticker import FormatStrFormatter

import envgeo_user_data
import envgeo_utils

# =============================================================================
# Page configuration / ページ設定
# =============================================================================
version = "1.3.4"
fig_title = "envgeo-seawater-database"

def main():
    # =============================================================================
    # Page header / ページ見出し
    # =============================================================================
    st.header(f'Depth Profile ({version})')

    st.button('Reload')

    # =============================================================================
    # Data-source selection / データソースの選択
    # =============================================================================
    data_source_JAPAN_SEA = envgeo_utils.data_source_JAPAN_SEA
    data_source_AROUND_JAPAN = envgeo_utils.data_source_AROUND_JAPAN
    data_source_GLOBAL = envgeo_utils.data_source_GLOBAL
    ref_data = st.radio("Data source (see Home > About):", (data_source_JAPAN_SEA, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True)

    # -----------------------------------------------------------------------------
    # Attribution / 出典表示
    # -----------------------------------------------------------------------------
    if ref_data == data_source_JAPAN_SEA:
        st.write(envgeo_utils.refs_JAPAN_SEA)
        
    elif ref_data == data_source_AROUND_JAPAN:
        st.write(envgeo_utils.refs_AROUND_JAPAN)

    elif ref_data == data_source_GLOBAL:
        st.write(envgeo_utils.refs_GLOBAL)
        
    else:
        st.warning("Invalid data source selection.")

    # =============================================================================
    # Profile parameter and display / プロファイルのパラメーターと表示
    # =============================================================================
    col1, col2 = st.columns([1,1])
    with col1:
        plot_all_data = st.radio(
            "Show background data",
            ("Yes", "No"),
            horizontal=True,
            help=getattr(envgeo_utils, "BACKGROUND_DATA_HELP_TEXT", "Show the unfiltered dataset behind the currently filtered data for context."),
        )
    
    with col2:
        plot_element = st.radio(
            "Profile parameter",
            ("d18O(VSMOW)", "dD(VSMOW)", "d-excess", "Temperature (°C)", "Salinity"),
            horizontal=True,
            help="Choose the seawater parameter plotted against water depth.",
        )

        
    if plot_element == "d18O(VSMOW)":
        X_data = "d18O"
        Y_data = "Depth_m"
        
        
        X_label = r"$\delta^{18}$O"
        Y_label = "water depth (m)"
        
        
        iso_scale_X = "(VSMOW)"
        iso_scale_Y = ""
     
        # Dataset-specific initial axis range / データセット別の初期軸範囲
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = -20.0, 4.0
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = -1.4, 0.6
        else:
            fig_x_min, fig_x_max = -1.4, 0.6

     
    elif plot_element == "dD(VSMOW)":
        X_data = "dD"
        Y_data = "Depth_m"
        
        X_label = r"$\delta$D"
        Y_label = "water depth (m)"
        
        iso_scale_X = "(VSMOW)"
        iso_scale_Y = ""
        
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = -150.0, 50.0
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = -20.0, 10.0
        else:
            fig_x_min, fig_x_max = -30.0, 20.0

    elif plot_element == "d-excess":
        X_data = "d-excess"
        Y_data = "Depth_m"
        
        X_label = "d-excess"
        Y_label = "water depth (m)"
        
        iso_scale_X = ""
        iso_scale_Y = ""
        
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = -30.0, 40.0
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = -5.0, 25.0
        else:
            fig_x_min, fig_x_max = -10.0, 30.0

    elif plot_element == "Temperature (°C)":
        X_data = "Temperature_degC"
        Y_data = "Depth_m"
        
        X_label = "Temperature(°C)"
        Y_label = "water depth (m)"
        
        iso_scale_X = ""
        iso_scale_Y = ""
        
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = -3, 35
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = -2, 30
        else:
            fig_x_min, fig_x_max = -2, 30

        
    elif plot_element == "Salinity":
        X_data = "Salinity"
        Y_data = "Depth_m"
        
        X_label = "Salinity"
        Y_label = "water depth (m)"
        
        iso_scale_X = ""
        iso_scale_Y = ""
        
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = 0, 40
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = 28, 36
        else:
            fig_x_min, fig_x_max = 28, 36
    
    else:
        pass

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
        == "37_Depth_Profile.py"
    )
    if embedded_in_integrated:
        uploaded_df = envgeo_utils.get_uploaded_data()
    else:
        uploaded_df = envgeo_user_data.render_upload_panel(
            "depth_profile",
            f"Requires {X_data} and Depth_m for the depth profile; "
            "latitude and longitude are optional and used for the location map.",
        )
    uploaded_df = envgeo_user_data.render_column_controls(
        uploaded_df,
        {
            f"X parameter ({X_data})": X_data,
            "Depth (Depth_m)": "Depth_m",
        },
        "depth_profile",
        optional_roles={
            "Month (optional)": "Month",
            "Latitude (Latitude_degN)": "Latitude_degN",
            "Longitude (Longitude_degE)": "Longitude_degE",
        },
    )
    uploaded_style = envgeo_user_data.render_marker_style_controls(
        uploaded_df,
        "depth_profile",
        include_line=True,
        marker_size_default=10,
        marker_size_min=1,
        marker_size_step=1,
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
         ref_data, data_source_JAPAN_SEA, data_source_AROUND_JAPAN,
         uploaded_df=uploaded_df, uploaded_filter_key="depth_profile",
         uploaded_dataset_label=envgeo_utils.UPLOADED_DATA_LABEL,
     )
    filtered_profile_df = df1.copy()
    df1, uploaded_df = envgeo_utils.split_uploaded_rows(
        filtered_profile_df, envgeo_utils.UPLOADED_DATA_LABEL
    )

    # Count rows with both the selected parameter and depth; dD/d-excess can be sparse.
    # アップロード行だけの選択にも対応するため、絞り込み後の統合テーブルで数える。
    data_found = len(filtered_profile_df.dropna(subset=[X_data, "Depth_m"]))
    if data_found == 1:
        st.warning('Only one data point was found. A depth profile could not be meaningfully generated.')
        st.stop()
    if data_found == 0:
        st.warning(f"No valid {plot_element} depth-profile data are available for the selected conditions.")
        st.stop()
    

    # =============================================================================
    # Gap rows and coordinate aliases / 区切り空白行と座標列の別名
    # =============================================================================
    # Gap rows separate different station/date groups in connected profiles.
    # 空白行で異なる地点・年月日のグループを分離し、線が誤って接続されるのを防ぐ。
    df_original = envgeo_utils.insert_gap_rows(df_original)
    df1 = envgeo_utils.insert_gap_rows(df1)
    df1['lat'] = df1['Latitude_degN']
    df1['lon'] = df1['Longitude_degE']

    # =============================================================================
    # Figure controls / 図の表示設定
    # =============================================================================
    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "FIGURE_CONTROLS_LABEL", "Figure controls"))
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        
        if ref_data == data_source_JAPAN_SEA:
            fig_depth_min, fig_depth_max = st.slider(label='Depth range',
                                        min_value=0,
                                        max_value=1000,
                                        value=(0, 500),
                                        step = 50,
                                        )
            
        elif ref_data == data_source_AROUND_JAPAN:
            fig_depth_min, fig_depth_max = st.slider(label='Depth range',
                                        min_value=0,
                                        max_value=3500,
                                        value=(0, 2000),
                                        step = 50,
                                        )
            
        else:
            fig_depth_min, fig_depth_max = st.slider(label='Depth range',
                                        min_value=0,
                                        max_value=9000,
                                        value=(0, 2000),
                                        step = 50,
                                        )
            

        

        axis_margin = max(abs(fig_x_min), abs(fig_x_max)) * 0.5
        lim_min_X, lim_max_X = st.slider(
            label=f'Axis Scale for {plot_element}',
            min_value=float(fig_x_min - axis_margin),
            max_value=float(fig_x_max + axis_margin),
            value=(float(fig_x_min), float(fig_x_max)),
            step=0.1,
            key=f"depth_profile_axis_scale::{plot_element}::{ref_data}",
        )
      
            
            

        # Figure size / 図の寸法
        fig_size_col1, fig_size_col2 = st.columns(2)
        with fig_size_col1:
            sld_fig_size_min_X = st.number_input(
                "Figure width",
                min_value=1.0,
                max_value=40.0,
                value=8.0,
                step=0.5,
                key=f"depth_profile_figure_width::{plot_element}",
                help="Set the figure width in inches.",
            )
        with fig_size_col2:
            sld_fig_size_max_Y = st.number_input(
                "Figure height",
                min_value=1.0,
                max_value=40.0,
                value=12.0,
                step=0.5,
                key=f"depth_profile_figure_height::{plot_element}",
                help="Set the figure height in inches.",
            )
        
    
    
        # Font sizes / フォントサイズ
        font_col1, font_col2 = st.columns(2)
        with font_col1:
            sld_font_size_min_S = st.number_input(
                "Tick font size",
                min_value=4,
                max_value=40,
                value=16,
                step=1,
                key=f"depth_profile_tick_font_size::{plot_element}",
            )
        with font_col2:
            sld_font_size_max_L = st.number_input(
                "Label font size",
                min_value=4,
                max_value=40,
                value=20,
                step=1,
                key=f"depth_profile_label_font_size::{plot_element}",
            )
                    
                    
        # Tick counts / 目盛り数
        tick_col1, tick_col2 = st.columns(2)
        with tick_col1:
            tick_interval_min_X = st.number_input(
                "X tick count",
                min_value=4,
                max_value=40,
                value=11,
                step=1,
                key=f"depth_profile_x_tick_count::{plot_element}",
            )
        with tick_col2:
            tick_interval_max_Y = st.number_input(
                "Y tick count",
                min_value=4,
                max_value=40,
                value=11,
                step=1,
                key=f"depth_profile_y_tick_count::{plot_element}",
            )

        # Uploaded rows without month information / 月情報がないアップロード行
        show_nodata_uploaded = st.checkbox(
            "Show uploaded data without Month information",
            value=True,
            key=f"depth_profile_show_nodata_uploaded::{plot_element}",
            help=(
                "When the uploaded data has no Month column (or has rows with missing Month), "
                "show those lines using the fixed marker color. "
                "Uncheck to hide lines that cannot be colored by month."
            ),
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
    # Figure and title / 図と図題
    # -----------------------------------------------------------------------------
    fig = plt.figure(figsize=(sld_fig_size_min_X, sld_fig_size_max_Y), dpi=150)
    
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.rcParams["font.size"] = 16
    
    

    
    # Figure title and file name / 図題とファイル名
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

    # Wrap long filter conditions so saved figures remain readable.
    # 保存図では長いフィルター条件を折り返し、図幅内に収める。
    wrapped_sub_title = "\n".join(textwrap.wrap(sub_title, width=62))
    title_head = f"{main_title}\n{wrapped_sub_title}"
    title_head2 = title_head.replace('_', ' ')

    fig.suptitle(title_head2, fontsize=max(14, sld_font_size_max_L - 3), y=0.97)
    fig.subplots_adjust(top=0.87)
    
    

    
    
    
    
    
    
    
    # -----------------------------------------------------------------------------
    # Profile drawing constants / プロファイル描画の共通設定
    # -----------------------------------------------------------------------------
    ax_length = 5
    
    # -----------------------------------------------------------------------------
    # Depth profile / 深度プロファイル
    # -----------------------------------------------------------------------------
    X_Y = 1
    
    X_Y_C = "gray"
    X_Y_M = " "

    
    X_Y_add = 1

    
    
    alpha_all = 0.4
    alpha_selected = 1
    X_Y_C_add_each = 1

    # Invert the vertical axis so increasing depth is downward.
    # 深くなる方向が下になるよう、鉛直軸を反転する。
    lim_min_Y = fig_depth_max
    lim_max_Y = fig_depth_min
    

 
    
    # """ここからdepth"""
    if X_Y == 1:

        grid = plt.GridSpec(1,1, )
        ax = fig.add_subplot(grid[0, 0])
        
        ax.set_xlabel(X_label + iso_scale_X, fontsize=sld_font_size_max_L)
        ax.set_ylabel(Y_label + iso_scale_Y, fontsize=sld_font_size_max_L)  
    
        

        # 対象列と水深があるデータだけをプロットする。
        # dD / d-excessは欠損が多いため、除外数を表示してから空白行を挿入する。
        df_all_no_gap = df_original.dropna(how='all')
        df_all_valid = df_all_no_gap.dropna(subset=[X_data, Y_data]).reset_index(drop=True)
        excluded_all = len(df_all_no_gap) - len(df_all_valid)
        if excluded_all > 0:
            st.caption(
                f":gray[Background profile: {len(df_all_valid):,} samples plotted "
                f"and {excluded_all:,} excluded due to missing {plot_element} or depth values.]"
            )

        # 同じ地点，同じ年月日，はグループにして他は1行開ける
        df_fig_ALL = envgeo_utils.insert_gap_rows(df_all_valid)
        
        
        # 特定の列に特定の変数を持つ行と空白行を残す

        if plot_all_data == "Yes":
            plt.plot(df_fig_ALL[X_data], df_fig_ALL[Y_data],c=X_Y_C, marker=X_Y_M, lw=0.5, alpha=alpha_all, label='ALL')
        else:
            pass

        plt.legend(fontsize = 16) #

        ax.set_xlim(lim_min_X, lim_max_X) 
        ax.set_ylim(lim_min_Y, lim_max_Y) 

        plt.tick_params(labelsize=sld_font_size_min_S)

        ax.set_xticks(np.linspace(lim_min_X, lim_max_X,tick_interval_min_X))
        ax.set_yticks(np.linspace(lim_min_Y, lim_max_Y, tick_interval_max_Y))
        
        
        

        
        if plot_element == "d18O(VSMOW)":
            ax.xaxis.set_major_formatter(FormatStrFormatter("%+.1f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))

        elif plot_element in ["dD(VSMOW)", "d-excess"]:
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))

         
        elif plot_element == "Temperature (°C)":
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))

            lim_max_X = 30

            
        elif plot_element == "Salinity":
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))

        
        else:
            pass

            

        ax.tick_params(length=ax_length)

        plt.legend(fontsize = 20) # 凡例の数字のフォントサイズを設定
        
    
        
        #追加で強調プロットをする場合
        if X_Y_add == 1:
        
            if X_Y_C_add_each == 1:
                
 
                df_add_no_gap = df1.dropna(how='all')
                df_add_valid = df_add_no_gap.dropna(subset=[X_data, Y_data]).reset_index(drop=True)
                excluded_add = len(df_add_no_gap) - len(df_add_valid)
                if excluded_add > 0:
                    st.caption(
                        f":blue[Selected profile: {len(df_add_valid):,} samples plotted "
                        f"and {excluded_add:,} excluded due to missing {plot_element} or depth values.]"
                    )

                df_fig_add = df_add_valid    

                # 同じ地点，同じ年月日，はグループにして他は1行開ける
                df_fig_add = envgeo_utils.insert_gap_rows(df_fig_add)

                # Month-band colors / 月帯ごとの色分け
                lw_add = 0.6
                df13 = df_fig_add[(df_fig_add['Month'] >= 1) & (df_fig_add['Month'] <= 3)
                          | df_fig_add.isnull().all(axis=1)]  
                plt.plot(df13[X_data], df13[Y_data],c='blue', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='1-3')
                
                df46 = df_fig_add[(df_fig_add['Month'] >= 4) & (df_fig_add['Month'] <= 6)
                          | df_fig_add.isnull().all(axis=1)]  
                plt.plot(df46[X_data], df46[Y_data],c='green', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='4-6')
                
                df79 = df_fig_add[(df_fig_add['Month'] >= 7) & (df_fig_add['Month'] <= 9)
                          | df_fig_add.isnull().all(axis=1)]  
                plt.plot(df79[X_data], df79[Y_data],c='orange', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='7-9')
                df1012 = df_fig_add[(df_fig_add['Month'] >= 10) & (df_fig_add['Month'] <= 12)
                          | df_fig_add.isnull().all(axis=1)]  
                plt.plot(df1012[X_data], df1012[Y_data],c='purple', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='10-12')
                

    
                plt.legend(fontsize=sld_font_size_min_S)

                
        else:
            pass

        # --- Uploaded data overlay: month-colored dotted lines and markers ---
        uploaded_depth_plot_count = 0
        if not uploaded_df.empty and {X_data, "Depth_m"}.issubset(uploaded_df.columns):
            uploaded_depth = uploaded_df.copy()
            uploaded_depth[X_data] = pd.to_numeric(uploaded_depth[X_data], errors="coerce")
            uploaded_depth["Depth_m"] = pd.to_numeric(uploaded_depth["Depth_m"], errors="coerce")
            uploaded_depth = uploaded_depth.dropna(subset=[X_data, "Depth_m"])
            uploaded_depth_excluded = len(uploaded_df) - len(uploaded_depth)
            if not uploaded_depth.empty:
                lw_up = float(uploaded_style["line_width"])
                ls_up = uploaded_style["line_style"]
                alpha_up = float(uploaded_style["alpha"])
                marker_size_up = max(1.0, math.sqrt(float(uploaded_style["size"])))
                marker_kwargs_up = {
                    "marker": uploaded_style["marker"],
                    "markersize": marker_size_up,
                    "markeredgecolor": uploaded_style["outline_color"],
                    "markeredgewidth": uploaded_style["outline_width"],
                }
                MONTH_BANDS = [
                    ((1,  3), 'blue',   '1-3'),
                    ((4,  6), 'green',  '4-6'),
                    ((7,  9), 'orange', '7-9'),
                    ((10, 12), 'purple', '10-12'),
                ]
                site_cols_up = [c for c in ['Latitude_degN', 'Longitude_degE'] if c in uploaded_depth.columns]
                if site_cols_up:
                    uploaded_depth['_site_key'] = (
                        uploaded_depth[site_cols_up].round(4).apply(
                            lambda row: '_'.join(str(value) for value in row),
                            axis=1,
                        )
                    )
                else:
                    uploaded_depth['_site_key'] = 'all'
                has_month = (
                    'Month' in uploaded_depth.columns
                    and pd.to_numeric(uploaded_depth['Month'], errors='coerce').notna().any()
                )
                if has_month:
                    uploaded_depth['Month'] = pd.to_numeric(uploaded_depth['Month'], errors='coerce')
                    valid_month = uploaded_depth['Month'].between(1, 12)
                    for (m_min, m_max), color, label_str in MONTH_BANDS:
                        df_m = uploaded_depth[uploaded_depth['Month'].between(m_min, m_max)]
                        uploaded_depth_plot_count += len(df_m)
                        label_used = False
                        for _sk, grp in df_m.groupby('_site_key', sort=False):
                            g = grp.sort_values('Depth_m')
                            ax.plot(
                                g[X_data], g['Depth_m'],
                                c=color, lw=lw_up, ls=ls_up, alpha=alpha_up,
                                label=f'Uploaded {label_str}' if not label_used else '_nolegend_',
                                zorder=10,
                                **marker_kwargs_up,
                            )
                            label_used = True
                    if show_nodata_uploaded:
                        no_month = uploaded_depth[~valid_month]
                        if not no_month.empty:
                            uploaded_depth_plot_count += len(no_month)
                            label_used = False
                            for _sk, grp in no_month.groupby('_site_key', sort=False):
                                g = grp.sort_values('Depth_m')
                                ax.plot(
                                    g[X_data], g['Depth_m'],
                                    c=uploaded_style["color"], lw=lw_up, ls=ls_up, alpha=alpha_up,
                                    label='Uploaded (no month)' if not label_used else '_nolegend_',
                                    zorder=10,
                                    **marker_kwargs_up,
                                )
                                label_used = True
                else:
                    if show_nodata_uploaded:
                        uploaded_depth_plot_count = len(uploaded_depth)
                        label_used = False
                        for _sk, grp in uploaded_depth.groupby('_site_key', sort=False):
                            g = grp.sort_values('Depth_m')
                            ax.plot(
                                g[X_data], g['Depth_m'],
                                c=uploaded_style["color"], lw=lw_up, ls=ls_up, alpha=alpha_up,
                                label='Uploaded data' if not label_used else '_nolegend_',
                                zorder=10,
                                **marker_kwargs_up,
                            )
                            label_used = True
    else:
        pass
    

     

    
    # =============================================================================
    # Image download / 図のダウンロード
    # =============================================================================
    # Keep the PNG in memory; no local file is written. / PNGはメモリ上だけで生成する。
    safe_parameter_name = envgeo_utils.safe_filename_text(X_data)
    fn = envgeo_utils.build_figure_filename(f"Fig_depth_{safe_parameter_name}", sub_title)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
     
    st.pyplot(fig)

    if not uploaded_df.empty and {X_data, "Depth_m"}.issubset(uploaded_df.columns):
        _ud_excluded = len(uploaded_df) - uploaded_depth_plot_count
        st.caption(
            f":blue[Uploaded overlay: {uploaded_depth_plot_count:,} / "
            f"{len(uploaded_df):,} plotted"
            + (f" ({_ud_excluded:,} excluded due to missing values)." if _ud_excluded else ".")
            + "]"
        )

    btn = st.download_button(
       label="Download image",
       data=img,
       file_name=fn,
       mime="image/png"
       )
    
    
    
    
    

    # =============================================================================
    # Sampling-location map / 採取地点の地図
    # =============================================================================
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
            key="map_style_depth_profile",
            help=getattr(envgeo_utils, "MAP_STYLE_HELP_TEXT", "Choose the background map style for the sampling-location map."),
        )
    _eff_37, _fell_37 = envgeo_utils.resolve_map_mode(map_mode)
    if _fell_37:
        st.warning(envgeo_utils.OFFLINE_FALLBACK_WARNING)

    # -------------------------------------------------------------------------
    # Coordinate validation / 座標の検証
    # -------------------------------------------------------------------------
    # Keep rows only when both coordinates are numeric and geographically valid.
    # 緯度・経度が数値かつ地理的に有効な行だけを地図用に残す。
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

    # Uploaded coordinates support the map guard and automatic extent only.
    # Page 37 passes ``uploaded_df`` itself to the overlay helper.
    # アップロード座標は地図表示可否と表示範囲の計算だけに用いる。
    # 実際の重ね表示には ``uploaded_df`` をヘルパーへ渡す。
    uploaded_map_df = pd.DataFrame(columns=["Longitude_degE", "Latitude_degN"])
    if not uploaded_df.empty and {"Longitude_degE", "Latitude_degN"}.issubset(uploaded_df.columns):
        uploaded_map_df = uploaded_df.copy()
        uploaded_map_df["Longitude_degE"] = pd.to_numeric(
            uploaded_map_df["Longitude_degE"], errors="coerce"
        )
        uploaded_map_df["Latitude_degN"] = pd.to_numeric(
            uploaded_map_df["Latitude_degN"], errors="coerce"
        )
        uploaded_map_df = uploaded_map_df.dropna(subset=["Longitude_degE", "Latitude_degN"])
        uploaded_map_df = uploaded_map_df.loc[
            uploaded_map_df["Latitude_degN"].between(-90, 90)
            & uploaded_map_df["Longitude_degE"].between(-180, 180)
        ]

    # Uploaded data alone can provide a usable map when reference rows are empty.
    # 参照データが空でもアップロードデータだけで地図を表示できるようにする。
    _has_valid_map_coords = _has_valid_map_coords or not uploaded_map_df.empty

    # -------------------------------------------------------------------------
    # Map extent and zoom / 地図範囲とズーム
    # -------------------------------------------------------------------------
    # Default view: Japan / 初期表示: 日本
    default_lat, default_lon, default_zoom = 36.0, 138.0, 4.0

    if not _has_valid_map_coords:
        center_lat, center_lon, auto_zoom = default_lat, default_lon, default_zoom
    else:
        # Combine reference and uploaded coordinates for the map extent.
        # 参照・アップロード座標を合わせて表示範囲を求める。
        _map_ext_sources = []
        if len(_valid_coords_df) > 0:
            _map_ext_sources.append(_valid_coords_df[["Longitude_degE", "Latitude_degN"]])
        if not uploaded_map_df.empty:
            _map_ext_sources.append(uploaded_map_df[["Longitude_degE", "Latitude_degN"]])
        _map_ext_df = pd.concat(_map_ext_sources, ignore_index=True)
        lat_min, lat_max = _map_ext_df["Latitude_degN"].min(), _map_ext_df["Latitude_degN"].max()
        lon_min, lon_max = _map_ext_df["Longitude_degE"].min(), _map_ext_df["Longitude_degE"].max()

        if pd.isna(lat_min) or pd.isna(lon_min):
            center_lat, center_lon, auto_zoom = default_lat, default_lon, default_zoom
        else:
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2

            lat_diff = max(lat_max - lat_min, 0.1)
            lon_diff = max(lon_max - lon_min, 0.1)

            # Pixel-based zoom estimate, consistent with Page 03.
            # Page 03と整合するピクセル基準のズーム推定。
            map_width_px, map_height_px = 1200, 700
            zoom_lon = math.log2((map_width_px * 360) / (lon_diff * 256))
            zoom_lat = math.log2((map_height_px * 180) / (lat_diff * 256))

            # Leave a margin so broad east-west datasets remain visible.
            # 東西に広いデータも収まるように余白を確保する。
            auto_zoom = min(zoom_lon, zoom_lat) - 2.0
            auto_zoom = max(1, min(15, auto_zoom))

            # Use a world-scale fallback for very broad longitude coverage.
            # 経度範囲が非常に広い場合は世界規模の表示に切り替える。
            if lon_diff > 100:
                center_lat, center_lon, auto_zoom = default_lat, default_lon, 1.5

    # -------------------------------------------------------------------------
    # Sampling-location map / 採取地点地図
    # -------------------------------------------------------------------------
    if not _has_valid_map_coords:
        st.info(
            "Map view is unavailable because the selected data contain no valid "
            "latitude/longitude coordinates."
        )
    else:
        c_scale_profile = envgeo_utils.get_custom_colorscale(X_data)

        # Use reference coordinates first; otherwise render the uploaded points.
        # px.scatter_mapboxには必ず空でないデータフレームを渡す。
        _map_plot_df = (
            _valid_coords_df if not _valid_coords_df.empty else uploaded_map_df
        )
        _x_color = X_data if X_data in _map_plot_df.columns else None
        hover_columns = [
            "Latitude_degN",
            "Longitude_degE",
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
        ]
        hover_data = {column: True for column in hover_columns if column in _map_plot_df.columns}

        fig_map = px.scatter_mapbox(
            _map_plot_df,
            lat="Latitude_degN",
            lon="Longitude_degE",
            color=_x_color,
            color_continuous_scale=c_scale_profile,
            hover_data=hover_data,
            opacity=0.6,
            height=500,
        )

        # Background style and offline overlays / 背景スタイルとオフライン重ね表示
        fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)
        envgeo_utils.add_coastline_overlay(fig_map)
        if _eff_37 == "Coastline (offline)":
            envgeo_utils.add_graticule_overlay(fig_map)

        _depth_color_range = (
            (lim_min_X, lim_max_X) if lim_min_X < lim_max_X else None
        )
        fig_map, uploaded_map_count = envgeo_user_data.add_uploaded_map_overlay(
            fig_map,
            uploaded_df,
            uploaded_style,
            color_column=X_data,
            colorscale=c_scale_profile,
            color_range=_depth_color_range,
            show_nodata=show_nodata_uploaded,
        )

        # Figure layout / 図のレイアウト
        fig_map.update_layout(
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=auto_zoom,
                domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            ),
            margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
            autosize=True,
            coloraxis_colorbar=dict(
                title=plot_element,
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

        # Render an explicitly keyed chart with mouse-wheel zoom enabled.
        # 一意のキーを指定し、マウスホイールによるズームを有効にして表示する。
        st.plotly_chart(
            fig_map,
            key="depth_profile",
            config={'scrollZoom': True, 'displayModeBar': True},
            **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
        )

        if not uploaded_df.empty:
            if uploaded_map_count:
                st.caption(
                    f":blue[Uploaded locations: {uploaded_map_count:,} / "
                    f"{len(uploaded_df):,} plotted on the map.]"
                )
            else:
                st.caption(
                    ":orange[Uploaded data: no rows with valid Latitude_degN "
                    "and Longitude_degE found.]"
                )

    # =============================================================================
    # Filtered-data table / 絞り込みデータ表
    # =============================================================================
    with st.expander("Filtered dataset for current view (CSV)", expanded=False):
        table_columns = [
            'reference',
            'Cruise',
            'Station',
            'Date',
            'Year',
            'Month',
            'Longitude_degE',
            'Latitude_degN',
            'Depth_m',
            'Temperature_degC',
            'Salinity',
            'd18O',
            'dD',
            'd-excess',
        ]
        df1_table = df_fig_add[[column for column in table_columns if column in df_fig_add.columns]].copy()
        # Remove gap rows before CSV display. / CSV表示前に区切り空白行を除外する。
        df1_table = df1_table.dropna(how='all')
        # Convert to text immediately before display for Arrow compatibility.
        # Arrow互換性のため、表示直前に文字列へ変換する。
        df1_table = df1_table.astype(str)

        st.dataframe(df1_table, **envgeo_utils.stretch_width_kwargs(st.dataframe))

if __name__ == '__main__':
    main()
    
