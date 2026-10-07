#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Salinity–δ18O relationship visualizer for EnvGeo-Seawater data.

EnvGeo-Seawater データの塩分–δ18O関係可視化ページです。

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score

import envgeo_user_data
import envgeo_utils

def main():
    # =============================================================================
    # Page header / ページ見出し
    # =============================================================================
    st.header(f'Salinity-δ18O Relationship ({version})')
    st.button('Reload')

    # =============================================================================
    # Data-source selection / データソースの選択
    # =============================================================================
    data_source_ENVGEO = envgeo_utils.data_source_ENVGEO
    data_source_AROUND_JAPAN = envgeo_utils.data_source_AROUND_JAPAN
    data_source_GLOBAL = envgeo_utils.data_source_GLOBAL
    ref_data = st.radio("Data source (see Home > About)", (data_source_ENVGEO, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True)

    # -----------------------------------------------------------------------------
    # Figure options / 図の表示オプション
    # -----------------------------------------------------------------------------

    col1, col2 = st.columns([1,1])
    with col1:
        plot_all_data = st.radio(
            "Show background data",
            ("Yes", "No"),
            index=1,
            horizontal=True,
            help=getattr(envgeo_utils, "BACKGROUND_DATA_HELP_TEXT", "Show the unfiltered dataset behind the currently filtered data for context."),
        )
    
    with col2:
        plot_reg_lines = st.radio(
            "Regression line",
            ("Yes", "No"),
            horizontal=True,
            help=getattr(envgeo_utils, "REGRESSION_HELP_TEXT", "Add a simple least-squares regression line for quick visual reference."),
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
        == "31_Salinity-d18O_Relationship.py"
    )
    if embedded_in_integrated:
        uploaded_df = envgeo_utils.get_uploaded_data()
    else:
        uploaded_df = envgeo_user_data.render_upload_panel(
            "sal_d18o",
            "The salinity-d18O overlay requires salinity and d18O columns.",
        )
    uploaded_df = envgeo_user_data.render_column_controls(
        uploaded_df,
        {
            "Salinity column": "Salinity",
            "d18O column": "d18O",
        },
        "sal_d18o",
    )
    uploaded_style = envgeo_user_data.render_marker_style_controls(
        uploaded_df,
        "sal_d18o",
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
             df_original, uploaded_df
         ),
         ref_data, data_source_ENVGEO, data_source_AROUND_JAPAN,
         uploaded_df=uploaded_df, uploaded_filter_key="sal_d18o",
         uploaded_dataset_label=envgeo_utils.UPLOADED_DATA_LABEL,
     )
    # Keep the sidebar-filtered integrated table for calculations and download.
    # 前景にアップロード行を重ね描きするため、表示用の参照データと分割する。
    filtered_integrated_df = df1.copy()
    df1, uploaded_df = envgeo_utils.split_uploaded_rows(
        filtered_integrated_df, envgeo_utils.UPLOADED_DATA_LABEL
    )
    loaded_user_excel_count = int(
        df_original["Dataset"].eq(envgeo_utils.USER_EXCEL_DATA_LABEL).sum()
    )
    filtered_user_excel_count = int(
        filtered_integrated_df["Dataset"]
        .eq(envgeo_utils.USER_EXCEL_DATA_LABEL)
        .sum()
    )
    if loaded_user_excel_count and not filtered_user_excel_count:
        st.warning(
            f"{envgeo_utils.USER_EXCEL_DATA_LABEL}: {loaded_user_excel_count:,} rows "
            "were loaded, but 0 remain after the current Data filtering settings."
        )
    # Regression requires at least two data points. / 回帰には2点以上が必要です。
    data_found = len(filtered_integrated_df["d18O"])
    if data_found == 1:
        st.warning('Only one data point was found. Regression analysis could not be performed.')
        st.stop()

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

        sal_d18o_color_candidates = [
            "Single color",
            "Depth_m",
            "Latitude_degN",
            "Longitude_degE",
            "Year",
            "Month",
            "d18O",
            "dD",
            "d-excess",
            "Salinity",
            "Temperature_degC",
        ]
        sal_d18o_color_options = [
            item for item in sal_d18o_color_candidates
            if item == "Single color" or item in filtered_integrated_df.columns
        ]
        sal_d18o_color_by = st.selectbox(
            "Color parameter",
            sal_d18o_color_options,
            index=0,
            help=(
                "Use a single blue marker color, or color the filtered "
                "salinity-d18O data by a numeric parameter."
            ),
        )

        sal_d18o_color_range = None
        if sal_d18o_color_by != "Single color":
            sal_d18o_matplotlib_colormap = envgeo_utils.get_matplotlib_colormap(sal_d18o_color_by)
            sal_d18o_color_source = pd.to_numeric(
                filtered_integrated_df[sal_d18o_color_by], errors="coerce"
            ).dropna()
            if not sal_d18o_color_source.empty:
                sal_d18o_color_min = float(sal_d18o_color_source.min())
                sal_d18o_color_max = float(sal_d18o_color_source.max())

                if sal_d18o_color_min == sal_d18o_color_max:
                    sal_d18o_color_range = (sal_d18o_color_min, sal_d18o_color_max)
                    st.caption(f"Colorbar range: {sal_d18o_color_min:g}")
                elif sal_d18o_color_by in ["Year", "Month"]:
                    sal_d18o_color_range = st.slider(
                        "Color range",
                        min_value=int(np.floor(sal_d18o_color_min)),
                        max_value=int(np.ceil(sal_d18o_color_max)),
                        value=(int(np.floor(sal_d18o_color_min)), int(np.ceil(sal_d18o_color_max))),
                        step=1,
                    )
                else:
                    color_step = 10.0 if sal_d18o_color_by == "Depth_m" else 0.1
                    sal_d18o_color_range = st.slider(
                        "Color range",
                        min_value=float(np.floor(sal_d18o_color_min)),
                        max_value=float(np.ceil(sal_d18o_color_max)),
                        value=(
                            float(np.floor(sal_d18o_color_min)),
                            float(np.ceil(sal_d18o_color_max)),
                        ),
                        step=color_step,
                    )
            else:
                st.caption(f"No valid {sal_d18o_color_by} values are available for the colorbar.")
        else:
            sal_d18o_matplotlib_colormap = None
    
        # Axis ranges / 軸の表示範囲
        if ref_data == data_source_GLOBAL:
          sal_min, sal_max = st.slider(label='Salinity scale',
                                      min_value=0,
                                      max_value=42,
                                      value=(0, 42),
                                      )
          
          d18O_min, d18O_max = st.slider(label=r'$\delta^{18}$O scale',
                                      min_value=-22.0,
                                      max_value=8.0,
                                      value=(-22.0, 8.0),
                                      )

        else:

            sal_min, sal_max = st.slider(label='Salinity scale',
                                        min_value=0,
                                        max_value=42,
                                        value=(20, 36),
                                        )
            
            d18O_min, d18O_max = st.slider(label=r'$\delta^{18}$O scale',
                                        min_value=-25.0,
                                        max_value=5.0,
                                        value=(-5.0, 1.0),
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
                key="sal_d18o_fig_width_x",
                help="Adjust the width of the Matplotlib salinity-d18O figure.",
            )
        with fig_size_col2:
            sld_fig_size_y = st.number_input(
                "Fig height (y)",
                min_value=4,
                max_value=24,
                value=9,
                step=1,
                key="sal_d18o_fig_height_y",
                help="Adjust the height of the Matplotlib salinity-d18O figure.",
            )

        font_col1, font_col2 = st.columns(2)
        with font_col1:
            sld_font_size_tick = st.number_input(
                "Tick font size",
                min_value=6,
                max_value=32,
                value=15,
                step=1,
                key="sal_d18o_tick_font_size",
                help="Adjust the tick-label font size.",
            )
        with font_col2:
            sld_font_size_label = st.number_input(
                "Label font size",
                min_value=6,
                max_value=32,
                value=16,
                step=1,
                key="sal_d18o_label_font_size",
                help="Adjust the axis-label and colorbar-label font size.",
            )

        tick_col1, tick_col2 = st.columns(2)
        with tick_col1:
            tick_count_x = st.number_input(
                "X tick count",
                min_value=3,
                max_value=30,
                value=9,
                step=1,
                key="sal_d18o_x_tick_count",
                help="Adjust the number of major tick marks on the salinity axis.",
            )
        with tick_col2:
            tick_count_y = st.number_input(
                "Y tick count",
                min_value=3,
                max_value=30,
                value=9,
                step=1,
                key="sal_d18o_y_tick_count",
                help="Adjust the number of major tick marks on the d18O axis.",
            )

        # Uploaded rows missing the color parameter / 色分け値がないアップロード行
        show_nodata_uploaded = st.checkbox(
            f"Show uploaded points without {sal_d18o_color_by} values",
            value=True,
            key="sal_d18o_show_nodata_uploaded",
            help="Show or hide uploaded data points that have no value for the selected color parameter.",
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

    uploaded_sal_d18o = pd.DataFrame()
    if not uploaded_df.empty and {"Salinity", "d18O"}.issubset(uploaded_df.columns):
        uploaded_sal_d18o = uploaded_df.dropna(
            subset=["Salinity", "d18O"]
        ).reset_index(drop=True)
        uploaded_excluded_count = len(uploaded_df) - len(uploaded_sal_d18o)
        st.caption(
            f":blue[Uploaded overlay: {len(uploaded_sal_d18o):,} / "
            f"{len(uploaded_df):,} plotted ({uploaded_excluded_count:,} excluded "
            "due to missing or invalid salinity/d18O).]"
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
    # Salinity–δ18O plot definition / 塩分–δ18O図の定義
    # -----------------------------------------------------------------------------
    X_data = "Salinity"
    Y_data = "d18O"
    

    X_label = "salinity"
    Y_label = r"$\delta^{18}$O"
    

    iso_scale_X = ""
    iso_scale_Y = "(VSMOW)"
    
    
    
    # Existing plot switches / 既存の描画切替
    X_Y = 1
    
    X_Y_C = "red"
    X_Y_M = "."
    X_Y_S = 100
    

    
    
    
    X_Y_add2 = 1

    
    selected_row = "Transect"
    
    
    
    fig_title_X_Y = X_label + " - " + Y_label
    sheet_names_add2 = "filtered data"
    alpha_all = 0.2
    X_Y_C_add = "blue"
    X_Y_C_add_each = 1

    # Axis limits / 軸範囲
    lim_min_X = sal_min
    lim_max_X = sal_max
    lim_min_Y = d18O_min
    lim_max_Y = d18O_max
    
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
            df_fig_ALL = df_original.dropna(subset=["Salinity", "d18O"]).reset_index(drop=True)

            excluded_count = len(df_original) - len(df_fig_ALL)
            if excluded_count > 0:
                st.caption(f":red[Background plot: {len(df_fig_ALL):,} / {len(df_original):,} plotted ({excluded_count:,} excluded due to missing d18O/salinity).]")
                    

    
            Ya = df_fig_ALL[Y_data]
            Xa = df_fig_ALL[X_data]
            

    
            d_select_main_sum = df_fig_ALL[selected_row].count().sum()

    
            ax.scatter(Xa, Ya, s=X_Y_S, c=X_Y_C, marker=X_Y_M, lw=0.5, ec="black", alpha=alpha_all)
        else:
            pass

        ax.set_xlim(lim_min_X, lim_max_X)
        ax.set_ylim(lim_min_Y, lim_max_Y)
        ax.set_xticks(np.linspace(lim_min_X, lim_max_X, tick_count_x))
        ax.set_yticks(np.linspace(lim_min_Y, lim_max_Y, tick_count_y))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%+.1f"))
        ax.tick_params(labelsize=sld_font_size_tick)
        ax.tick_params(length=ax_length)

        plt.title(fig_title_X_Y)

        
        
        
        
        if plot_all_data == "Yes":
    
            # Background regression / 背景データの回帰
            if plot_reg_lines == "Yes":
                coef = np.polyfit(Xa, Ya, 1)
                y1 = np.poly1d(coef)(Xa)
                plt.plot(Xa, y1, label='regression line (ALL)', c=X_Y_C)
            
                reg_line = 'ALL:  y' + ' = ' + '{:.2f}'.format(coef[0]) + 'x ' +' + (' + '{:.2f}'.format(coef[1]) 
                line_r = np.corrcoef(Xa, Ya)
            
                ax.text(0.99, 0.05+0.01, reg_line + ")   (R=" + '{:.2f}'.format(line_r[0,1])+', N=' + str(d_select_main_sum)+')', horizontalalignment='right', transform=ax.transAxes, fontsize=max(8, sld_font_size_tick - 3))
        
  
        
            else:
                pass
        else:
            pass
        
        
        
        
  
        
        # -------------------------------------------------------------------------
        # Filtered data and regression / 絞り込みデータと回帰
        # -------------------------------------------------------------------------

        selected_regression_available = False
        if X_Y_add2 == 1:
        
            if X_Y_C_add_each == 1:
                X_Y_C_add = 'blue'
                # Rows need both plot axes and, when selected, the color variable.
                # 両軸と、選択時には色分け変数を満たす行だけを描く。
                filtered_required_columns = ["Salinity", "d18O"]
                if sal_d18o_color_by != "Single color":
                    filtered_required_columns.append(sal_d18o_color_by)
                # Regression uses the same integrated table that the sidebar filtered.
                # 回帰には、サイドバーで絞り込んだ統合テーブルを用いる。
                df_fig_add = filtered_integrated_df.dropna(
                    subset=filtered_required_columns
                ).reset_index(drop=True)

                excluded_count_add = len(filtered_integrated_df) - len(df_fig_add)
                if excluded_count_add > 0:
                    missing_label = "d18O/salinity"
                    if sal_d18o_color_by != "Single color":
                        missing_label = f"d18O/salinity/{sal_d18o_color_by}"
                    st.caption(f":blue[Filtered plot: {len(df_fig_add):,} / {len(filtered_integrated_df):,} plotted ({excluded_count_add:,} excluded due to missing {missing_label}).]")

                if filtered_user_excel_count:
                    user_excel_plot_count = int(
                        df_fig_add["Dataset"]
                        .eq(envgeo_utils.USER_EXCEL_DATA_LABEL)
                        .sum()
                    )
                    if user_excel_plot_count < filtered_user_excel_count:
                        st.warning(
                            f"{envgeo_utils.USER_EXCEL_DATA_LABEL}: "
                            f"{user_excel_plot_count:,} / {filtered_user_excel_count:,} "
                            "filtered rows can be plotted. This figure requires both "
                            "Salinity and d18O values."
                        )
                        

                

                Y_add = df_fig_add[Y_data]
                X_add = df_fig_add[X_data]
                d_select_add2_sum = filtered_integrated_df[selected_row].count().sum()

                
                if sal_d18o_color_by != "Single color" and sal_d18o_color_range is not None:
                    color_values = pd.to_numeric(df_fig_add[sal_d18o_color_by], errors="coerce")
                    filtered_scatter = ax.scatter(
                        X_add,
                        Y_add,
                        s=X_Y_S,
                        c=color_values,
                        cmap=sal_d18o_matplotlib_colormap,
                        vmin=sal_d18o_color_range[0],
                        vmax=sal_d18o_color_range[1],
                        marker=X_Y_M,
                        alpha=alpha_selected,
                        lw=0.5,
                        ec="black",
                        label=sheet_names_add2,
                    )
                    cbar = fig.colorbar(filtered_scatter, ax=ax, pad=0.02, fraction=0.045)
                    cbar.set_label(sal_d18o_color_by, fontsize=sld_font_size_label)
                    cbar.ax.tick_params(labelsize=sld_font_size_tick)
                else:
                    ax.scatter(X_add, Y_add, s=X_Y_S,c=X_Y_C_add,marker=X_Y_M, alpha=alpha_selected,lw=0.5, ec="black", label= sheet_names_add2)

                
                if (
                    plot_reg_lines == "Yes"
                    and len(X_add) >= 2
                    and X_add.nunique() > 1
                ):
                    # Selected-data regression / 選択データの回帰
                    coef_add = np.polyfit(X_add, Y_add, 1)
                    y1_add = np.poly1d(coef_add)(X_add)
                    plt.plot(X_add, y1_add, label='regression line (' + sheet_names_add2 +')', c=X_Y_C_add,)
                
                    reg_line_add = sheet_names_add2 + ':  y' + ' = ' + '{:.2f}'.format(coef_add[0]) + 'x ' +' + (' + '{:.2f}'.format(coef_add[1]) 
                    line_r_add = np.corrcoef(X_add, Y_add)
                    selected_regression_available = True
                
                    ax.text(0.99, 0.05*3+0.01, reg_line_add + ")   (R=" + '{:.2f}'.format(line_r_add[0,1])+', N=' + str(d_select_add2_sum)+')', horizontalalignment='right', transform=ax.transAxes, fontsize=max(8, sld_font_size_tick - 3))
                
                
       
                elif plot_reg_lines == "Yes":
                    st.caption(
                        ":gray[Regression line was skipped because fewer than "
                        "two valid selected data points are available.]"
                    )
                
            else:
                pass
            
        else:
            pass
    
    
    
    

    
    
    

        
        # -------------------------------------------------------------------------
        # Regression statistics / 回帰統計量
        # -------------------------------------------------------------------------
        if plot_reg_lines == "Yes": 
        
            if plot_all_data == "Yes" and "coef" in locals():
                Y_all_pred = coef[0]*Xa + coef[1]

                MSE_all = mean_squared_error(Ya, Y_all_pred)
                RMES_all = np.sqrt(mean_squared_error(Ya, Y_all_pred))

                R2_all =  r2_score(Ya, Y_all_pred)  
                
                ax.text(0.99, 0+0.01, 'RMSE_all: ' + '{:.3f}'.format(RMES_all)+', R$^{2}$_all: ' + '{:.2f}'.format(R2_all), horizontalalignment='right', transform=ax.transAxes, fontsize=max(8, sld_font_size_tick - 3), c='red')
            else:
                pass
                
            
            
            if selected_regression_available:
                Y_add_pred = coef_add[0]*X_add + coef_add[1]

                MSE_add = mean_squared_error(Y_add, Y_add_pred)
                RMES_add = np.sqrt(mean_squared_error(Y_add, Y_add_pred))

                R2_add =  r2_score(Y_add, Y_add_pred)

                ax.text(0.99, 0.05*2+0.01, 'RMSE_add: ' + '{:.3f}'.format(RMES_add)+', R$^{2}$_add: ' + '{:.2f}'.format(R2_add), horizontalalignment='right', transform=ax.transAxes, fontsize=max(8, sld_font_size_tick - 3), c='blue')
        
        else:
            pass
    
        # -------------------------------------------------------------------------
        # Uploaded overlay / アップロードデータの重ね表示
        # -------------------------------------------------------------------------
        if not uploaded_sal_d18o.empty:
            use_shared_colorbar = (
                uploaded_style["color_mode"] == "Use current colorbar when possible"
                and sal_d18o_color_by != "Single color"
                and sal_d18o_color_range is not None
                and sal_d18o_color_by in uploaded_sal_d18o.columns
            )
            uploaded_color_valid = pd.Series(
                False,
                index=uploaded_sal_d18o.index,
            )
            if use_shared_colorbar:
                uploaded_color_values = pd.to_numeric(
                    uploaded_sal_d18o[sal_d18o_color_by],
                    errors="coerce",
                )
                uploaded_color_valid = uploaded_color_values.notna()
                if uploaded_color_valid.any():
                    ax.scatter(
                        uploaded_sal_d18o.loc[uploaded_color_valid, "Salinity"],
                        uploaded_sal_d18o.loc[uploaded_color_valid, "d18O"],
                        s=uploaded_style["size"],
                        c=uploaded_color_values[uploaded_color_valid],
                        cmap=sal_d18o_matplotlib_colormap,
                        vmin=sal_d18o_color_range[0],
                        vmax=sal_d18o_color_range[1],
                        marker=uploaded_style["marker"],
                        alpha=uploaded_style["alpha"],
                        linewidths=uploaded_style["outline_width"],
                        edgecolors=uploaded_style["outline_color"],
                        label="Uploaded data",
                        zorder=20,
                    )

            fixed_color_rows = ~uploaded_color_valid
            if fixed_color_rows.any() and (not use_shared_colorbar or show_nodata_uploaded):
                ax.scatter(
                    uploaded_sal_d18o.loc[fixed_color_rows, "Salinity"],
                    uploaded_sal_d18o.loc[fixed_color_rows, "d18O"],
                    s=uploaded_style["size"],
                    c=uploaded_style["color"],
                    marker=uploaded_style["marker"],
                    alpha=uploaded_style["alpha"],
                    linewidths=uploaded_style["outline_width"],
                    edgecolors=uploaded_style["outline_color"],
                    label=(
                        f"Uploaded data (no {sal_d18o_color_by})"
                        if use_shared_colorbar
                        else "Uploaded data"
                    ),
                    zorder=20,
                )
    
    

    

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
    fn = envgeo_utils.build_figure_filename("Fig_sal_d18O_SW", main_title2)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
     
    st.pyplot(fig)
    plt.close(fig)

    btn = st.download_button(
       label="Download image",
       data=img,
       file_name=fn,
       mime="image/png")
   

    

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
            key="map_style_31_auto",
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
    # Map drawing / 地図の描画
    # -----------------------------------------------------------------------------
    if not _has_valid_map_coords:
        st.info(
            "Map view is unavailable because the selected data contain no valid "
            "latitude/longitude coordinates."
        )
    else:
        c_scale_d18o = envgeo_utils.get_custom_colorscale("d18O")

        fig_map = px.scatter_mapbox(
            _valid_coords_df,
            lat="Latitude_degN",
            lon="Longitude_degE",
            color="d18O",
            color_continuous_scale=c_scale_d18o,
            hover_data={
                "Latitude_degN": True,
                "Longitude_degE": True,
                "d18O": True,
                "dD": True,
                "Salinity": True,
                "Temperature_degC": True,
                "Year": True,
                "Month": True,
                "Day": True,
                "Cruise": True,
                "Station": True,
                "Depth_m": True,
                "reference": True,
            },
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
            key="sal_d18O_plot",
            config={'scrollZoom': True, 'displayModeBar': True},
        )

    # =============================================================================
    # Filtered-data table / 絞り込みデータ表
    # =============================================================================
    # Show every row that passed filtering, including rows without this figure's axes.
    # 図の軸に欠損がある行も含め、絞り込みを通過した全行を表示・ダウンロードする。
    envgeo_utils.display_isotope_table(filtered_integrated_df)
    

if __name__ == '__main__':
    main()
    

    
    
    
