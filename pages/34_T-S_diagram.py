#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature–salinity diagram visualizer for EnvGeo-Seawater data.

Created: 2023-04-22
Author: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""


# --- Version info ---
version = "1.3.2"  # 2026-09-22

# ToDo




fig_title = "envgeo-seawater-database"  
    
    
    
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import plotly.express as px
import math
import gsw
import io
import envgeo_utils  
import envgeo_user_data

def main():
    
    # タイトル
    st.header(f'Temperature-Salinity Diagram ({version})')
    

    # リロードボタン
    st.button('Reload')
    
 

    ##############################################################################
    # データソースの変数、envgeo_utilsから読み出す
    ##############################################################################
    data_source_JAPAN_SEA = envgeo_utils.data_source_JAPAN_SEA
    data_source_AROUND_JAPAN = envgeo_utils.data_source_AROUND_JAPAN
    data_source_GLOBAL = envgeo_utils.data_source_GLOBAL
    

    ##############################################################################
    # データソース選択
    ##############################################################################
    ref_data = st.radio("Data source (see Home > About):", (data_source_JAPAN_SEA, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True)


    ##############################################################################
    # データソース選択
    ##############################################################################

    # 全データと凡例を表示するかどうか
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
    


    ##############################################################################
    # 選択したデータセットの文献表示
    ##############################################################################
    
    if ref_data == data_source_JAPAN_SEA:
        st.write(envgeo_utils.refs_JAPAN_SEA)
        
    elif ref_data == data_source_AROUND_JAPAN:
        st.write(envgeo_utils.refs_AROUND_JAPAN)
      
    elif ref_data == data_source_GLOBAL:
        st.write(envgeo_utils.refs_GLOBAL)
        
    else:
        st.warning("Invalid data source selection.")


    ##############################################################################
    # envgeo_utilsからデータフレーム読み込み
    ##############################################################################
    df_original = envgeo_utils.load_isotope_data(ref_data) # フィルターしないデータ
    df1 = df_original # このあとフィルターするデータ

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



    ##############################################################################
    # サイドバーここから　　df1フィルタリング　も一括で
    ##############################################################################

    # envgeo_utils を使って一括フィルタリングとサイドバー生成
    # すべての変数を順番通りに受け取る
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
         uploaded_df=uploaded_df, uploaded_filter_key="ts_diagram",
         uploaded_dataset_label=envgeo_utils.UPLOADED_DATA_LABEL,
     )
    df1, uploaded_df = envgeo_utils.split_uploaded_rows(
        df1, envgeo_utils.UPLOADED_DATA_LABEL
    )

    # データが一つだけの時に警告　近似直線を引くなどの必要がある図の場合のみ使用，d18Oなどは適宜変更
    data_found = len(df1["d18O"])
    if data_found == 1:
        st.warning('Only one data point was found. A T–S diagram could not be meaningfully generated.')
        st.stop()
    


    ##############################################################################
    # 後半の定義用
    ##############################################################################
        
    df1['lat'] = df1['Latitude_degN']
    df1['lon'] = df1['Longitude_degE']
    # df1['Depth_m'] = df1['Depth_m']*(-1) # 以前の3D-4Dでは，depthをマイナス表示にしてた


    ##############################################################################
    # 図のスケール変更
    ##############################################################################
   

    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "FIGURE_CONTROLS_LABEL", "Figure controls"))
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        
        # マーカーの問明度調整
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
        
        # --- 図のスケール設定  ---
        if ref_data == data_source_JAPAN_SEA:
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

        # --- Matplotlib figure appearance ---
        # Depth Profileと同じ考え方で、論文図向けの見た目をページ上で調整する。
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

        # アップロードデータのうちカラーバー要素が無いポイントの表示切替
        show_nodata_uploaded = st.checkbox(
            f"Show uploaded points without {ts_color_by} values",
            value=True,
            key="ts_diagram_show_nodata_uploaded",
            help="Show or hide uploaded data points that have no value for the T-S color parameter.",
        )


    ##############################################################################
    # キャッシュクリア
    ##############################################################################
        
    # キャッシュのクリア　サイドバーの一番下などに配置
    if st.sidebar.button("🔄 Clear cache"):
        envgeo_utils.clear_app_cache()
        st.rerun() # アプリを再実行して最新のExcelを読み込ませる




        
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################


    ##############################################################################
    #  ここから図の設定と描画
    ##############################################################################


    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
    
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
    
    
    

    
    ############################################
    ######      font size line etc..       #####
    ############################################
    
    plt.rcParams["font.size"] = sld_font_size_tick
    fig_size = [sld_fig_size_x, sld_fig_size_y] #図のサイズ
    fig_dpi = 150 #図の解像度
    ax_length = 15
    

    
    ########################################
    ######    FIG: x vs y    #####
    ########################################
    
    
    X_data = "Salinity"
    Y_data = "Temperature_degC"
    
    X_label = "salinity"
    Y_label = "Temperature"
    
    iso_scale_X = ""
    iso_scale_Y = "(C)"
    

    ###### 別途，X_Yのプロットをするかどうか,色を一括にするか ######
    #する場合は1,しない場合は2
    X_Y = 1
    
    #メインプロットの設定
    X_Y_C = "red" #色の設定
    X_Y_M = "." #現時点で色は変更設定なしマーカーの種類
    X_Y_S = 100
    

  
    
    #タイトル
    fig_title_X_Y= X_label + " - "+ Y_label + "" # 書き出し専用
    
    #追加データのlabel
    sheet_names_add2 = "filtered data"
    
    #プロットの透明度
    alpha_all = 0.2 #メインプロット
    
    #強調プロットの色の指定
    X_Y_C_add =  "blue" 

    
    #############################################
    ######      data range for  FIG         #####
    #############################################
    

    lim_min_X = sal_min
    lim_max_X = sal_max
    lim_min_Y = temp_min
    lim_max_Y = temp_max
    
    

    
    
    
    ############################################
    ######      　　　設定ここまで！！　　       #####
    ############################################
    
    
    
    

    

    ##############################################################################
    # 全データを背景にプロット
    ##############################################################################

    
    # 全データプロット
    if X_Y == 1:

        fig = plt.figure(figsize = (fig_size),dpi=fig_dpi)
        ax = plt.subplot(111)
    
        ax.set_xlabel(X_label + iso_scale_X, fontsize=sld_font_size_label)
        ax.set_ylabel(Y_label + iso_scale_Y, fontsize=sld_font_size_label)  

        if plot_all_data == "Yes":
            ax.scatter(-1000, -1000, s=X_Y_S,c=X_Y_C,marker=X_Y_M, alpha=alpha_all, label='ALL') #凡例等のダミー
        else:
            pass
        

        
        # --------------------------------------
        # 全プロット用のデータフレーム読み込みと整理
        # --------------------------------------
        # 塩分と温度が無いデータを削除
        
        df_fig_ALL = df_original.dropna(subset=["Salinity", "Temperature_degC"]).reset_index(drop=True)
        
        # 排除したサンプル数を計算（オプション：前述の英語メッセージなどで使う用）
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
                plt.legend(fontsize = sld_font_size_tick) # 凡例の数字のフォントサイズを設定
        else:
            pass
            
            
        plt.title(fig_title_X_Y) 
        
    

        ##############################################################################
        # フィルターしたデータを重ね書き
        ##############################################################################


                
        X_Y_C_add  =  'blue' #カラーを選ぶ
        
        df_fig_add = df_original
        


        # --------------------------------------
        # フィルターデータ用のデータフレーム読み込みと整理
        # --------------------------------------
        # 塩分とd18Oが無いデータを削除
        df_fig_add = df1.dropna(subset=["Salinity", "Temperature_degC"]).reset_index(drop=True)

        # 排除したサンプル数を計算（オプション：前述の英語メッセージなどで使う用）
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
                plt.legend(fontsize = sld_font_size_tick) # 凡例の数字のフォントサイズを設定
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
                plt.legend(fontsize = sld_font_size_tick) # 凡例の数字のフォントサイズを設定

 
        ##############################################################################
        # 密度曲線を描く
        ##############################################################################
            
        #######   密度曲線を描く ###########
        # data=pd.read_excel(excel_file, sheet_name=sheet_num)
        ts=df_fig_ALL[['Temperature_degC', 'Salinity']]
        df=ts.sort_values('Temperature_degC',ascending=True)
        mint=np.min(df['Temperature_degC'])
        maxt=np.max(df['Temperature_degC'])
        mins=np.min(df['Salinity'])
        maxs=np.max(df['Salinity'])
        tempL=np.linspace(mint-5,maxt+5)
        salL=np.linspace(mins-5,maxs+5)
        Tg, Sg = np.meshgrid(tempL,salL)
        sigma_theta = gsw.sigma0(Sg, Tg)

        
        cs = ax.contour(Sg, Tg, sigma_theta, colors='lightgrey', linestyles='dashed', zorder=0, levels=50)
        

        
        plt.clabel(cs,fontsize=sld_font_size_tick,inline=True,fmt='%.1f',zorder=0, )

        ##############################################################################
        # Uploaded data overlay (always drawn last / 常に最前面)
        ##############################################################################

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

    
    
        #######   水塊分類 ###########
        # # NPIW
        # ax.plot([34.0,34.4],[4,8], color="blue", lw=2)
        # ax.text(34.25,6,"NPIW", color="blue")
        
        # # Kuroshio
        # ax.plot([34.5,35.0],[20,28], color="red", lw=2)
        # ax.text(34.7,24,"Kuroshio", color="red")
        
        # # Oyashio
        # ax.plot([33.0,34.0],[0,5], color="green", lw=2)
        # ax.text(33.4,2,"Oyashio", color="green")
            
            
    

    
        #==========  以下，図のファイル名用　============
        #全体のタイトル名
        main_title = fig_title
        
        # --- 月 (スライダー用) ---  
        # sub_title = 'Lon:'+str(sld_lon_min)+'-'+str(sld_lon_max)+', Lat:'+str(sld_lat_min)+'-'+str(sld_lat_max)+', Y:'+str(sld_year_min)+'-'+str(sld_year_max)+', M:'+str(sld_month_min)+'-'+str(sld_month_max)+', S:'+str(sld_sal_min)+'-'+str(sld_sal_max)+', D:'+str(sld_depth_min)+'-'+str(sld_depth_max)+'m'
        # --- 月 (multiselect用) ---
        # 月の表示用テキストを作成（選択されたリストをカンマ区切りにする）
        month_text = ", ".join(map(str, sorted(selected_months))) if selected_months else "None"
        
        
        ### もし「月が多すぎてサブタイトルが長くなる」のが嫌な場合
        # 月の表示ロジック
        if len(selected_months) == 12:
            month_display = "All"
        elif len(selected_months) == 0:
            month_display = "None"
        else:
            # 標準機能だけで「1-3」のように短縮するロジック
            sorted_m = sorted(list(set(selected_months)))
            ranges = []
            if sorted_m:
                start = sorted_m[0]
                for i in range(len(sorted_m)):
                    # 次の要素が連続していない、または最後の要素の場合に書き出し
                    if i + 1 == len(sorted_m) or sorted_m[i+1] != sorted_m[i] + 1:
                        end = sorted_m[i]
                        ranges.append(f"{start}-{end}" if start != end else str(start))
                        if i + 1 < len(sorted_m):
                            start = sorted_m[i+1]
            month_display = ", ".join(ranges)
            
            

        sub_title = f"Lon:{sld_lon_min}-{sld_lon_max}, Lat:{sld_lat_min}-{sld_lat_max}, Y:{sld_year_min}-{sld_year_max}, M:{month_display}, S:{sld_sal_min}-{sld_sal_max}, D:{sld_depth_min}-{sld_depth_max}m"
        
        main_title2 = sub_title
        
        sub_title2 = ''
        
        title_head = str(main_title+'\n'+main_title2+'\n'+sub_title2)
        
        title_head2 = title_head.replace('_', ' ') #図のタイトル表示用
        fig.suptitle(title_head2,fontsize=sld_font_size_label + 4)
        
 
    else:
        pass
            
    
    
    
    #Save to memory first. の場合は，ローカルに保存されないので安心
    fn = envgeo_utils.build_figure_filename("Fig_T-S_SW", main_title2)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
     
    st.pyplot(fig)

    btn = st.download_button(
       label="Download image",
       data=img,
       file_name=fn,
       mime="image/png")
   



    ###############################################################################################
    ###############################################################################################
    # Map section
    ###############################################################################################
    ###############################################################################################


    fig = px.scatter_mapbox(df_fig_add, lat="Latitude_degN", lon="Longitude_degE", zoom=3,
                            # color='Month',
                            hover_data=["d18O","Salinity",'Temperature_degC','Date','Cruise','Station','Depth_m', 'reference'],
                            opacity=0.4,
                            )
    

    # 選択されたデータの地点プロット
    # --- Location map / 採取地点の地図表示 ---
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

    # 2. データの範囲から中心座標とズームレベルを計算
    map_extent_sources = [df_fig_add[["Longitude_degE", "Latitude_degN"]]]
    if not uploaded_map_df.empty:
        map_extent_sources.append(
            uploaded_map_df[["Longitude_degE", "Latitude_degN"]]
        )
    map_extent_df = pd.concat(map_extent_sources, ignore_index=True)
    lat_min, lat_max = map_extent_df["Latitude_degN"].min(), map_extent_df["Latitude_degN"].max()
    lon_min, lon_max = map_extent_df["Longitude_degE"].min(), map_extent_df["Longitude_degE"].max()

    # 初期値（日本）の設定
    default_lat, default_lon, default_zoom = 36.0, 138.0, 4.0

    # --- 判定と計算を一本化 ---
    if pd.isna(lat_min) or pd.isna(lon_min):
        # 【抽出前】データがない場合は日本を中心に固定
        center_lat, center_lon, auto_zoom = default_lat, default_lon, default_zoom
    else:
        # 【抽出後】データがある場合
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        lat_diff = max(lat_max - lat_min, 0.1)
        lon_diff = max(lon_max - lon_min, 0.1)
        
        # 03番準拠のピクセル計算
        map_width_px, map_height_px = 1200, 700
        zoom_lon = math.log2((map_width_px * 360) / (lon_diff * 256))
        zoom_lat = math.log2((map_height_px * 180) / (lat_diff * 256))
        
        # 東西に広範囲な場合に全プロットを収めるため、マージンを少し多めに引く (-1.8)
        # この 1.5 を 1.8 や 2.0 にすると、さらに一歩「引いた」視点になります。
        auto_zoom = min(zoom_lon, zoom_lat) - 2.0
        auto_zoom = max(1, min(15, auto_zoom))

        # もしデータが世界規模（100度以上）に広がっているなら、日本中心の引きの絵にする
        if lon_diff > 100:
              center_lat, center_lon, auto_zoom = default_lat, default_lon, 1.5
    
 


    # 3. 地図の作成 (px.scatter_mapbox内ではwidthを指定しない)
    c_scale_d18o = envgeo_utils.get_custom_colorscale("d18O")

    fig_map = px.scatter_mapbox(
        df_fig_add, 
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
        height=500  # 高さはここで固定
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

    # 4. 背景スタイルの適用
    fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)
    
    

    

    # 5. レイアウト設定 (ここが幅を広げる決め手)
    # カラーバーと凡例を地図内オーバーレイにして、外側余白で地図が圧縮されないようにする。
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

    # 6. 表示 (use_container_width=True を確実に使う)
    # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）　
    # マウスホイールでのズームが強制的に有効
    st.plotly_chart(
        fig_map, 
        key="TS_plot",
        config={'scrollZoom': True, 'displayModeBar': True}, # ズームを有効化
        **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
    )





    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    ##選ばれたデータを表示
    # 例：特定の列だけを選択して新しいデータフレームを作成
    
    # with表記 (推奨)
    # with st.expander("selected dataset (CSV)", expanded=False):
                
    #     df1_table = df1[['reference','Cruise', 'Station', 'Date', 'Longitude_degE', 'Latitude_degN', 'Depth_m', 'Temperature_degC', 'Salinity', 'd18O', 'dD']]

    #     st.dataframe(df1_table)    
        
    # with st.expander("selected dataset (CSV)", expanded=False):
    #     df1_table = df1[['reference','Cruise', 'Station', 'Date', 'Year', 'Month', 'Longitude_degE', 'Latitude_degN', 'Depth_m', 'Temperature_degC', 'Salinity', 'd18O', 'dD']].copy()        
    #     # 【重要】表示直前に全列を文字列化（これでArrowエラーは100%消えます）
    #     df1_table = df1_table.astype(str) 
        
    #     # 最新の width='stretch' を使用
    #     st.dataframe(df1_table, use_container_width=True)
    
        
    # envgeo_utilsから読み出すとき   
    envgeo_utils.display_isotope_table(df_fig_add)
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################





if __name__ == '__main__':
    main()
    
