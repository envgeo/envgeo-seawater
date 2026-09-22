#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Salinity–δ18O relationship visualizer for EnvGeo-Seawater data.

Created: 2023-05-21
Author: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""



# --- Version info ---
version = "1.3.2"  # 2026-09-22

# ToDo




fig_title = "envgeo-seawater-database"  # 2026/02/12
    
    
    
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import io
import envgeo_utils  
import envgeo_user_data



def main():
    

    
    
    # タイトル
    st.header(f'Salinity-δ18O Relationship ({version})')
  

    ############################################################
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
    ref_data = st.radio("Data source (see Home > About)", (data_source_JAPAN_SEA, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True)



    ##############################################################################
    # 全データを背景にプロットするかどうか
    # 近似直線にプロットするかどうか
    ##############################################################################

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




    ##############################################################################
    # サイドバーここから　　df1フィルタリング　も一括で
    #  緯度経度などは型変換をせず、そのまま最小・最大を取得
    ##############################################################################

    
    # 関数の呼び出し
    # すべての変数を順番通りに受け取ります
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
         ref_data, data_source_JAPAN_SEA, data_source_AROUND_JAPAN,
         uploaded_df=uploaded_df, uploaded_filter_key="sal_d18o",
         uploaded_dataset_label=envgeo_utils.UPLOADED_DATA_LABEL,
     )
    # Keep the sidebar-filtered integrated table for calculations.  The split
    # copy is only for drawing the uploaded rows again in the foreground.
    filtered_integrated_df = df1.copy()
    df1, uploaded_df = envgeo_utils.split_uploaded_rows(
        filtered_integrated_df, envgeo_utils.UPLOADED_DATA_LABEL
    )


    # データが一つだけの時に警告　近似直線を引くなどの必要がある図の場合のみ使用，d18Oなどは適宜変更
    data_found = len(filtered_integrated_df["d18O"])
    if data_found == 1:
        st.warning('Only one data point was found. Regression analysis could not be performed.')
        st.stop()



    ##############################################################################
    # 3D-4Dでは，depthをマイナス表示にする場合あり，それ以外は後半の定義用　今後の為に残置
    ##############################################################################
        
    # df_original['lat'] = df_original['Latitude_degN']
    # df_original['lon'] = df_original['Longitude_degE']
    # df_original['Depth_m'] = df_original['Depth_m']*(-1)


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
    
        #図の描画範囲
        if ref_data == data_source_GLOBAL:
          sal_min, sal_max = st.slider(label='Salinity scale',
                                      min_value=0,
                                      max_value=42,
                                      value=(0, 42),
                                      )
          
          # # st.sidebar.subheader('地図の緯度の範囲（拡大）')
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
            
            # # st.sidebar.subheader('地図の緯度の範囲（拡大）')
            d18O_min, d18O_max = st.slider(label=r'$\delta^{18}$O scale',
                                        min_value=-25.0,
                                        max_value=5.0,
                                        value=(-5.0, 1.0),
                                        )

        # --- Matplotlib figure appearance ---
        # T-S Diagramと同じ考え方で、論文図向けの見た目を調整する。
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

        # アップロードデータのうちカラーバー要素が無いポイントの表示切替
        show_nodata_uploaded = st.checkbox(
            f"Show uploaded points without {sal_d18o_color_by} values",
            value=True,
            key="sal_d18o_show_nodata_uploaded",
            help="Show or hide uploaded data points that have no value for the selected color parameter.",
        )


    ##############################################################################
    # キャッシュクリア
    ##############################################################################
        
    # キャッシュのクリア　サイドバーの一番下などに配置
    if st.sidebar.button("🔄 Clear cache"):
        envgeo_utils.clear_app_cache()
        # st.sidebar.success("キャッシュをクリアしました！再読み込みします...")
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



    
    
    
    ############################################
    ######      font size line etc..       #####
    ############################################

    plt.rcParams["font.size"] = sld_font_size_tick
    
    fig_size = [sld_fig_size_x, sld_fig_size_y] #図のサイズ
    fig_dpi = 150 #図の解像度
    ax_length = 15
    
    
    
    ########################################
    ######    FIG: salinity vs d18O    #####
    ########################################

    X_data = "Salinity"
    Y_data = "d18O"
    

    X_label = "salinity"
    Y_label = r"$\delta^{18}$O"
    

    iso_scale_X = ""
    iso_scale_Y = "(VSMOW)"
    
    
    
    ###### プロットをするかどうか,色を一括にするか ######
    #する場合は1,しない場合は2
    X_Y = 1
    
    #メインプロットの設定
    X_Y_C = "red" #色の設定
    X_Y_M = "." #現時点で色は変更設定なしマーカーの種類
    X_Y_S = 100
    

    
    
    
    #追加で強調プロットをする場合は「1」しない場合は「2」　
    X_Y_add2 = 1

    
    selected_row = "Transect"
    
    
    
    #タイトル
    # 書き出し専用
    fig_title_X_Y= X_label + " - "+ Y_label + "" 
    
    #追加データのlabel
    sheet_names_add2 = "filtered data"
    
    #プロットの透明度
    alpha_all = 0.2 #メインプロット
    
    #強調プロットの色の指定
    X_Y_C_add =  "blue" #単色にしたい場合
    X_Y_C_add_each = 1  #シート毎に塗り分けたい場合は「１」　そうでなければ「２」
    
    
    
    #############################################
    ######      data range for SUB FIG      #####
    #############################################
    
    lim_min_X = sal_min
    lim_max_X = sal_max
    lim_min_Y = d18O_min
    lim_max_Y = d18O_max
    
    
    
    
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
            
            # --------------------------------------
            # 全プロット用のデータフレーム読み込みと整理
            # --------------------------------------
            # 塩分とd18Oが無いデータを削除
            df_fig_ALL = df_original.dropna(subset=["Salinity", "d18O"]).reset_index(drop=True)

            # 排除したサンプル数を計算（オプション：前述の英語メッセージなどで使う用）
            excluded_count = len(df_original) - len(df_fig_ALL)
            if excluded_count > 0:
                st.caption(f":red[Background plot: {len(df_fig_ALL):,} / {len(df_original):,} plotted ({excluded_count:,} excluded due to missing d18O/salinity).]")
                    

    
            Ya = df_fig_ALL[Y_data]
            Xa = df_fig_ALL[X_data]
            

    
            #列の要素を表示
            d_select_main = df_fig_ALL[selected_row].value_counts().to_dict()
            d_select_main_sum = df_fig_ALL[selected_row].count().sum()

    
            ax.scatter(Xa, Ya, s=X_Y_S,c=X_Y_C,marker=X_Y_M,lw=0.5, ec="black", alpha=alpha_all)
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

        plt.title(fig_title_X_Y) #

        
        
        
        
        if plot_all_data == "Yes":
    
            # 回帰直線を追加-------------------------------------
            if plot_reg_lines == "Yes":
  
            
            # 一次関数で多項式近似を行う
            #近似式の係数
                coef = np.polyfit(Xa, Ya, 1)
            #近似式の計算
                y1 = np.poly1d(coef)(Xa) #1次
            #グラフ表示
                plt.plot(Xa, y1, label='regression line (ALL)', c=X_Y_C)
            
                reg_line = 'ALL:  y' + ' = ' + '{:.2f}'.format(coef[0]) + 'x ' +' + (' + '{:.2f}'.format(coef[1]) 
                line_r = np.corrcoef(Xa, Ya)
            
                ax.text(0.99, 0.05+0.01, reg_line + ")   (R=" + '{:.2f}'.format(line_r[0,1])+', N=' + str(d_select_main_sum)+')', horizontalalignment='right', transform=ax.transAxes, fontsize=max(8, sld_font_size_tick - 3))
            # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
        
  
        
            else:
                pass
        else:
            pass
        
        
        
        
  
        
        ##############################################################################
        # フィルターしたデータを重ね書き
        ##############################################################################

        selected_regression_available = False
        if X_Y_add2 == 1:
        
            if X_Y_C_add_each == 1:     
                X_Y_C_add  =  'blue' #カラーを選ぶ

                
                # --------------------------------------
                # フィルターデータ用のデータフレーム読み込みと整理
                # --------------------------------------
                # 塩分、d18O、色分け列に必要なデータだけを残す
                filtered_required_columns = ["Salinity", "d18O"]
                if sal_d18o_color_by != "Single color":
                    filtered_required_columns.append(sal_d18o_color_by)
                # The selected regression is calculated from the same
                # reference-plus-upload table that the sidebar has filtered.
                df_fig_add = filtered_integrated_df.dropna(
                    subset=filtered_required_columns
                ).reset_index(drop=True)

                # 排除したサンプル数を計算（オプション：前述の英語メッセージなどで使う用）
                excluded_count_add = len(filtered_integrated_df) - len(df_fig_add)
                if excluded_count_add > 0:
                    missing_label = "d18O/salinity"
                    if sal_d18o_color_by != "Single color":
                        missing_label = f"d18O/salinity/{sal_d18o_color_by}"
                    st.caption(f":blue[Filtered plot: {len(df_fig_add):,} / {len(filtered_integrated_df):,} plotted ({excluded_count_add:,} excluded due to missing {missing_label}).]")
                        

                

                # 描画したいXとYの両方にデータが入っている行だけを残す
                # df_fig_add2 = df_fig_add.dropna(subset=[X_data, Y_data])

                Y_add = df_fig_add[Y_data]
                X_add = df_fig_add[X_data]

                
                #列の要素を表示
                d_select_add2 = df_fig_add[selected_row].value_counts().to_dict()
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
                # 一次関数で多項式近似を行う
                #近似式の係数
                    coef_add = np.polyfit(X_add, Y_add, 1)
                #近似式の計算
                    y1_add = np.poly1d(coef_add)(X_add) #1次
                #グラフ表示
                    plt.plot(X_add, y1_add, label='regression line (' + sheet_names_add2 +')', c=X_Y_C_add,)
                
                    reg_line_add = sheet_names_add2 + ':  y' + ' = ' + '{:.2f}'.format(coef_add[0]) + 'x ' +' + (' + '{:.2f}'.format(coef_add[1]) 
                    line_r_add = np.corrcoef(X_add, Y_add)
                    selected_regression_available = True
                
                    ax.text(0.99, 0.05*3+0.01, reg_line_add + ")   (R=" + '{:.2f}'.format(line_r_add[0,1])+', N=' + str(d_select_add2_sum)+')', horizontalalignment='right', transform=ax.transAxes, fontsize=max(8, sld_font_size_tick - 3))
                    # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
                
                
       
                elif plot_reg_lines == "Yes":
                    st.caption(
                        ":gray[Regression line was skipped because fewer than "
                        "two valid selected data points are available.]"
                    )
                
            else:
                pass
            
        else:
            pass
    
    
    
    
        ###############################################################################################
        ############################################################################################### 
        ###############################################################################################
        ###############################################################################################

    
    
    

        
        #==========  以下，近似直線の計算　============
        if plot_reg_lines == "Yes": 
        
            if plot_all_data == "Yes" and "coef" in locals():
                Y_all_pred = coef[0]*Xa + coef[1]

                MSE_all = mean_squared_error(Ya, Y_all_pred)
                RMES_all = np.sqrt(mean_squared_error(Ya, Y_all_pred))

                #　R2の計算
                R2_all =  r2_score(Ya, Y_all_pred)  
                
                ax.text(0.99, 0+0.01, 'RMSE_all: ' + '{:.3f}'.format(RMES_all)+', R$^{2}$_all: ' + '{:.2f}'.format(R2_all), horizontalalignment='right', transform=ax.transAxes, fontsize=max(8, sld_font_size_tick - 3), c='red')
            else:
                pass
                
            
            
            if selected_regression_available:
                Y_add_pred = coef_add[0]*X_add + coef_add[1]

                MSE_add = mean_squared_error(Y_add, Y_add_pred)
                RMES_add = np.sqrt(mean_squared_error(Y_add, Y_add_pred))

                #　R2の計算
                R2_add =  r2_score(Y_add, Y_add_pred)

                ax.text(0.99, 0.05*2+0.01, 'RMSE_add: ' + '{:.3f}'.format(RMES_add)+', R$^{2}$_add: ' + '{:.2f}'.format(R2_add), horizontalalignment='right', transform=ax.transAxes, fontsize=max(8, sld_font_size_tick - 3), c='blue')
        
        else:
            pass
    
        #==========  ここまで，近似直線の計算　============

        # Uploaded data overlay (always drawn last / 常に最前面)
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
    
    

    

        #==========  以下，図のファイル名用　============
        #全体のタイトル名　　手入力
        main_title = fig_title
        
        # --- 月 (スライダー用) ---  
        # sub_title = 'Lon:'+str(sld_lon_min)+'-'+str(sld_lon_max)+', Lat:'+str(sld_lat_min)+'-'+str(sld_lat_max)+', Y:'+str(sld_year_min)+'-'+str(sld_year_max)+', M:'+str(sld_month_min)+'-'+str(sld_month_max)+', S:'+str(sld_sal_min)+'-'+str(sld_sal_max)+', D:'+str(sld_depth_min)+'-'+str(sld_depth_max)+'m'
        # --- 月 (multiselect用) ---
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
            
    
    
    
    
    
    
    
    ##############################################################################
    # 画像保存
    ##############################################################################
    
    #Save to memory first. の場合は，ローカルに保存されないので安心
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
   

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    

    # 選択されたデータの地点プロット
    # --- Location map / 採取地点の地図表示 ---
    st.divider()
    st.subheader('Sampling Location Map')

    import math

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
        # この 1.5 を 1.8 や 2.0 にすると、さらに一歩「引いた」視点に
        auto_zoom = min(zoom_lon, zoom_lat) - 2.0
        auto_zoom = max(1, min(15, auto_zoom))

        # もしデータが世界規模（100度以上）に広がっているなら、日本中心の引きの画に
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

    # 6. 表示 (st.plotly_chart(fig, width='stretch'))
    # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）　
    # マウスホイールでのズームが強制的に有効
    st.plotly_chart(
        fig_map, 
        # width='stretch', # Streamlitのバージョン上げたら復活させる。今はwarningになる
        key="sal_d18O_plot",
        config={'scrollZoom': True, 'displayModeBar': True} # ズームを有効化
    )







    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    ##選ばれたデータを表示
    # 例：特定の列だけを選択して新しいデータフレームを作成
        
        
    # envgeo_utilsから読み出すとき   
    envgeo_utils.display_isotope_table(df_fig_add)
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

if __name__ == '__main__':
    main()
    



    
    
    
