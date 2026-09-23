#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-profile visualizer for EnvGeo-Seawater data.

Created: 2023-04-22
Author: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""


# --- Version info ---
version = "1.3.3"  # 2026-09-23

# ToDo



fig_title = "envgeo-seawater-database"  # 2026/02/12
    


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
import io
import textwrap
from matplotlib.ticker import FormatStrFormatter
import envgeo_utils
import envgeo_user_data

    

def main():
    
    # タイトル
    st.header(f'Depth Profile ({version})')

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
    # 選択したデータセットの文献表示
    ##############################################################################
    
    if ref_data == data_source_JAPAN_SEA:
        st.write(envgeo_utils.refs_JAPAN_SEA)
        
    elif ref_data == data_source_AROUND_JAPAN:
        # st.text('including data from previous reports')
        st.write(envgeo_utils.refs_AROUND_JAPAN)

    elif ref_data == data_source_GLOBAL:
        st.write(envgeo_utils.refs_GLOBAL)
        
    else:
        st.warning("Invalid data source selection.")


    ##############################################################################
    # --- 3. プロットのエレメント選択　---　　 
    ##############################################################################

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
        #水深-d18Oの時
        X_data = "d18O"
        Y_data = "Depth_m"
        
        
        # """XYの表示用のラベルを指定"""
        X_label = r"$\delta^{18}$O"
        Y_label = "water depth (m)"
        
        
        # """XYの表示用のラベルのスケールを指定"""
        iso_scale_X = "(VSMOW)"
        iso_scale_Y = ""
     
        #水深-d18Oの時
        # lim_min_X = -1.4
        # lim_max_X = 0.6
        
        # 海域別の初期値設定
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = -20.0, 4.0  # GLOBALは幅を広く
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = -1.4, 0.6  # 日本海はズーム
        else:
            fig_x_min, fig_x_max = -1.4, 0.6  # Around JAPAN(標準)

     
    elif plot_element == "dD(VSMOW)":
        # 水深-dDの時
        X_data = "dD"
        Y_data = "Depth_m"
        
        # """XYの表示用のラベルを指定"""
        X_label = r"$\delta$D"
        Y_label = "water depth (m)"
        
        # """XYの表示用のラベルのスケールを指定"""
        iso_scale_X = "(VSMOW)"
        iso_scale_Y = ""
        
        # 海域別の初期値設定
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = -150.0, 50.0
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = -20.0, 10.0
        else:
            fig_x_min, fig_x_max = -30.0, 20.0

    elif plot_element == "d-excess":
        # 水深-d-excessの時
        X_data = "d-excess"
        Y_data = "Depth_m"
        
        # """XYの表示用のラベルを指定"""
        X_label = "d-excess"
        Y_label = "water depth (m)"
        
        # """XYの表示用のラベルのスケールを指定"""
        iso_scale_X = ""
        iso_scale_Y = ""
        
        # 海域別の初期値設定
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = -30.0, 40.0
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = -5.0, 25.0
        else:
            fig_x_min, fig_x_max = -10.0, 30.0

    elif plot_element == "Temperature (°C)":
        #水深-水温の時
        X_data = "Temperature_degC"
        Y_data = "Depth_m"
        
        # """XYの表示用のラベルを指定"""
        X_label = "Temperature(°C)"
        Y_label = "water depth (m)"
        
        # """XYの表示用のラベルのスケールを指定"""
        iso_scale_X = ""
        iso_scale_Y = ""
        
    	# #水深-d18Oの時
        # lim_min_X = -2
        # lim_max_X = 30
        
        # 海域別の初期値設定
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = -3, 35  # GLOBALは幅を広く
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = -2, 30  # 日本海はズーム
        else:
            fig_x_min, fig_x_max = -2, 30  # Around JAPAN(標準)

        
    elif plot_element == "Salinity":
        #水深-塩分の時
        X_data = "Salinity"
        Y_data = "Depth_m"
        
        # """XYの表示用のラベルを指定"""
        X_label = "Salinity"
        Y_label = "water depth (m)"
        
        # """XYの表示用のラベルのスケールを指定"""
        iso_scale_X = ""
        iso_scale_Y = ""
        
    	# #水深-d18Oの時
        # lim_min_X = 28
        # lim_max_X = 36
        
        # 海域別の初期値設定
        if ref_data == data_source_GLOBAL:
            fig_x_min, fig_x_max = 0, 40  # GLOBALは幅を広く
        elif ref_data == data_source_JAPAN_SEA:
            fig_x_min, fig_x_max = 28, 36  # 日本海はズーム
        else:
            fig_x_min, fig_x_max = 28, 36  # Around JAPAN(標準)
    
    else:
        pass





    ##############################################################################
    # envgeo_utilsからデータフレーム読み込み
    ##############################################################################
    df_original = envgeo_utils.load_isotope_data(ref_data) # フィルターしないデータ

    df1 = df_original # このあとフィルターするデータ

    if df_original.empty:
        st.warning("No data available for the selected conditions.")
        return

    


    ##############################################################################
    # --- Upload overlay ---
    ##############################################################################

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


    ##############################################################################
    # サイドバーここから　　df1フィルタリング　も一括で
    #　2026/03/06　Min-Maxをdfから取得に変更
    #  緯度経度などは型変換をせず、そのまま最小・最大を取得
    ##############################################################################


    # 関数の呼び出し
    # すべての変数を順番通りに受け取り
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


    # データが一つだけの時に警告。
    # dD / d-excessは欠損が多いため、対象列と水深が両方ある点だけを数える。
    # The selected Dataset list can contain only Uploaded data.  Use the
    # complete filtered input here; df1 below remains reference-only so the
    # existing foreground uploaded trace stays visually distinct.
    data_found = len(filtered_profile_df.dropna(subset=[X_data, "Depth_m"]))
    if data_found == 1:
        st.warning('Only one data point was found. A depth profile could not be meaningfully generated.')
        st.stop()
    if data_found == 0:
        st.warning(f"No valid {plot_element} depth-profile data are available for the selected conditions.")
        st.stop()
    

    ##############################################################################
    # [重要] 同じ地点，同じ年月日，はグループにして他は1行開ける
    ##############################################################################

    
    df_original = envgeo_utils.insert_gap_rows(df_original)     # [重要]　同じ地点，同じ年月日，はグループにして他は1行開ける
    
    df1 = envgeo_utils.insert_gap_rows(df1)     # [重要]　同じ地点，同じ年月日，はグループにして他は1行開ける
    
    

    ##############################################################################
    # 後半の定義用
    ##############################################################################
        
    df1['lat'] = df1['Latitude_degN']
    df1['lon'] = df1['Longitude_degE']



    
    #######################################################################
    #######################################################################   
    ###  図の調整　サイドバー
    #######################################################################        
    #######################################################################       
    
    
    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "FIGURE_CONTROLS_LABEL", "Figure controls"))
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        
        # st.sidebar.subheader('描画水深の範囲')
        if ref_data == data_source_JAPAN_SEA:
            fig_depth_min, fig_depth_max = st.slider(label='Depth range',
                                        min_value=0,
                                        max_value=1000,
                                        value=(0, 500),
                                        step = 50,
                                        )
            # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
            
        elif ref_data == data_source_AROUND_JAPAN:
            fig_depth_min, fig_depth_max = st.slider(label='Depth range',
                                        min_value=0,
                                        max_value=3500,
                                        value=(0, 2000),
                                        step = 50,
                                        )
            # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
            
        else:
            fig_depth_min, fig_depth_max = st.slider(label='Depth range',
                                        min_value=0,
                                        max_value=9000,
                                        value=(0, 2000),
                                        step = 50,
                                        )
            # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
            

        

        # st.sidebar.subheader('X軸の描画範囲の範囲')
        axis_margin = max(abs(fig_x_min), abs(fig_x_max)) * 0.5
        lim_min_X, lim_max_X = st.slider(
            label=f'Axis Scale for {plot_element}',
            min_value=float(fig_x_min - axis_margin),
            max_value=float(fig_x_max + axis_margin),
            value=(float(fig_x_min), float(fig_x_max)),
            step=0.1,
            key=f"depth_profile_axis_scale::{plot_element}::{ref_data}",
        )
      
            
            

        #図のサイズXY
        # st.sidebar.subheader('図のサイズ')
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
        
    
    
        #フォントサイズ
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
                    
                    
        #メモリ間隔
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

        # アップロードデータのうち月情報が無い行の表示切替
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
        
    fig = plt.figure(figsize = (sld_fig_size_min_X, sld_fig_size_max_Y),dpi=150)
    
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.rcParams["font.size"] = 16
    
    

    
    # ==========  以下，図のファイル名用　============

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

    sub_title2 = ''
    
    # Keep the saved Depth Profile title inside the figure width.
    # Depth Profileはフィルタ条件が長くなりやすいため、保存図では少し短めに折り返す。
    wrapped_sub_title = "\n".join(textwrap.wrap(sub_title, width=62))
    title_head = str(main_title+'\n'+wrapped_sub_title+'\n'+sub_title2)
    
    title_head2 = title_head.replace('_', ' ') #図のタイトル表示用

    fig.suptitle(title_head2, fontsize=max(14, sld_font_size_max_L - 3), y=0.97)
    fig.subplots_adjust(top=0.87)
    
    

    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    
    
    
    
    
    #####################################
    ######    EXCEL SHEET select    #####
    #####################################

    sheet_num = [2]  #ここ必要ない，過去に複数のエクセルシート読み込んだなごり
    
  
    
    ############################################
    ######      font size line etc..       #####
    ############################################
    
    ax_length = 5
    
   
    
    
    ########################################
    ######     FIG: depth profile    #####
    ########################################
    


    
    sheet_num_XY = sheet_num #ここは変更しない
    X_Y = 1
    
    #メインプロットの設定
    X_Y_C = "gray" #色の設定
    X_Y_M = " " #現時点で色は変更設定なしマーカーの種類

    
    # 追加で強調プロットをする場合は「1」
    X_Y_add = 1

    
    
    #プロットの透明度
    alpha_all = 0.4 #メインプロット
    alpha_selected = 1 #強調プロット
    
    #強調プロットの色の指定
    # X_Y_C_add =  "blue" #単色にしたい場合
    X_Y_C_add_each = 1 #シート毎に塗り分けたい場合は「１」　そうでなければ「２」

    
    #############################################
    ######      data range for SUB FIG      #####
    #############################################
    
    
    # #水深-d18Oの時
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


                #########月ごとに色分けする場合######################
                lw_add = 0.6 #線の太さ
                # 描画する月範囲を指定 and指定
                df13 = df_fig_add[(df_fig_add['Month'] >= 1) & (df_fig_add['Month'] <= 3)
                          | df_fig_add.isnull().all(axis=1)]  
                plt.plot(df13[X_data], df13[Y_data],c='blue', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='1-3')
                
                # 描画する月範囲を指定 and指定
                df46 = df_fig_add[(df_fig_add['Month'] >= 4) & (df_fig_add['Month'] <= 6)
                          | df_fig_add.isnull().all(axis=1)]  
                plt.plot(df46[X_data], df46[Y_data],c='green', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='4-6')
                
                # 描画する月範囲を指定 and指定
                df79 = df_fig_add[(df_fig_add['Month'] >= 7) & (df_fig_add['Month'] <= 9)
                          | df_fig_add.isnull().all(axis=1)]  
                plt.plot(df79[X_data], df79[Y_data],c='orange', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='7-9')
                # 描画する月範囲を指定 and指定
                df1012 = df_fig_add[(df_fig_add['Month'] >= 10) & (df_fig_add['Month'] <= 12)
                          | df_fig_add.isnull().all(axis=1)]  
                plt.plot(df1012[X_data], df1012[Y_data],c='purple', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='10-12')
                

    
                plt.legend(fontsize = sld_font_size_min_S) # 凡例の数字のフォントサイズを設定


                
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
    

     
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################

    
    #Save to memory first. の場合は，ローカルに保存されないので安心
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
    
    
    
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    


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
            index=envgeo_utils.MAP_MODE_DEFAULT_INDEX,
            horizontal=True,
            key="map_style_depth_profile",
            help=getattr(envgeo_utils, "MAP_STYLE_HELP_TEXT", "Choose the background map style for the sampling-location map."),
        )
    _eff_37, _fell_37 = envgeo_utils.resolve_map_mode(map_mode)
    if _fell_37:
        st.warning(envgeo_utils.OFFLINE_FALLBACK_WARNING)

    # 2. 有効座標の抽出（同一行に緯度・経度が両方有効、かつ範囲内）
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

    # Build uploaded_map_df for the guard and center/zoom calculation.
    # (page 37 uses uploaded_df directly in add_uploaded_map_overlay, so this
    #  variable is local to the map extent logic and does not affect the overlay.)
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

    # Uploaded data alone may provide valid coordinates even when df_fig_add
    # (reference rows only) is empty — extend the guard to cover that case.
    _has_valid_map_coords = _has_valid_map_coords or not uploaded_map_df.empty

    # 3. データの範囲から中心座標とズームレベルを計算
    # 初期値（日本）の設定
    default_lat, default_lon, default_zoom = 36.0, 138.0, 4.0

    if not _has_valid_map_coords:
        center_lat, center_lon, auto_zoom = default_lat, default_lon, default_zoom
    else:
        # Combine reference and uploaded coords for map extent
        _map_ext_sources = []
        if len(_valid_coords_df) > 0:
            _map_ext_sources.append(_valid_coords_df[["Longitude_degE", "Latitude_degN"]])
        if not uploaded_map_df.empty:
            _map_ext_sources.append(uploaded_map_df[["Longitude_degE", "Latitude_degN"]])
        _map_ext_df = pd.concat(_map_ext_sources, ignore_index=True)
        lat_min, lat_max = _map_ext_df["Latitude_degN"].min(), _map_ext_df["Latitude_degN"].max()
        lon_min, lon_max = _map_ext_df["Longitude_degE"].min(), _map_ext_df["Longitude_degE"].max()

        # --- 判定と計算を一本化 ---
        if pd.isna(lat_min) or pd.isna(lon_min):
            center_lat, center_lon, auto_zoom = default_lat, default_lon, default_zoom
        else:
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

    # 4. 地図の作成 (px.scatter_mapbox内ではwidthを指定しない)
    if not _has_valid_map_coords:
        st.info(
            "Map view is unavailable because the selected data contain no valid "
            "latitude/longitude coordinates."
        )
    else:
        c_scale_profile = envgeo_utils.get_custom_colorscale(X_data)

        # Use reference coords as base when available; fall back to uploaded coords
        # so that px.scatter_mapbox always receives a non-empty DataFrame.
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
            height=500  # 高さはここで固定
        )

        # 4. 背景スタイルの適用
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

        # 5. レイアウト設定 (ここが幅を広げる決め手)
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

        # 6. 表示
        # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）
        # マウスホイールでのズームが強制的に有効
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



    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    ##フィルタ後・現在表示中のデータを表示
    # 例：特定の列だけを選択して新しいデータフレームを作成
        
    with st.expander("Filtered dataset for current view (CSV)", expanded=False):
        # 1. 必要な列をコピー
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
        # --- [追加] 空白行（すべての列が欠損値の行）を削除 --- CSV用
        df1_table = df1_table.dropna(how='all')
        # 【重要】表示直前に全列を文字列化（これでArrowエラーは消える）
        df1_table = df1_table.astype(str) 
        
        # 3. テーブルを表示
        # 最新の width='stretch' を使用すべきか？
        st.dataframe(df1_table, 
                     # width="stretch", #Streramlitあげたら復活させる  
                     )
        
        
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################


if __name__ == '__main__':
    main()
    
    
