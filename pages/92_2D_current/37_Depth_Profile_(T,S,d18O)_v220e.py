#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:15:03 2023
@author: Toyoho Ishimura @Kyoto-U
2026/02/10 update 
"""


# --- Version info ---
version = "1.0.0" #v220_20260317

# ToDo



fig_title = "envgeo-seawater-database"  # 2026/02/12
    


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
from matplotlib.ticker import FormatStrFormatter
import envgeo_utils  
pd.set_option('future.no_silent_downcasting', True)

    

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
    ref_data = st.radio("Data source (see Home > About):", (data_source_JAPAN_SEA, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True, args=[1, 0])



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
        plot_all_data = st.radio("Show all data in background (gray lines):", ("Yes", "No"), horizontal=True, args=[1, 0])
    
    with col2:
        plot_element = st.radio("Target Parameter:", ("d18O(VSMOW)", "Temperature (°C)", "Salinity"), horizontal=True, args=[1, 0])


        
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
    
    else:()





    ##############################################################################
    # envgeo_utilsからデータフレーム読み込み
    ##############################################################################
    df_original = envgeo_utils.load_isotope_data(ref_data) # フィルターしないデータ

    df1 = df_original # このあとフィルターするデータ

    if df_original.empty:
        st.warning("No data available for the selected conditions.")
        return

    


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
     submitted) = envgeo_utils.sidebar_filter_and_display(df1, ref_data, data_source_JAPAN_SEA, data_source_AROUND_JAPAN)


    # データが一つだけの時に警告　近似直線を引くなどの必要がある図の場合のみ使用，d18Oなどは適宜変更
    data_found = len(df1["d18O"])
    if data_found == 1:
        st.warning('Only one data point was found. A depth profile could not be meaningfully generated.')
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
        st.subheader(':blue[--- for fig scale only ---]')
        
        # st.sidebar.subheader('描画水深の範囲')
        if ref_data == data_source_JAPAN_SEA:
            fig_depth_min, fig_depth_max = st.slider(label='Depth scale',
                                        min_value=0,
                                        max_value=1000,
                                        value=(0, 500),
                                        step = 50,
                                        )
            # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
            
        elif ref_data == data_source_AROUND_JAPAN:
            fig_depth_min, fig_depth_max = st.slider(label='Depth scale',
                                        min_value=0,
                                        max_value=3500,
                                        value=(0, 2000),
                                        step = 50,
                                        )
            # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
            
        else:
            fig_depth_min, fig_depth_max = st.slider(label='Depth scale',
                                        min_value=0,
                                        max_value=9000,
                                        value=(0, 2000),
                                        step = 50,
                                        )
            # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
            

        

        # st.sidebar.subheader('X軸の描画範囲の範囲')
        if ref_data == data_source_JAPAN_SEA:
            lim_min_X, lim_max_X = st.slider(label=f'Axis Scale for {plot_element}',
                                        min_value=-fig_x_min,
                                        max_value=fig_x_max,
                                        value=(fig_x_min, fig_x_max),
                                        )
            # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
            
        elif ref_data == data_source_AROUND_JAPAN:
            lim_min_X, lim_max_X = st.slider(label=f'Axis Scale for {plot_element}',
                                        min_value=fig_x_min,
                                        max_value=fig_x_max,
                                        value=(fig_x_min, fig_x_max),
                                        )
            # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
            
        else:
            lim_min_X, lim_max_X = st.slider(label=f'Axis Scale for {plot_element}',
                                        min_value=-fig_x_min,
                                        max_value=fig_x_max,
                                        value=(fig_x_min, fig_x_max),
                                        )
            # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
      
            
            

        #図のサイズXY
        # st.sidebar.subheader('図のサイズ')
        sld_fig_size_min_X, sld_fig_size_max_Y = st.slider(label='Fig size (x vs y)',
                                    min_value=1,
                                    max_value=40,
                                    value=(8, 12),
                                    )
        # st.sidebar.write(f'Selected: {sld_fig_size_min_X} ~ {sld_fig_size_max_Y}')
        
    
    
        #フォントサイズ
        # st.sidebar.subheader('図のサイズ')
        sld_font_size_min_S, sld_font_size_max_L = st.slider(label='Font size (scale and label)',
                                    min_value=4,
                                    max_value=40,
                                    value=(16, 20),
                                    )
        # st.sidebar.write(f'Selected: {sld_fig_size_min_X} ~ {sld_fig_size_max_Y}')
                    
                    
        #メモリ間隔
        # st.sidebar.subheader('図のサイズ')
        tick_interval_min_X, tick_interval_max_Y = st.slider(label='Tick interval (x and y)',
                                    min_value=4,
                                    max_value=40,
                                    value=(11, 11),
                                    )
        # st.sidebar.write(f'Selected: {sld_fig_size_min_X} ~ {sld_fig_size_max_Y}')           
                    
                    
                    
    
      
    

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
    st.markdown("##### :red[--- The map area can be adjusted using the [fig scale] setting in the sidebar---]")



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
    
    title_head = str(main_title+'\n'+sub_title+'\n'+sub_title2)
    
    title_head2 = title_head.replace('_', ' ') #図のタイトル表示用

    fig.suptitle(title_head2,fontsize=20)
    
    

    
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
    
        


        # 同じ地点，同じ年月日，はグループにして他は1行開ける
        df_fig_ALL = envgeo_utils.insert_gap_rows(df_original)
        
        
        # 特定の列に特定の変数を持つ行と空白行を残す

        if plot_all_data == "Yes":
            plt.plot(df_fig_ALL[X_data], df_fig_ALL[Y_data],c=X_Y_C, marker=X_Y_M, lw=0.5, alpha=alpha_all, label='ALL')
        else:()



        plt.legend(fontsize = 16) #


        ax.set_xlim(lim_min_X, lim_max_X) 
        ax.set_ylim(lim_min_Y, lim_max_Y) 

        plt.tick_params(labelsize=sld_font_size_min_S)

        ax.set_xticks(np.linspace(lim_min_X, lim_max_X,tick_interval_min_X))
        ax.set_yticks(np.linspace(lim_min_Y, lim_max_Y, tick_interval_max_Y))
        
        
        

        
        if plot_element == "d18O(VSMOW)":
            ax.xaxis.set_major_formatter(FormatStrFormatter("%+.1f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))

         
        elif plot_element == "Temperature (°C)":
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))

            lim_max_X = 30

            
        elif plot_element == "Salinity":
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))

        
        else:()

            

        ax.tick_params(length=ax_length)

        plt.legend(fontsize = 20) # 凡例の数字のフォントサイズを設定
        
    
        
        #追加で強調プロットをする場合
        if X_Y_add == 1:
        
            if X_Y_C_add_each == 1:
                
 
                df_fig_add = df1    

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


                
        else:()
    else:()
    

     
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################

    
    #######################画像を保存するためのボタン作成########################
    sub_title2 = sub_title
    sub_title2 = sub_title2.replace(':', '') #pdf書き出し用
    sub_title2 = sub_title2.replace(',', '_') #pdf書き出し用
    sub_title2 = sub_title2.replace(' ', '') #pdf書き出し用
    sub_tite = str('Fig_depth'+'_'+sub_title2+".png")




    #Save to memory first. の場合は，ローカルに保存されないので安心
    import io
    fn = sub_tite
    img = io.BytesIO()
    plt.savefig(img, format='png')
     
    btn = st.download_button(
       label="Download image",
       data=img,
       file_name=fn,
       mime="image/png"
       )
    
    

    st.pyplot(fig)
    
    
    
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    


    # 選択されたデータの地点プロット
    # --- 採取地点の地図表示 (Auto-Zoom & 幅広設定) ---
    st.divider()
    st.subheader('Location Map (Auto-Zoom)')


    # 1. 地図背景の選択
    map_mode = st.radio(
        "Map Style:", 
        ["Standard", "Satellite", "Bathymetry (Sea)", "Contour (GSI)"], 
        horizontal=True,
        key="map_style_31_auto"
    )

 # 2. データの範囲から中心座標とズームレベルを計算
    lat_min, lat_max = df_fig_add["Latitude_degN"].min(), df_fig_add["Latitude_degN"].max()
    lon_min, lon_max = df_fig_add["Longitude_degE"].min(), df_fig_add["Longitude_degE"].max()

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

    # 4. 背景スタイルの適用
    fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)
    
    


    # 5. レイアウト設定 (ここが幅を広げる決め手)
    fig_map.update_layout(
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=auto_zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        # widthを指定せず autosize を True にすることで、コンテナいっぱいに広がる
        autosize=True, 
        coloraxis_colorbar=dict(
            title="δ18O (‰)",
            x=1.0,           # カラーバーを右端に寄せる
            xanchor='right'
        )
    )

    # 6. 表示 
    # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）　
    # マウスホイールでのズームが強制的に有効
    st.plotly_chart(
        fig_map, 
        # width="stretch", #Streramlitあげたら復活させる  
        key="depth_profile",
        config={'scrollZoom': True, 'displayModeBar': True} # ズームを有効化
    )



    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    ##選ばれたデータを表示
    # 例：特定の列だけを選択して新しいデータフレームを作成
        
    with st.expander("selected dataset (CSV)", expanded=False):
        # 1. 必要な列をコピー
        df1_table = df_fig_add[['reference','Cruise', 'Station', 'Date','Year', 'Month', 'Longitude_degE', 'Latitude_degN', 'Depth_m', 'Temperature_degC', 'Salinity', 'd18O', 'dD']].copy()        
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
    
    

