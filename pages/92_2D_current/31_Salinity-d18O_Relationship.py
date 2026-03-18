#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:00:21 2023

@author: Toyoho Ishimura @Kyoto-U

2026/03/18 update
"""



# --- Version info ---
version = "1.0.0" #v220_20260317

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
import envgeo_utils  
pd.set_option('future.no_silent_downcasting', True)



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
    ref_data = st.radio("Data source (see Home > About)", (data_source_JAPAN_SEA, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True, args=[1, 0])



    ##############################################################################
    # 全データを背景にプロットするかどうか
    # 近似直線にプロットするかどうか
    ##############################################################################

    col1, col2 = st.columns([1,1])
    with col1:
        plot_all_data = st.radio("Show all data in background (red):", ("Yes", "No"), horizontal=True, args=[1, 0])
    
    with col2:
        plot_reg_lines = st.radio("Add regression line(s):", ("Yes", "No"), horizontal=True, args=[1, 0])



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
     submitted) = envgeo_utils.sidebar_filter_and_display(df_original, ref_data, data_source_JAPAN_SEA, data_source_AROUND_JAPAN)


    # データが一つだけの時に警告　近似直線を引くなどの必要がある図の場合のみ使用，d18Oなどは適宜変更
    data_found = len(df1["d18O"])
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
        st.subheader(':blue[--- for fig scale only ---]')
    
    
        # マーカーの問明度調整
        alpha_selected = st.slider(label='Transparency (Filtered Plot)',
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=(0.9),
                                    step=0.05
                                    )
    
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

    plt.rcParams["font.size"] = 15
    
    fig_size = [12,9] #図のサイズ
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
    sheet_names_add2 = "selected data"
    
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
    

    
        ax.set_xlabel(X_label + iso_scale_X)
        ax.set_ylabel(Y_label + iso_scale_Y)  
        
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
        else:()
            
            

        ax.set_xlim(lim_min_X, lim_max_X) 
        ax.set_ylim(lim_min_Y, lim_max_Y) 
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%+.1f"))
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
            
                ax.text(0.99, 0.05+0.01, reg_line + ")   (R=" + '{:.2f}'.format(line_r[0,1])+', N=' + str(d_select_main_sum)+')', horizontalalignment='right', transform=ax.transAxes)
            # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
        
  
        
            else:()    
        else:()    
        
        
        
        
  
        
        ##############################################################################
        # フィルターしたデータを重ね書き
        ##############################################################################

        if X_Y_add2 == 1:
        
            if X_Y_C_add_each == 1:     
                X_Y_C_add  =  'blue' #カラーを選ぶ

                
                # --------------------------------------
                # フィルターデータ用のデータフレーム読み込みと整理
                # --------------------------------------
                # 塩分とd18Oが無いデータを削除
                df_fig_add = df1.dropna(subset=["Salinity", "d18O"]).reset_index(drop=True)

                # 排除したサンプル数を計算（オプション：前述の英語メッセージなどで使う用）
                excluded_count_add = len(df1) - len(df_fig_add)
                if excluded_count_add > 0:
                    st.caption(f":blue[Filtered plot: {len(df_fig_add):,} / {len(df1):,} plotted ({excluded_count_add:,} excluded due to missing d18O/salinity).]")
                        

                

                # 描画したいXとYの両方にデータが入っている行だけを残す
                # df_fig_add2 = df_fig_add.dropna(subset=[X_data, Y_data])

                Y_add = df_fig_add[Y_data]
                X_add = df_fig_add[X_data]

                
                #列の要素を表示
                d_select_add2 = df_fig_add[selected_row].value_counts().to_dict()
                d_select_add2_sum = df1[selected_row].count().sum()

                
                ax.scatter(X_add, Y_add, s=X_Y_S,c=X_Y_C_add,marker=X_Y_M, alpha=alpha_selected,lw=0.5, ec="black", label= sheet_names_add2)


                
                if plot_reg_lines == "Yes":
                # 一次関数で多項式近似を行う
                #近似式の係数
                    coef_add = np.polyfit(X_add, Y_add, 1)
                #近似式の計算
                    y1_add = np.poly1d(coef_add)(X_add) #1次
                #グラフ表示
                    plt.plot(X_add, y1_add, label='regression line (' + sheet_names_add2 +')', c=X_Y_C_add,)
                
                    reg_line_add = sheet_names_add2 + ':  y' + ' = ' + '{:.2f}'.format(coef_add[0]) + 'x ' +' + (' + '{:.2f}'.format(coef_add[1]) 
                    line_r_add = np.corrcoef(X_add, Y_add)
                
                    ax.text(0.99, 0.05*3+0.01, reg_line_add + ")   (R=" + '{:.2f}'.format(line_r_add[0,1])+', N=' + str(d_select_add2_sum)+')', horizontalalignment='right', transform=ax.transAxes)
                    # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
                
                
       
                else:()
                
            else:()
            
        else:()
    
    
    
    
        ###############################################################################################
        ############################################################################################### 
        ###############################################################################################
        ###############################################################################################

    
    
    

        
        #==========  以下，近似直線の計算　============
        if plot_reg_lines == "Yes": 
        
            if plot_all_data == "Yes":
                Y_all_pred = coef[0]*Xa + coef[1]
      
                MSE_all = mean_squared_error(Ya, Y_all_pred)
                RMES_all = np.sqrt(mean_squared_error(Ya, Y_all_pred))

                #　R2の計算
                R2_all =  r2_score(Ya, Y_all_pred)  
                
                ax.text(0.99, 0+0.01, 'RMSE_all: ' + '{:.3f}'.format(RMES_all)+', R$^{2}$_all: ' + '{:.2f}'.format(R2_all), horizontalalignment='right', transform=ax.transAxes, fontsize=12, c='red')
            else:()
                
            
            
            Y_add_pred = coef_add[0]*X_add + coef_add[1]
  
            MSE_add = mean_squared_error(Y_add, Y_add_pred)
            RMES_add = np.sqrt(mean_squared_error(Y_add, Y_add_pred))
            
            #　R2の計算
            R2_add =  r2_score(Y_add, Y_add_pred)  
            
            ax.text(0.99, 0.05*2+0.01, 'RMSE_add: ' + '{:.3f}'.format(RMES_add)+', R$^{2}$_add: ' + '{:.2f}'.format(R2_add), horizontalalignment='right', transform=ax.transAxes, fontsize=12, c='blue')
        
        else: ()
    
        #==========  ここまで，近似直線の計算　============
    
    

    

        #==========  以下，図のファイル名用　============
        #全体のタイトル名　　手入力
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
        fig.suptitle(title_head2,fontsize=20)
        

    else:()
            
    
    
    
    
    
    
    
    ##############################################################################
    # 画像保存
    ##############################################################################
    
    #######################画像を保存するためのボタン作成########################
    sub_title2 = main_title2
    sub_title2 = sub_title2.replace(':', '') 
    sub_title2 = sub_title2.replace(',', '_') 
    sub_title2 = sub_title2.replace(' ', '') 
    sub_tite = str('Fig_sal_d18O_SW'+'_'+sub_title2+".png")


        

    #Save to memory first. の場合は，ローカルに保存されないので安心
    import io
    fn = sub_tite
    img = io.BytesIO()
    plt.savefig(img, format='png')
     
    btn = st.download_button(
       label="Download image",
       data=img,
       file_name=fn,
       mime="image/png")
    

    st.pyplot(fig)
   

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    

    # 選択されたデータの地点プロット
    # --- 採取地点の地図表示 (Auto-Zoom & 幅広設定) ---
    st.divider()
    st.subheader('Location Map (Auto-Zoom)')

    import math

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
    



    
    
    
