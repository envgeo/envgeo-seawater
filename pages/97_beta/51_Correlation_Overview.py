#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:15:03 2023
@author: Toyoho Ishimura @Kyoto-U
2026/02/10 update 
"""

# --- バージョン管理の設定 ---
version = "0.2.12" #2026/02/23
fig_title = "envgeo-seawater-database"  # 2026/02/12
    

# compiledのみ，各図でExcelからの読み込みをしているので，ちょっと重いかも



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import cartopy.crs as ccrs
import envgeo_utils  # 作った設定ファイルを読み込む


print((f"--------13_compiled_figs(saliniry-d18O-dD-depth_etc)_({version})--------"))
# datetimeモジュールを使った現在の日付と時刻の取得
import datetime
dt = datetime.datetime.today()  # ローカルな現在の日付と時刻を取得
print(dt)  # 2021-10-29 15:58:08.356501


st.header(f'Compiled figs ({version})')
# st.subheader('(saliniry-d18O-dD-depth_etc)')




############ページタイトル設定############

# st.set_page_config(
#     page_title="d18O mapping",
#     # page_icon="🗾",
#     layout="wide"
#     )









def main():
    
    
    # リロードボタン
    st.button('Reload')
    
    
    # データソース選択
    # st.write("data source:", (data_source_JAPAN_SEA))
    st.write(':blue[data source:]  Kodama et al. (2024) [ECS - Japan Sea] ')
    # ref_data = st.radio("data source (see home>about):", (data_source_JAPAN_SEA, "Kodama et al. (2024) with other reports", data_source_GLOBAL), horizontal=True, args=[1, 0])
    # ref_data = st.radio("data source (see home>about):", (data_source_JAPAN_SEA, "Kodama et al. (2024) with other reports"), horizontal=True, args=[1, 0])

    
    # データソースの変数、envgeo_utilsから読み出す
    data_source_JAPAN_SEA = envgeo_utils.data_source_JAPAN_SEA
    data_source_AROUND_JAPAN = envgeo_utils.data_source_AROUND_JAPAN
    data_source_GLOBAL = envgeo_utils.data_source_GLOBAL
    

    # データソース選択
    # ref_data = st.radio("data source (see home>about):", (data_source_JAPAN_SEA, data_source_AROUND_JAPAN), horizontal=True, args=[1, 0])



    ref_data = data_source_JAPAN_SEA

    # # 注意書き
    # if ref_data == data_source_JAPAN_SEA:
    #     st.write(':blue[data source:]  Kodama et al. (2024)')
        
    # elif ref_data == "Kodama et al. (2024) with other reports":
    #     # st.text('including data from previous reports')
    #     st.write(':blue[data source:] Kodama et al. (2024), Yamamoto et al. (2001), Sakamoto et al. (2019), Kodaira et al. (2016), Horikawa et al. (2023).')

        
    # else:
    #     # st.text('including data from previous reports')

    #     st.write(':blue[data source:] Kodama et al. (2024), Yamamoto et al. (2001), Sakamoto et al. (2019), Kodaira et al. (2016), Horikawa et al. (2023)')
    #     st.write(':blue[with:] NASA_database (Jan.23, 2025)]https://data.giss.nasa.gov/cgi-bin/o18data/geto18.cgi')




    # メインのDF,これは改変しない
    df_original = envgeo_utils.load_isotope_data(ref_data)
    
    
    
    
######  scalebarで制御してsubmitする場合 ################

    with st.sidebar.form("parameter", clear_on_submit=False):
        
        st.header('select parameters ➡ submit')
        

##############################################################################
# サイドバーここから
##############################################################################

        
        
        st.form_submit_button(":red[submit]")

        ###このセットでsubmitボタン二つに出来る
        # st.form_submit_button(":red[submit (TOP)]")
        # submitted = st.form_submit_button(":red[submit (BOTTOM)]")
        
        
        #　一つだけの時は以下
        # submitted = st.form_submit_button(":red[submit]")
        
        
        
        st.subheader(':blue[--- for data range ---]') 
        
        
        
        
        
    
    
        #年の範囲       # サブレベルヘッダ
        # st.sidebar.subheader('年の範囲')
        
        sld_year_min, sld_year_max = st.slider(label='Year selected',
                                    min_value=2013,
                                    max_value=2022,
                                    value=(2013, 2022),
                                    )
        # st.sidebar.write(f'Selected: {sld_year_min} ~ {sld_year_max}')
        
        #月の範囲       # サブレベルヘッダ
        # st.sidebar.subheader('月の範囲')
        
        
        # --- 月　スライダー版（これを書き換え） ---
        # sld_month_min, sld_month_max = st.slider(label='Month selected',
        #                             min_value=1,
        #                             max_value=12,
        #                             value=(1, 12),
        #                             )
        # # st.sidebar.write(f'Selected: {sld_month_min} ~ {sld_month_max}')
        

        # --- 月　multiselect版 ---
        selected_months = st.multiselect(
            label='Month selected',
            options=list(range(1, 13)),  # 1〜12の選択肢
            default=list(range(1, 13))   # 初期状態は全選択
        )
        
        
        #経度longitudeの範囲   
        # st.sidebar.subheader('経度の範囲')
        sld_lon_min, sld_lon_max = st.slider(label='Longitude selected',
                                    min_value=115,
                                    max_value=145,
                                    value=(115, 145),
                                    )
        # st.sidebar.write(f'Selected: {sld_lon_min} ~ {sld_lon_max}')
        
        
        #緯度の範囲   
        # st.sidebar.subheader('緯度の範囲')
        sld_lat_min, sld_lat_max = st.slider(label='Latitude selected',
                                    min_value=20,
                                    max_value=45,
                                    value=(20, 45),
                                    )
        # st.sidebar.write(f'Selected: {sld_lat_min} ~ {sld_lat_max}')
        
        
        #水深の範囲   
        # st.sidebar.subheader('水深の範囲')
        sld_depth_min, sld_depth_max = st.slider(label='Water depth selected',
                                    min_value=0,
                                    max_value=1000,
                                    value=(0, 1000),
                                    )
        # st.sidebar.write(f'Selected: {sld_depth_min} ~ {sld_depth_max}')
        
        #塩分の範囲   
        # st.sidebar.subheader('塩分の範囲')
        sld_sal_min, sld_sal_max = st.slider(label='Salinity selected',
                                    min_value=0,
                                    max_value=40,
                                    value=(20, 38),
                                    )
        # st.sidebar.write(f'Selected: {sld_sal_min} ~ {sld_sal_max}')
        
        
        
        
        # envgeo_utilsからデータフレームを読み出し
        # compiledだけ他と異なる設定ですよ
        df_sideber = df_original
        
        # st.sidebar.subheader('航海区の範囲')dfから要素抽出
        # 例：列名が "RockType" の場合
        Transect_list = df_sideber["Transect"].dropna().unique().tolist()
        print(Transect_list,"AAA")
        
        selected_cruise = st.multiselect('Choose cruise area', Transect_list,default=Transect_list)
        
        

        
        
        
        st.write('Cruise Area (2015-2021)')
        st.image("data/sites_20230515.gif") 
    
          
          
        
        #スペース入れる
        # st.subheader(':blue[  ]')
        # st.subheader(':blue[  ]')
        st.subheader(':blue[--- for fig scale only ---]')
        
        
        #地図の描画範囲（拡大）
        # # 120-0.001, 145+0.001, 20-0.001, 45+0.001
        
        # # st.sidebar.subheader('地図の経度の範囲（拡大）')
        # map_lon_min, map_lon_max = st.sidebar.slider(label='Map Longitude selected',
        #                             min_value=120-0.001,
        #                             max_value=145+0.001,
        #                             value=(120-0.001, 145+0.001),
        #                             )
        # # st.sidebar.write(f'Selected: {map_lon_min} ~ {map_lon_max}')
        
        # # st.sidebar.subheader('地図の緯度の範囲（拡大）')
        # map_lat_min, map_lat_max = st.sidebar.slider(label='Map Latitude selected',
        #                             min_value=20-0.001,
        #                             max_value=45+0.001,
        #                             value=(20-0.001, 45+0.001),
                                    # )
        # # st.sidebar.write(f'Selected: {map_lat_min} ~ {map_lat_max}')
        
        
        # st.sidebar.subheader('描画水深の範囲')
        fig_depth_min, fig_depth_max = st.slider(label='Water depth selected',
                                    min_value=0,
                                    max_value=1000,
                                    value=(0, 500),
                                    )
        # st.sidebar.write(f'Selected: {fig_depth_min} ~ {fig_depth_max}')
    
                            
        submitted = st.form_submit_button(":red[submit!]")

##############################################################################
# サイドバーここまで
##############################################################################





    
    
    # """手動設定項目"""
    #####################################
    ######    EXCEL BOOK import     #####
    #####################################
                    
                        
        
    fig = plt.figure(figsize = (18, 24),dpi=150)
    
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    
    
    
    #元データ読み込み
    # excel_file = 'd18O_20230513-1_NA2_2021add.xlsx'
    # sheet_num = 2
    
    # df = pd.read_excel(excel_file, sheet_name=sheet_num)
    
    
    # """図のフォント設定、サイズも"""
    ##### ベースのフォントとフォントサイズの指定
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams["font.size"] = 10
    
    
    
    
    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    

    
    # df_original = envgeo_utils.load_isotope_data(ref_data, sheet_num=1)
    
    # """選択描画範囲の設定用"""
    def data_limit(sheet_num = 0):
    
            # sheet_num = 1
            # df1 = pd.read_excel(excel_file, sheet_name=sheet_num)
            df1 = envgeo_utils.load_isotope_data(ref_data)
            
            # df1 = df_fig_ALL
        
            #緯度経度と水深とTransectで制限
            # df1 = df_fig_add
            
            df1 = df1[(df1['Depth_m'] == 'xxx') 
                        
                        |(df1['Depth_m'] <= sld_depth_max) & (df1['Depth_m'] >= sld_depth_min )#調整用
                        | df1.isnull().all(axis=1)]      
            
            df1 = df1[(df1['Transect'].isin(selected_cruise))
                        | df1.isnull().all(axis=1)]
                




            # Transectのデータ制限せず，サイドバーで制限されたまま出力する場合
            df1 = df1[df1['Transect'].isin(selected_cruise) | df1['Transect'].isna()].copy()
                # なぜこれで青色の線のスキマが復活するのか
                # 元のデータ (df):
                # 　　「データA」→「データA」→「空白行」→「データA」
                # これまでの抽出 (df1):
                #　　 reference が「データA」のものだけを抽出した結果、「空白行」が条件に合わず消されてしまい、「データA」同士が隣り合って線が繋がってしまった
                # 新しい抽出:
                #　　| df['reference'].isna() （または reference が空であること）という条件を加えることで、「データA」の間にあった「空白行」も一緒に df1 にコピーされるようになる

            
            # # Transectの手動でデータ制限する場合
            # df1 = df1[ (df1['Transect'] == 0) 
            #             | (df1['Transect'] == Transect001) 
            #             | (df1['Transect'] == Transect002)
            #             | (df1['Transect'] == Transect003)
            #             | (df1['Transect'] == Transect004)
            #             | (df1['Transect'] == Transect005)
            #             | (df1['Transect'] == Transect006)
            #             | (df1['Transect'] == Transect007)
            #             | (df1['Transect'] == Transect008)
            #             | (df1['Transect'] == Transect009)
            #             | (df1['Transect'] == Transect010)
            #             | (df1['Transect'] == Transect011)
            #             | (df1['Transect'] == Transect012)
            #             | (df1['Transect'] == Transect013)
            #             | (df1['Transect'] == Transect014)
            #             # | (df1['Transect'] == Transect015)
            #             # | (df1['Transect'] == Transect016)
            #             # | (df1['Transect'] == Transect017)
            #             # | (df1['Transect'] == Transect018)
            #             # | (df1['Transect'] == Transect019)
            #             # | (df1['Transect'] == Transect020)
            #             # | (df1['Transect'] == Transect021)
            #             # | (df1['Transect'] == Transect022)
            #             # | (df1['Transect'] == Transect023)
            #             # | (df1['Transect'] == Transect024)
            #             # | (df1['Transect'] == Transect025)
            #             # | (df1['Transect'] == Transect026)
            #             # | (df1['Transect'] == Transect027)
            #             # | (df1['Transect'] == Transect028)
            #             # | (df1['Transect'] == Transect029)
            #             # | (df1['Transect'] == Transect030)
            #             | df1.isnull().all(axis=1)]      
              

            df1 = df1[(df1['Longitude_degE'] == 'xxx') 
                        |(df1['Longitude_degE'] <= sld_lon_max) & (df1['Longitude_degE'] >= sld_lon_min) #調整用
                        | df1.isnull().all(axis=1)]      
              
            df1 = df1[(df1['Latitude_degN'] == 'xxx')
                        |(df1['Latitude_degN'] <= sld_lat_max) & (df1['Latitude_degN'] >= sld_lat_min) #調整用
                        | df1.isnull().all(axis=1)]      
        
            # --- 月 (スライダー用) ---
            # df1 = df1[(df1['Month'] == 'xxx')
                        
            #             |(df1['Month'] <= sld_month_max) & (df1['Month'] >= sld_month_min)
            #             | df1.isnull().all(axis=1)]
              
            # --- 月 (multiselect用) ---
            if selected_months:
                # isin で選ばれた月を抽出
                # | (または)
                # df1.isnull().all(axis=1) で全ての列が空の行（挿入した空白行）を抽出
                df1 = df1[df1['Month'].isin(selected_months) | df1.isnull().all(axis=1)]
            else:
                # 月が一つも選ばれていない場合でも、空白行だけは残す
                df1 = df1[df1.isnull().all(axis=1)]        
              
            df1 = df1[(df1['Salinity'] == 'xxx')        
                        |(df1['Salinity'] >= sld_sal_min) & (df1['Salinity'] <= sld_sal_max)
                        | df1.isnull().all(axis=1)]      
            
            #描画する年範囲を指定
            df1 = df1[(df1['Year'] <= sld_year_max) & (df1['Year'] >= sld_year_min) 
                        | df1.isnull().all(axis=1)] 

     
            return df1
    
    
    
    
        
    #全体のタイトル名　　手入力
    # main_title = 'DEPTH PROFILE (V02)'
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

    # sub_title = (
    #     f"Lon:{sld_lon_min}-{sld_lon_max}, Lat:{sld_lat_min}-{sld_lat_max}, "
    #     f"Y:{sld_year_min}-{sld_year_max}, M:[{month_text}], "
    #     f"S:{sld_sal_min}-{sld_sal_max}, D:{sld_depth_min}-{sld_depth_max}m"
    # )
    main_title2 = sub_title
    
    sub_title2 = ''
    
    title_head = str(main_title+'\n'+sub_title+'\n'+sub_title2)
    title_head2 = title_head.replace('_', ' ') #図のタイトル表示用
    fig.suptitle(title_head2,fontsize=30)
    
    

    
    # """選択描画範囲のlabel，手入力"""

    
    sheet_names_add2 = 'selected'
    # sheet_names_add2 =  'N:25-130,E:135-140,D:>10m'
    
    
    
    #追加でさらに特定のクルーズのみplotする場合1
    selected_add3 = 2
    selected_row2 = 'Cruise'
    selected_value = 'YK1606'
    # selected_value = 'xxx'
    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """top right """
    # """水深"""
    
    
    
    #####################################
    ######    EXCEL SHEET select    #####
    #####################################
    
    # """EXCELブックのシート選択、シートごとの描画色も"""
    #シート２はdepthプロファイルのみ抽出したシート
    # sheet_num = [2]
    
    sheet_num = [0]
    ######　プロットの色選択，sheet_numの順番に対応 10以上の数がある場合には色を追加 ######
    color = ["red","lime","blue","green","darkcyan","cyan","orange","yellow","fuchsia","violet","greenyellow"]
    
    # color = ["blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue"]
    
    
    ############################################
    ######      font size line etc..       #####
    ############################################
    
    # """凡例（legend）を入れるかどうか"""
    legend = 1 #1以外だと凡例無し
    
    
    # """図のサイズと解像度"""
    # fig_size = [4,9] #図のサイズ
    # fig_dpi = 150 #図の解像度
    
    # """線の太さとマーカーのサイズ"""
    ##### 線の太さとマーカーのサイズと種類
    # lw_select = 2 #プロットラインの太さ
    # marker_select = '.' #プロットマーカーの種類，種類はweb参照
    # ms_select = 15 #プロットマーカーのサイズ
    
    #軸のメモリの長さ
    ax_length = 5
    
    #ラインとプロットの透明度
    # fig_alpha = 0.85 #不透明度、1は透過なし,0.1-1の間の数
    
    
    
    
    ########################################
    ######    SUB FIG: d13C vs d18O    #####
    ########################################
    
    
    # """XYの列を指定 エクセルシートから"""
    
    #水深-d18Oの時
    X_data = "d18O"
    Y_data = "Depth_m"
    
    
    
    # """XYの表示用のラベルを指定"""
    
    X_label = r"$\delta^{18}$O"
    Y_label = "water depth (m)"
    
    
    
    # """XYの表示用のラベルのスケールを指定"""
    
    iso_scale_X = "(VSMOW)"
    iso_scale_Y = ""
    
    
    
    # """作図用,色とマーカーとサイズやタイトルも"""
    
    sheet_num_XY = sheet_num #ここは変更しない
    ###### 別途，X_Yのプロットをするかどうか,色を一括にするか ######
    #する場合は1,しない場合は2
    X_Y = 1
    
    #メインプロットの設定
    X_Y_C = "gray" #色の設定
    X_Y_M = " " #現時点で色は変更設定なしマーカーの種類
    X_Y_S = 1
    
    
    #全体の回帰直線を書く場合「１」　書かない場合には「２」
    # reg_line_write = 2
    
    
    
    
    
    # 追加で強調プロットをする場合は,d13C_d18O_addを「1」しない場合は「2」　シートナンバー選択、
    X_Y_add = 1
    sheet_num_add = sheet_num
    
    
    
    # 条件抽出する場合「1」しない場合は「2」　
    fig_add_sort = 1
    selected_row = "Transect"
    
    # selected_area = 'CK'
    # selected_area = 'Nansei'
    # selected_area = 'nECS'
    # selected_area = 'Noto'
    # selected_area = 'Pacific'
    # selected_area = 'Pacific_west'
    # selected_area = 'sECS'
    # selected_area = 'Shimane&Tottori'
    # selected_area = 'SI'
    # selected_area = 'Toyama'
    # selected_area = 'Tsushima'
    # selected_area = 'Yamato'
    # selected_area = 'NA2'
    # selected_area = 'ECS2021'
    
    
    
    
    
    
    
    #タイトル
    fig_title_X_Y= X_label + " - "+ Y_label + "" #d13C_d18O書き出し専用
    
    #プロットの透明度
    alpha_all = 0.4 #メインプロット
    alpha_selected = 1 #強調プロット
    
    #強調プロットの色の指定
    # X_Y_C_add =  "blue" #単色にしたい場合
    X_Y_C_add_each = 1 #シート毎に塗り分けたい場合は「１」　そうでなければ「２」
    
    #個別に近似直線を引く場合「１」，引かない場合はそれ以外の数字
    # reg_line_add_write = 2
    
    
    
    #############################################
    ######      data range for SUB FIG      #####
    #############################################
    
    
    # #水深-d18Oの時
    lim_min_X = -1.4
    lim_max_X = 0.6
    lim_min_Y = fig_depth_max
    # lim_min_Y = 1000
    lim_max_Y = fig_depth_min
    

    
    # """------------------ここから先はさわらない！----------------------"""
    # """以下の設定は基本的に変更しない"""
    
    
    #もとのエクセルファイルのシートリストを表示
    # print()
    # sheet_all = pd.read_excel(excel_file, sheet_name=None)
    # print("選択したExcelのSheetリスト:",list(sheet_all.keys()))
    
    
    
    
    
    
    # """ここからdepth"""
    if X_Y == 1:
        # sheet_num_XY = [3,4,5,6,7,8]
        
        print('-------------SUB_FIG   depth vs d18O-------------')
        
        # ax = plt.subplot(323)
        # fig = plt.figure()
        # grid = plt.GridSpec(3,2, wspace=0.76, hspace=0.45)
        grid = plt.GridSpec(3,2, )
        ax = fig.add_subplot(grid[1:3, 0])
        
        ax.set_xlabel(X_label + iso_scale_X, fontsize=15)
        ax.set_ylabel(Y_label + iso_scale_Y, fontsize=15)  #LateX形式で特殊文字
    
        
        # input_sheet_name = pd.ExcelFile(excel_file).sheet_names
        # for sheet_num in sheet_num:    
        #     print("読み込まれたSheet:", [sheet_num], input_sheet_name[sheet_num])
        
        for sheet_num_XY in sheet_num_XY:
            # df_fig_ALL = pd.read_excel(excel_file, sheet_name=input_sheet_name[sheet_num_XY])
    
     #Excelファイルの読み込み
            # df_fig_ALL = pd.read_excel(excel_file, sheet_name=sheet_num_XY)
            # df_fig_ALL = envgeo_utils.load_isotope_data(ref_data, sheet_num=2)
            
            
            # 同じ地点，同じ年月日，はグループにして他は1行開ける
            df_fig_ALL = envgeo_utils.insert_gap_rows(df_original)
            
            
            # 特定の列に特定の変数を持つ行と空白行を残す
            # df_fig_ALL = df_fig_ALL[(df_fig_ALL[selected_row] == () | df_fig_ALL.isnull().all(axis=1)]
            plt.plot(df_fig_ALL[X_data], df_fig_ALL[Y_data],c=X_Y_C, marker=X_Y_M, lw=0.5, alpha=alpha_all, label='ALL')
            
            #列の要素を表示
            # d_select = df_fiｇ_add[selected_row].value_counts().to_dict()
            # print('要素と出現数:', d_select)
            # print('---------------')
    
    
            plt.legend(fontsize = 15) # 凡例の数字のフォントサイズを設定
    
    
            ax.set_xlim(lim_min_X, lim_max_X) 
            ax.set_ylim(lim_min_Y, lim_max_Y) 
            plt.tick_params(labelsize=15)
            ax.set_xticks(np.linspace(lim_min_X, lim_max_X,11))
            ax.set_yticks(np.linspace(lim_min_Y, lim_max_Y, 11))
            
            
    
            ax.xaxis.set_major_formatter(FormatStrFormatter("%+.1f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.f"))
                
    
            ax.tick_params(length=ax_length)
            # ax.annotate("point A", xy = (-7, 0), size = 15,
            #             color = "red", arrowprops = dict())
    
        plt.title(fig_title_X_Y, fontsize=20) #
        plt.legend(fontsize = 15) # 凡例の数字のフォントサイズを設定
        
    
        
        #追加で強調プロットをする場合
        if X_Y_add == 1:
            # sheet_num_add = [1]
            # sheet_num_add = [2,3]
            for sheet_num_add in sheet_num_add:    
        
                if X_Y_C_add_each == 1:
                    
                    # plt.title(fig_title_X_Y+'_with_selected', fontsize=20) #
                    
                    X_Y_C_add  =  color[sheet_num_add] #メインFigと同じくシート毎に分けたい場合
                    
           
                    # # Excelファイルの読み込み
                    # df_fiｇ_add = pd.read_excel(excel_file, sheet_name=sheet_num_add)
                    
                    # df1 = df_fiｇ_add
                    
                    # df1 = df1[(df1['Depth_m'] == 'xxx') 
                    #         |(df1['Depth_m'] <= 10) & (df1['Depth_m'] >= 0)
                    #         |(df1['Depth_m'] <= 200) & (df1['Depth_m'] > 10)
                    #         |(df1['Depth_m'] <= 500) & (df1['Depth_m'] > 200)
                    #         # |(df1['Depth_m'] <= 1000) & (df1['Depth_m'] > 500)
                    #           | df_fiｇ_add.isnull().all(axis=1)] 
                    
                    # df1 = df1[ (df1['Transect'] == 0) 
                    #             | (df1['Transect'] == 'CK') 
                    #             # | (df1['Transect'] == 'Nansei') 
                    #             # | (df1['Transect'] == 'nECS') 
                    #             | (df1['Transect'] == 'Noto') 
                    #             # | (df1['Transect'] == 'Pacific') 
                    #             # | (df1['Transect'] == 'Pacific_west') 
                    #             # | (df1['Transect'] == 'sECS')          
                    #             # | (df1['Transect'] == 'Shimane&Tottori')          
                    #             # | (df1['Transect'] == 'SI')
                    #             | (df1['Transect'] == 'Toyama')
                    #             # | (df1['Transect'] == 'Tsushima')
                    #             | (df1['Transect'] == 'Yamato')
                    #             # | (df1['Transect'] == 'NA2') 
                    #             # | (df1['Transect'] == 'ECS2021') 
                    #             | df_fiｇ_add.isnull().all(axis=1)]
    
    
                    # # #描画する緯度経度を指定 
                    # df1 = df1[(df1['Longitude_degE'] == 'xxx') 
                    #             |(df1['Longitude_degE'] <= 145) & (df1['Longitude_degE'] > 140)    
                    #             |(df1['Longitude_degE'] <= 140) & (df1['Longitude_degE'] > 135)         
                    #             |(df1['Longitude_degE'] <= 135) & (df1['Longitude_degE'] > 130)
                    #             |(df1['Longitude_degE'] <= 130) & (df1['Longitude_degE'] > 125)
                    #             |(df1['Longitude_degE'] <= 125) & (df1['Longitude_degE'] > 120)
                    #             |(df1['Longitude_degE'] <= 120) & (df1['Longitude_degE'] >= 115)
                    #           | df_fiｇ_add.isnull().all(axis=1)] 
    
                    # df1 = df1[(df1['Latitude_degN'] == 'xxx')
                    #           |(df1['Latitude_degN'] <= 45) & (df1['Latitude_degN'] > 40)          
                    #           |(df1['Latitude_degN'] <= 40) & (df1['Latitude_degN'] > 35)
                    #           |(df1['Latitude_degN'] <= 35) & (df1['Latitude_degN'] > 30)
                    #           |(df1['Latitude_degN'] <= 30) & (df1['Latitude_degN'] > 25)
                    #           |(df1['Latitude_degN'] <= 25) & (df1['Latitude_degN'] >= 20)
                    #           | df_fiｇ_add.isnull().all(axis=1)]
                    # df_fiｇ_add = df1
                    
                    # df1 = data_limit(sheet_num=2)
                    df1 = data_limit()

                    # df_fig_add = df1
                    
                    # 同じ地点，同じ年月日，はグループにして他は1行開ける
                    df_fig_add = envgeo_utils.insert_gap_rows(df1)
                    df1 = df_fig_add
                    
                    #df1が空になっているかどうかを確認する
                    df30m = df1[(df1['Depth_m'] == 'xxx') 
                            |(df1['Depth_m'] <= 1000) & (df1['Depth_m'] >= 30)]
                            # |(df1['Depth_m'] <= 200) & (df1['Depth_m'] > 10)
                            # |(df1['Depth_m'] <= 500) & (df1['Depth_m'] > 200)
                            # |(df1['Depth_m'] <= 1000) & (df1['Depth_m'] > 500)
                              # | df_fiｇ_add.isnull().all(axis=1)] 
   
                    
                    # df30m_empty = df30m.empty
                
                    # st.write(df_empty)
                    data_found_num = str(len(df30m["d18O"]))
                
                    
                
                    #選んだパラメーター表示
                    # st.write('YEAR:'+str(sld_year_min)+'-'+str(sld_year_max)+', ' 
                    #           +'MONTH:'+str(sld_month_min)+'-'+str(sld_month_max)+', '
                    #           +'Longitude:'+str(sld_lon_min)+'-'+str(sld_lon_max)+', '
                    #           +'Latitude:'+str(sld_lat_min)+'-'+str(sld_lat_max)+', '
                    #           +'Water_depth:'+str(sld_depth_min)+'-'+str(sld_depth_max)+', '
                    #           +'Salinity:'+str(sld_sal_min)+'-'+str(sld_sal_max))
                    # st.write('Area(Cruise)',selected_cruise)
                    # selected_cruise_indicate =str(list(selected_cruise[:]))
                    # st.write('Selected Area (Cruise)', selected_cruise_indicate)
                    # print('Area(Cruise)', list(selected_cruise[:]))
                    #テキストの色変更
                    # st.write(""":red['test']""")
                
                
                
                
        
                
                    # # バリデーション処理
                    # if df30m_empty == 1:  #データが無かったとき
                    #     st.warning('no data found')
                    #     # 条件を満たないときは処理を停止する
                    #     st.stop()
                    # elif df30m_empty == 0: #データがあったとき
                    #     st.write(data_found_num,'data found for depth profile (below 30m)')
                    st.write(data_found_num,'data found for depth profile (below 30m)')
                    
                    
                    
                    
       
                    
                    
                    
                       
                    #########月ごとに色分けする場合######################
                    lw_add = 0.6 #線の太さ
                    # 描画する月範囲を指定 and指定
                    df13 = df1[(df1['Month'] >= 1) & (df1['Month'] <= 3)
                              | df_fiｇ_add.isnull().all(axis=1)]  
                    plt.plot(df13[X_data], df13[Y_data],c='blue', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='1-3')
                    
                    # 描画する月範囲を指定 and指定
                    df46 = df1[(df1['Month'] >= 4) & (df1['Month'] <= 6)
                              | df_fiｇ_add.isnull().all(axis=1)]  
                    plt.plot(df46[X_data], df46[Y_data],c='green', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='4-6')
                    
                    # 描画する月範囲を指定 and指定
                    df79 = df1[(df1['Month'] >= 7) & (df1['Month'] <= 9)
                              | df_fiｇ_add.isnull().all(axis=1)]  
                    plt.plot(df79[X_data], df79[Y_data],c='orange', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='7-9')
                    # 描画する月範囲を指定 and指定
                    df1012 = df1[(df1['Month'] >= 10) & (df1['Month'] <= 12)
                              | df_fiｇ_add.isnull().all(axis=1)]  
                    plt.plot(df1012[X_data], df1012[Y_data],c='purple', marker=X_Y_M, lw=lw_add, alpha=alpha_selected, label='10-12')
                    
                    
                    # 特定の列に特定の変数を持つ行と空白行を残す
                    # df_fiｇ_add = df_fiｇ_add[(df_fiｇ_add[selected_row] == selected_area) | df_fiｇ_add.isnull().all(axis=1)]
                    # df_fiｇ_add = df_fiｇ_add[(df_fiｇ_add[selected_row] == selected_area) | df_fiｇ_add.isnull().all(axis=1)]
                    
                    
                    
                    
                    
                    #########全部plotする場合######################    
                    
                    #列の要素を表示
                    d_select = df_fiｇ_add[selected_row].value_counts().to_dict()
                    print('要素と出現数:', d_select)
                    print('---------------')
    
        
                    plt.legend(fontsize = 15) # 凡例の数字のフォントサイズを設定
    
                    
                    
                    #############################
                    # 今は使っていない，今後の為。追加プロット用
                    if selected_add3 == 1:
    
                        #個別に色を変えてもう一つプロット
                        # Excelファイルの読み込み
                        # df_fiｇ_add = pd.read_excel(excel_file, sheet_name=sheet_num_add)
                        # df_fiｇ_add = envgeo_utils.load_isotope_data(ref_data, sheet_num=2)
                        
                        # 同じ地点，同じ年月日，はグループにして他は1行開ける
                        df_fiｇ_add = envgeo_utils.insert_gap_rows(df_original)
                        
                        selected_row2 = selected_row2
                        selected_value = selected_value
                        
                        # 特定の列に特定の変数を持つ行と空白行を残す
                        df_fiｇ_add = df_fiｇ_add[(df_fiｇ_add[selected_row2] == selected_value) | df_fiｇ_add.isnull().all(axis=1)]
                        plt.plot(df_fiｇ_add[X_data], df_fiｇ_add[Y_data],c='red', marker=X_Y_M, lw=2, alpha=alpha_selected, label=selected_value)
                        plt.legend(fontsize = 15) # 凡例の数字のフォントサイズを設定
                        
                    else:()
                      
                    #############################
            
                else:() 
                
        else:()
    else:()
    
    print("############ DONE ############")
    # """DONE"""
    
        
    
    
    
    
    
    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """middle-bottom left """
    # """採取地点図 水深30m以深のデプスプロファイルがあるもの"""
    
    
    
    #日本地図描画
    ax = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree())
    
    ax.set_global()
    ax.coastlines()
    # ax.set_extent([120-0.001, 145+0.001, 20-0.001, 45+0.001], crs=ccrs.PlateCarree())
    # ax.set_extent([map_lon_min, map_lon_max, map_lat_min, map_lat_max], crs=ccrs.PlateCarree())
    
    ax.set_xlim([120-0.01, 145+0.01])
    ax.set_ylim([20-0.01, 45+0.01])
    gl = ax.gridlines(draw_labels=True)
    
    
    
    #描画する水深範囲を指定 m
    df1 = df1[(df1['Depth_m'] == 'xxx') 
                # |(df1['Depth_m'] < 30) & (df1['Depth_m'] >= 0)
              |(df1['Depth_m'] <= 500) & (df1['Depth_m'] >= 30)
              |(df1['Depth_m'] <= 1000) & (df1['Depth_m'] > 500)
              ] 
    
    
    # """水深図から条件を引用する場合には以下はいらない"""
    
    
    
    # #描画　鉛直サンプリングの全観測点
    # df_depth_all = pd.read_excel(excel_file, sheet_name=sheet_num_add)
    df_depth_all = envgeo_utils.load_isotope_data(ref_data, sheet_num=sheet_num_add) 
    

    
    
    plt.scatter(df_depth_all["Longitude_degE"], df_depth_all["Latitude_degN"], c='lightblue', s=10, alpha=1, transform=ccrs.PlateCarree(), label="ALL")
    
    #描画　選択した観測点　単一職
    plt.scatter(df1["Longitude_degE"], df1["Latitude_degN"], c='red', s=10, alpha=1, transform=ccrs.PlateCarree(), label='selected')
    
    
    
    
    ax.set_title('vertical sampling sites (below 30m)', fontsize=20) #Transectでソートした場合           
    plt.legend(fontsize = 15,loc='lower right',bbox_to_anchor=(1, 0.13)) # 凡例の数字のフォントサイズを設定
    
    # #列の要素を表示
    # d_select = df1['Transect'].value_counts().to_dict()
    # print('要素と出現数:', d_select)
    # print('---------------')
    
    
    
    
    
    
    
    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # # """"""""""""""""""""""""""" 　　　右の図はここから　　　　"""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # #ここからシートナンバー１   右側の３つの図
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    
    
    
    
    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # # """middle right """
    
    
    
    # # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # # """top right """
    # # 全てのサンプリングポイント
    
    
 
    #元データ読み込み
    # df = pd.read_excel("d18O_20210626-2_test.xlsx")
    
    # excel_file = 'd18O_20210626-3_NA2.xlsx'
    #全体のplotは元データ全てを使うので，シート１
    sheet_num = 0
    
    # df = pd.read_excel(excel_file, sheet_name=sheet_num)
    
    df = envgeo_utils.load_isotope_data(ref_data, sheet_num=sheet_num) 
    
    #PDFに書き出すかどうか
    # PDF_export_SUB = 2
    
    
    
    #日本地図描画
    ax = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree())
    # ax = fig.add_subplot(grid[0, 0])
    
    ax.set_global()
    ax.coastlines()
    # ax.set_extent([120-0.001, 145+0.001, 20-0.001, 45+0.001], crs=ccrs.PlateCarree())
    # ax.set_extent([map_lon_min, map_lon_max, map_lat_min, map_lat_max], crs=ccrs.PlateCarree()) #これがあるとstreamlit cloud動かない？
        
    ax.set_xlim([120-0.01, 145+0.01])
    ax.set_ylim([20-0.01, 45+0.01])
    
    gl = ax.gridlines(draw_labels=True)
    
    
    
    #描画する水深範囲を指定 m　全データのプロット用
    df1 = df[(df['Depth_m'] == 'xxx') 
              |(df['Depth_m'] <=30) & (df['Depth_m'] >= 0)
                # |(df['Depth_m'] <= 500) & (df['Depth_m'] >= 30)
                # |(df['Depth_m'] <= 1000) & (df['Depth_m'] >= 501)
              ] 
    #描画する年範囲を指定
    # df1 = df1[(df1['Year'] <= 2022) & (df['Year'] >= 2014)] 
    
    #描画する月範囲を指定 and指定
    # df1 = df1[(df1['Month'] >= 5) & (df['Month'] <= 10)] 
    #描画する月範囲を指定 or指定
    # df1 = df1[(df1['Month'] >= 11) | (df['Month'] <= 4)] 
    #描画する月範囲を指定
    # df1 = df1[(df1['Transect'] == "Noto") & (df['Transect'] == "Noto")] 
    #描画するPI指定
    # df1 = df1[(df1['PI'] == "Kodama") | (df['PI'] == "Kitajima")] 
    
    
    #描画　鉛直サンプリングの全観測点
    plt.scatter(df1["Longitude_degE"], df1["Latitude_degN"], c='gray', s=2, alpha=1, transform=ccrs.PlateCarree(), label="ALL")
    
    
    # #描画するTransectを指定 一つだけの場合
    
    # # selected_area = 'CK'
    # # selected_area = 'Nansei'
    # # selected_area = 'nECS'
    # # selected_area = 'Noto'
    # # selected_area = 'Pacific'
    # # selected_area = 'Pacific_west'
    # # selected_area = 'sECS'
    # # selected_area = 'Shimane&Tottori'
    # # selected_area = 'SI'
    # # selected_area = 'Toyama'
    # # selected_area = 'Tsushima'
    # # selected_area = 'Yamato'
    # # selected_area = 'NA2'
    # # selected_area = 'ECS2021'
    
    # # df1 = df1[(df1['Transect'] == selected_area)] 
    
    
    # df1 = df1[(df1['Depth_m'] == 'xxx') 
    #             |(df1['Depth_m'] <= 10) & (df1['Depth_m'] >= 0)
    #             |(df1['Depth_m'] <= 200) & (df1['Depth_m'] > 10)
    #             |(df1['Depth_m'] <= 500) & (df1['Depth_m'] > 200)
    #             |(df1['Depth_m'] <= 1000) & (df1['Depth_m'] > 500)
    #           ] 
    
    # df1 = df1[ (df1['Transect'] == 0) 
    #             | (df1['Transect'] == 'CK') 
    #             # | (df1['Transect'] == 'Nansei') 
    #             # | (df1['Transect'] == 'nECS') 
    #             | (df1['Transect'] == 'Noto') 
    #             # | (df1['Transect'] == 'Pacific') 
    #             # | (df1['Transect'] == 'Pacific_west') 
    #             # | (df1['Transect'] == 'sECS')          
    #             # | (df1['Transect'] == 'Shimane&Tottori')          
    #             # | (df1['Transect'] == 'SI')
    #             | (df1['Transect'] == 'Toyama')
    #             | (df1['Transect'] == 'Tsushima')
    #             | (df1['Transect'] == 'Yamato')
    #             # | (df1['Transect'] == 'NA2') 
    #             ] 
    
    
    # # #描画する緯度経度を指定 
    # df1 = df1[(df1['Longitude_degE'] == 'xxx') 
    #             |(df1['Longitude_degE'] <= 145) & (df1['Longitude_degE'] >= 140)    
    #             |(df1['Longitude_degE'] <= 140) & (df1['Longitude_degE'] >= 135)         
    #             |(df1['Longitude_degE'] <= 135) & (df1['Longitude_degE'] >= 130)
    #             |(df1['Longitude_degE'] <= 130) & (df1['Longitude_degE'] >= 125)
    #             |(df1['Longitude_degE'] <= 125) & (df1['Longitude_degE'] >= 120)
    #             |(df1['Longitude_degE'] <= 120) & (df1['Longitude_degE'] >= 115)
    #             # |(df1['Longitude_degE'] <= 130) & (df1['Longitude_degE'] >= 128) #調整用
    #             ] 
    
    # df1 = df1[(df1['Latitude_degN'] == 'xxx')
    #             |(df1['Latitude_degN'] <= 45) & (df1['Latitude_degN'] >= 40)          
    #             |(df1['Latitude_degN'] <= 40) & (df1['Latitude_degN'] >= 35)
    #             |(df1['Latitude_degN'] <= 35) & (df1['Latitude_degN'] >= 30)
    #             |(df1['Latitude_degN'] <= 30) & (df1['Latitude_degN'] >= 25)
    #             |(df1['Latitude_degN'] <= 25) & (df1['Latitude_degN'] >= 20)
    #             # |(df1['Latitude_degN'] <= 33) & (df1['Latitude_degN'] >= 31) #調整用
    #           ]
              
    # df1 = df1[(df1['Month'] == 'xxx')
    #             # |(df1['Month'] <= 12) & (df1['Month'] >= 10)          
    #             |(df1['Month'] <= 9) & (df1['Month'] >= 7)
    #             # |(df1['Month'] <= 6) & (df1['Month'] >= 4)
    #             # |(df1['Month'] <= 3) & (df1['Month'] >= 1)      
              # ] 
                    
    df1 = data_limit()
    df_fig_add_salinity_d18O = df1
    
    #df1が空になっているかどうかを確認する
    df_empty = df1.empty

    # st.write(df_empty)
    data_found = len(df1["d18O"])
    data_found_num = str(len(df1["d18O"]))

    
    # バリデーション処理
    if df_empty == 1:  #データが無かったとき
        st.warning('no data found')
        # 条件を満たないときは処理を停止する
        st.stop()
    elif df_empty == 0: #データがあったとき
        st.write(data_found_num,'data found for all data')
    
              
    if data_found == 1:
        st.warning('only 1 data found , could not analyze the data')
        st.stop()
                    
    
    
    
    
                 
                     
    
                    
     
    
    
    
    
    
    
    
    
    #選択したデータをプロット
    df1 = df1[(df1['Depth_m'] == 'xxx') 
            |(df1['Depth_m'] <= 30) & (df1['Depth_m'] >= 0)
            |(df1['Depth_m'] <= 200) & (df1['Depth_m'] > 10)
            |(df1['Depth_m'] <= 500) & (df1['Depth_m'] > 200)
            |(df1['Depth_m'] <= 1000) & (df1['Depth_m'] > 500)
            | df1.isnull().all(axis=1)] 
    
    
    
    
    
    
    
##################################選択データ表示　2024/10/07###################################################################################################################

    
    
    selected_row = "Transect"

    #列の要素を表示
    d_select_add2 = df1[selected_row].value_counts().to_dict()
    d_select_add2_sum = df1[selected_row].count().sum()
    print('要素と出現数:', d_select_add2)
    print('要素と出現数:', d_select_add2_sum)
    print('---------------')
                            
##################################選択データ表示　2024/10/07###################################################################################################################
            
    # with表記 (推奨)
    with st.expander("selected data", expanded=False):

    #選んだパラメーター表示
    
    
    # 月がスライダーの場合
        # st.write(':green[YEAR]:'+str(sld_year_min)+'-'+str(sld_year_max)+', '
        #           +':green[MONTH]:'+str(sld_month_min)+'-'+str(sld_month_max)+', '
        #           +':green[Longitude]:'+str(sld_lon_min)+'-'+str(sld_lon_max)+', '
        #           +':green[Latitude]:'+str(sld_lat_min)+'-'+str(sld_lat_max)+', '
        #           +':green[Water_depth]:'+str(sld_depth_min)+'-'+str(sld_depth_max)+', '
        #           +':green[Salinity]:'+str(sld_sal_min)+'-'+str(sld_sal_max))
        
    # 月がマルチセレクトの場合　　リストを文字列に変換（例: [1, 2] -> "1, 2"）
        month_display = ", ".join(map(str, sorted(selected_months))) if selected_months else "None"
    
        st.write(':green[YEAR]:' + str(sld_year_min) + '-' + str(sld_year_max) + ', '
                 + ':green[MONTH]:' + '[' + month_display + ']' + ', '
                 + ':green[Longitude]:' + str(sld_lon_min) + '-' + str(sld_lon_max) + ', '
                 + ':green[Latitude]:' + str(sld_lat_min) + '-' + str(sld_lat_max) + ', '
                 + ':green[Water_depth]:' + str(sld_depth_min) + '-' + str(sld_depth_max) + ', '
                 + ':green[Salinity]:' + str(sld_sal_min) + '-' + str(sld_sal_max))
        
        
        
        # st.write('Area(Cruise)',selected_cruise)
        selected_cruise_indicate =str(list(selected_cruise[:]))
        st.write(':green[Selected Data (Cruise, papers)]', selected_cruise_indicate)
    
        st.write(':green[Selected Data (detail)]',d_select_add2)
        
        
        st.write(':green[Average values]')
                                
        #平均値と標準偏差
        col1, col2, col3, col4 = st.columns(4)
    
        with col2:
            average = np.mean(df1['d18O'])
            average = round(average,3)
            st.write('d18O _ave:', average)
    
        with col3:
            stdev = np.std(df1['d18O'])
            stdev = round(stdev,3)
            st.write('stdev: ±', stdev)  
            
        
        col1, col2, col3, col4 = st.columns(4)
    
        with col2:
            average = np.mean(df1['Salinity'])
            average = round(average,2)
            st.write('Sal_ave:', average)
    
        with col3:
            stdev = np.std(df1['Salinity'])
            stdev = round(stdev,2)
            st.write('stdev ±:', stdev)  
            
        col1, col2, col3, col4 = st.columns(4)
    
        with col2:
            average = np.mean(df1['Temperature_degC'])
            average = round(average,2)
            st.write('Temp_ave:', average)
    
        with col3:
            stdev = np.std(df1['Temperature_degC'])
            stdev = round(stdev,2)
            st.write('stdev ±:', stdev)  
                
        
##################################選択データ表示　2024/10/07###################################################################################################################

    
                    
                    
    
    
    
    
    
    
    
    
    #描画
    ax_cmap = ax.scatter(df1["Longitude_degE"], df1["Latitude_degN"], c=df1['d18O'],cmap='jet', s=20, alpha=0.7, vmin=-1.5, vmax=1, transform=ccrs.PlateCarree(), label='selected')
    plt.legend(fontsize = 15,loc='center right',bbox_to_anchor=(1, 0.22)) # 凡例の数字のフォントサイズを設定
   
    

# =============================================================================
#     2023/07/22ここから不具合　修正
# =============================================================================
    
    # カラーバーの位置調整
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # axins1 = inset_axes(ax,
    #                     width="50%",  # width = 10% of parent_bbox width
    #                     height="2%",  # height : 50%
    #                     loc='lower right',
    #                     bbox_to_anchor=(-0.02, 0.1, 1, -0.7),
    #                     bbox_transform=ax.transAxes,
    #                     )
    
    
    # fig.colorbar(ax_cmap, shrink=0.65, cax=axins1,orientation='horizontal',label=r"$\delta^{18}$O"+' (VSMOW)')
    
    cax = fig.add_axes((0.7, 0.69, 0.15, 0.005))  # [left, bottom, width, height]
    
    fig.colorbar(ax_cmap, shrink=0.2, orientation='horizontal',label=r"$\delta^{18}$O"+' (VSMOW)',cax=cax)

    

# =============================================================================
# 　2023/07/22ここまで不具合 修正
# =============================================================================

    
    # ax.set_title('title', fontsize=20)
    ax.set_title('surface sampling sites', fontsize=20) #Transectでソートした場合
    
    
    #列の要素を表示
    d_select = df1['Transect'].value_counts().to_dict()
    print('要素と出現数:', d_select)
    print('---------------')
    
    
    
    
    
    
    
    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """middle right """
    
    # 塩分-18O
    
    
    
    # """手動設定項目"""
    
    
    #####################################
    ######    EXCEL SHEET select    #####
    #####################################
    
    # """EXCELブックのシート選択、シートごとの描画色も"""
    sheet_num = [0]
    
    
    ######　プロットの色選択，sheet_numの順番に対応 10以上の数がある場合には色を追加 ######
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    # color = ["red","blue","lime","green","darkcyan","cyan","orange","yellow","fuchsia","violet","greenyellow"]
    color = ["blue","blue","blue","blue","blue","blue","blue","lime","green","darkcyan","cyan","orange","yellow","fuchsia"]
    
    
    
    
    
    
    ############################################
    ######      font size line etc..       #####
    ############################################
    
    # """凡例（legend）を入れるかどうか"""
    legend = 1 #1以外だと凡例無し
    
    # """図のフォント設定、サイズも"""
    ##### ベースのフォントとフォントサイズの指定
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams["font.size"] = 15
    
    # """図のサイズと解像度"""
    fig_size = [12,9] #図のサイズ
    fig_dpi = 150 #図の解像度
    
    # """線の太さとマーカーのサイズ"""
    ##### 線の太さとマーカーのサイズと種類
    # lw_select = 2 #プロットラインの太さ
    # marker_select = '.' #プロットマーカーの種類，種類はweb参照
    # ms_select = 15 #プロットマーカーのサイズ
    
    #軸のメモリの長さ
    ax_length = 5
    
    #ラインとプロットの透明度
    # fig_alpha = 0.85 #不透明度、1は透過なし,0.1-1の間の数
    
    
    
    
    ########################################
    ######    SUB FIG: d13C vs d18O    #####
    ########################################
    
    
    # """XYの列を指定 エクセルシートから"""
    
    # # d13C, d18Oの時
    # X_data = "d13C"
    # Y_data = "d18O"
    
    # # #d18O-dDの時
    # X_data = "d18O"
    # Y_data = "dD"
    
    #塩分-d18Oの時
    X_data = "Salinity"
    Y_data = "d18O"
    
    
    # """XYの表示用のラベルを指定"""
    
    # # x=d13C y=d18O
    # X_label = r"$\delta^{13}$C"
    # Y_label = r"$\delta^{18}$O"
    
    # # # x=d18O, Y=dD
    # X_label = r"$\delta^{18}$O"
    # Y_label = r"$\delta$D"
    
    # # x=salinity, Y=d18O
    X_label = "salinity"
    Y_label = r"$\delta^{18}$O"
    
    
    
    
    
    # """XYの表示用のラベルのスケールを指定"""
    
    
    # salinity-dD
    iso_scale_X = ""
    iso_scale_Y = "(VSMOW)"
    
    
    
    # """作図用,色とマーカーとサイズやタイトルも"""
    
    sheet_num_XY = sheet_num #ここは変更しない
    ###### 別途，X_Yのプロットをするかどうか,色を一括にするか ######
    #する場合は1,しない場合は2
    X_Y = 1
    
    #メインプロットの設定
    X_Y_C = "red" #色の設定
    X_Y_M = "." #現時点で色は変更設定なしマーカーの種類
    X_Y_S = 100
    
    
    #全体の回帰直線を書く場合「１」　書かない場合には「２」
    reg_line_write = 1
    
    
    
    
    
    #追加でTransect毎の強調プロットをする場合は,d13C_d18O_addを「1」しない場合は「2」　シートナンバー選択、
    X_Y_add1 = 2
    # sheet_num_add = [3,4,5,6]
    # sheet_num_add = [2,3]
    
    # 条件抽出する場合「1」しない場合は「2」　
    fig_add_sort = 1
    selected_row = "Transect"
    
    # selected_area = 'CK'
    # selected_area = 'Nansei'
    # selected_area = 'nECS'
    # selected_area = 'Noto'
    # selected_area = 'Pacific'
    # selected_area = 'Pacific_west'
    # selected_area = 'sECS'
    # selected_area = 'Shimane&Tottori'
    # selected_area = 'SI'
    # selected_area = 'Toyama'
    # selected_area = 'Tsushima'
    # selected_area = 'Yamato'
    # selected_area = 'NA2'
    # selected_area = 'ECS2021'
    
    
    
    #追加で緯度経度、海域毎毎の強調プロットをする場合は,d13C_d18O_addを「1」しない場合は「2」　シートナンバー選択、
    X_Y_add2 = 1
    # sheet_num_add = [3,4,5,6]
    # sheet_num_add = [2,3]
    
    #追加用の参照シート、
    sheet_num_add = [0]
    
    
    
    
    #タイトル
    # fig_title_X_Y = "XY_PLOT" #d13C_d18O書き出し専用
    fig_title_X_Y= X_label + " - "+ Y_label + "" #d13C_d18O書き出し専用
    
    #プロットの透明度
    alpha_all = 0.2 #メインプロット
    alpha_selected = 0.9 #強調プロット
    
    #強調プロットの色の指定
    X_Y_C_add =  "blue" #単色にしたい場合
    X_Y_C_add_each = 1  #シート毎に塗り分けたい場合は「１」　そうでなければ「２」
    
    #個別に近似直線を引く場合「１」，引かない場合はそれ以外の数字
    reg_line_add_write = 1
    
    
    
    #############################################
    ######      data range for SUB FIG      #####
    #############################################
    
    
    
    
    # salinity-d18Oの時
    lim_min_X = 20
    lim_max_X = 36
    lim_min_Y = -4
    lim_max_Y = 1
    
    
    
    
    
    
    ############################################
    ######      　　　設定ここまで！！　　       #####
    ############################################
    
    
    
    
    
    
    
    
    
    
    
    
    # """------------------ここから先はさわらない！----------------------"""
    # """------------------ここから先はさわらない！----------------------"""
    # """------------------ここから先はさわらない！----------------------"""
    # """------------------ここから先はさわらない！----------------------"""
    # """以下の設定は基本的に変更しない"""
    
    
    #もとのエクセルファイルのシートリストを表示
    # print()
    # sheet_all = pd.read_excel(excel_file, sheet_name=None)
    # print("選択したExcelのSheetリスト:",list(sheet_all.keys()))
    
    
    
    
    # """salinity-d18Oのプロットをする場合，回帰直線付き　変更しない"""
    if X_Y == 1:
        # sheet_num_XY = [3,4,5,6,7,8]
        
        print('-------------SUB_FIG   salinity vs d18O-------------')
        # fig = plt.figure(figsize = (fig_size),dpi=fig_dpi)
        ax = plt.subplot(324)
        # ax1 = fig.add_subplot(3, 2, 2)
    
        ax.set_xlabel(X_label + iso_scale_X)
        ax.set_ylabel(Y_label + iso_scale_Y)  #LateX形式で特殊文字
        ax.scatter(-1000, -1000, s=X_Y_S,c=X_Y_C,marker=X_Y_M, alpha=alpha_all, label='ALL') #凡例等のダミー
        # #軸のラベル
    
        
        # input_sheet_name = pd.ExcelFile(excel_file).sheet_names
        # for sheet_num in sheet_num:    
        #     print("読み込まれたSheet:", [sheet_num], input_sheet_name[sheet_num])
        
        for sheet_num_XY in sheet_num_XY:
            # df_fig_ALL = pd.read_excel(excel_file, sheet_name=input_sheet_name[sheet_num_XY])
            df_fig_ALL = envgeo_utils.load_isotope_data(ref_data, sheet_num=sheet_num_XY) 
      
            
    
            
            
            # # print(d18Oa)
            # Ya = Ya.dropna()
            # Xa = Xa.dropna()
            # # print(d18Oa)
            
            
            # これだとOK
            # 1. 描画したいXとYの両方にデータが入っている行だけを残す
            df_fig_ALL2 = df_fig_ALL.dropna(subset=[X_data, Y_data])

            Ya = df_fig_ALL2[Y_data]
            Xa = df_fig_ALL2[X_data]
    
    
    
    
    
            #列の要素を表示
            d_select_main = df_fig_ALL[selected_row].value_counts().to_dict()
            d_select_main_sum = df_fig_ALL[selected_row].count().sum()
            print('要素と出現数:', d_select_main)
            print('要素と出現数:', d_select_main_sum)
            print('---------------')
    
            
    
            ax.set_xlim(lim_min_X, lim_max_X) 
            ax.set_ylim(lim_min_Y, lim_max_Y) 
            plt.tick_params(labelsize=15)
    
            
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.f"))      
            ax.yaxis.set_major_formatter(FormatStrFormatter("%+.1f"))
            ax.tick_params(length=ax_length)
            # ax.annotate("point A", xy = (-7, 0), size = 15,
            #             color = "red", arrowprops = dict())
            
            
            ax.scatter(Xa, Ya, s=X_Y_S,c=X_Y_C,marker=X_Y_M,lw=0.5, ec="black", alpha=alpha_all)
            
        plt.title(fig_title_X_Y, fontsize=20) #
        plt.legend(fontsize = 20) # 凡例の数字のフォントサイズを設定
        
        
    
        # 回帰直線を追-------------------------------------
        if reg_line_write ==1: 
    
        
        
        # 一次関数で多項式近似を行う
        #近似式の係数
            coef = np.polyfit(Xa, Ya, 1)
        #近似式の計算
            y1 = np.poly1d(coef)(Xa) #1次
        #グラフ表示
            plt.plot(Xa, y1, label='regression line (ALL)', c=X_Y_C)
        
            reg_line = 'ALL:  y' + ' = ' + '{:.2f}'.format(coef[0]) + 'x ' +' + (' + '{:.2f}'.format(coef[1]) 
            line_r = np.corrcoef(Xa, Ya)
        
            ax.text(0.99, 0.05, reg_line + ")    (R=" + '{:.2f}'.format(line_r[0,1])+', N=' + str(d_select_main_sum)+')', horizontalalignment='right', transform=ax.transAxes)
        # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
    
    
    
        # 作成した多項式近似を表示
            print("回帰直線　ALL:", Y_data + '=' + '{:.2f}'.format(coef[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef[1]))
            print("相関係数（ｒ）:", np.corrcoef(Xa, Ya))
            print("----------------")
    
        else:()    
        
        
    
        # 回帰直線を追-------------------------------------
    
        
        # #全体のプロットをする場合
        # if X_Y_add1 == 1:
        #     # sheet_num_add = [1]
        #     # sheet_num_add = [2,3]
        #     for sheet_num_add in sheet_num_add:    
        
        #         if X_Y_C_add_each == 1:
                    
        #             plt.title(fig_title_X_Y+'_with_selected', fontsize=20) #
                    
        #             X_Y_C_add  =  color[sheet_num_add] #メインFigと同じくシート毎に分けたい場合
    
        #             df_fig_add = pd.read_excel(excel_file, sheet_name=sheet_num_add)
                    
                    
      
    
        #             if fig_add_sort == 1:
        #                 df_fig_add = df_fig_add[df_fig_add[selected_row] == selected_area]
        #                 plt.title(fig_title_X_Y+'_with_'+ selected_area, fontsize=20) #
        #             else:()
                        
        #             #列の要素を表示
        #             d_select_add = df_fig_add[selected_row].value_counts().to_dict()
        #             d_select_add_sum = df_fig_add[selected_row].count().sum()
        #             print('要素と出現数:', d_select_add)
        #             print('要素と出現数:', d_select_add_sum)
        #             print('---------------')
                    
                    
                    
        #             Y_add = df_fig_add[Y_data]
        #             X_add = df_fig_add[X_data]
        #             Y_add = Y_add.dropna()
        #             X_add = X_add.dropna()
                    
        #             ax.scatter(X_add, Y_add, s=X_Y_S,c=X_Y_C_add,marker=X_Y_M, alpha=alpha_selected,lw=0.5, ec="black", label= pd.ExcelFile(excel_file).sheet_names[sheet_num_add])
        #             # plt.legend(fontsize = 15) # 凡例の数字のフォントサイズを設定
    
                    
        #             #読み込んだシート名の表示用
        #             input_file = pd.ExcelFile(excel_file)
        #             sheet_names = input_file.sheet_names
        #             print("d13C_d18O強調用に読み込まれたSheet:", [sheet_num_add], sheet_names[sheet_num_add])
        #             print("　　　　サンプルID:",df_fiｇ_add.iloc[1,0])
                    
        #             if reg_line_add_write ==1: 
        #             # 一次関数で多項式近似を行う
        #             #近似式の係数
        #                 coef_add = np.polyfit(X_add, Y_add, 1)
        #             #近似式の計算
        #                 y1_add = np.poly1d(coef_add)(X_add) #1次
        #             #グラフ表示
        #                 plt.plot(X_add, y1_add, label='regression line (' + sheet_names[sheet_num_add]+')', c=X_Y_C_add,)
                    
        #                 reg_line_add = sheet_names[sheet_num_add] + ':  y' + ' = ' + '{:.2f}'.format(coef_add[0]) + 'x ' +' + (' + '{:.2f}'.format(coef_add[1]) 
        #                 line_r_add = np.corrcoef(X_add, Y_add)
                    
        #                 ax.text(0.99, 0.05*sheet_num_add, reg_line_add + ")    (R=" + '{:.2f}'.format(line_r_add[0,1])+', N=' + str(d_select_add_sum)+')', horizontalalignment='right', transform=ax.transAxes)
        #             # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
                    
                    
        #                 plt.legend(fontsize = 10) # 凡例の数字のフォントサイズを設定
        #             # 作成した多項式近似を表示
        #                 print("回帰直線　add:", Y_data + '=' + '{:.2f}'.format(coef_add[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef_add[1]))
        #                 print("相関係数（ｒ）:", np.corrcoef(X_add, Y_add))
        #                 print("----------------")
        #             else:()
                    
        #         else:()
                
        # else:()
    
        
    
    
    
        #追加で緯度経度とTransevtごとの強調プロットをする場合
        if X_Y_add2 == 1:
            # sheet_num_add = [1]
            # sheet_num_add = [2,3]
            for sheet_num_add in sheet_num_add:    
        
                if X_Y_C_add_each == 1:
                    
                    # plt.title(fig_title_X_Y+'_with_selected', fontsize=20) #
                    
                    X_Y_C_add  =  color[sheet_num_add] #メインFigと同じくシート毎に分けたい場合
    
                    # df_fig_add = pd.read_excel(excel_file, sheet_name=sheet_num_add)
                    df_fig_add =  envgeo_utils.load_isotope_data(ref_data, sheet_num=sheet_num_add)
                    
                    
                    
                    ####################################################################################################################################################
    
                    #緯度経度と水深とTransectで制限
                    # df1 = df_fig_add
                    
                    # df1 = df1[(df1['Depth_m'] == 'xxx') 
                    #             |(df1['Depth_m'] <= 10) & (df1['Depth_m'] >= 0)
                    #             |(df1['Depth_m'] <= 200) & (df1['Depth_m'] > 10)
                    #             |(df1['Depth_m'] <= 500) & (df1['Depth_m'] > 200)
                    #             |(df1['Depth_m'] <= 1000) & (df1['Depth_m'] > 500)
                    #          ] 
                    
                    # df1 = df1[ (df1['Transect'] == 0) 
                    #             # | (df1['Transect'] == 'CK') 
                    #             # | (df1['Transect'] == 'Nansei') 
                    #             # | (df1['Transect'] == 'nECS') 
                    #             # | (df1['Transect'] == 'Noto') 
                    #             | (df1['Transect'] == 'Pacific') 
                    #             | (df1['Transect'] == 'Pacific_west') 
                    #             # | (df1['Transect'] == 'sECS')          
                    #             # | (df1['Transect'] == 'Shimane&Tottori')          
                    #             # | (df1['Transect'] == 'SI')
                    #             # | (df1['Transect'] == 'Toyama')
                    #             # | (df1['Transect'] == 'Tsushima')
                    #             # | (df1['Transect'] == 'Yamato')
                    #             # | (df1['Transect'] == 'NA2') 
                    #             # | (df1['Transect'] == 'ECS2021') 
                    #             ] 
    
    
                    # # #描画する緯度経度を指定 
                    # df1 = df1[(df1['Longitude_degE'] == 'xxx') 
                    #             |(df1['Longitude_degE'] <= 145) & (df1['Longitude_degE'] >= 140)    
                    #             |(df1['Longitude_degE'] <= 140) & (df1['Longitude_degE'] >= 135)         
                    #             |(df1['Longitude_degE'] <= 135) & (df1['Longitude_degE'] >= 130)
                    #             |(df1['Longitude_degE'] <= 130) & (df1['Longitude_degE'] >= 125)
                    #             |(df1['Longitude_degE'] <= 125) & (df1['Longitude_degE'] >= 120)
                    #             |(df1['Longitude_degE'] <= 120) & (df1['Longitude_degE'] >= 115)
                    #             # |(df1['Longitude_degE'] <= 130) & (df1['Longitude_degE'] >= 128) #調整用
                    #            ] 
    
                    # df1 = df1[(df1['Latitude_degN'] == 'xxx')
                    #             |(df1['Latitude_degN'] <= 45) & (df1['Latitude_degN'] >= 40)          
                    #             |(df1['Latitude_degN'] <= 40) & (df1['Latitude_degN'] >= 35)
                    #             |(df1['Latitude_degN'] <= 35) & (df1['Latitude_degN'] >= 30)
                    #             |(df1['Latitude_degN'] <= 30) & (df1['Latitude_degN'] >= 25)
                    #             |(df1['Latitude_degN'] <= 25) & (df1['Latitude_degN'] >= 20)
                    #            # |(df1['Latitude_degN'] <= 33) & (df1['Latitude_degN'] >= 31) #調整用
                    #           ]
                              
                    # df1 = df1[(df1['Month'] == 'xxx')
                    #             |(df1['Month'] <= 12) & (df1['Month'] >= 10)          
                    #             |(df1['Month'] <= 9) & (df1['Month'] >= 7)
                    #             |(df1['Month'] <= 6) & (df1['Month'] >= 4)
                    #             |(df1['Month'] <= 3) & (df1['Month'] >= 1)      
                    #           ] 
                    
                    # df_fig_add = df1
                    
                    df_fig_add = df_fig_add_salinity_d18O 
                    df_fig_add_for_d18O_dD = df_fig_add
                    
                    #上記の制限要素用の名前
                    # sheet_names_add2 = 'Area B (N:25-130,E:135-140,D:>10m)'
                    # sheet_names_add2 = 'N:25-30, E:135-140, WD:0-10m'
                    sheet_names_add2 = sheet_names_add2
                    
                    
                    ####################################################################################################################################################
                    
                    
                    
                    
          
                    # if fig_add_sort == 1:
                    #     df_fig_add = df_fig_add[df_fig_add[selected_row] == selected_area]
                    #     plt.title(fig_title_X_Y+'_with_'+ selected_area, fontsize=20) #
                    # else:()
                        

                    
                    # これだと欠損値でエラーになる。
                    # Y_add = df_fig_add[Y_data]
                    # X_add = df_fig_add[X_data]
                    # Y_add = Y_add.dropna()
                    # X_add = X_add.dropna()
                    
                    
                    # これだとOK
                    # 1. 描画したいXとYの両方にデータが入っている行だけを残す
                    df_fig_add2 = df_fig_add.dropna(subset=[X_data, Y_data])

                    Y_add = df_fig_add2[Y_data]
                    X_add = df_fig_add2[X_data]
                    
                    
                    
                    
                    #列の要素を表示
                    d_select_add2 = df_fig_add[selected_row].value_counts().to_dict()
                    d_select_add2_sum = df1[selected_row].count().sum()
                    print('要素と出現数:', d_select_add2)
                    print('要素と出現数:', d_select_add2_sum)
                    print('---------------')
    
                    
                    ax.scatter(X_add, Y_add, s=X_Y_S,c=X_Y_C_add,marker=X_Y_M, alpha=alpha_selected,lw=0.5, ec="black", label= sheet_names_add2)
                    # plt.legend(fontsize = 15) # 凡例の数字のフォントサイズを設定
    
                    
                    # #読み込んだシート名の表示用
                    # input_file = pd.ExcelFile(excel_file)
                    # sheet_names = input_file.sheet_names
                    # print("d13C_d18O強調用に読み込まれたSheet:", [sheet_num_add], sheet_names[sheet_num_add])
                    # print("　　　　サンプルID:",df_fiｇ_add.iloc[1,0])
                    
                    if reg_line_add_write ==1: 
                    # 一次関数で多項式近似を行う
                    #近似式の係数
                        coef_add = np.polyfit(X_add, Y_add, 1)
                    #近似式の計算
                        y1_add = np.poly1d(coef_add)(X_add) #1次
                    #グラフ表示
                        plt.plot(X_add, y1_add, label='regression line (' + sheet_names_add2 +')', c=X_Y_C_add,)
                    
                        reg_line_add = sheet_names_add2 + ':  y' + ' = ' + '{:.2f}'.format(coef_add[0]) + 'x ' +' + (' + '{:.2f}'.format(coef_add[1]) 
                        line_r_add = np.corrcoef(X_add, Y_add)
                    
                        ax.text(0.99, 0.05*3+0.01, reg_line_add + ")    (R=" + '{:.2f}'.format(line_r_add[0,1])+', N=' + str(d_select_add2_sum)+')', horizontalalignment='right', transform=ax.transAxes)
                        # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
                    
                    
                        plt.legend(fontsize = 10) # 凡例の数字のフォントサイズを設定
                    # 作成した多項式近似を表示
                        print("回帰直線　add:", Y_data + '=' + '{:.2f}'.format(coef_add[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef_add[1]))
                        print("相関係数（ｒ）:", np.corrcoef(X_add, Y_add))
                        print("----------------")
                    else:()
                    
                else:()
                
        else:()
    
    
    
    
    
        #==========  以下，検証　============
        
        print(("--------MES RMSE R2 (all)--------"))
        
        Y_all_pred = coef[0]*Xa + coef[1]
        print("回帰直線　ALL:", Y_data + '=' + '{:.2f}'.format(coef[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef[1]))
        print('a=',coef[0])
        print('b=',coef[1])
        
        # print(Ya)
        # print(Y_all_pred)
        print()
        ###############　MSE，RMSEの計算
        #https://pythondatascience.plavox.info/scikit-learn/回帰モデルの評価
        
        # from sklearn.metrics import mean_squared_error
        # import numpy as np
        MSE_all = mean_squared_error(Ya, Y_all_pred)
        RMES_all = np.sqrt(mean_squared_error(Ya, Y_all_pred))
            
        print('MSE_all:', '{:.3f}'.format(MSE_all))
        print('RMSE_all:', '{:.3f}'.format(RMES_all))
        
        ###############　R2の計算
        # from sklearn.metrics import r2_score
        R2_all =  r2_score(Ya, Y_all_pred)  
        print('R2_all:', '{:.3f}'.format(R2_all))
        
    
        ax.text(0.99, 0+0.01, 'RMSE_all: ' + '{:.3f}'.format(RMES_all)+', R$^{2}$_all: ' + '{:.2f}'.format(R2_all), horizontalalignment='right', transform=ax.transAxes, fontsize=12, c='red')
        
        
        
        print(("--------MES RMSE R2 (add)--------"))
        
        Y_add_pred = coef_add[0]*X_add + coef_add[1]
        print("回帰直線　add:", Y_data + '=' + '{:.2f}'.format(coef_add[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef_add[1]))
        print('a=',coef_add[0])
        print('b=',coef_add[1])
        
        # print(Y_add)
        # print(Y_add_pred)
        print()
        ###############　MSE，RMSEの計算
        #https://pythondatascience.plavox.info/scikit-learn/回帰モデルの評価
        
        # from sklearn.metrics import mean_squared_error
        # import numpy as np
        MSE_add = mean_squared_error(Y_add, Y_add_pred)
        RMES_add = np.sqrt(mean_squared_error(Y_add, Y_add_pred))
            
        print('MSE_add:', '{:.3f}'.format(MSE_add))
        print('RMSE_add:', '{:.3f}'.format(RMES_add))
        
        ###############　R2の計算
        # from sklearn.metrics import r2_score
        R2_add =  r2_score(Y_add, Y_add_pred)  
        print('R2_add:', '{:.3f}'.format(R2_add))
        
        ax.text(0.99, 0.05*2+0.01, 'RMSE_add: ' + '{:.3f}'.format(RMES_add)+', R$^{2}$_add: ' + '{:.2f}'.format(R2_add), horizontalalignment='right', transform=ax.transAxes, fontsize=12, c='blue')
    
    
    
        
            
        # if PDF_export_SUB == 1:
        #     plt.savefig('Fig_'+X_data+'_'+Y_data+'_'+selected_area+'.pdf')
        #     print('pdfに書き出しました')
        # else:
        #     print('pdf書き出し無し')
        
        # if PNG_export_SUB == 1:
        #     plt.savefig('Fig_'+X_data+'_'+Y_data+'_'+selected_area+'.png')
        #     print('pngに書き出しました')
        # else:
        #     print('png書き出し無し')
    
    
        # plt.show()
    
    
    else:()
    
    
    
    
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # """bottom right """
    #  d18O-dD
    
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # # import datetime
    # # import matplotlib.dates as dates
    # from matplotlib.ticker import FormatStrFormatter
    # # from matplotlib.ticker import MultipleLocator
    # # import matplotlib.ticker as ticker
    # import seaborn as sns
    
    
    
    
    # """      Spyderのメニューボタンの実行ボタンを使わないと作業ディレクトリが反映されないので注意！     """
    # """      Spyderのメニューボタンの実行ボタンを使わないと作業ディレクトリが反映されないので注意！     """
    # """      Spyderのメニューボタンの実行ボタンを使わないと作業ディレクトリが反映されないので注意！     """
    # """      ファイルの読み込みと保存が別の場所を参照してしまう！！！！     """
    
    
    
    
    # """手動設定項目"""
    #####################################
    ######    EXCEL BOOK import     #####
    #####################################
    # """"Excelブックインポート"""
    #読み込みExcelブックファイルを指定  (xls, xlsxのどちらでも可能?)
    #Spyderのメニューボタンの実行ボタンを使わないと作業ディレクトリが反映されないので注意！
    
    #耳石
    # excel_file = 'otolith_test.xlsx'
    
    #海水
    # excel_file = 'd18O_20210626-3_NA2.xlsx'
    # 
    
    
    #####################################
    ######    EXCEL SHEET select    #####
    #####################################
    
    # """EXCELブックのシート選択、シートごとの描画色も"""
    #読み込まれたExcelファイルから，シート番号を指定。一番左のシートが[0],２番目が[1]
    
    sheet_num = [0]
    
    
    ######　プロットの色選択，sheet_numの順番に対応 10以上の数がある場合には色を追加 ######
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    color = ["red","blue","lime","green","darkcyan","cyan","orange","yellow","fuchsia","violet","greenyellow"]
    # color = ["red","blue","blue","blue","blue","blue","blue","lime","green","darkcyan","cyan","orange","yellow","fuchsia"]
    
    
    
    
    
    ########################################
    ######    SUB FIG: d13C vs d18O    #####
    ########################################
    
    
    # """XYの列を指定 エクセルシートから"""
    
    # #d18O-dDの時
    X_data = "d18O"
    Y_data = "dD"
    
    
    
    # """XYの表示用のラベルを指定"""
    
    
    
    # # x=d18O, Y=dD
    X_label = r"$\delta^{18}$O"
    Y_label = r"$\delta$D"
    
    
    
    
    
    # """XYの表示用のラベルのスケールを指定"""
    
    
    # dD, d18O
    iso_scale_X = "(VSMOW)"
    iso_scale_Y = "(VSMOW)"
    
    
    
    
    # """作図用,色とマーカーとサイズやタイトルも"""
    
    sheet_num_XY = sheet_num #ここは変更しない
    ###### 別途，X_Yのプロットをするかどうか,色を一括にするか ######
    #する場合は1,しない場合は2
    X_Y = 1
    
    #メインプロットの設定
    X_Y_C = "red" #色の設定
    X_Y_M = "." #現時点で色は変更設定なしマーカーの種類
    X_Y_S = 100
    
    
    #全体の回帰直線を書く場合「１」　書かない場合には「２」
    reg_line_write = 1
    
    
    
    
    
    #追加でTransect毎の強調プロットをする場合は,d13C_d18O_addを「1」しない場合は「2」　シートナンバー選択、
    X_Y_add1 = 2
    # sheet_num_add = [3,4,5,6]
    # sheet_num_add = [2,3]
    
    # 条件抽出する場合「1」しない場合は「2」　
    fig_add_sort = 1
    selected_row = "Transect"
    
    # selected_area = 'CK'
    # selected_area = 'Nansei'
    # selected_area = 'nECS'
    # selected_area = 'Noto'
    # selected_area = 'Pacific'
    # selected_area = 'Pacific_west'
    # selected_area = 'sECS'
    # selected_area = 'Shimane&Tottori'
    # selected_area = 'SI'
    # selected_area = 'Toyama'
    # selected_area = 'Tsushima'
    # selected_area = 'Yamato'
    # selected_area = 'NA2'
    
    
    
    
    #追加で緯度経度、海域毎毎の強調プロットをする場合は,d13C_d18O_addを「1」しない場合は「2」　シートナンバー選択、
    X_Y_add2 = 1
    # sheet_num_add = [3,4,5,6]
    # sheet_num_add = [2,3]
    
    #追加用の参照シート、色の設定とかに必要
    sheet_num_add = [1]
    
    
    
    
    
    #タイトル
    # fig_title_X_Y = "XY_PLOT" #d13C_d18O書き出し専用
    fig_title_X_Y= X_label + " - "+ Y_label + "" #d13C_d18O書き出し専用
    
    #プロットの透明度
    alpha_all = 0.2 #メインプロット
    alpha_selected = 0.9 #強調プロット
    
    #強調プロットの色の指定
    X_Y_C_add =  "blue" #単色にしたい場合
    X_Y_C_add_each = 1  #シート毎に塗り分けたい場合は「１」　そうでなければ「２」
    
    #個別に近似直線を引く場合「１」，引かない場合はそれ以外の数字
    reg_line_add_write = 1
    
    
    
    #############################################
    ######      data range for SUB FIG      #####
    #############################################
    
    
    
    #d18O-dDの時
    lim_min_X = -5
    lim_max_X = 1
    lim_min_Y = -30
    lim_max_Y = 4
    
    
    
    
    
    
    ############################################
    ######      　　　設定ここまで！！　　       #####
    ############################################
    
    
    
    
    
    
    
    
    
    
    
    
    # """------------------ここから先はさわらない！----------------------"""
    # """------------------ここから先はさわらない！----------------------"""
    # """------------------ここから先はさわらない！----------------------"""
    # """------------------ここから先はさわらない！----------------------"""
    # """以下の設定は基本的に変更しない"""
    
    
    #もとのエクセルファイルのシートリストを表示
    # print()
    # sheet_all = pd.read_excel(excel_file, sheet_name=None)
    # print("選択したExcelのSheetリスト:",list(sheet_all.keys()))
    
    
    
    
    # """dD-d18Oのプロットをする場合，回帰直線付き　変更しない"""
    if X_Y == 1:
        # sheet_num_XY = [3,4,5,6,7,8]
        
        print('-------------SUB_FIG   d13C vs d18O-------------')
        # fig = plt.figure(figsize = (sfig_size),dpi=fig_dpi)
        ax = plt.subplot(326)
        
    
        ax.set_xlabel(X_label + iso_scale_X)
        ax.set_ylabel(Y_label + iso_scale_Y)  #LateX形式で特殊文字
        ax.scatter(-1000, -1000, s=X_Y_S,c=X_Y_C,marker=X_Y_M, alpha=alpha_all, label='ALL') #凡例等のダミー
        # #軸のラベル
        # ax.set_xlabel(r"$\delta^{13}$C (VPDB)")
        # ax.set_ylabel(r"$\delta^{18}$O (VPDB)")  #LateX形式で特殊文字
        
        # input_sheet_name = pd.ExcelFile(excel_file).sheet_names
        # for sheet_num in sheet_num:    
        #     print("読み込まれたSheet:", [sheet_num], input_sheet_name[sheet_num])
        
        for sheet_num_XY in sheet_num_XY:
            # df_fig_ALL = pd.read_excel(excel_file, sheet_name=input_sheet_name[sheet_num_XY])
            
            # df_fig_ALL =   envgeo_utils.load_isotope_data(ref_data, sheet_num=sheet_num_XY)

            df_fig_ALL =   envgeo_utils.load_isotope_data(ref_data)

            
    
            # print(df_fig_ALL)
            Ya = df_fig_ALL[Y_data]
            Xa = df_fig_ALL[X_data]
            # print(d18Oa)
            Ya = Ya.dropna()
            Xa = Xa.dropna()
            # print(d18Oa)
            
    
            #列の要素を表示
            d_select_main = df_fig_ALL[selected_row].value_counts().to_dict()
            d_select_main_sum = df_fig_ALL[selected_row].count().sum()
            print('要素と出現数:', d_select_main)
            print('要素と出現数:', d_select_main_sum)
            print('---------------')
    
            
        
            # for sheet_num_add in sheet_num_add:    
            #     df_fig = pd.read_excel(excel_file, sheet_name=sheet_num)
            #     d18O = df_fig["d18O"]
            #     d13C = df_fig["d13C"]
            ax.set_xlim(lim_min_X, lim_max_X) 
            # ax.set_xticks(np.linspace(-11, -3, 9))
            ax.set_ylim(lim_min_Y, lim_max_Y) 
            plt.tick_params(labelsize=15)
            # ax.set_xticks(np.linspace(-1.4, 0.6,11))
            # ax.set_yticks(np.linspace(1000, 0, 11))
            
            if X_data == 'Salinity':
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.f"))
            else:
                ax.xaxis.set_major_formatter(FormatStrFormatter("%+.1f"))
            
            
            ax.yaxis.set_major_formatter(FormatStrFormatter("%+.1f"))
            ax.tick_params(length=ax_length)
            # ax.annotate("point A", xy = (-7, 0), size = 15,
            #             color = "red", arrowprops = dict())
            
            
            ax.scatter(Xa, Ya, s=X_Y_S,c=X_Y_C,marker=X_Y_M,lw=0.5, ec="black", alpha=alpha_all)
        plt.title(fig_title_X_Y, fontsize=20) #
        plt.legend(fontsize = 20) # 凡例の数字のフォントサイズを設定
        
        
    
        # 回帰直線を追-------------------------------------
        if reg_line_write ==1: 
    
        
        # 一次関数で多項式近似を行う
        #近似式の係数
            coef = np.polyfit(Xa, Ya, 1)
        #近似式の計算
            y1 = np.poly1d(coef)(Xa) #1次
        #グラフ表示
            plt.plot(Xa, y1, label='regression line (ALL)', c=X_Y_C)
        
            reg_line = 'ALL:  y' + ' = ' + '{:.2f}'.format(coef[0]) + 'x ' +' + (' + '{:.2f}'.format(coef[1]) 
            line_r = np.corrcoef(Xa, Ya)
        
            ax.text(0.99, 0.05, reg_line + ")    (R=" + '{:.2f}'.format(line_r[0,1])+', N=' + str(d_select_main_sum)+')', horizontalalignment='right', transform=ax.transAxes)
        # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
    
    
    
        # 作成した多項式近似を表示
            print("回帰直線　ALL:", Y_data + '=' + '{:.2f}'.format(coef[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef[1]))
            print("相関係数（ｒ）:", np.corrcoef(Xa, Ya))
            print("----------------")
    
        else:()    
        
        
    
        # 回帰直線を追-------------------------------------
    
        # 条件抽出する場合
        # fig_add_sort = 1
        # selected_row = "Transect"
        # selected_area = 'Noto'
    
        
        # #追加でTransectごとの強調プロットをする場合
        # if X_Y_add1 == 1:
        #     # sheet_num_add = [1]
        #     # sheet_num_add = [2,3]
        #     for sheet_num_add in sheet_num_add:    
        
        #         if X_Y_C_add_each == 1:
                    
        #             plt.title(fig_title_X_Y+'_with_selected', fontsize=20) #
                    
        #             X_Y_C_add  =  color[sheet_num_add] #メインFigと同じくシート毎に分けたい場合
    
        #             df_fig_add = pd.read_excel(excel_file, sheet_name=sheet_num_add)
                    
                    
      
    
        #             if fig_add_sort == 1:
        #                 df_fig_add = df_fig_add[df_fig_add[selected_row] == selected_area]
        #                 plt.title(fig_title_X_Y+'_with_'+ selected_area, fontsize=20) #
        #             else:()
                        
        #             #列の要素を表示
        #             d_select_add = df_fig_add[selected_row].value_counts().to_dict()
        #             d_select_add_sum = df_fig_add[selected_row].count().sum()
        #             print('要素と出現数:', d_select_add)
        #             print('要素と出現数:', d_select_add_sum)
        #             print('---------------')
                    
                    
                    
        #             Y_add = df_fig_add[Y_data]
        #             X_add = df_fig_add[X_data]
        #             Y_add = Y_add.dropna()
        #             X_add = X_add.dropna()
                    
        #             ax.scatter(X_add, Y_add, s=X_Y_S,c=X_Y_C_add,marker=X_Y_M, alpha=alpha_selected,lw=0.5, ec="black", label= pd.ExcelFile(excel_file).sheet_names[sheet_num_add])
        #             # plt.legend(fontsize = 15) # 凡例の数字のフォントサイズを設定
    
                    
        #             #読み込んだシート名の表示用
        #             input_file = pd.ExcelFile(excel_file)
        #             sheet_names = input_file.sheet_names
        #             print("d13C_d18O強調用に読み込まれたSheet:", [sheet_num_add], sheet_names[sheet_num_add])
        #             print("　　　　サンプルID:",df_fiｇ_add.iloc[1,0])
                    
        #             if reg_line_add_write ==1: 
        #             # 一次関数で多項式近似を行う
        #             #近似式の係数
        #                 coef_add = np.polyfit(X_add, Y_add, 1)
        #             #近似式の計算
        #                 y1_add = np.poly1d(coef_add)(X_add) #1次
        #             #グラフ表示
        #                 plt.plot(X_add, y1_add, label='regression line (' + sheet_names[sheet_num_add]+')', c=X_Y_C_add,)
                    
        #                 reg_line_add = sheet_names[sheet_num_add] + ':  y' + ' = ' + '{:.2f}'.format(coef_add[0]) + 'x ' +' + ' + '{:.2f}'.format(coef_add[1]) 
        #                 line_r_add = np.corrcoef(X_add, Y_add)
                    
        #                 ax.text(0.99, 0.05*sheet_num_add, reg_line_add + "    (R=" + '{:.2f}'.format(line_r_add[0,1])+', N=' + str(d_select_add_sum)+')', horizontalalignment='right', transform=ax.transAxes)
        #             # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
                    
                    
        #                 plt.legend(fontsize = 10) # 凡例の数字のフォントサイズを設定
        #             # 作成した多項式近似を表示
        #                 print("回帰直線　add:", Y_data + '=' + '{:.2f}'.format(coef_add[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef_add[1]))
        #                 print("相関係数（ｒ）:", np.corrcoef(X_add, Y_add))
        #                 print("----------------")
        #             else:()
                    
        #         else:()
                
        # else:()
    
        
    
    
    
        #追加で緯度経度とTransevtごとの強調プロットをする場合
        if X_Y_add2 == 1:
            # sheet_num_add = [1]
            # sheet_num_add = [2,3]
            for sheet_num_add in sheet_num_add:    
        
                if X_Y_C_add_each == 1:
                    
                    # plt.title(fig_title_X_Y+'_with_selected', fontsize=20) #
                    
                    X_Y_C_add  =  color[sheet_num_add] #メインFigと同じくシート毎に分けたい場合
    
                    # df_fig_add = pd.read_excel(excel_file, sheet_name=sheet_num_add)
                    df_fig_add = df_fig_add_for_d18O_dD
                    
    
                    # sheet_names_add2 = 'N:25-30, E:135-140, WD:0-10m'
                    sheet_names_add2 = sheet_names_add2
                    
                    
                    
                    ####################################################################################################################################################
                    
                    
                    
                    
          
                    # if fig_add_sort == 1:
                    #     df_fig_add = df_fig_add[df_fig_add[selected_row] == selected_area]
                    #     plt.title(fig_title_X_Y+'_with_'+ selected_area, fontsize=20) #
                    # else:()
                        
                    Y_add = df_fig_add[Y_data]
                    X_add = df_fig_add[X_data]
                    Y_add = Y_add.dropna()
                    X_add = X_add.dropna()
                    
                    
                    #列の要素を表示
                    d_select_add2 = df_fig_add[selected_row].value_counts().to_dict()
                    d_select_add2_sum = df1[selected_row].count().sum()
                    print('要素と出現数:', d_select_add2)
                    print('要素と出現数:', d_select_add2_sum)
                    print('---------------')
    
                    
                    ax.scatter(X_add, Y_add, s=X_Y_S,c=X_Y_C_add,marker=X_Y_M, alpha=alpha_selected,lw=0.5, ec="black", label= sheet_names_add2)
                    # plt.legend(fontsize = 15) # 凡例の数字のフォントサイズを設定
    
                    
                    #読み込んだシート名の表示用
                    # input_file = pd.ExcelFile(excel_file)
                    # sheet_names = input_file.sheet_names
                    # print("d13C_d18O強調用に読み込まれたSheet:", [sheet_num_add], sheet_names[sheet_num_add])
                    # print("　　　　サンプルID:",df_fiｇ_add.iloc[1,0])
                    
                    if reg_line_add_write ==1: 
                    # 一次関数で多項式近似を行う
                    #近似式の係数
                        coef_add = np.polyfit(X_add, Y_add, 1)
                    #近似式の計算
                        y1_add = np.poly1d(coef_add)(X_add) #1次
                    #グラフ表示
                        plt.plot(X_add, y1_add, label='regression line (' + sheet_names_add2 +')', c=X_Y_C_add,)
                    
                        reg_line_add = sheet_names_add2 + ':  y' + ' = ' + '{:.2f}'.format(coef_add[0]) + 'x ' +' + (' + '{:.2f}'.format(coef_add[1]) 
                        line_r_add = np.corrcoef(X_add, Y_add)
                    
                        ax.text(0.99, 0.05*3+0.01, reg_line_add + ")    (R=" + '{:.2f}'.format(line_r_add[0,1])+', N=' + str(d_select_add2_sum)+')', horizontalalignment='right', transform=ax.transAxes)
                        # ax.text(0.99, 0.01, line_r, horizontalalignment='right', transform=ax.transAxes)
                    
                    
                        plt.legend(fontsize = 10) # 凡例の数字のフォントサイズを設定
                    # 作成した多項式近似を表示
                        print("回帰直線　add:", Y_data + '=' + '{:.2f}'.format(coef_add[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef_add[1]))
                        print("相関係数（ｒ）:", np.corrcoef(X_add, Y_add))
                        print("----------------")
                    else:()
                    
                else:()
                
        else:()
    
    
    
    
    
        #==========  以下，検証　============
        
        print(("--------MES RMSE R2 (all)--------"))
        
        Y_all_pred = coef[0]*Xa + coef[1]
        print("回帰直線　ALL:", Y_data + '=' + '{:.2f}'.format(coef[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef[1]))
        print('a=',coef[0])
        print('b=',coef[1])
        
        # print(Ya)
        # print(Y_all_pred)
        print()
        ###############　MSE，RMSEの計算
        #https://pythondatascience.plavox.info/scikit-learn/回帰モデルの評価
        
        # from sklearn.metrics import mean_squared_error
        # import numpy as np
        MSE_all = mean_squared_error(Ya, Y_all_pred)
        RMES_all = np.sqrt(mean_squared_error(Ya, Y_all_pred))
            
        print('MSE_all:', '{:.3f}'.format(MSE_all))
        print('RMSE_all:', '{:.3f}'.format(RMES_all))
        
        ###############　R2の計算
        # from sklearn.metrics import r2_score
        R2_all =  r2_score(Ya, Y_all_pred)  
        print('R2_all:', '{:.3f}'.format(R2_all))
        
    
        ax.text(0.99, 0+0.01, 'RMSE_all: ' + '{:.3f}'.format(RMES_all)+', R$^{2}$_all: ' + '{:.2f}'.format(R2_all), horizontalalignment='right', transform=ax.transAxes, fontsize=12, c='red')
        
        
        
        print(("--------MES RMSE R2 (add)--------"))
        
        Y_add_pred = coef_add[0]*X_add + coef_add[1]
        print("回帰直線　add:", Y_data + '=' + '{:.2f}'.format(coef_add[0]) + ' * ' + Y_data + '+' + '{:.2f}'.format(coef_add[1]))
        print('a=',coef_add[0])
        print('b=',coef_add[1])
        
        # print(Y_add)
        # print(Y_add_pred)
        print()
        ###############　MSE，RMSEの計算
        #https://pythondatascience.plavox.info/scikit-learn/回帰モデルの評価
        
        # from sklearn.metrics import mean_squared_error
        # import numpy as np
        MSE_add = mean_squared_error(Y_add, Y_add_pred)
        RMES_add = np.sqrt(mean_squared_error(Y_add, Y_add_pred))
            
        print('MSE_add:', '{:.3f}'.format(MSE_add))
        print('RMSE_add:', '{:.3f}'.format(RMES_add))
        
        ###############　R2の計算
        # from sklearn.metrics import r2_score
        R2_add =  r2_score(Y_add, Y_add_pred)  
        print('R2_add:', '{:.3f}'.format(R2_add))
        
        ax.text(0.99, 0.05*2+0.01, 'RMSE_add: ' + '{:.3f}'.format(RMES_add)+', R$^{2}$_add: ' + '{:.2f}'.format(R2_add), horizontalalignment='right', transform=ax.transAxes, fontsize=12, c='blue')
    
    
    
    else:()
    
    
    
    
    
    
    
    
    
    
    
    
    #######################画像を保存するためのボタン作成########################
    sub_title2 = sub_title
    sub_title2 = sub_title2.replace(':', '') #pdf書き出し用
    # sub_title2 = sub_title2.replace('>', '') #pdf書き出し用
    # sub_title2 = sub_title2.replace('<', '') #pdf書き出し用
    sub_title2 = sub_title2.replace(',', '_') #pdf書き出し用
    sub_title2 = sub_title2.replace(' ', '') #pdf書き出し用
    sub_tite = str('Fig_compiled_SW'+'_'+sub_title2+".png")


    
    #画像を保存，以下の方法だとローカルにも保存されてしまう
    # fn = sub_tite
    # # plt.savefig(fn)

    # with open(fn, "rb") as img:
    #     btn = st.download_button(
    #         label="Download image",
    #         data=img,
    #         file_name=fn,
    #         mime="image/png"
    #     )

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
    
    
    
    
# show plots
# fig.tight_layout()
# fig.show()

       
            
    
    # st.subheader('Area Chart')
    # st.area_chart(df_fig_ALL)
    
    # Matplotlib の Figure を指定して可視化する
    st.pyplot(fig)
    
    
    
    # ##########採取地点のmap　拡大可能##################
    # df1 = data_limit()
    # df1['lat'] = df1['Latitude_degN']
    # df1['lon'] = df1['Longitude_degE']
    
    # # df = pd.DataFrame(
    # #     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    # #     columns=['lat', 'lon'])
    
    # st.map(df1)



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
    envgeo_utils.display_isotope_table(df1)
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################




if __name__ == '__main__':
    main()
    

