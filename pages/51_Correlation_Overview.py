#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:15:03 2023
@author: Toyoho Ishimura @Kyoto-U
2026/02/10 update 
"""

# --- バージョン管理の設定 ---
version = "1.3.0" #2026/02/23
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


@st.cache_data(show_spinner=False)
def load_isotope_data_cached(ref_data, sheet_num=None):
    if sheet_num is None:
        return envgeo_utils.load_isotope_data(ref_data)
    return envgeo_utils.load_isotope_data(ref_data, sheet_num=sheet_num)


def build_month_display(selected_months):
    if len(selected_months) == 12:
        return "All"
    if len(selected_months) == 0:
        return "None"

    sorted_months = sorted(set(selected_months))
    ranges = []
    start = sorted_months[0]

    for i, month in enumerate(sorted_months):
        is_last = i + 1 == len(sorted_months)
        next_is_consecutive = (not is_last) and sorted_months[i + 1] == month + 1
        if next_is_consecutive:
            continue

        end = month
        ranges.append(f"{start}-{end}" if start != end else str(start))
        if not is_last:
            start = sorted_months[i + 1]

    return ", ".join(ranges)


def filter_observation_data(
    df,
    *,
    selected_cruise,
    selected_months,
    year_range,
    lon_range,
    lat_range,
    depth_range,
    salinity_range,
):
    year_min, year_max = year_range
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    depth_min, depth_max = depth_range
    sal_min, sal_max = salinity_range

    keep_blank = df.isnull().all(axis=1)

    filtered = df[
        (df["Depth_m"] == "xxx")
        | ((df["Depth_m"] <= depth_max) & (df["Depth_m"] >= depth_min))
        | keep_blank
    ]

    filtered = filtered[
        filtered["Transect"].isin(selected_cruise) | filtered["Transect"].isna()
    ].copy()

    filtered = filtered[
        (filtered["Longitude_degE"] == "xxx")
        | ((filtered["Longitude_degE"] <= lon_max) & (filtered["Longitude_degE"] >= lon_min))
        | filtered.isnull().all(axis=1)
    ]

    filtered = filtered[
        (filtered["Latitude_degN"] == "xxx")
        | ((filtered["Latitude_degN"] <= lat_max) & (filtered["Latitude_degN"] >= lat_min))
        | filtered.isnull().all(axis=1)
    ]

    if selected_months:
        filtered = filtered[
            filtered["Month"].isin(selected_months) | filtered.isnull().all(axis=1)
        ]
    else:
        filtered = filtered[filtered.isnull().all(axis=1)]

    filtered = filtered[
        (filtered["Salinity"] == "xxx")
        | ((filtered["Salinity"] >= sal_min) & (filtered["Salinity"] <= sal_max))
        | filtered.isnull().all(axis=1)
    ]

    filtered = filtered[
        ((filtered["Year"] <= year_max) & (filtered["Year"] >= year_min))
        | filtered.isnull().all(axis=1)
    ]

    return filtered


def prepare_xy_frame(df, x_col, y_col):
    return df.dropna(subset=[x_col, y_col]).copy()


def count_valid_rows(df, required_columns):
    return len(df.dropna(subset=required_columns))


def plot_xy_with_regression(
    *,
    ax,
    base_df,
    selected_df,
    x_col,
    y_col,
    x_label,
    y_label,
    x_scale,
    y_scale,
    x_limits,
    y_limits,
    x_formatter,
    y_formatter,
    tick_length,
    main_style,
    selected_style,
    figure_title,
    selected_label,
    selected_row,
    write_main_regression,
    write_selected_regression,
    metric_colors=("red", "blue"),
):
    base_xy = prepare_xy_frame(base_df, x_col, y_col)
    selected_xy = prepare_xy_frame(selected_df, x_col, y_col)

    ax.set_xlabel(x_label + x_scale)
    ax.set_ylabel(y_label + y_scale)
    ax.scatter(
        -1000,
        -1000,
        s=main_style["size"],
        c=main_style["color"],
        marker=main_style["marker"],
        alpha=main_style["alpha"],
        label="ALL",
    )

    ax.scatter(
        base_xy[x_col],
        base_xy[y_col],
        s=main_style["size"],
        c=main_style["color"],
        marker=main_style["marker"],
        lw=0.5,
        ec="black",
        alpha=main_style["alpha"],
    )

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    plt.tick_params(labelsize=15)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(length=tick_length)
    plt.title(figure_title, fontsize=20)
    plt.legend(fontsize=20)

    base_count = base_df[selected_row].count().sum()
    selected_count = selected_df[selected_row].count().sum()
    main_coef = None
    selected_coef = None

    if write_main_regression and len(base_xy) >= 2:
        main_coef = np.polyfit(base_xy[x_col], base_xy[y_col], 1)
        main_fit = np.poly1d(main_coef)(base_xy[x_col])
        plt.plot(base_xy[x_col], main_fit, label="regression line (ALL)", c=main_style["color"])
        main_r = np.corrcoef(base_xy[x_col], base_xy[y_col])
        reg_line = f"ALL:  y = {main_coef[0]:.2f}x + ({main_coef[1]:.2f}"
        ax.text(
            0.99,
            0.05,
            reg_line + f")    (R={main_r[0,1]:.2f}, N={base_count})",
            horizontalalignment="right",
            transform=ax.transAxes,
        )
        print("回帰直線　ALL:", f"{y_col}={main_coef[0]:.2f} * {x_col}+{main_coef[1]:.2f}")
        print("相関係数（ｒ）:", np.corrcoef(base_xy[x_col], base_xy[y_col]))
        print("----------------")

    if len(selected_xy) >= 1:
        ax.scatter(
            selected_xy[x_col],
            selected_xy[y_col],
            s=selected_style["size"],
            c=selected_style["color"],
            marker=selected_style["marker"],
            alpha=selected_style["alpha"],
            lw=0.5,
            ec="black",
            label=selected_label,
        )

    if write_selected_regression and len(selected_xy) >= 2:
        selected_coef = np.polyfit(selected_xy[x_col], selected_xy[y_col], 1)
        selected_fit = np.poly1d(selected_coef)(selected_xy[x_col])
        plt.plot(
            selected_xy[x_col],
            selected_fit,
            label=f"regression line ({selected_label})",
            c=selected_style["color"],
        )
        selected_r = np.corrcoef(selected_xy[x_col], selected_xy[y_col])
        reg_line = f"{selected_label}:  y = {selected_coef[0]:.2f}x + ({selected_coef[1]:.2f}"
        ax.text(
            0.99,
            0.16,
            reg_line + f")    (R={selected_r[0,1]:.2f}, N={selected_count})",
            horizontalalignment="right",
            transform=ax.transAxes,
        )
        plt.legend(fontsize=10)
        print("回帰直線　add:", f"{y_col}={selected_coef[0]:.2f} * {x_col}+{selected_coef[1]:.2f}")
        print("相関係数（ｒ）:", np.corrcoef(selected_xy[x_col], selected_xy[y_col]))
        print("----------------")

    if main_coef is not None:
        y_pred = main_coef[0] * base_xy[x_col] + main_coef[1]
        mse_all = mean_squared_error(base_xy[y_col], y_pred)
        rmse_all = np.sqrt(mse_all)
        r2_all = r2_score(base_xy[y_col], y_pred)
        print("--------MES RMSE R2 (all)--------")
        print("MSE_all:", f"{mse_all:.3f}")
        print("RMSE_all:", f"{rmse_all:.3f}")
        print("R2_all:", f"{r2_all:.3f}")
        ax.text(
            0.99,
            0.01,
            f"RMSE_all: {rmse_all:.3f}, R$^{{2}}$_all: {r2_all:.2f}",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=12,
            c=metric_colors[0],
        )

    if selected_coef is not None:
        y_pred = selected_coef[0] * selected_xy[x_col] + selected_coef[1]
        mse_add = mean_squared_error(selected_xy[y_col], y_pred)
        rmse_add = np.sqrt(mse_add)
        r2_add = r2_score(selected_xy[y_col], y_pred)
        print("--------MES RMSE R2 (add)--------")
        print("MSE_add:", f"{mse_add:.3f}")
        print("RMSE_add:", f"{rmse_add:.3f}")
        print("R2_add:", f"{r2_add:.3f}")
        ax.text(
            0.99,
            0.11,
            f"RMSE_add: {rmse_add:.3f}, R$^{{2}}$_add: {r2_add:.2f}",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=12,
            c=metric_colors[1],
        )

    return base_xy, selected_xy




############ページタイトル設定############

# st.set_page_config(
#     page_title="d18O mapping",
#     # page_icon="🗾",
#     layout="wide"
#     )









def main():
    st.header(f'Correlation Overview ({version})')
    # Preserve this page as a research-prototype view of the original exploratory workflow.
    st.caption("This page preserves the original exploratory workflow used during development.")
    
    
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
    df_original = load_isotope_data_cached(ref_data)
    
    
    
    
######  scalebarで制御してsubmitする場合 ################

    st.sidebar.header("Correlation controls")
    st.sidebar.caption(f"Correlation Overview ({version})")

    with st.sidebar.form("parameter", clear_on_submit=False):

##############################################################################
# サイドバーここから
##############################################################################

        st.subheader(getattr(envgeo_utils, "DATA_FILTERING_LABEL", "Data filtering"))
        st.caption("Set the shared filters used for the compiled correlation figures.")
        submit_top = st.form_submit_button(
            "Apply settings",
            use_container_width=True,
        )
        
        
        
        
        
    
    
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
        
        

        
        
        
        st.caption('Cruise area reference (2015-2021)')
        st.image("data/sites_20230515.gif") 
    
          
          
        
        #スペース入れる
        # st.subheader(':blue[  ]')
        # st.subheader(':blue[  ]')
        st.subheader(getattr(envgeo_utils, "FIGURE_CONTROLS_LABEL", "Figure controls"))
        
        
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
    
                            
        submit_bottom = st.form_submit_button(
            "Apply settings!",
            use_container_width=True,
        )
        submitted = submit_top or submit_bottom

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
            return filter_observation_data(
                df_original,
                selected_cruise=selected_cruise,
                selected_months=selected_months,
                year_range=(sld_year_min, sld_year_max),
                lon_range=(sld_lon_min, sld_lon_max),
                lat_range=(sld_lat_min, sld_lat_max),
                depth_range=(sld_depth_min, sld_depth_max),
                salinity_range=(sld_sal_min, sld_sal_max),
            )
    
    
    
    
        
    #全体のタイトル名　　手入力
    # main_title = 'DEPTH PROFILE (V02)'
    main_title = fig_title
    
    # --- 月 (スライダー用) ---  
    # sub_title = 'Lon:'+str(sld_lon_min)+'-'+str(sld_lon_max)+', Lat:'+str(sld_lat_min)+'-'+str(sld_lat_max)+', Y:'+str(sld_year_min)+'-'+str(sld_year_max)+', M:'+str(sld_month_min)+'-'+str(sld_month_max)+', S:'+str(sld_sal_min)+'-'+str(sld_sal_max)+', D:'+str(sld_depth_min)+'-'+str(sld_depth_max)+'m'
    # --- 月 (multiselect用) ---
    # 月の表示用テキストを作成（選択されたリストをカンマ区切りにする）
    month_text = ", ".join(map(str, sorted(selected_months))) if selected_months else "None"
    month_display = build_month_display(selected_months)
        
        

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
    data_found = count_valid_rows(df1, ["d18O"])
    data_found_num = str(data_found)

    
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
        print('-------------SUB_FIG   salinity vs d18O-------------')
        ax = plt.subplot(324)
        base_frames = [
            load_isotope_data_cached(ref_data, sheet_num=sheet_num_XY)
            for sheet_num_XY in sheet_num_XY
        ]
        df_fig_all = pd.concat(base_frames, ignore_index=True)
        df_fig_add = df_fig_add_salinity_d18O
        df_fig_add_for_d18O_dD = df_fig_add.copy()

        print('要素と出現数:', df_fig_all[selected_row].value_counts().to_dict())
        print('要素と出現数:', df_fig_all[selected_row].count().sum())
        print('---------------')
        print('要素と出現数:', df_fig_add[selected_row].value_counts().to_dict())
        print('要素と出現数:', df_fig_add[selected_row].count().sum())
        print('---------------')

        selected_color = color[sheet_num_add[0]] if X_Y_add2 == 1 and X_Y_C_add_each == 1 else X_Y_C_add
        plot_xy_with_regression(
            ax=ax,
            base_df=df_fig_all,
            selected_df=df_fig_add,
            x_col=X_data,
            y_col=Y_data,
            x_label=X_label,
            y_label=Y_label,
            x_scale=iso_scale_X,
            y_scale=iso_scale_Y,
            x_limits=(lim_min_X, lim_max_X),
            y_limits=(lim_min_Y, lim_max_Y),
            x_formatter=FormatStrFormatter("%.f"),
            y_formatter=FormatStrFormatter("%+.1f"),
            tick_length=ax_length,
            main_style={"size": X_Y_S, "color": X_Y_C, "marker": X_Y_M, "alpha": alpha_all},
            selected_style={"size": X_Y_S, "color": selected_color, "marker": X_Y_M, "alpha": alpha_selected},
            figure_title=fig_title_X_Y,
            selected_label=sheet_names_add2,
            selected_row=selected_row,
            write_main_regression=(reg_line_write == 1),
            write_selected_regression=(X_Y_add2 == 1 and reg_line_add_write == 1),
        )
    
    
    
        
            
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
        print('-------------SUB_FIG   d13C vs d18O-------------')
        ax = plt.subplot(326)
        base_frames = [
            load_isotope_data_cached(ref_data, sheet_num=sheet_num_XY)
            for sheet_num_XY in sheet_num_XY
        ]
        df_fig_all = pd.concat(base_frames, ignore_index=True)
        df_fig_add = df_fig_add_for_d18O_dD.copy()

        print('要素と出現数:', df_fig_all[selected_row].value_counts().to_dict())
        print('要素と出現数:', df_fig_all[selected_row].count().sum())
        print('---------------')
        print('要素と出現数:', df_fig_add[selected_row].value_counts().to_dict())
        print('要素と出現数:', df_fig_add[selected_row].count().sum())
        print('---------------')

        selected_color = color[sheet_num_add[0]] if X_Y_add2 == 1 and X_Y_C_add_each == 1 else X_Y_C_add
        plot_xy_with_regression(
            ax=ax,
            base_df=df_fig_all,
            selected_df=df_fig_add,
            x_col=X_data,
            y_col=Y_data,
            x_label=X_label,
            y_label=Y_label,
            x_scale=iso_scale_X,
            y_scale=iso_scale_Y,
            x_limits=(lim_min_X, lim_max_X),
            y_limits=(lim_min_Y, lim_max_Y),
            x_formatter=FormatStrFormatter("%+.1f"),
            y_formatter=FormatStrFormatter("%+.1f"),
            tick_length=ax_length,
            main_style={"size": X_Y_S, "color": X_Y_C, "marker": X_Y_M, "alpha": alpha_all},
            selected_style={"size": X_Y_S, "color": selected_color, "marker": X_Y_M, "alpha": alpha_selected},
            figure_title=fig_title_X_Y,
            selected_label=sheet_names_add2,
            selected_row=selected_row,
            write_main_regression=(reg_line_write == 1),
            write_selected_regression=(X_Y_add2 == 1 and reg_line_add_write == 1),
        )

    
    
    else:()
    
    
    
    
    
    
    
    
    
    
    
    
    #画像を保存，以下の方法だとローカルにも保存されてしまう
    # fn = envgeo_utils.build_figure_filename("Fig_compiled_SW", sub_title)
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
    fn = envgeo_utils.build_figure_filename("Fig_compiled_SW", sub_title)
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
    
