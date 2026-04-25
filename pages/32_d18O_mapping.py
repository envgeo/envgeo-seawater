#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:15:03 2023
@author: Toyoho Ishimura @Kyoto-U
2026/02/10 update 
"""




# --- Version info ---
version = "1.0.2" #v220f_20260425

# ToDo
# このバージョンは補完計算の調整が必要



fig_title = "envgeo-seawater-database"  # 2026/02/12


import streamlit as st
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import plotly.express as px
import math
import envgeo_utils  
from scipy.interpolate import griddata # コンターマップ用
import cartopy.feature as cfeature  # 陸地塗りつぶし用
import io # ファイル処理用
pd.set_option('future.no_silent_downcasting', True)



def main():
    
        
    # タイトル
    st.header(f'δ18O Mapping ({version})')
  
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
    # envgeo_utilsからデータフレーム読み込み
    ##############################################################################
    df1 = envgeo_utils.load_isotope_data(ref_data)
   
    if df1.empty:
        st.warning("No data available for the selected conditions.")
        return


    ##############################################################################
    # サイドバーここから　　df1フィルタリング　も一括で
    #　2026/03/06　Min-Maxをdfから取得に変更
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
     submitted) = envgeo_utils.sidebar_filter_and_display(df1, ref_data, data_source_JAPAN_SEA, data_source_AROUND_JAPAN)




    ##############################################################################
    # 図の中心とスケール変更
    ##############################################################################

    # サイドバーの中にコンテナを作成し、境界線（border）を有効にする
    with st.sidebar.container(border=True):
        st.subheader(':blue[--- Map Display Settings ---]')
        
        center_option = st.radio(
            ":blue[Map Center:]",
            ("Atlantic (0°)", "Pacific (180°)"),
            horizontal=True
        )
        
        lon_center = 0 if "Atlantic" in center_option else 180

        # Re-map longitudes into the visible 360-degree window of the selected map center.
        # 選択した地図中心で見えている 360 度の範囲に経度を並べ替える。
        def normalize_lon_to_center(lon, center):
            return ((np.asarray(lon) - (center - 180)) % 360) + (center - 180)

        # Pacific-centered maps use 0-360 so regional windows such as 120-240E remain selectable.
        # 太平洋中心では 0-360 を使い、120-240E のような範囲をそのまま選べるようにする。
        lon_slider_min = -180 if lon_center == 0 else 0
        lon_slider_max = 180 if lon_center == 0 else 360
        
        # --- 図のスケール設定 (表示は整数、内部計算は微小オフセットあり) ---
        if ref_data == data_source_JAPAN_SEA:
            # 地図の描画範囲（日本海）
            map_lon_default = (120, 145)
            map_lat_raw = st.slider('Map Latitude ', 20, 45, (20, 45), step=1)
            cbar_val = (-1.5, 1.0) # カラーバー初期値
    
        elif ref_data == data_source_AROUND_JAPAN:
            # 地図の描画範囲（日本周辺）
            map_lon_default = (120, 180) if lon_center == 0 else (120, 240)
            map_lat_raw = st.slider('Map Latitude ', -70, 55, (0, 55), step=1)
            cbar_val = (-1.5, 1.0)
    
        else:
            # 地図の描画範囲（全体）
            map_lon_default = (-180, 180) if lon_center == 0 else (0, 360)
            map_lat_raw = st.slider('Map Latitude', -90, 90, (-90, 90), step=1)
            cbar_val = (-5.0, 2.0)

        map_lon_raw = st.slider(
            'Map Longitude ',
            lon_slider_min,
            lon_slider_max,
            map_lon_default,
            step=1
        )
    
        # 内部計算用に0.001のオフセットを適用
        map_lon_min, map_lon_max = map_lon_raw[0] - 0.001, map_lon_raw[1] + 0.001
        map_lat_min, map_lat_max = map_lat_raw[0] - 0.001, map_lat_raw[1] + 0.001
    
        # --- カラーバー用d18O範囲 ---
        # ここも step=1 にすることで整数表示になる（小数が必要な場合は step=0.1 に変更）
        fig_d18O_min, fig_d18O_max = st.slider(
            label='d18O range for colorbar',
            min_value=-20,
            max_value=5,
            value=(int(cbar_val[0]), int(cbar_val[1])),
            step=1
        )

   



    ##############################################################################
    #  ここから図の設定と描画
    ##############################################################################


    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################

    st.markdown("##### :red[--- Adjust the map view in [Map Display Settings]. ---]")
    st.caption("Even samples lacking salinity/temperature values are included in the plot.")



    
    ###############################################################################################
    ############################################################################################### 
    # 計算できない，もしくはカラーバー用のデータが無い場合に除外

    # 1. 計算する
    original_len_df1 = len(df1)
    # 2. 【追加】計算できなかった行（null）をその場で除外する
    df1 = df1.dropna(subset=['d18O'])
   
    # 消えた数を出力
    removed_len_df1 = original_len_df1 - len(df1)
    plotted_len_df1 = original_len_df1 - removed_len_df1

    if removed_len_df1 > 0:
        st.caption(f":red[Note: {plotted_len_df1} samples were plotted and {removed_len_df1} samples were excluded due to no d18O data.]")
       
    ###############################################################################################
    ############################################################################################### 
    
    

  


    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
    
    
    #ファイル名用の項目
    
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
    
    title_head = str(main_title+'\n'+sub_title+'\n'+sub_title2)
    title_head2 = title_head.replace('_', ' ') #図のタイトル表示用

    # ファイルのタイトルを自動設定
    sub_title2 = sub_title
    sub_title2 = sub_title2.replace(':', '') #pdf書き出し用
    sub_title2 = sub_title2.replace(',', '_') #pdf書き出し用
    sub_title2 = sub_title2.replace(' ', '') #pdf書き出し用
    sub_tite = str('Fig_d18O_map'+'_'+sub_title2+".png")

    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################


    plt.rcParams["font.size"] = 15

    #############################################################
    # Scatter Map
    #############################################################
    fig = plt.figure(figsize=(12, 8), facecolor="white", dpi=150)
    
    ax = fig.add_subplot(
        1, 1, 1,
        projection=ccrs.PlateCarree(central_longitude=lon_center)
    )
    
    # Keep the requested extent inside the longitude domain used by the current center option.
    # 指定された表示範囲が、現在の地図中心で使う経度範囲から外れないようにする。
    map_lon_min = max(lon_slider_min, map_lon_min)
    map_lon_max = min(lon_slider_max, map_lon_max)
    map_lat_min = max( -90, map_lat_min)
    map_lat_max = min(  90, map_lat_max)
    
    ax.set_extent(
        [map_lon_min, map_lon_max, map_lat_min, map_lat_max],
        crs=ccrs.PlateCarree()
    )
    
    ax.coastlines(resolution="50m", zorder=3)
    ax.add_feature(cfeature.LAND,
                   facecolor="white",
                   edgecolor="black",
                   linewidth=0.5,
                   zorder=2)
    
    ax.gridlines(draw_labels=True, zorder=4)
    
    # Plot sample points in the same longitude frame as the selected map extent.
    # 観測点も、選択した表示範囲と同じ経度系に変換してから描画する。
    lon_wrapped = normalize_lon_to_center(df1["Longitude_degE"].values, lon_center)
    
    ax_scatter = ax.scatter(
        lon_wrapped,
        df1["Latitude_degN"],
        c=df1["d18O"],
        cmap="jet",
        s=10,
        alpha=0.7,
        vmin=fig_d18O_min,
        vmax=fig_d18O_max,
        transform=ccrs.PlateCarree(),
        zorder=1
    )
    
    
    # ---- Scatter Map用のカラーバーを追加 ----
    cbar_scatter = fig.colorbar(
        ax_scatter,
        ax=ax,
        orientation="horizontal", # 横向き
        pad=0.08,                  # 地図との隙間
        fraction=0.04,             # カラーバーの大きさ
        aspect=25,                 # カラーバーの細長さ
        extend="neither"           # 【重要】ここを "neither" にすると両端が□になります
    )
    cbar_scatter.set_label(r"$\delta^{18}$O (VSMOW)", fontsize=12)
    
    
    ax.set_title(title_head2,fontsize=15)
    
    # PNG保存（Scatter）
    img_scatter = io.BytesIO()
    fig.savefig(img_scatter, format="png", dpi=300, bbox_inches="tight")
    img_scatter.seek(0)
    
    

    #############################################################
    # 補間（周期拡張オプション付き）
    #############################################################
    
    edge_wrap_relevant = (
        lon_center == 180
        and (map_lon_min <= lon_slider_min + 1.0 or map_lon_max >= lon_slider_max - 1.0)
    )
    lon_original = df1["Longitude_degE"].values
    # Interpolation also needs the center-adjusted longitude frame to match the displayed window.
    # 補間計算でも、表示中のウィンドウと同じ経度系を使う必要がある。
    lon_for_interp = normalize_lon_to_center(lon_original, lon_center)
    # Build the interpolation grid in the same longitude domain as the slider and set_extent.
    # 補間グリッドも slider / set_extent と同じ経度範囲で作る。
    grid_lon = np.linspace(lon_slider_min, lon_slider_max, 360)
    lat_vals     = df1["Latitude_degN"].values
    val          = df1["d18O"].values
    
    #############################################################
    #  データ拡張（必要なときだけ）
    #############################################################
    
    lon_ext = lon_for_interp
    lat_ext = lat_vals
    val_ext = val
    
    # ---- グリッド ----
    grid_lat = np.linspace(map_lat_min, map_lat_max, 250)
    
    X, Y = np.meshgrid(grid_lon, grid_lat)
    
    # ---- 補間 ----
    Z = griddata(
        (lon_ext, lat_ext),
        val_ext,
        (X, Y),
        method="linear"
    )
    
    Z = np.ma.masked_invalid(Z)
    
    Z_plot = Z
    lon_plot = grid_lon
        
    #############################################################
    # Contour Map Figure 作成
    #############################################################
    fig_contour = plt.figure(figsize=(12, 8), facecolor="white", dpi=150)
    
    ax2 = fig_contour.add_subplot(
        1, 1, 1,
        projection=ccrs.PlateCarree(central_longitude=lon_center)
    )
    
    # extent
    ax2.set_extent(
        [map_lon_min, map_lon_max, map_lat_min, map_lat_max],
        crs=ccrs.PlateCarree()
        )   
    #############################################################
    # 描画
    #############################################################
    
    levels = np.linspace(fig_d18O_min, fig_d18O_max, 51)
    
    ax_cntr = ax2.contourf(
        lon_plot,
        grid_lat,
        Z_plot,
        levels=levels,
        cmap="jet",
        transform=ccrs.PlateCarree(),
        alpha=0.85,
        zorder=1
    )
    
    ax2.add_feature(
        cfeature.LAND,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        zorder=2
    )
    
    ax2.coastlines(resolution="50m", zorder=3)
    ax2.gridlines(draw_labels=True, zorder=4)
    
    # ---- 観測点（表示用はwrap）----
    lon_wrapped = normalize_lon_to_center(lon_original, lon_center)
    
    ax2.scatter(
        lon_wrapped,
        lat_vals,
        c="black",
        s=2,
        alpha=0.3,
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    ax2.set_title(title_head2,fontsize=15)
    
    
    # ---- カラーバーの目盛り固定設定 ----
    num_ticks = 6  # ★ここに出したい目盛りの数を指定（例：6個なら5等分）
    # 1. スライダーの範囲を単純に分割
    raw_ticks = np.linspace(fig_d18O_min, fig_d18O_max, num_ticks)

    # 2. 分割した値を「キリの良い数字」に丸める（小数点第1位など）
    # ※丸めないと 0.1000000000004 のような表示になる
    fixed_ticks = [round(t, 1) for t in raw_ticks]

    
    # ---- カラーバーの描画 ----
    cbar = fig_contour.colorbar(
        ax_cntr,
        ax=ax2,
        orientation="horizontal",
        pad=0.08,
        fraction=0.05,
        ticks=fixed_ticks,   # 固定された目盛り位置
        extend="neither"     # 両端は□
    )

    
    # ---- ラベルの書式を小数点第1位に統一 ----
    cbar.ax.set_xticklabels([f"{t:.1f}" for t in fixed_ticks])
    
    cbar.set_label(r"$\delta^{18}$O (VSMOW)")
    

    
    # ---- PNG保存 ----
    img_contour = io.BytesIO()
    fig_contour.savefig(img_contour, format="png", dpi=300, bbox_inches="tight")
    img_contour.seek(0)
        
    #############################################################
    # 表示切替
    #############################################################
    map_type = st.radio(
        ":blue[Map Type:]",
        ("Scatter Map", "Contour Map"),
        horizontal=True
    )
    
    if map_type == "Scatter Map":
        st.download_button(
            "Download Scatter Map",
            img_scatter,
            f"Fig_d18O_scatter_{sub_title2}_center{lon_center}.png",
            "image/png"
        )
        st.pyplot(fig)
    
    else:
        st.download_button(
            "Download Contour Map",
            img_contour,
            f"Fig_d18O_contour_{sub_title2}_center{lon_center}.png",
            "image/png"
        )
        st.pyplot(fig_contour)


    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
    
    # Map section 
    
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
    lat_min, lat_max = df1["Latitude_degN"].min(), df1["Latitude_degN"].max()
    lon_min, lon_max = df1["Longitude_degE"].min(), df1["Longitude_degE"].max()

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
        df1, 
        lat="Latitude_degN", 
        lon="Longitude_degE",
        color="d18O", 
        color_continuous_scale=c_scale_d18o,
        hover_data={
            "Latitude_degN": True,  # 名前を表示
            "Longitude_degE": True,  # 値を表示
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
            "reference": True,  # カテゴリを表示

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
            xanchor='right',
        ),
        # --- ここで初期値を設定 ---
        coloraxis_cmin=fig_d18O_min, # 最小値
        coloraxis_cmax=fig_d18O_max   # 最大値
    )
    

    # 6. 表示 (use_container_width=True を確実に使う)
    # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）　
    # マウスホイールでのズームが強制的に有効
    st.plotly_chart(
        fig_map, 
        use_container_width=True, # クラウドではTrueの方が見やすいです
        key="d18O_map",
        config={'scrollZoom': True, 'displayModeBar': True} # ズームを有効化
    )






    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

        
    # Sidebar-filtered datasetを読み出し
    envgeo_utils.display_isotope_table(df1)
    
  
    
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################

if __name__ == '__main__':
    main()
    
    
