#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:15:03 2023
@author: Toyoho Ishimura @Kyoto-U
2026/02/10 update
"""




# --- Version info ---
version = "1.3.0" #v220f_20260425

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


MAP_PARAMETER_LABELS = {
    "d18O": r"$\delta^{18}$O (VSMOW)",
    "dD": r"$\delta$D (VSMOW)",
    "d-excess": "d-excess",
    "Salinity": "Salinity",
    "Temperature_degC": "Temperature (degC)",
}

MAP_PARAMETER_PLOTLY_LABELS = {
    "d18O": "δ18O (‰)",
    "dD": "δD (‰)",
    "d-excess": "d-excess (‰)",
    "Salinity": "Salinity",
    "Temperature_degC": "Temperature (degC)",
}


def get_parameter_color_range_defaults(parameter, ref_data, data_source_global):
    """
    Return slider limits, default color range, and step for each map parameter.

    地図で色分けするパラメーターごとに、カラーバー範囲の初期値を返します。
    """
    if parameter == "d18O":
        default_range = (-5.0, 2.0) if ref_data == data_source_global else (-1.5, 1.0)
        return -20.0, 5.0, default_range, 0.1
    if parameter == "dD":
        default_range = (-50.0, 20.0) if ref_data == data_source_global else (-20.0, 10.0)
        return -200.0, 100.0, default_range, 1.0
    if parameter == "d-excess":
        return -30.0, 40.0, (-10.0, 25.0), 0.5
    if parameter == "Salinity":
        default_range = (0.0, 40.0) if ref_data == data_source_global else (30.0, 36.0)
        return 0.0, 50.0, default_range, 0.1
    if parameter == "Temperature_degC":
        default_range = (-5.0, 35.0) if ref_data == data_source_global else (-2.0, 30.0)
        return -5.0, 45.0, default_range, 0.5

    return -20.0, 20.0, (-5.0, 5.0), 0.1



def main():
    
        
    # タイトル
    st.header(f'Isotope & Hydrographic Mapping ({version})')
    st.caption(
        "Map d18O, dD, d-excess, salinity, temperature, and related seawater parameters."
    )
  
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

    map_parameter_candidates = [
        "d18O",
        "dD",
        "d-excess",
        "Salinity",
        "Temperature_degC",
    ]
    map_parameter_options = [
        item for item in map_parameter_candidates
        if item in df1.columns
    ]
    if not map_parameter_options:
        st.warning("No mappable isotope or hydrographic parameters are available.")
        return

    st.subheader("Map")
    parameter_col, map_type_col = st.columns([1, 1])
    with parameter_col:
        selected_parameter = st.selectbox(
            "Mapped parameter",
            map_parameter_options,
            index=0,
            help=(
                "Choose the seawater parameter plotted on the map. "
                "Rows without values for the selected parameter are excluded from the map."
            ),
        )
    with map_type_col:
        map_type = st.radio(
            "Map type",
            ("Scatter Map", "Contour Map"),
            horizontal=True,
        )
    parameter_label = MAP_PARAMETER_LABELS.get(selected_parameter, selected_parameter)
    parameter_plotly_label = MAP_PARAMETER_PLOTLY_LABELS.get(selected_parameter, selected_parameter)




    ##############################################################################
    # 図の中心とスケール変更
    ##############################################################################

    # サイドバーの中にコンテナを作成し、境界線（border）を有効にする
    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "MAP_DISPLAY_SETTINGS_LABEL", "Map display settings"))
        
        center_option = st.radio(
            ":blue[Map Center:]",
            ("Atlantic (0°)", "Pacific (180°)"),
            horizontal=True
        )
        
        lon_center = 0 if "Atlantic" in center_option else 180

        colormap_options = envgeo_utils.get_plotly_colormap_options(selected_parameter)
        selected_colormap_label = st.selectbox(
            "Colormap",
            list(colormap_options.keys()),
            index=list(colormap_options.keys()).index("Jet"),
            help=(
                "Choose the color palette used for parameter maps. "
                "cmocean palettes are designed for oceanographic data."
            ),
        )
        map_plotly_colorscale = envgeo_utils.get_plotly_colormap(
            selected_parameter,
            selected_colormap_label,
        )
        map_matplotlib_colormap = envgeo_utils.get_matplotlib_colormap(
            selected_parameter,
            selected_colormap_label,
        )

        # Re-map longitudes into the visible 360-degree window of the selected map center.
        # 選択した地図中心で見えている 360 度の範囲に経度を並べ替える。
        def normalize_lon_to_center(lon, center):
            return ((np.asarray(lon) - (center - 180)) % 360) + (center - 180)

        # Pacific-centered maps use 0-360 so regional windows such as 120-240E remain selectable.
        # 太平洋中心では 0-360 を使い、120-240E のような範囲をそのまま選べるようにする。
        lon_slider_min = -180 if lon_center == 0 else 0
        lon_slider_max = 180 if lon_center == 0 else 360

        map_state_key = f"map_display_settings::{ref_data}::{lon_center}"
        
        # --- 図のスケール設定 (表示は整数、内部計算は微小オフセットあり) ---
        if ref_data == data_source_JAPAN_SEA:
            # 地図の描画範囲（日本海）
            map_lon_default = (120, 145)
            map_lat_default = (20, 45)
            lat_slider_min, lat_slider_max = 0, 70
    
        elif ref_data == data_source_AROUND_JAPAN:
            # 地図の描画範囲（日本周辺）
            map_lon_default = (120, 180) if lon_center == 0 else (120, 240)
            map_lat_default = (0, 55)
            lat_slider_min, lat_slider_max = -70, 70
    
        else:
            # 地図の描画範囲（全体）
            map_lon_default = (-180, 180) if lon_center == 0 else (0, 360)
            map_lat_default = (-90, 90)
            lat_slider_min, lat_slider_max = -90, 90
        color_range_min, color_range_max, color_range_default, color_range_step = (
            get_parameter_color_range_defaults(
                selected_parameter,
                ref_data,
                data_source_GLOBAL,
            )
        )

        if map_state_key not in st.session_state:
            st.session_state[map_state_key] = {
                "map_lon_raw": map_lon_default,
                "map_lat_raw": map_lat_default,
                "color_ranges": {},
                "colorbar_thickness": 4,
                "colorbar_length": 90,
                "colorbar_font_size": 12,
            }

        map_settings = st.session_state[map_state_key]
        color_ranges = map_settings.setdefault("color_ranges", {})
        current_color_range = color_ranges.get(selected_parameter, color_range_default)

        with st.form(key=f"map_display_form::{ref_data}::{lon_center}"):
            map_lon_raw_form = st.slider(
                'Map Longitude ',
                lon_slider_min,
                lon_slider_max,
                map_settings["map_lon_raw"],
                step=1
            )
            map_lat_raw_form = st.slider(
                'Map Latitude ',
                lat_slider_min,
                lat_slider_max,
                map_settings["map_lat_raw"],
                step=1
            )
        
            selected_color_range_form = st.slider(
                label=f'{selected_parameter} range for colorbar',
                min_value=color_range_min,
                max_value=color_range_max,
                value=current_color_range,
                step=color_range_step
            )
            colorbar_thickness_form = st.slider(
                "Colorbar thickness",
                min_value=2,
                max_value=10,
                value=map_settings.get("colorbar_thickness", 4),
                step=1,
                help=(
                    "Adjust the thickness of the horizontal parameter colorbar "
                    "in the Matplotlib scatter and contour maps."
                ),
            )
            colorbar_length_form = st.slider(
                "Colorbar length",
                min_value=40,
                max_value=100,
                value=map_settings.get("colorbar_length", 90),
                step=5,
                help=(
                    "Adjust the displayed length of the horizontal parameter colorbar. "
                    "100 uses the full available width."
                ),
            )
            colorbar_font_size_form = st.slider(
                "Colorbar font size",
                min_value=8,
                max_value=20,
                value=map_settings.get("colorbar_font_size", 12),
                step=1,
                help="Adjust the label and tick font size of the parameter colorbar.",
            )
            apply_map_settings = st.form_submit_button("Apply map settings")

        if apply_map_settings:
            map_settings = {
                "map_lon_raw": map_lon_raw_form,
                "map_lat_raw": map_lat_raw_form,
                "color_ranges": {
                    **map_settings.get("color_ranges", {}),
                    selected_parameter: selected_color_range_form,
                },
                "colorbar_thickness": colorbar_thickness_form,
                "colorbar_length": colorbar_length_form,
                "colorbar_font_size": colorbar_font_size_form,
            }
            st.session_state[map_state_key] = map_settings

        map_lon_raw = map_settings["map_lon_raw"]
        map_lat_raw = map_settings["map_lat_raw"]
        selected_color_range = map_settings.get("color_ranges", {}).get(
            selected_parameter,
            current_color_range,
        )
        parameter_min, parameter_max = selected_color_range
        colorbar_thickness = map_settings.get("colorbar_thickness", 4) / 100
        colorbar_length = map_settings.get("colorbar_length", 90) / 100
        colorbar_font_size = map_settings.get("colorbar_font_size", 12)

        # 内部計算用に0.001のオフセットを適用
        map_lon_min, map_lon_max = map_lon_raw[0] - 0.001, map_lon_raw[1] + 0.001
        map_lat_min, map_lat_max = map_lat_raw[0] - 0.001, map_lat_raw[1] + 0.001

   



    ##############################################################################
    #  ここから図の設定と描画
    ##############################################################################


    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    st.caption(getattr(envgeo_utils, "MAP_AREA_HELP_TEXT", "Map center, extent, colormap, and figure settings can be adjusted in the sidebar."))



    
    ###############################################################################################
    ###############################################################################################
    # 計算できない，もしくはカラーバー用のデータが無い場合に除外

    # 1. 計算する
    original_len_df1 = len(df1)
    # 2. 【追加】計算できなかった行（null）をその場で除外する
    df1 = df1.dropna(subset=[selected_parameter])
   
    # 消えた数を出力
    removed_len_df1 = original_len_df1 - len(df1)
    plotted_len_df1 = original_len_df1 - removed_len_df1

    if removed_len_df1 > 0:
        st.caption(
            f":red[Note: {plotted_len_df1} samples were plotted and "
            f"{removed_len_df1} samples were excluded due to no {selected_parameter} data.]"
        )

    if df1.empty:
        st.warning(f"No valid {selected_parameter} data are available for the selected conditions.")
        return
       
    ###############################################################################################
    ###############################################################################################
    
    

  


    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    
    
    #ファイル名用の項目
    
    #全体のタイトル名
    main_title = f"{fig_title} - {selected_parameter}"
    
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

    # File names for downloaded figures / 図保存用のファイル名
    safe_parameter_name = envgeo_utils.safe_filename_text(selected_parameter)

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################


    plt.rcParams["font.size"] = 15

    # Keep the requested extent inside the longitude domain used by the current center option.
    # 指定された表示範囲が、現在の地図中心で使う経度範囲から外れないようにする。
    map_lon_min = max(lon_slider_min, map_lon_min)
    map_lon_max = min(lon_slider_max, map_lon_max)
    map_lat_min = max( -90, map_lat_min)
    map_lat_max = min(  90, map_lat_max)
    
    if map_type == "Scatter Map":
        #############################################################
        # Scatter Map
        #############################################################
        fig = plt.figure(figsize=(12, 8), facecolor="white", dpi=150)
        
        ax = fig.add_subplot(
            1, 1, 1,
            projection=ccrs.PlateCarree(central_longitude=lon_center)
        )
        
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
            c=df1[selected_parameter],
            cmap=map_matplotlib_colormap,
            s=10,
            alpha=0.7,
            vmin=parameter_min,
            vmax=parameter_max,
            transform=ccrs.PlateCarree(),
            zorder=1
        )
        
        # ---- Scatter Map用のカラーバーを追加 ----
        cbar_scatter = fig.colorbar(
            ax_scatter,
            ax=ax,
            orientation="horizontal", # 横向き
            pad=0.08,                  # 地図との隙間
            fraction=colorbar_thickness, # カラーバーの太さ
            shrink=colorbar_length,    # カラーバーの長さ
            aspect=25,                 # カラーバーの細長さ
            extend="neither"           # 【重要】ここを "neither" にすると両端が□になります
        )
        cbar_scatter.set_label(parameter_label, fontsize=colorbar_font_size)
        cbar_scatter.ax.tick_params(labelsize=colorbar_font_size)
        
        ax.set_title(title_head2,fontsize=15)
        
        # PNG保存（Scatter）
        img_scatter = io.BytesIO()
        fig.savefig(img_scatter, format="png", dpi=300, bbox_inches="tight")
        img_scatter.seek(0)
        
        st.download_button(
            "Download Scatter Map",
            img_scatter,
            envgeo_utils.build_figure_filename(
                f"Fig_{safe_parameter_name}_scatter",
                f"{sub_title}_center{lon_center}"
            ),
            "image/png"
        )
        st.pyplot(fig)
    
    else:
        #############################################################
        # Contour Map
        #############################################################
        lon_original = df1["Longitude_degE"].values
        # Interpolation also needs the center-adjusted longitude frame to match the displayed window.
        # 補間計算でも、表示中のウィンドウと同じ経度系を使う必要がある。
        lon_for_interp = normalize_lon_to_center(lon_original, lon_center)
        # Build the interpolation grid in the same longitude domain as the slider and set_extent.
        # 補間グリッドも slider / set_extent と同じ経度範囲で作る。
        grid_lon = np.linspace(lon_slider_min, lon_slider_max, 360)
        lat_vals     = df1["Latitude_degN"].values
        val          = df1[selected_parameter].values
        
        # ---- グリッド ----
        grid_lat = np.linspace(map_lat_min, map_lat_max, 250)
        X, Y = np.meshgrid(grid_lon, grid_lat)
        
        # ---- 補間 ----
        Z = griddata(
            (lon_for_interp, lat_vals),
            val,
            (X, Y),
            method="linear"
        )
        Z_plot = np.ma.masked_invalid(Z)
        lon_plot = grid_lon
        
        fig_contour = plt.figure(figsize=(12, 8), facecolor="white", dpi=150)
        ax2 = fig_contour.add_subplot(
            1, 1, 1,
            projection=ccrs.PlateCarree(central_longitude=lon_center)
        )
        
        ax2.set_extent(
            [map_lon_min, map_lon_max, map_lat_min, map_lat_max],
            crs=ccrs.PlateCarree()
        )
        
        levels = np.linspace(parameter_min, parameter_max, 51)
        ax_cntr = ax2.contourf(
            lon_plot,
            grid_lat,
            Z_plot,
            levels=levels,
            cmap=map_matplotlib_colormap,
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
        
        num_ticks = 6
        raw_ticks = np.linspace(parameter_min, parameter_max, num_ticks)
        fixed_ticks = [round(t, 1) for t in raw_ticks]
        
        cbar = fig_contour.colorbar(
            ax_cntr,
            ax=ax2,
            orientation="horizontal",
            pad=0.08,
            fraction=colorbar_thickness,
            shrink=colorbar_length,
            ticks=fixed_ticks,
            extend="neither"
        )
        cbar.ax.set_xticklabels([f"{t:.1f}" for t in fixed_ticks])
        cbar.set_label(parameter_label, fontsize=colorbar_font_size)
        cbar.ax.tick_params(labelsize=colorbar_font_size)
        
        img_contour = io.BytesIO()
        fig_contour.savefig(img_contour, format="png", dpi=300, bbox_inches="tight")
        img_contour.seek(0)
        
        st.download_button(
            "Download Contour Map",
            img_contour,
            envgeo_utils.build_figure_filename(
                f"Fig_{safe_parameter_name}_contour",
                f"{sub_title}_center{lon_center}"
            ),
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
    # --- Location map / 採取地点の地図表示 ---
    st.divider()
    st.subheader('Location Map')



    # Keep map controls compact so the map remains visible after Streamlit reruns.
    # Streamlitの再実行後も地図が見つけやすいよう、地図設定をポップオーバーに集約する。
    with st.popover("Map controls", use_container_width=True):
        map_mode = st.radio(
            "Map Style:",
            envgeo_utils.MAP_MODE_OPTIONS,
            horizontal=True,
            key="map_style_31_auto"
        )
    st.caption(f"Map Style: {map_mode}")

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
    hover_data = {column: True for column in hover_columns if column in df1.columns}

    # 3. 地図の作成 (px.scatter_mapbox内ではwidthを指定しない)
    fig_map = px.scatter_mapbox(
        df1,
        lat="Latitude_degN",
        lon="Longitude_degE",
        color=selected_parameter,
        color_continuous_scale=map_plotly_colorscale,
        hover_data=hover_data,
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
            title=parameter_plotly_label,
            x=1.0,           # カラーバーを右端に寄せる
            xanchor='right',
        ),
        # --- ここで初期値を設定 ---
        coloraxis_cmin=parameter_min, # 最小値
        coloraxis_cmax=parameter_max   # 最大値
    )
    

    # 6. 表示 (use_container_width=True を確実に使う)
    # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）
    # マウスホイールでのズームが強制的に有効
    st.plotly_chart(
        fig_map,
        use_container_width=True, # クラウドではTrueの方が見やすいです
        key=f"parameter_map_{selected_parameter}",
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
    
    
