#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive 3D/4D visualizer for EnvGeo-Seawater data.

EnvGeo-Seawater データの対話型3D/4D可視化ページです。

Created: 2023-05-21
Author: Toyoho Ishimura, Kyoto University
Last reviewed: 2026-09-30
"""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import envgeo_utils

# =============================================================================
# Page configuration / ページ設定
# =============================================================================
version = "1.3.4"

def main():
    # -------------------------------------------------------------------------
    # Page introduction / ページの概要
    # -------------------------------------------------------------------------
    st.header(f'Interactive 3D/4D Visualizer ({version})')
    st.caption(
        "Use this Plotly page for interactive data exploration. "
        "For publication- or presentation-ready static figures, use the corresponding individual pages."
    )

    st.button('Reload')

    def label_for_column(column):
        """Return a concise label for a data column.

        データ列に対応する簡潔な表示名を返します。
        """
        labels = {
            "Longitude_degE": "Longitude",
            "Latitude_degN": "Latitude",
            "Depth_m": "Water Depth",
            "Temperature_degC": "Temperature(C)",
            "Salinity": "Salinity",
            "d18O": "d18O",
            "dD": "dD",
            "d-excess": "d-excess",
            "Year": "Year",
            "Month": "Month",
        }
        return labels.get(column, column)

    def available_numeric_columns(df, preferred_columns):
        """Return preferred numeric columns present in the current data.

        現在のデータに存在する優先順の数値列を返します。
        """
        return [
            column for column in preferred_columns
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
        ]

    def safe_option_index(options_list, preferred):
        """Return an option index, falling back to the first item.

        指定候補がなければ先頭項目のindexを返します。
        """
        return options_list.index(preferred) if preferred in options_list else 0

    # -------------------------------------------------------------------------
    # Data-source selection / データソース選択
    # -------------------------------------------------------------------------
    data_source_ENVGEO = envgeo_utils.data_source_ENVGEO
    data_source_AROUND_JAPAN = envgeo_utils.data_source_AROUND_JAPAN
    data_source_GLOBAL = envgeo_utils.data_source_GLOBAL
    

    ref_data = st.radio("Data source (see Home > About)", (data_source_ENVGEO, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True)

    # Citation display / 引用表示
    if ref_data == data_source_ENVGEO:
        st.write(envgeo_utils.refs_ENVGEO)
        
    elif ref_data == data_source_AROUND_JAPAN:
        st.write(envgeo_utils.refs_AROUND_JAPAN)

    elif ref_data == data_source_GLOBAL:
        st.write(envgeo_utils.refs_GLOBAL)
        
    else:
        st.warning("Invalid data source selection.")

    # Reference-data loading / 参照データの読込
    df1 = envgeo_utils.load_isotope_data(ref_data)
   
    if df1.empty:
        st.warning("No data available for the selected conditions.")
        return

    # Derived isotope parameter / 派生同位体パラメータ
    df1 = envgeo_utils.add_d_excess(df1)

    # -------------------------------------------------------------------------
    # Shared sidebar filtering / 共通sidebarによる絞り込み
    # -------------------------------------------------------------------------

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
     submitted) = envgeo_utils.sidebar_filter_and_display(df1, ref_data, data_source_ENVGEO, data_source_AROUND_JAPAN)

    # -------------------------------------------------------------------------
    # Coordinate helpers / 座標補助処理
    # -------------------------------------------------------------------------
    df1['lat'] = df1['Latitude_degN']
    df1['lon'] = df1['Longitude_degE']
    # Re-map longitudes into the visible 360-degree window of the selected map center.
    # 選択した地図中心で見えている 360 度の範囲に経度を並べ替える。
    def normalize_lon_to_center(lon, center):
        return ((np.asarray(lon) - (center - 180)) % 360) + (center - 180)

    # Convert shared map-region presets to the longitude frame used by Fig.3-Fig.6.
    # 共通の海域プリセットを、Fig.3-Fig.6で使う経度系に合わせる。
    def region_bounds_for_4d(bounds, lon_min, lon_max, lat_min, lat_max, center):
        preset_lon_min, preset_lon_max, preset_lat_min, preset_lat_max = bounds
        original_lon_span = abs(preset_lon_max - preset_lon_min)

        if original_lon_span >= 300:
            converted_lon = (lon_min, lon_max)
        else:
            converted_pair = normalize_lon_to_center([preset_lon_min, preset_lon_max], center)
            converted_lon_min = float(converted_pair[0])
            converted_lon_max = float(converted_pair[1])
            if converted_lon_min > converted_lon_max:
                converted_lon = (lon_min, lon_max)
            else:
                converted_lon = (
                    max(lon_min, int(np.floor(converted_lon_min))),
                    min(lon_max, int(np.ceil(converted_lon_max))),
                )

        converted_lat = (
            max(lat_min, int(np.floor(preset_lat_min))),
            min(lat_max, int(np.ceil(preset_lat_max))),
        )

        return converted_lon, converted_lat

    # Insert line breaks at large longitude jumps so coastlines do not draw false horizontal connectors.
    # 海岸線の大きな経度ジャンプで線を切り、不要な横線が描かれないようにする。
    def wrap_coastline_with_breaks(lon, lat, center, jump_threshold=180):
        lon_wrapped = normalize_lon_to_center(lon, center)
        lat_arr = np.asarray(lat, dtype=float)
        lon_segments = [lon_wrapped[0]]
        lat_segments = [lat_arr[0]]
        for i in range(1, len(lon_wrapped)):
            prev_lon, curr_lon = lon_wrapped[i - 1], lon_wrapped[i]
            prev_lat, curr_lat = lat_arr[i - 1], lat_arr[i]
            if (
                np.isnan(prev_lon) or np.isnan(curr_lon)
                or np.isnan(prev_lat) or np.isnan(curr_lat)
                or abs(curr_lon - prev_lon) > jump_threshold
            ):
                lon_segments.append(np.nan)
                lat_segments.append(np.nan)
            lon_segments.append(curr_lon)
            lat_segments.append(curr_lat)
        return lon_segments, lat_segments

    # Keep the longitude-latitude aspect close to the selected geographic window.
    # 選択した緯度経度の範囲に合わせて、地図の縦横比が崩れないようにする。
    def get_geo_aspectratio(x_range, y_range, z_ratio=1.0):
        lon_span = max(float(x_range[1] - x_range[0]), 0.1)
        lat_span = max(float(y_range[1] - y_range[0]), 0.1)
        mean_lat = (float(y_range[0]) + float(y_range[1])) / 2.0
        x_ratio = max((lon_span * np.cos(np.deg2rad(mean_lat))) / lat_span, 0.2)
        return dict(x=x_ratio, y=1.0, z=z_ratio)

    # -------------------------------------------------------------------------
    # Map-depth view controls / 地図・深度表示の操作
    # -------------------------------------------------------------------------
    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "MAP_DISPLAY_SETTINGS_LABEL", "Map display settings"))
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)

        # Match the map-center selector used in the 2D mapping page.
        # 2D マップページと同じ地図中心の切り替え UI を使う。
        center_option_3d = st.radio(
            "Map center for map-depth views",
            ("Atlantic (0°)", "Pacific (180°)"),
            horizontal=True,
            help="Used for map-depth views and custom longitude-latitude-depth plots.",
        )
        lon_center_3d = 0 if "Atlantic" in center_option_3d else 180

        # Offer shared colormaps, including cmocean options for oceanographic data.
        # 海洋データ向けのcmocean候補を含む共通カラーマップを使う。
        colormap_options = envgeo_utils.get_plotly_colormap_options("d18O")
        selected_colormap_label = st.selectbox(
            "Colormap",
            list(colormap_options.keys()),
            index=list(colormap_options.keys()).index(
                envgeo_utils.recommended_plotly_colormap_label("d18O")
            ),
            help="Applied to the 4D plot and the sampling-location map.",
        )
        map_colorscale = colormap_options[selected_colormap_label]

        st.subheader("Figure scale settings for Fig.3-Fig.6 map-depth views")

        # Allow Fig3-Fig6 map views to use explicit lon/lat windows from the sidebar.
        # Fig3-Fig6 の地図表示範囲を、サイドバーから緯度経度で直接調整できるようにする。
        if ref_data == data_source_ENVGEO:
            map_lon_default = (120, 145)
            map_lat_default = (20, 45)
            lon_slider_min, lon_slider_max = (0, 360) if lon_center_3d == 180 else (-180, 180)
            lat_slider_min, lat_slider_max = 0, 70
        elif ref_data == data_source_AROUND_JAPAN:
            map_lon_default = (120, 160)
            map_lat_default = (20, 60)
            lon_slider_min, lon_slider_max = (0, 360) if lon_center_3d == 180 else (-180, 180)
            lat_slider_min, lat_slider_max = -70, 70
        else:
            map_lon_default = (0, 360) if lon_center_3d == 180 else (-180, 180)
            map_lat_default = (-90, 90)
            lon_slider_min, lon_slider_max = (0, 360) if lon_center_3d == 180 else (-180, 180)
            lat_slider_min, lat_slider_max = -90, 90

        region_preset_4d = st.selectbox(
            "Map-depth region preset (Fig.3-Fig.6)",
            ["Dataset default"] + list(envgeo_utils.MAP_REGION_PRESETS),
            help=(
                "Set the longitude and latitude range for Fig.3-Fig.6 map-depth views. "
                "You can still fine-tune the range with the sliders below."
            ),
        )

        if region_preset_4d != "Dataset default":
            map_lon_default, map_lat_default = region_bounds_for_4d(
                envgeo_utils.MAP_REGION_PRESETS[region_preset_4d]["bounds"],
                lon_slider_min,
                lon_slider_max,
                lat_slider_min,
                lat_slider_max,
                lon_center_3d,
            )

        figure_scale_state_key = (
            f"figure_scale::{ref_data}::{lon_center_3d}::{region_preset_4d}"
        )
        map_lon_raw = st.slider(
            "Map longitude range",
            lon_slider_min,
            lon_slider_max,
            map_lon_default,
            step=1,
            key=f"{figure_scale_state_key}::longitude",
        )
        map_lat_raw = st.slider(
            "Map latitude range",
            lat_slider_min,
            lat_slider_max,
            map_lat_default,
            step=1,
            key=f"{figure_scale_state_key}::latitude",
        )

        depth_slider_max = int(sld_depth_max + 100)
        fig_depth_min, fig_depth_max = st.slider(
            label="Depth range for 3D view",
            min_value=0,
            max_value=depth_slider_max,
            value=(0, int(sld_depth_max)),
            step=50,
            key=f"{figure_scale_state_key}::depth::{depth_slider_max}",
        )

        marker_size = st.slider(
            "Marker size",
            1,
            10,
            3,
            key=f"{figure_scale_state_key}::marker_size",
        )

        map_x_range = [map_lon_raw[0], map_lon_raw[1]]
        map_y_range = [map_lat_raw[0], map_lat_raw[1]]
        map_aspectratio = get_geo_aspectratio(
            map_x_range,
            map_y_range,
            z_ratio=0.5 if ref_data == data_source_GLOBAL else 1.0
        )
    # -------------------------------------------------------------------------
    # Cache control / キャッシュ制御
    # -------------------------------------------------------------------------

    if st.sidebar.button("🔄 Clear cache"):
        envgeo_utils.clear_app_cache()
        st.rerun()

    # =============================================================================
    # Standard 4D figure definitions / 標準4D図の定義
    # =============================================================================

    # -------------------------------------------------------------------------
    # Fig.1: Salinity–δ18O–depth–temperature / 塩分–δ18O–深度–水温
    # -------------------------------------------------------------------------
    # Keep rows that contain all variables required by this figure.
    # この図に必要な全変数を持つ行だけを描画対象にする。
    original_len_df1 = len(df1)
    df_fig1 = df1.dropna(subset=['Temperature_degC','d18O', 'Depth_m', 'Salinity'])
    removed_num_fig1 = original_len_df1 - len(df_fig1)
    plotted_num_fig1 = original_len_df1 - removed_num_fig1

    fig1=px.scatter_3d(df_fig1, x='Salinity', y='d18O', z='Depth_m',
                    color='Temperature_degC', 
                    width=700,
                    height=600,
                    color_continuous_scale=map_colorscale,
                    hover_data={
                        "lat": True,  
                        "lon": True,  
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
                    }
                )
                

    # Marker style / マーカー表示
    fig1.update_traces(
        mode='markers',
        marker = dict(size = 3),
    )
    

    
    # Figure layout / 図のレイアウト
    fig1.update_layout(
        scene=dict(
            # Axis labels / 軸ラベル
            xaxis_title='Salinity',
            yaxis_title='d18O',
            zaxis_title='Water Depth',
            
            # Reverse depth so deeper samples are lower in the view.
            # 深い試料が図の下側になるよう深度軸を反転する。
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min], 
                autorange=False
            ),
            
            # Aspect ratio / 縦横比
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            
            # Initial camera angle / 初期カメラ角度
            camera=dict(
                eye=dict(x=-0.6, y=-1.1, z=1.9),
                center=dict(x=0, y=0, z=-0.1)
            )
        ),
        margin=dict(r=20, l=10, b=10, t=10)
    )
    
    
    fig1.update_traces(marker=dict(size=marker_size))
    

    

    # -------------------------------------------------------------------------
    # Fig.2: Salinity–temperature–depth–δ18O / 塩分–水温–深度–δ18O
    # -------------------------------------------------------------------------
    # Keep rows that contain all variables required by this figure.
    # この図に必要な全変数を持つ行だけを描画対象にする。
    original_len_df1 = len(df1)
    df_fig2 = df1.dropna(subset=['Temperature_degC','d18O', 'Depth_m', 'Salinity'])
    removed_num_fig2 = original_len_df1 - len(df_fig2)
    plotted_num_fig2 = original_len_df1 - removed_num_fig2

    fig2=px.scatter_3d(df_fig2, x='Salinity', y='Temperature_degC', z='Depth_m',
                    color='d18O', 
                    width=700,
                    height=600,
                    color_continuous_scale=map_colorscale,
                    hover_data={
                        "lat": True,  
                        "lon": True,  
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
                    }
                )
                

    # Marker style / マーカー表示
    fig2.update_traces(
        mode='markers',
        marker = dict(size = 3),
    )
    

    # Figure layout / 図のレイアウト
    fig2.update_layout(
        scene=dict(
            # Axis labels / 軸ラベル
            xaxis_title='Salinity',
            yaxis_title='Temperature (C)',
            zaxis_title='Water Depth',
            
            # Reverse depth so deeper samples are lower in the view.
            # 深い試料が図の下側になるよう深度軸を反転する。
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            
            # Aspect ratio and camera / 縦横比とカメラ
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=-0.6, y=-1.1, z=1.9),
                center=dict(x=0, y=0, z=-0.1)
            )
        ),
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig2.update_traces(marker=dict(size=marker_size))
    

    # -------------------------------------------------------------------------
    # Fig.3: Map–depth–δ18O / 地図–深度–δ18O
    # -------------------------------------------------------------------------
    

    original_len_df1 = len(df1)
    df_fig3 = df1.dropna(subset=['d18O', 'Depth_m']).copy()
    # Use center-adjusted longitude only for plotting; keep original Longitude_degE for hover/readout.
    # 描画用だけ中心に合わせた経度を使い、表示値は元の Longitude_degE を保つ。
    df_fig3['lon_plot'] = normalize_lon_to_center(df_fig3['lon'], lon_center_3d)

    removed_num_fig3 = original_len_df1 - len(df_fig3)
    plotted_num_fig3 = original_len_df1 - removed_num_fig3

    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    # Coastline segments also need center-adjusted longitude with explicit breaks at wrap boundaries.
    # 海岸線も中心に合わせた経度へ変換し、折り返し境界では明示的に線を切る。
    coastline_x_plot, coastline_y_plot = wrap_coastline_with_breaks(coastline_x, coastline_y, lon_center_3d)
    
    fig3=px.scatter_3d(df_fig3, x='lon_plot', y='lat', z='Depth_m',
                    color='d18O', 
                    width=700,
                    height=600,
                    color_continuous_scale=map_colorscale,
                    hover_data={
                        "lat": True,  
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
                    }
                )
   

    
    # Marker style / マーカー表示
    fig3.update_traces(
        mode='markers',
        marker = dict(size = 3),
        name='d18O'
        )
    
    fig3.update_traces(marker=dict(size=marker_size))
    
    # Figure layout / 図のレイアウト
    fig3.update_layout(
        scene=dict(
            # Reverse depth so deeper samples are lower in the view.
            # 深い試料が図の下側になるよう深度軸を反転する。
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            )
        )
    
  
    
    # Apply shared map-depth layout / 共通の地図・深度レイアウトを適用する。
    x_range = map_x_range
    y_range = map_y_range
    
    fig3 = envgeo_utils.apply_common_layout(
        fig3, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    fig3.update_layout(scene=dict(aspectmode='manual', aspectratio=map_aspectratio))
    
    
    # Coastline guides / 海岸線の補助表示
    fig3.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_min] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none'
        ))
    # Place the gray coastline on the figure bottom instead of the dataset deepest sample.
    # 灰色の海岸線はデータ最深点ではなく、図の底面に合わせて描画する。
    fig3.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_max] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        name='coastline', line=dict(color='gray', width=0.5),
        hoverinfo='none'
        ))
    
    
    
    # -------------------------------------------------------------------------
    # Fig.4: Map–depth–temperature / 地図–深度–水温
    # -------------------------------------------------------------------------
        
    original_len_df1 = len(df1)
    df_fig4 = df1.dropna(subset=['Temperature_degC', 'Depth_m']).copy()
    # Use center-adjusted longitude only for plotting; keep original Longitude_degE for hover/readout.
    # 描画用だけ中心に合わせた経度を使い、表示値は元の Longitude_degE を保つ。
    df_fig4['lon_plot'] = normalize_lon_to_center(df_fig4['lon'], lon_center_3d)
    
    removed_num_fig4 = original_len_df1 - len(df_fig4)
    plotted_num_fig4 = original_len_df1 - removed_num_fig4

    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    # Coastline segments also need center-adjusted longitude with explicit breaks at wrap boundaries.
    # 海岸線も中心に合わせた経度へ変換し、折り返し境界では明示的に線を切る。
    coastline_x_plot, coastline_y_plot = wrap_coastline_with_breaks(coastline_x, coastline_y, lon_center_3d)

    fig4=px.scatter_3d(df_fig4, x='lon_plot', y='lat', z='Depth_m',
                    color='Temperature_degC', 
                    width=700,
                    height=600,
                    color_continuous_scale=map_colorscale,
                    hover_data={
                        "lat": True,  
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
                    }
                )
    
    
    
    # Marker style / マーカー表示
    fig4.update_traces(
        mode='markers',
        marker = dict(size = 3),
        name='Temperature'
        )
        
    fig4.update_traces(marker=dict(size=marker_size))
    
    
        
    # Figure layout / 図のレイアウト
    fig4.update_layout(
        scene=dict(
            # Reverse depth so deeper samples are lower in the view.
            # 深い試料が図の下側になるよう深度軸を反転する。
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            )
        )

    

    # Apply shared map-depth layout / 共通の地図・深度レイアウトを適用する。
    x_range = map_x_range
    y_range = map_y_range
    
    fig4 = envgeo_utils.apply_common_layout(
        fig4, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    fig4.update_layout(scene=dict(aspectmode='manual', aspectratio=map_aspectratio))
    
    
    
    # Coastline guides / 海岸線の補助表示
    fig4.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_min] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none'
        ))
    
    # Place the gray coastline on the figure bottom instead of the dataset deepest sample.
    # 灰色の海岸線はデータ最深点ではなく、図の底面に合わせて描画する。
    fig4.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_max] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        name='coastline', line=dict(color='gray', width=0.5),
        hoverinfo='none'
        ))
    
    
    
    # -------------------------------------------------------------------------
    # Fig.5: Map–depth–salinity / 地図–深度–塩分
    # -------------------------------------------------------------------------

        
    original_len_df1 = len(df1)
    df_fig5 = df1.dropna(subset=['Salinity', 'Depth_m']).copy()
    # Use center-adjusted longitude only for plotting; keep original Longitude_degE for hover/readout.
    # 描画用だけ中心に合わせた経度を使い、表示値は元の Longitude_degE を保つ。
    df_fig5['lon_plot'] = normalize_lon_to_center(df_fig5['lon'], lon_center_3d)

    removed_num_fig5 = original_len_df1 - len(df_fig5)
    plotted_num_fig5 = original_len_df1 - removed_num_fig5

     
    
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    # Coastline segments also need center-adjusted longitude with explicit breaks at wrap boundaries.
    # 海岸線も中心に合わせた経度へ変換し、折り返し境界では明示的に線を切る。
    coastline_x_plot, coastline_y_plot = wrap_coastline_with_breaks(coastline_x, coastline_y, lon_center_3d)
    
    fig5=px.scatter_3d(df_fig5, x='lon_plot', y='lat', z='Depth_m',
                    color='Salinity', 
                    width=700,
                    height=600,
                    color_continuous_scale=map_colorscale,
                    
                    hover_data={
                        "lat": True,  # 名前を表示
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
                    }
                )
    

    
    # Marker style / マーカー表示
    fig5.update_traces(
        mode='markers',
        marker = dict(size = 3),
        name='Salinity'
        )
        
    fig5.update_traces(marker=dict(size=marker_size))

        
    # Figure layout / 図のレイアウト
    fig5.update_layout(
        scene=dict(
            # Reverse depth so deeper samples are lower in the view.
            # 深い試料が図の下側になるよう深度軸を反転する。
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            )
        )

    
    
    # Apply shared map-depth layout / 共通の地図・深度レイアウトを適用する。
    x_range = map_x_range
    y_range = map_y_range
    
    fig5 = envgeo_utils.apply_common_layout(
        fig5, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    fig5.update_layout(scene=dict(aspectmode='manual', aspectratio=map_aspectratio))
    
    
    
    # Coastline guides / 海岸線の補助表示
    fig5.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_min] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none'
        ))
    
    # Place the gray coastline on the figure bottom instead of the dataset deepest sample.
    # 灰色の海岸線はデータ最深点ではなく、図の底面に合わせて描画する。
    fig5.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_max] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        name='coastline', line=dict(color='gray', width=0.5),
        hoverinfo='none'
        ))
    
    
    
    
    # -------------------------------------------------------------------------
    # Fig.6: Map–depth–d-excess / 地図–深度–d-excess
    # -------------------------------------------------------------------------
    

    original_len_df1 = len(df1)
    
    df_fig6 = df1.dropna(subset=['d-excess','Depth_m']).copy()
    # Use center-adjusted longitude only for plotting; keep original Longitude_degE for hover/readout.
    # 描画用だけ中心に合わせた経度を使い、表示値は元の Longitude_degE を保つ。
    df_fig6['lon_plot'] = normalize_lon_to_center(df_fig6['lon'], lon_center_3d)

    removed_num_fig6 = original_len_df1 - len(df_fig6)
    plotted_num_fig6 = original_len_df1 - removed_num_fig6

        
    
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    # Coastline segments also need center-adjusted longitude with explicit breaks at wrap boundaries.
    # 海岸線も中心に合わせた経度へ変換し、折り返し境界では明示的に線を切る。
    coastline_x_plot, coastline_y_plot = wrap_coastline_with_breaks(coastline_x, coastline_y, lon_center_3d)

    fig6=px.scatter_3d(df_fig6, x='lon_plot', y='lat', z='Depth_m',
                    color='d-excess', 
                    width=700,
                    height=600,
                    color_continuous_scale=map_colorscale,
                    hover_data={
                        "lat": True,  
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
                        "d-excess": True, 
                    }
                )
    
   
    
    # Marker style / マーカー表示
    fig6.update_traces(
        mode='markers',
        marker = dict(size = 3),
        name='d-excess'
        )
        
    fig6.update_traces(marker=dict(size=marker_size))
    
    
        
    # Figure layout / 図のレイアウト
    fig6.update_layout(
        scene=dict(
            # Reverse depth so deeper samples are lower in the view.
            # 深い試料が図の下側になるよう深度軸を反転する。
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            )
        )
    
    
    
    # Apply shared map-depth layout / 共通の地図・深度レイアウトを適用する。
    x_range = map_x_range
    y_range = map_y_range
    
    fig6 = envgeo_utils.apply_common_layout(
        fig6, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    fig6.update_layout(scene=dict(aspectmode='manual', aspectratio=map_aspectratio))
    
    
    # Coastline guides / 海岸線の補助表示
    fig6.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_min] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none'
        ))
    # Place the gray coastline on the figure bottom instead of the dataset deepest sample.
    # 灰色の海岸線はデータ最深点ではなく、図の底面に合わせて描画する。
    fig6.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_max] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        name='coastline', line=dict(color='gray', width=0.5),
        hoverinfo='none'
        ))
    
    

    
    # =============================================================================
    # Figure selection and custom 4D view / 図の選択とカスタム4D表示
    # =============================================================================
    options = [
        "Salinity-δ18O-depth-temperature (Fig.1)", 
        "Salinity-temperature-depth-δ18O (Fig.2)", 
        "Map-depth-δ18O (Fig.3)", 
        "Map-depth-temperature (Fig.4)", 
        "Map-depth-salinity (Fig.5)", 
        "Map-depth-d-excess (Fig.6)",
        "Custom 4D view beta",
    ]
    custom_option = options[-1]
    custom_salinity_d18o_mode = "Salinity-δ18O with custom Z and color"
    custom_ts_mode = "Temperature-salinity with custom Z and color"
    custom_map_depth_mode = "Map-depth with custom color"
    custom_full_mode = "Full custom X-Y-Z-color"

    st.markdown(
        """
        <style>
        div[role="radiogroup"][aria-label="4D view"] {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem 1.6rem;
        }
        div[role="radiogroup"][aria-label="4D view"] > label {
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # View selector / 表示選択
    display_option = st.radio(
        "4D view",
        options,
        help="Choose one of the standard 4D views or the custom 4D view.",
    )

    # Resolve selected figure and data / 選択した図とデータを決定する。
    custom_mode = None
    custom_x = custom_y = custom_z = custom_color = None

    if display_option == custom_option:
        preferred_numeric_columns = [
            "Longitude_degE",
            "Latitude_degN",
            "Depth_m",
            "Salinity",
            "Temperature_degC",
            "d18O",
            "dD",
            "d-excess",
            "Year",
            "Month",
        ]
        numeric_options = available_numeric_columns(df1, preferred_numeric_columns)
        if len(numeric_options) < 4:
            st.warning("Custom 4D view requires at least four numeric columns.")
            return

        custom_mode = st.radio(
            "Custom view type",
            [
                custom_salinity_d18o_mode,
                custom_ts_mode,
                custom_map_depth_mode,
                custom_full_mode,
            ],
            horizontal=True,
            help="Choose the base layout for the custom 4D plot.",
        )

        custom_is_map_template = custom_mode == custom_map_depth_mode

        if custom_mode == custom_salinity_d18o_mode:
            custom_x = "Salinity"
            custom_y = "d18O"
            custom_cols = st.columns(2)
            with custom_cols[0]:
                custom_z = st.selectbox(
                    "Z axis",
                    numeric_options,
                    index=safe_option_index(numeric_options, "Depth_m"),
                    key="custom_4d_sal_d18o_z",
                )
            with custom_cols[1]:
                custom_color = st.selectbox(
                    "Color parameter",
                    numeric_options,
                    index=safe_option_index(numeric_options, "Temperature_degC"),
                    key="custom_4d_sal_d18o_color",
                    help=getattr(envgeo_utils, "COLOR_PARAMETER_HELP_TEXT", "Choose the variable used to color the plotted points or map markers."),
                )
        elif custom_mode == custom_ts_mode:
            custom_x = "Salinity"
            custom_y = "Temperature_degC"
            custom_cols = st.columns(2)
            with custom_cols[0]:
                custom_z = st.selectbox(
                    "Z axis",
                    numeric_options,
                    index=safe_option_index(numeric_options, "Depth_m"),
                    key="custom_4d_ts_z",
                )
            with custom_cols[1]:
                custom_color = st.selectbox(
                    "Color parameter",
                    numeric_options,
                    index=safe_option_index(numeric_options, "d18O"),
                    key="custom_4d_ts_color",
                    help=getattr(envgeo_utils, "COLOR_PARAMETER_HELP_TEXT", "Choose the variable used to color the plotted points or map markers."),
                )
        elif custom_mode == custom_map_depth_mode:
            custom_x = "Longitude_degE"
            custom_y = "Latitude_degN"
            custom_z = "Depth_m"
            custom_color = st.selectbox(
                "Color parameter",
                numeric_options,
                index=safe_option_index(numeric_options, "d18O"),
                key="custom_4d_map_color",
                help=getattr(envgeo_utils, "COLOR_PARAMETER_HELP_TEXT", "Choose the variable used to color the plotted points or map markers."),
            )
        else:
            custom_cols_top = st.columns(2)
            with custom_cols_top[0]:
                custom_x = st.selectbox(
                    "X axis",
                    numeric_options,
                    index=safe_option_index(numeric_options, "d18O"),
                    key="custom_4d_full_x",
                )
            with custom_cols_top[1]:
                custom_y = st.selectbox(
                    "Y axis",
                    numeric_options,
                    index=safe_option_index(numeric_options, "dD"),
                    key="custom_4d_full_y",
                )
            custom_cols_bottom = st.columns(2)
            with custom_cols_bottom[0]:
                custom_z = st.selectbox(
                    "Z axis",
                    numeric_options,
                    index=safe_option_index(numeric_options, "Depth_m"),
                    key="custom_4d_full_z",
                )
            with custom_cols_bottom[1]:
                custom_color = st.selectbox(
                    "Color parameter",
                    numeric_options,
                    index=safe_option_index(numeric_options, "Temperature_degC"),
                    key="custom_4d_full_color",
                    help=getattr(envgeo_utils, "COLOR_PARAMETER_HELP_TEXT", "Choose the variable used to color the plotted points or map markers."),
                )

        if custom_mode == custom_full_mode and len({custom_x, custom_y, custom_z}) < 3:
            st.warning("Full custom 4D view requires different X, Y, and Z axes.")
            return

        custom_required_columns = list(dict.fromkeys([custom_x, custom_y, custom_z, custom_color]))
        df_custom = df1.dropna(subset=custom_required_columns).copy()
        removed_num_custom = len(df1) - len(df_custom)
        plotted_num_custom = len(df_custom)
        if removed_num_custom > 0:
            st.caption(
                f":red[{plotted_num_custom} samples plotted; "
                f"{removed_num_custom} rows excluded because selected variables were incomplete.]"
            )

        if df_custom.empty:
            st.warning("No valid rows remain for the selected custom variables.")
            return

        plot_x = custom_x
        plot_y = custom_y
        if custom_is_map_template:
            # Use the same map-centered longitude frame as Fig.3-Fig.6.
            # Fig.3-Fig.6 と同じ地図中心の経度系と海岸線を使う。
            df_custom["lon_plot"] = normalize_lon_to_center(df_custom["lon"], lon_center_3d)
            plot_x = "lon_plot"
            plot_y = "lat"

        target_fig = px.scatter_3d(
            df_custom,
            x=plot_x,
            y=plot_y,
            z=custom_z,
            color=custom_color,
            width=700,
            height=600,
            color_continuous_scale=map_colorscale,
            hover_data={
                column: True for column in [
                    "Longitude_degE",
                    "Latitude_degN",
                    "Depth_m",
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
                    "reference",
                ] if column in df_custom.columns
            },
        )
        target_fig.update_traces(mode="markers", marker=dict(size=marker_size))
        z_axis_settings = {}
        if custom_z == "Depth_m":
            z_axis_settings = dict(range=[fig_depth_max, fig_depth_min], autorange=False)
        if custom_is_map_template:
            target_fig.update_layout(scene=dict(zaxis=z_axis_settings))
            target_fig = envgeo_utils.apply_common_layout(
                target_fig,
                ref_data,
                fig_depth_max,
                fig_depth_min,
                x_range=map_x_range,
                y_range=map_y_range,
            )
            target_fig.update_layout(scene=dict(aspectmode="manual", aspectratio=map_aspectratio))

            custom_coastline_x, custom_coastline_y = envgeo_utils.load_coastline_data(ref_data)
            custom_coastline_x_plot, custom_coastline_y_plot = wrap_coastline_with_breaks(
                custom_coastline_x,
                custom_coastline_y,
                lon_center_3d,
            )
            target_fig.add_traces(
                go.Scatter3d(
                    x=custom_coastline_x_plot,
                    y=custom_coastline_y_plot,
                    z=[fig_depth_min] * len(custom_coastline_x_plot),
                    mode="lines",
                    name="coastline",
                    line=dict(color="blue", width=0.8),
                    hoverinfo="none",
                )
            )
            target_fig.add_traces(
                go.Scatter3d(
                    x=custom_coastline_x_plot,
                    y=custom_coastline_y_plot,
                    z=[fig_depth_max] * len(custom_coastline_x_plot),
                    mode="lines",
                    name="coastline",
                    line=dict(color="gray", width=0.5),
                    hoverinfo="none",
                )
            )
        else:
            target_fig.update_layout(
                scene=dict(
                    xaxis_title=label_for_column(custom_x),
                    yaxis_title=label_for_column(custom_y),
                    zaxis_title=label_for_column(custom_z),
                    zaxis=z_axis_settings,
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=1),
                    camera=dict(
                        eye=dict(x=-0.6, y=-1.1, z=1.9),
                        center=dict(x=0, y=0, z=-0.1),
                    ),
                ),
                margin=dict(r=20, l=10, b=10, t=10),
            )
        plot_key = "p_custom_4d"
        df_map = df_custom
    elif display_option == options[0]:
        target_fig = fig1
        plot_key = "p1"
        df_map = df_fig1
        if removed_num_fig1 > 0:
            st.caption(f":red[{plotted_num_fig1} samples plotted; {removed_num_fig1} rows excluded because required variables were incomplete.]")
    elif display_option == options[1]:
        target_fig = fig2
        plot_key = "p2"
        df_map = df_fig2
        if removed_num_fig2 > 0:
            st.caption(f":red[{plotted_num_fig2} samples plotted; {removed_num_fig2} rows excluded because required variables were incomplete.]")
    elif display_option == options[2]:
        target_fig = fig3
        plot_key = "p3"
        df_map = df_fig3
        if removed_num_fig3 > 0:
            st.caption(f":red[{plotted_num_fig3} samples plotted; {removed_num_fig3} rows excluded because required variables were incomplete.]")
    elif display_option == options[3]:
        target_fig = fig4
        plot_key = "p4"
        df_map = df_fig4
        if removed_num_fig4 > 0:
            st.caption(f":red[{plotted_num_fig4} samples plotted; {removed_num_fig4} rows excluded because required variables were incomplete.]")
    elif display_option == options[4]:
        target_fig = fig5
        plot_key = "p5"
        df_map = df_fig5
        if removed_num_fig5 > 0:
            st.caption(f":red[{plotted_num_fig5} samples plotted; {removed_num_fig5} rows excluded because required variables were incomplete.]")
    else:
        target_fig = fig6
        plot_key = "p6"
        df_map = df_fig6
        if removed_num_fig6 > 0:
            st.caption(f":red[{plotted_num_fig6} samples plotted; {removed_num_fig6} rows excluded because d-excess could not be calculated.]")

    
    st.write("---")
    

    st.subheader(display_option)

    # =============================================================================
    # Colorbar controls and figure output / カラーバー操作と図の出力
    # =============================================================================
    # Standard figures use these short color-variable labels.
    # 標準図には以下の短い色変数ラベルを用いる。
    c_int = ["Temperature_degC", "d18O", "d18O", "Temperature_degC", "Salinity", "d-excess"]
    c_lbl = ["Temperature(C)", "d18O", "d18O", "Temperature(C)", "Salinity", "d-excess"]

    # Dataset-specific default color ranges / データセット別の初期色範囲
    if ref_data == data_source_ENVGEO:
        default_ranges = {
            "Temperature_degC": (5.0, 28.0),
            "d18O": (-1.5, 1.0),
            "Salinity": (33.5, 35.0),
            "d-excess": (-2.0, 2.0)
        }
    else:
        default_ranges = {
            "Temperature_degC": (-2.0, 30.0),
            "d18O": (-5.0, 2.0),
            "Salinity": (30.0, 37.0),
            "d-excess": (-5.0, 20.0)
        }

    # Resolve the active color variable / 現在の色変数を決定する。
    if display_option == custom_option:
        t_col = custom_color
        t_lbl = label_for_column(custom_color)
    else:
        idx = options.index(display_option) if display_option in options else 0
        t_col = c_int[idx]
        t_lbl = c_lbl[idx]

    # Active colorbar range / 現在のカラーバー範囲
    v_min_actual = float(df_map[t_col].min())
    v_max_actual = float(df_map[t_col].max())

    # Fall back to data limits when no default range is defined.
    # 初期範囲が未定義ならデータの最小・最大値を用いる。
    d_range = default_ranges.get(t_col, (v_min_actual, v_max_actual))

    r_3d = st.slider(
        f"Colorbar range: {t_lbl}",
        min_value=float(math.floor(v_min_actual * 10) / 10),
        max_value=float(math.ceil(v_max_actual * 10) / 10),
        value=d_range,
        step=0.1,
        key=f"c3_slider_{envgeo_utils.safe_filename_text(t_col)}_{envgeo_utils.safe_filename_text(display_option)}"
    )

    # Apply colorbar settings to standard figures / 標準図へカラーバー設定を適用する。
    for f_idx, f_name in enumerate(['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6']):
        if f_name in locals() and locals()[f_name] is not None:
            f = locals()[f_name]
            
            # Color variable for this standard figure / この標準図の色変数
            current_fig_col = c_int[f_idx]
            current_fig_lbl = c_lbl[f_idx]

            # Update colorbar layout / カラーバー配置を更新する。
            f.update_layout(
                coloraxis_colorbar=dict(
                    title=current_fig_lbl,
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    x=0.5,
                    xanchor="center",
                    thickness=15
                ),
                margin=dict(b=100)
            )

            # Apply the active range to figures using the same color variable.
            # 同じ色変数を用いる図へ現在の範囲を適用する。
            if current_fig_col == t_col:
                f.update_coloraxes(cmin=r_3d[0], cmax=r_3d[1])
            else:
                # Keep defaults for other color variables / 他の色変数は初期範囲を使う。
                c_min, c_max = default_ranges.get(current_fig_col, (None, None))
                if c_min is not None:
                    f.update_coloraxes(cmin=c_min, cmax=c_max)

    if display_option == custom_option:
        target_fig.update_layout(
            coloraxis_colorbar=dict(
                title=t_lbl,
                orientation="h",
                yanchor="top",
                y=-0.15,
                x=0.5,
                xanchor="center",
                thickness=15,
            ),
            margin=dict(b=100),
        )
        target_fig.update_coloraxes(cmin=r_3d[0], cmax=r_3d[1])

    # Render selected 3D/4D figure / 選択した3D/4D図を描画する。
    st.caption(
        "3D camera controls: drag to rotate. Use the Plotly toolbar at the upper "
        "right to switch rotation, pan, zoom, or reset the camera. Modifier keys "
        "(Shift, Control, or Command) can change drag behavior; details vary by "
        "browser and operating system."
    )
    st.plotly_chart(
        target_fig,
        key=plot_key,
        config={'scrollZoom': True}
    )
    st.download_button(
        "Download interactive HTML",
        envgeo_utils.figure_to_self_contained_html(target_fig),
        envgeo_utils.build_figure_filename("p04_3d4d", extension="html"),
        "text/html",
        key=f"{plot_key}_html_dl",
    )

    
    

    # =============================================================================
    # Linked sampling-location map / 連動する採水地点地図
    # =============================================================================
    st.divider()
    st.subheader("Sampling Location Map")
    st.caption("Map detail settings below apply only to this sampling-location map.")

    # Keep map controls compact so the map remains visible after Streamlit reruns.
    # Streamlitの再実行後も地図が見つけやすいよう、地図設定をポップオーバーに集約する。
    with st.popover(
        "Map detail settings", **envgeo_utils.stretch_width_kwargs(st.popover)
    ):
        map_mode = st.radio(
            "Map style",
            envgeo_utils.MAP_MODE_OPTIONS,
            index=envgeo_utils.MAP_MODE_DEFAULT_INDEX,
            horizontal=True,
            key="map_style_04_auto",
            help=getattr(envgeo_utils, "MAP_STYLE_HELP_TEXT", "Choose the background map style for the sampling-location map."),
        )
    st.caption(f"Map style: {map_mode}")

    # Map extent and zoom / 地図範囲とズーム
    lat_min, lat_max = df_map["Latitude_degN"].min(), df_map["Latitude_degN"].max()
    lon_min, lon_max = df_map["Longitude_degE"].min(), df_map["Longitude_degE"].max()

    # Fallback location / データがない場合の既定位置
    default_lat, default_lon, default_zoom = 36.0, 138.0, 4.0

    if pd.isna(lat_min) or pd.isna(lon_min):
        center_lat, center_lon, auto_zoom = default_lat, default_lon, default_zoom
    else:
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        lat_diff = max(lat_max - lat_min, 0.1)
        lon_diff = max(lon_max - lon_min, 0.1)
        
        # Calculate zoom from data extent / データ範囲からズームを計算する。
        map_width_px, map_height_px = 1200, 700
        zoom_lon = math.log2((map_width_px * 360) / (lon_diff * 256))
        zoom_lat = math.log2((map_height_px * 180) / (lat_diff * 256))
        
        # Leave margin around the selected extent / 選択範囲の周囲に余白を残す。
        auto_zoom = min(zoom_lon, zoom_lat) - 2.0
        auto_zoom = max(1, min(15, auto_zoom))

        # Use a broad fallback view for globally distributed points.
        # 全球規模に分布する点では広域の既定表示を用いる。
        if lon_diff > 100:
              center_lat, center_lon, auto_zoom = default_lat, default_lon, 1.5
    
    
    

    
    # Map color settings / 地図の色設定

    m_cols = ["Temperature_degC", "d18O", "d18O", "Temperature_degC", "Salinity", "d-excess"]
    m_lbls = ["Temperature(C)", "d18O", "d18O", "Temperature(C)", "Salinity", "d-excess"]
    
    # Resolve the active map color variable / 地図の色変数を決定する。
    if display_option == custom_option:
        m_target = custom_color
        m_label = label_for_column(custom_color)
    else:
        m_idx = options.index(display_option) if display_option in options else 0
        m_target = m_cols[m_idx]
        m_label = m_lbls[m_idx]
    
    # Map colorbar range / 地図カラーバー範囲
    mv1, mv2 = float(df_map[m_target].min()), float(df_map[m_target].max())
    r_map = st.slider(
        f"Map colorbar range: {m_label}", 
        float(math.floor(mv1*10)/10), float(math.ceil(mv2*10)/10), (mv1, mv2), 
        0.1, key=f"slider_map_{envgeo_utils.safe_filename_text(m_target)}_{envgeo_utils.safe_filename_text(display_option)}"
    )
    
    # Build the sampling-location map / 採水地点地図を作成する。
    # Reuse the sidebar colormap choice for the final 2D location map as well.
    # 最後の 2D 地図でも、サイドバーで選んだカラーマップを共通利用する。
    c_scale_map = map_colorscale
    fig_map = px.scatter_mapbox(
        df_map, lat="Latitude_degN", lon="Longitude_degE",
        color=m_target,
        color_continuous_scale=c_scale_map,
        hover_data={
            "lat": True,  
            "lon": True,  
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
            "d-excess": True,
        },
        opacity=0.6, height=500
    )
    
    

    # Apply the selected background style / 選択した背景スタイルを適用する。
    fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)
    
    

    
    # Map layout / 地図レイアウト
    fig_map.update_layout(
        coloraxis_colorbar=dict(
            title=m_label,
            orientation="h",
            yanchor="top", y=-0.15,
            x=0.5, xanchor="center",
            thickness=15
        ),
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
        margin=dict(l=0, r=0, t=0, b=100),
        autosize=True
    )
    
    # Apply the selected colorbar range / 選択したカラーバー範囲を適用する。
    fig_map.update_coloraxes(cmin=r_map[0], cmax=r_map[1])

    # Use a unique widget key and enable wheel zoom.
    # 一意のwidget keyを用い、マウスホイールズームを有効にする。

    st.plotly_chart(
        fig_map,
        key="dynamic_map_final",
        config={'scrollZoom': True, 'displayModeBar': True}
    )
    st.download_button(
        "Download interactive HTML",
        envgeo_utils.figure_to_self_contained_html(fig_map),
        envgeo_utils.build_figure_filename("p04_map", extension="html"),
        "text/html",
        key="p04_map_html_dl",
    )

    # -------------------------------------------------------------------------
    # Current-view data table / 現在の表示対象データ表
    # -------------------------------------------------------------------------

    with st.expander("Filtered dataset for current view (CSV)", expanded=False):
        
        table_columns = [
            'reference', 'Cruise', 'Station', 'Date', 'Year', 'Month',
            'Longitude_degE', 'Latitude_degN', 'Depth_m',
            'Temperature_degC', 'Salinity', 'd18O', 'dD',
            envgeo_utils.QUALITY_FLAG_COLUMN,
            envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN,
        ]

        # Add d-excess when available / 使用可能な場合だけd-excessを追加する。
        if 'd-excess' in df_map.columns:
            table_columns.append('d-excess')

        available_columns = [col for col in table_columns if col in df_map.columns]
        df1_table = df_map[available_columns].copy()

      
        # Convert mixed columns to strings for Streamlit-table compatibility.
        # Streamlit表での混在型互換性のため、列を文字列へ変換する。
        df1_table = df1_table.astype(str) 
        
        st.dataframe(df1_table)
        
    
    

if __name__ == '__main__':
    main()
    
