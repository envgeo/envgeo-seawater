
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:00:21 2023

@author: Toyoho Ishimura @Kyoto-U

2026/04/29 update 
"""

# --- Version info ---
version = "1.3.0" #v220_20260429　mapセンター調整済

# ToDo
# 最後のマップのカラーバーの初期値を調整必要
# 最後に，各図を定義して，選ばれたときに個別に実行すれば軽くなるはず


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import envgeo_utils    
pd.set_option('future.no_silent_downcasting', True)


def main():
    
    st.header(f'4D Visualizer ({version})')

    st.button('Reload')

    def label_for_column(column):
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
        return [
            column for column in preferred_columns
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
        ]

    def safe_option_index(options_list, preferred):
        return options_list.index(preferred) if preferred in options_list else 0


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
    df1 = envgeo_utils.load_isotope_data(ref_data)
   
    if df1.empty:
        st.warning("No data available for the selected conditions.")
        return

    ##############################################################################
    # d-excessを計算
    ##############################################################################
    df1 = envgeo_utils.add_d_excess(df1)


    ##############################################################################
    # サイドバーここから　　df1フィルタリング　も一括で
    ##############################################################################

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
    # 後半の定義用
    ##############################################################################
        
    df1['lat'] = df1['Latitude_degN']
    df1['lon'] = df1['Longitude_degE']
    # df1['Depth_m'] = df1['Depth_m']*(-1)

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


    ##############################################################################
    # 図のスケール変更
    ##############################################################################
   

   # サイドバーの中にコンテナを作成し、境界線（border）を有効にする
    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "MAP_DISPLAY_SETTINGS_LABEL", "Map display settings"))

        # Match the map-center selector used in the 2D mapping page.
        # 2D マップページと同じ地図中心の切り替え UI を使う。
        center_option_3d = st.radio(
            "Map Center: (Fig.3-Fig.6)",
            ("Atlantic (0°)", "Pacific (180°)"),
            horizontal=True
        )
        lon_center_3d = 0 if "Atlantic" in center_option_3d else 180

        # Offer shared colormaps, including cmocean options for oceanographic data.
        # 海洋データ向けのcmocean候補を含む共通カラーマップを使う。
        colormap_options = envgeo_utils.get_plotly_colormap_options("d18O")
        selected_colormap_label = st.selectbox(
            "Colormap for map figures",
            list(colormap_options.keys()),
            index=list(colormap_options.keys()).index(
                envgeo_utils.recommended_plotly_colormap_label("d18O")
            )
        )
        map_colorscale = colormap_options[selected_colormap_label]

        st.subheader(getattr(envgeo_utils, "FIGURE_SCALE_SETTINGS_LABEL", "Figure scale settings"))

        # Allow Fig3-Fig6 map views to use explicit lon/lat windows from the sidebar.
        # Fig3-Fig6 の地図表示範囲を、サイドバーから緯度経度で直接調整できるようにする。
        if ref_data == data_source_JAPAN_SEA:
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
            "Region preset (Fig.3-Fig.6)",
            ["Dataset default"] + list(envgeo_utils.MAP_REGION_PRESETS),
            help=(
                "Set the longitude and latitude range for Fig.3-Fig.6. "
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

        figure_scale_state_key = f"figure_scale_settings::{ref_data}::{lon_center_3d}::{region_preset_4d}"
        if figure_scale_state_key not in st.session_state:
            st.session_state[figure_scale_state_key] = {
                "map_lon_raw": map_lon_default,
                "map_lat_raw": map_lat_default,
                "fig_depth_raw": (0, int(sld_depth_max)),
                "marker_size": 3
            }

        figure_scale_settings = st.session_state[figure_scale_state_key]

        with st.form(key=f"figure_scale_form::{ref_data}::{lon_center_3d}::{region_preset_4d}"):
            map_lon_raw_form = st.slider(
                'Map Longitude (Fig.3-Fig.6)',
                lon_slider_min,
                lon_slider_max,
                figure_scale_settings["map_lon_raw"],
                step=1
            )
            map_lat_raw_form = st.slider(
                'Map Latitude (Fig.3-Fig.6)',
                lat_slider_min,
                lat_slider_max,
                figure_scale_settings["map_lat_raw"],
                step=1
            )

            fig_depth_raw_form = st.slider(
                label='Depth scale',
                min_value=0,
                max_value=int(sld_depth_max + 100), # 整数化して小数点を防止
                value=figure_scale_settings["fig_depth_raw"],
                step=50
            )
            
            # サイドバーにサイズ調整を追加
            marker_size_form = st.slider("Marker Size", 1, 10, figure_scale_settings["marker_size"])
            apply_figure_scale_settings = st.form_submit_button("Apply figure scale")

        if apply_figure_scale_settings:
            figure_scale_settings = {
                "map_lon_raw": map_lon_raw_form,
                "map_lat_raw": map_lat_raw_form,
                "fig_depth_raw": fig_depth_raw_form,
                "marker_size": marker_size_form
            }
            st.session_state[figure_scale_state_key] = figure_scale_settings

        map_lon_raw = figure_scale_settings["map_lon_raw"]
        map_lat_raw = figure_scale_settings["map_lat_raw"]
        fig_depth_min, fig_depth_max = figure_scale_settings["fig_depth_raw"]
        marker_size = figure_scale_settings["marker_size"]

        map_x_range = [map_lon_raw[0], map_lon_raw[1]]
        map_y_range = [map_lat_raw[0], map_lat_raw[1]]
        map_aspectratio = get_geo_aspectratio(
            map_x_range,
            map_y_range,
            z_ratio=0.5 if ref_data == data_source_GLOBAL else 1.0
        )



    ##############################################################################
    # キャッシュクリア
    ##############################################################################

    if st.sidebar.button("🔄 Clear cache"):
        envgeo_utils.clear_app_cache()
        st.rerun() 
    
    
    
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
    
    
    
    
    ###############################################################################################
    ###### Fig1 4D salinity-d18O-depth-temperature #######
    ###############################################################################################

    # 計算できない，もしくはカラーバー用のデータが無い場合に除外
    
 

    # 1. 計算する
    original_len_df1 = len(df1)
    # 2. 【追加】計算できなかった行（null）をその場で除外する
    df_fig1 = df1.dropna(subset=['Temperature_degC','d18O', 'Depth_m', 'Salinity'])
    # 消えた数を出力
    removed_num_fig1 = original_len_df1 - len(df_fig1)
    plotted_num_fig1 = original_len_df1 - removed_num_fig1
    # if removed_num_fig1 > 0:
    #     st.sidebar.info(f"{plotted_num_fig1} samples were plotted and {removed_num_fig1} samples were excluded due to no data.")


    # XYZC
    y = df_fig1['lat']
    x = df_fig1['lon']
    z = df_fig1['Depth_m']
    c = df_fig1['Temperature_degC']

    fig1=px.scatter_3d(df_fig1, x='Salinity', y='d18O', z='Depth_m',
                    color='Temperature_degC', 
                    #symbol='species'
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
                


    # マーカー、ラインの設定
    fig1.update_traces(
        mode = 'markers', # 'markers+lines', 'markers'
        marker = dict(size = 3),
    )
    

    
    # ---  【重要】レイアウトの一括設定（ここでスライダーを反映） ---
    fig1.update_layout(
        scene=dict(
            # 軸のタイトル
            xaxis_title='Salinity',
            yaxis_title='d18O',
            zaxis_title='Water Depth',
            
            # Z軸の範囲と反転設定 (スライダーの値をここに集約)
            # 逆順 [max, min] にすることで自動的に反転（Deepest at bottom）になります
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min], 
                autorange=False
            ),
            
            # アスペクト比
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            
            # カメラアングル
            camera=dict(
                eye=dict(x=-0.6, y=-1.1, z=1.9),
                center=dict(x=0, y=0, z=-0.1)
            )
        ),
        margin=dict(r=20, l=10, b=10, t=10)
    )
    
    
    fig1.update_traces(marker=dict(size=marker_size))
    


    # st.write(fig1)
    # st.plotly_chart(fig1, 
        # width="stretch" #Streramlitあげたら復活させる
        # )  # ブラウザの幅に合わせる


    
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################

    ###############################################################################################
    ###### Fig2 4D salinity-temperature-depth-d18O #######
    ###############################################################################################
        
    # 計算できない，もしくはカラーバー用のデータが無い場合に除外
    


    # 1. 計算する
    original_len_df1 = len(df1)
    # 【追加】計算できなかった行（null）をその場で除外
    df_fig2 = df1.dropna(subset=['Temperature_degC','d18O', 'Depth_m', 'Salinity'])
    # 消えた数を出力
    removed_num_fig2 = original_len_df1 - len(df_fig2)
    plotted_num_fig2 = original_len_df1 - removed_num_fig2
    # if removed_num_fig2 > 0:
    #     st.sidebar.info(f"{plotted_num_fig2} samples were plotted and {removed_num_fig2} samples were excluded due to no data.")


    # XYZC
    y = df_fig2['lat']
    x = df_fig2['lon']
    z = df_fig2['Depth_m']
    c = df_fig2['d18O']
    
   # 3. プロット作成
    fig2=px.scatter_3d(df_fig2, x='Salinity', y='Temperature_degC', z='Depth_m',
                    color='d18O', 
                    #symbol='species'
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
                


    # マーカー、ラインの設定
    fig2.update_traces(
        mode = 'markers', # 'markers+lines', 'markers'
        marker = dict(size = 3),
    )
    


    # 4. 【重要】レイアウトの一括設定
    # スライダーの値をここで適用することで、設定がリセットされるのを防ぐ
    fig2.update_layout(
        scene=dict(
            # 各軸のタイトル
            xaxis_title='Salinity',
            yaxis_title='Temperature (C)',
            zaxis_title='Water Depth',
            
            # Z軸の範囲設定（スライダー値を反映し、逆順 [max, min] で反転表示）
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            
            # アスペクト比とカメラ角度
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
    


    # st.write(Fig2)
    # st.plotly_chart(fig2,  
    #         width="stretch" #Streramlitあげたら復活させる  
    #         )
    

    
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################

    ###############################################################################################
    ###### Fig3 4D map-depth-d18O #######
    ###############################################################################################
    
    # 計算できない，もしくはカラーバー用のデータが無い場合に除外

    # 1. 計算する
    original_len_df1 = len(df1)
    # 2. 【追加】計算できなかった行（null）をその場で除外する
    df_fig3 = df1.dropna(subset=['d18O', 'Depth_m']).copy()
    # Use center-adjusted longitude only for plotting; keep original Longitude_degE for hover/readout.
    # 描画用だけ中心に合わせた経度を使い、表示値は元の Longitude_degE を保つ。
    df_fig3['lon_plot'] = normalize_lon_to_center(df_fig3['lon'], lon_center_3d)

    # 消えた数を出力
    removed_num_fig3 = original_len_df1 - len(df_fig3)
    plotted_num_fig3 = original_len_df1 - removed_num_fig3
    # if removed_num_fig3 > 0:
    #     st.sidebar.info(f"{plotted_num_fig3} samples were plotted and {removed_num_fig3} samples were excluded due to no data.")


    # XYZC
    y = df_fig3['lat']
    x = df_fig3['lon_plot']
    z = df_fig3['Depth_m']
    c = df_fig3['d18O']
    


    # --- envgeo_utils を使って読み込み ---
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    # Coastline segments also need center-adjusted longitude with explicit breaks at wrap boundaries.
    # 海岸線も中心に合わせた経度へ変換し、折り返し境界では明示的に線を切る。
    coastline_x_plot, coastline_y_plot = wrap_coastline_with_breaks(coastline_x, coastline_y, lon_center_3d)
    
    fig3=px.scatter_3d(df_fig3, x='lon_plot', y='lat', z='Depth_m',
                    color='d18O', 
                    #symbol='species'
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
   

    
    # マーカー、ラインの設定
    fig3.update_traces(
        mode = 'markers', # 'markers+lines', 'markers'
        marker = dict(size = 3),
        name='d18O'
        )
    
    fig3.update_traces(marker=dict(size=marker_size))
    
    # 4. 【重要】レイアウトの一括設定
    # スライダーの値をここで適用することで、設定がリセットされるのを防ぐ
    fig3.update_layout(
        scene=dict(
            # Z軸の範囲設定（スライダー値を反映し、逆順 [max, min] で反転表示）
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            )
        )
    
  
    
    # --- envgeo_utilsから呼び出して，レイアウトの一括更新 ---
     # 1. まず変数を定義する（NASAなどの場合は None、日本近海なら数値が入るように）
    x_range = map_x_range
    y_range = map_y_range
    
    # 2. その後で共通レイアウトを呼び出す
    fig3 = envgeo_utils.apply_common_layout(
        fig3, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    fig3.update_layout(scene=dict(aspectmode='manual', aspectratio=map_aspectratio))
    
    
    # 海岸線を底面に追加する
    # データの最上部の場合
    fig3.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_min] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    #スケールの底面の場合
    # Place the gray coastline on the figure bottom instead of the dataset deepest sample.
    # 灰色の海岸線はデータ最深点ではなく、図の底面に合わせて描画する。
    fig3.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_max] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='gray', width=0.5),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    
    
    
    # グラフを表示する
    # fig.show()
    # st.write(fig3)
    # st.plotly_chart(fig3,  
    #         width="stretch" #Streramlitあげたら復活させる  
    #         )
        
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
    
    ###############################################################################################
    ###### Fig4 map-depth-temperature #######
    ###############################################################################################
        
    # 計算できない，もしくはカラーバー用のデータが無い場合に除外
    # 1. 計算する
    original_len_df1 = len(df1)
    # # 2. 【追加】計算できなかった行（null）をその場で除外する
    df_fig4 = df1.dropna(subset=['Temperature_degC', 'Depth_m']).copy()
    # Use center-adjusted longitude only for plotting; keep original Longitude_degE for hover/readout.
    # 描画用だけ中心に合わせた経度を使い、表示値は元の Longitude_degE を保つ。
    df_fig4['lon_plot'] = normalize_lon_to_center(df_fig4['lon'], lon_center_3d)
    
    # 消えた数を出力
    removed_num_fig4 = original_len_df1 - len(df_fig4)
    plotted_num_fig4 = original_len_df1 - removed_num_fig4
    # if removed_num_fig4 > 0:
    #     st.sidebar.info(f"{plotted_num_fig4} samples were plotted and {removed_num_fig4} samples were excluded due to no data.")



    # XYZC
    y = df_fig4['lat']
    x = df_fig4['lon_plot']
    z = df_fig4['Depth_m']
    c = df_fig4['Temperature_degC']


    # --- envgeo_utils を使って読み込み ---
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    # Coastline segments also need center-adjusted longitude with explicit breaks at wrap boundaries.
    # 海岸線も中心に合わせた経度へ変換し、折り返し境界では明示的に線を切る。
    coastline_x_plot, coastline_y_plot = wrap_coastline_with_breaks(coastline_x, coastline_y, lon_center_3d)

    fig4=px.scatter_3d(df_fig4, x='lon_plot', y='lat', z='Depth_m',
                    color='Temperature_degC', 
                    #symbol='species'
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
    
    
    
    # マーカー、ラインの設定
    fig4.update_traces(
        mode = 'markers', # 'markers+lines', 'markers'
        marker = dict(size = 3),
        name='Temperature'
        )
        
    fig4.update_traces(marker=dict(size=marker_size))
    
    
        
    # 4. 【重要】レイアウトの一括設定
    # スライダーの値をここで適用することで、設定がリセットされるのを防ぐ
    fig4.update_layout(
        scene=dict(
            # Z軸の範囲設定（スライダー値を反映し、逆順 [max, min] で反転表示）
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            )
        )

    

    # --- envgeo_utilsから呼び出して，レイアウトの一括更新 ---
     # 1. まず変数を定義する（NASAなどの場合は None、日本近海なら数値が入るように）
    x_range = map_x_range
    y_range = map_y_range
    
    # 2. その後で共通レイアウトを呼び出す
    fig4 = envgeo_utils.apply_common_layout(
        fig4, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    fig4.update_layout(scene=dict(aspectmode='manual', aspectratio=map_aspectratio))
    
    
    
    # 海岸線を底面に追加する
    # データの最上部の場合
    fig4.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_min] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    
    #スケールの底面の場合
    # Place the gray coastline on the figure bottom instead of the dataset deepest sample.
    # 灰色の海岸線はデータ最深点ではなく、図の底面に合わせて描画する。
    fig4.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_max] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='gray', width=0.5),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    
    
    
    # グラフを表示する
    # fig.show()
    # st.write(fig4)
    # st.plotly_chart(fig4,  
    #         width="stretch" #Streramlitあげたら復活させる  
    #         )
    
    
        
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
    
    ###############################################################################################
    ###### Fig5 4D map-depth-salinity #######
    ###############################################################################################

        
    # 計算できない，もしくはカラーバー用のデータが無い場合に除外
    # 1. 計算する
    original_len_df1 = len(df1)
    # 2. 【追加】計算できなかった行（null）をその場で除外する
    df_fig5 = df1.dropna(subset=['Salinity', 'Depth_m']).copy()
    # Use center-adjusted longitude only for plotting; keep original Longitude_degE for hover/readout.
    # 描画用だけ中心に合わせた経度を使い、表示値は元の Longitude_degE を保つ。
    df_fig5['lon_plot'] = normalize_lon_to_center(df_fig5['lon'], lon_center_3d)

    # 消えた数を出力
    removed_num_fig5 = original_len_df1 - len(df_fig5)
    plotted_num_fig5 = original_len_df1 - removed_num_fig5
    # if removed_num_fig5 > 0:
    #     st.sidebar.info(f"{plotted_num_fig5} samples were plotted and {removed_num_fig5} samples were excluded due to no data.")

     
    
    # XYZC
    y = df_fig5['lat']
    x = df_fig5['lon_plot']
    z = df_fig5['Depth_m']
    c = df_fig5['Salinity']


    
    # --- envgeo_utils を使って読み込み ---
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    # Coastline segments also need center-adjusted longitude with explicit breaks at wrap boundaries.
    # 海岸線も中心に合わせた経度へ変換し、折り返し境界では明示的に線を切る。
    coastline_x_plot, coastline_y_plot = wrap_coastline_with_breaks(coastline_x, coastline_y, lon_center_3d)
    
    fig5=px.scatter_3d(df_fig5, x='lon_plot', y='lat', z='Depth_m',
                    color='Salinity', 
                    #symbol='species'
                    width=700,
                    height=600,
                    color_continuous_scale=map_colorscale,
                    
                    #############################ポップアップ情報ここから##########################
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
                        # "x": False,  # X座標はツールチップから除外
                        # "y": False  # Y座標はツールチップから除外
                    }
                    #############################ポップアップ情報ここまで##########################
                )
    

    
    # マーカー、ラインの設定
    fig5.update_traces(
        mode = 'markers', # 'markers+lines', 'markers'
        marker = dict(size = 3),
        name='Salinity'
        )
        
    fig5.update_traces(marker=dict(size=marker_size))

        
    # 4. 【重要】レイアウトの一括設定
    # スライダーの値をここで適用することで、設定がリセットされるのを防ぐ
    fig5.update_layout(
        scene=dict(
            # Z軸の範囲設定（スライダー値を反映し、逆順 [max, min] で反転表示）
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            )
        )

    
    
    # --- envgeo_utilsから呼び出して，レイアウトの一括更新 ---
     # 1. まず変数を定義する（NASAなどの場合は None、日本近海なら数値が入るように）
    x_range = map_x_range
    y_range = map_y_range
    
    # 2. その後で共通レイアウトを呼び出す
    fig5 = envgeo_utils.apply_common_layout(
        fig5, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    fig5.update_layout(scene=dict(aspectmode='manual', aspectratio=map_aspectratio))
    
    
    
    # 海岸線を底面に追加する
    # データの最上部の場合
    fig5.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_min] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    
    #スケールの底面の場合
    # Place the gray coastline on the figure bottom instead of the dataset deepest sample.
    # 灰色の海岸線はデータ最深点ではなく、図の底面に合わせて描画する。
    fig5.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_max] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='gray', width=0.5),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    
    
    
    
    # グラフを表示する
    # fig.show()
    # st.write(fig5)
    # st.plotly_chart(fig5,  
    #         width="stretch" #Streramlitあげたら復活させる  
    #         )
    
    
        
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
    
    
    ##############################################################################################
    ##### Fig6 4D map-depth-dexcess #######
    ##############################################################################################
    
    # 計算できない，もしくはカラーバー用のデータが無い場合に除外

    # 1. 計算する
    original_len_df1 = len(df1)
    # df_dexcess['d-excess'] = df_dexcess['dD'] - 8 * df_dexcess['d18O']
    
    # 2. 【追加】計算できなかった行（null）をその場で除外する
    df_fig6 = df1.dropna(subset=['d-excess','Depth_m']).copy()
    # Use center-adjusted longitude only for plotting; keep original Longitude_degE for hover/readout.
    # 描画用だけ中心に合わせた経度を使い、表示値は元の Longitude_degE を保つ。
    df_fig6['lon_plot'] = normalize_lon_to_center(df_fig6['lon'], lon_center_3d)

    # 消えた数を出力
    removed_num_fig6 = original_len_df1 - len(df_fig6)
    plotted_num_fig6 = original_len_df1 - removed_num_fig6
    # if removed_num_fig6 > 0:
    #     st.sidebar.info(f" {plotted_num_fig6} samples were plotted and {removed_num_fig6} samples were excluded due to calculation errors.")

        
    
    # XYZC
    y = df_fig6['lat']
    x = df_fig6['lon_plot']
    z = df_fig6['Depth_m']
    c = df_fig6['d-excess']
    

    
    # --- 海岸線の座標データをenvgeo_utils を使って読み込み ---
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    # Coastline segments also need center-adjusted longitude with explicit breaks at wrap boundaries.
    # 海岸線も中心に合わせた経度へ変換し、折り返し境界では明示的に線を切る。
    coastline_x_plot, coastline_y_plot = wrap_coastline_with_breaks(coastline_x, coastline_y, lon_center_3d)


    fig6=px.scatter_3d(df_fig6, x='lon_plot', y='lat', z='Depth_m',
                    color='d-excess', 
                    #symbol='species'
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
    
   
    
    # マーカー、ラインの設定
    fig6.update_traces(
        # mode = 'markers+lines', # 'markers+lines', 'markers'
        mode = 'markers', # 'markers+lines', 'markers'
        marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='d-excess'
        )
        
    fig6.update_traces(marker=dict(size=marker_size))
    
    
        
    # 4. 【重要】レイアウトの一括設定
    # スライダーの値をここで適用することで、設定がリセットされるのを防ぐ
    fig6.update_layout(
        scene=dict(
            # Z軸の範囲設定（スライダー値を反映し、逆順 [max, min] で反転表示）
            zaxis=dict(
                range=[fig_depth_max, fig_depth_min],
                autorange=False
            ),
            )
        )
    
    
    
    # --- envgeo_utilsから呼び出して，レイアウトの一括更新 ---
     # 1. まず変数を定義する（NASAなどの場合は None、日本近海なら数値が入るように）
    x_range = map_x_range
    y_range = map_y_range
    
    # 2. その後で共通レイアウトを呼び出す
    fig6 = envgeo_utils.apply_common_layout(
        fig6, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    fig6.update_layout(scene=dict(aspectmode='manual', aspectratio=map_aspectratio))
    
    
    # 海岸線を底面に追加する
    # データの最上部の場合
    fig6.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_min] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    #スケールの底面の場合
    # Place the gray coastline on the figure bottom instead of the dataset deepest sample.
    # 灰色の海岸線はデータ最深点ではなく、図の底面に合わせて描画する。
    fig6.add_traces(go.Scatter3d(x=coastline_x_plot, y=coastline_y_plot, z=[fig_depth_max] * len(coastline_x_plot), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='gray', width=0.5),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    
    

    
    # グラフを表示する
    # st.plotly_chart(fig6,  
    #         width="stretch" #Streramlitあげたら復活させる  
    #         )
    
        
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################

    # ここから本格的に表示用の図
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
    
    
    # ここから本格的に表示用の図
    
    # fig1-fig6を選択して表示
    # 個別のfigは st.plotly_chartで書き出さず，ここで選ぶようにする
        
    # --- 1. 二段組み（グリッド）にするためのCSS ---
    st.markdown("""
        <style>
        /* ラジオボタンの項目を2列の並びにする */
        div[role="radiogroup"] {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # --- 2. 選択肢の定義 ---
    # リストにしておくことで、if文での判定ミス（一文字違いなど）を物理的に防ぐ
    options = [
        "4D salinity-δ18O-depth-temperature (Fig.1)", 
        "4D salinity-temperature-depth-d18O (Fig.2)", 
        "4D map-depth-d18O (Fig.3)", 
        "4D map-depth-temperature (Fig.4)", 
        "4D map-depth-salinity (Fig.5)", 
        "4D map-depth-d-excess (Fig.6)",
        "Custom 4D plot beta"
    ]
    custom_option = options[-1]

    # --- 3. ラジオボタンの設置 ---
    display_option = st.radio(
        "Choose data to display:",
        options
    )

    # --- 4. 選択された項目に応じて表示するfigとdfを決定する ---
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
            st.warning("Custom 4D plot requires at least four numeric columns.")
            return

        custom_mode = st.radio(
            "Custom template",
            [
                "Salinity-d18O-[custom]-[custom]",
                "T-S-[custom]-[custom]",
                "Lon-Lat-depth-[custom]",
            ],
            horizontal=True,
        )

        custom_is_map_template = custom_mode == "Lon-Lat-depth-[custom]"

        if custom_mode == "Salinity-d18O-[custom]-[custom]":
            custom_x = "Salinity"
            custom_y = "d18O"
            custom_cols = st.columns(2)
            with custom_cols[0]:
                custom_z = st.selectbox(
                    "Z",
                    numeric_options,
                    index=safe_option_index(numeric_options, "Depth_m"),
                    key="custom_4d_sal_d18o_z",
                )
            with custom_cols[1]:
                custom_color = st.selectbox(
                    "Color",
                    numeric_options,
                    index=safe_option_index(numeric_options, "Temperature_degC"),
                    key="custom_4d_sal_d18o_color",
                )
        elif custom_mode == "T-S-[custom]-[custom]":
            custom_x = "Salinity"
            custom_y = "Temperature_degC"
            custom_cols = st.columns(2)
            with custom_cols[0]:
                custom_z = st.selectbox(
                    "Z",
                    numeric_options,
                    index=safe_option_index(numeric_options, "Depth_m"),
                    key="custom_4d_ts_z",
                )
            with custom_cols[1]:
                custom_color = st.selectbox(
                    "Color",
                    numeric_options,
                    index=safe_option_index(numeric_options, "d18O"),
                    key="custom_4d_ts_color",
                )
        else:
            custom_x = "Longitude_degE"
            custom_y = "Latitude_degN"
            custom_z = "Depth_m"
            custom_color = st.selectbox(
                "Color",
                numeric_options,
                index=safe_option_index(numeric_options, "d18O"),
                key="custom_4d_map_color",
            )

        custom_required_columns = [custom_x, custom_y, custom_z, custom_color]
        df_custom = df1.dropna(subset=custom_required_columns).copy()
        removed_num_custom = len(df1) - len(df_custom)
        plotted_num_custom = len(df_custom)
        if removed_num_custom > 0:
            st.caption(
                f":red[Note: {plotted_num_custom} samples were plotted and "
                f"{removed_num_custom} samples were excluded due to incomplete data "
                f"for the selected custom variables.]"
            )

        if df_custom.empty:
            st.warning("No valid rows remain for the selected custom 4D variables.")
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
                    marker=dict(size=3),
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
                    marker=dict(size=3),
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
            st.caption(f":red[Note: {plotted_num_fig1} samples were plotted and {removed_num_fig1} samples were excluded due to incomplete data (missing one or more variables).]")
    elif display_option == options[1]:
        target_fig = fig2
        plot_key = "p2"
        df_map = df_fig2
        if removed_num_fig2 > 0:
            st.caption(f":red[Note: {plotted_num_fig2} samples were plotted and {removed_num_fig2} samples were excluded due to incomplete data (missing one or more variables).]")
    elif display_option == options[2]:
        target_fig = fig3
        plot_key = "p3"
        df_map = df_fig3
        if removed_num_fig3 > 0:
            st.caption(f":red[Note: {plotted_num_fig3} samples were plotted and {removed_num_fig3} samples were excluded due to incomplete data (missing one or more variables).]")
    elif display_option == options[3]:
        target_fig = fig4
        plot_key = "p4"
        df_map = df_fig4
        if removed_num_fig4 > 0:
            st.caption(f":red[Note: {plotted_num_fig4} samples were plotted and {removed_num_fig4} samples were excluded due to incomplete data (missing one or more variables).]")
    elif display_option == options[4]:
        target_fig = fig5
        plot_key = "p5"
        df_map = df_fig5
        if removed_num_fig5 > 0:
            st.caption(f":red[Note: {plotted_num_fig5} samples were plotted and {removed_num_fig5} samples were excluded due to incomplete data (missing one or more variables).]")
    else:
        target_fig = fig6
        plot_key = "p6"
        df_map = df_fig6
        if removed_num_fig6 > 0:
            st.caption(f":red[Note: {plotted_num_fig6} samples were plotted and {removed_num_fig6} samples were excluded due to calculation errors (missing one or more variables).]")



    
    st.write("---")
    

    st.subheader(display_option) # タイトルを表示

    
    
    # スライダーの設置　2026/03/11改訂
    import math

    # --- 3D図（Fig1-6）共通：短縮ラベルの設定 ---
    c_int = ["Temperature_degC", "d18O", "d18O", "Temperature_degC", "Salinity", "d-excess"]
    c_lbl = ["Temperature(C)", "d18O", "d18O", "Temperature(C)", "Salinity", "d-excess"]

    # --- 要素ごとのデフォルト・カラーレンジ設定 (外れ値対策) ---
    # ref_data の条件に合わせて数値を調整
    if ref_data == data_source_JAPAN_SEA:
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

    # display_option から現在のインデックスを取得して短縮名に変換
    if display_option == custom_option:
        t_col = custom_color
        t_lbl = label_for_column(custom_color)
    else:
        idx = options.index(display_option) if display_option in options else 0
        t_col = c_int[idx]
        t_lbl = c_lbl[idx]

    # --- 3D図専用のスライダー ---
    # データの絶対的な最小・最大
    v_min_actual = float(df_map[t_col].min())
    v_max_actual = float(df_map[t_col].max())

    # スライダーの初期位置を辞書から取得（辞書にない場合はデータの最小・最大）
    d_range = default_ranges.get(t_col, (v_min_actual, v_max_actual))

    r_3d = st.slider(
        f"Colorbar scale adjustment: {t_lbl}",
        min_value=float(math.floor(v_min_actual * 10) / 10),
        max_value=float(math.ceil(v_max_actual * 10) / 10),
        value=d_range, # ここに自動設定された初期値が入る
        step=0.1,
        key=f"c3_slider_{envgeo_utils.safe_filename_text(t_col)}_{envgeo_utils.safe_filename_text(display_option)}"
    )

    # --- 各Figの更新と反映 ---
    for f_idx, f_name in enumerate(['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6']):
        if f_name in locals() and locals()[f_name] is not None:
            f = locals()[f_name]
            
            # この図が担当しているカラムとラベルを取得
            current_fig_col = c_int[f_idx]
            current_fig_lbl = c_lbl[f_idx]

            # レイアウト更新
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

            # 現在表示対象(t_col)の図、または同じ要素の図にはスライダー値を反映
            if current_fig_col == t_col:
                f.update_coloraxes(cmin=r_3d[0], cmax=r_3d[1])
            else:
                # それ以外の図はデフォルトレンジを適用
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

    # --- 5. 最後に一回だけ表示を実行 ---
    st.plotly_chart(
        target_fig, 
        # width="stretch" #Streramlitあげたら復活させる
        key=plot_key,
        config={'scrollZoom': True}
    )
    
    
           
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
    
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################


    # 選択されたデータの地点プロット
    # --- Location map / 採取地点の地図表示 ---
    st.divider()
    # st.subheader('Location Map')
    st.subheader("Geographical Distribution Map")
    import math

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

    # データの範囲から中心座標とズームレベルを計算
    lat_min, lat_max = df_map["Latitude_degN"].min(), df_map["Latitude_degN"].max()
    lon_min, lon_max = df_map["Longitude_degE"].min(), df_map["Longitude_degE"].max()

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
    
    
    

    
    # --- Map 専用設定 ---
    # 内部列名と短縮ラベルのリスト（Map側で独自に定義）

    m_cols = ["Temperature_degC", "d18O", "d18O", "Temperature_degC", "Salinity", "d-excess"]
    m_lbls = ["Temperature(C)", "d18O", "d18O", "Temperature(C)", "Salinity", "d-excess"]
    
    # 現在の選択から項目を特定
    if display_option == custom_option:
        m_target = custom_color
        m_label = label_for_column(custom_color)
    else:
        m_idx = options.index(display_option) if display_option in options else 0
        m_target = m_cols[m_idx]
        m_label = m_lbls[m_idx]
    
    # Map専用スライダーの作成（ここで r_map を定義）
    # st.write(f"### Map Scale Control ({m_label})")
    mv1, mv2 = float(df_map[m_target].min()), float(df_map[m_target].max())
    r_map = st.slider(
        f"Colorbar scale adjustment: {t_lbl} ", 
        float(math.floor(mv1*10)/10), float(math.ceil(mv2*10)/10), (mv1, mv2), 
        0.1, key=f"slider_map_{envgeo_utils.safe_filename_text(m_target)}_{envgeo_utils.safe_filename_text(display_option)}"
    )
    
    # 地図作成 (color="d18O" 固定を解除)
    # 地図の作成
    # Reuse the sidebar colormap choice for the final 2D location map as well.
    # 最後の 2D 地図でも、サイドバーで選んだカラーマップを共通利用する。
    c_scale_map = map_colorscale
    fig_map = px.scatter_mapbox(
        df_map, lat="Latitude_degN", lon="Longitude_degE",
        color=m_target,                   # 選択項目で色付け
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
            "d-excess": True, # d-excess用
        },
        opacity=0.6, height=500
    )
    
    


    # 背景スタイルの適用
    fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)
    
    

    
    # 地図のレイアウト設定（カラーバーを水平に下配置）
    fig_map.update_layout(
        coloraxis_colorbar=dict(
            title=m_label,
            orientation="h",       # 水平
            yanchor="top", y=-0.15, # 図の下
            x=0.5, xanchor="center",
            thickness=15
        ),
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
        margin=dict(l=0, r=0, t=0, b=100),
        autosize=True
    )
    
    # スライダー r_map の値を地図に反映
    fig_map.update_coloraxes(cmin=r_map[0], cmax=r_map[1])





    # 表示 
    # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）　
    # マウスホイールでのズームが強制的に有効

    st.plotly_chart(
        fig_map, 
        # width="stretch" #Streramlitあげたら復活させる
        key="dynamic_map_final", # キーも一応ユニークに
        config={'scrollZoom': True, 'displayModeBar': True}
    )




    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    ##選ばれたデータを表示
    # 例：特定の列だけを選択して新しいデータフレームを作成
        

    with st.expander("selected dataset (CSV)", expanded=False):
        
        table_columns = [
            'reference', 'Cruise', 'Station', 'Date', 'Year', 'Month',
            'Longitude_degE', 'Latitude_degN', 'Depth_m',
            'Temperature_degC', 'Salinity', 'd18O', 'dD',
            envgeo_utils.QUALITY_FLAG_COLUMN,
            envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN,
        ]

        # d-excess が使える時だけ追加
        if 'd-excess' in df_map.columns:
            table_columns.append('d-excess')

        available_columns = [col for col in table_columns if col in df_map.columns]
        df1_table = df_map[available_columns].copy()

      
        # 【重要】表示直前に全列を文字列化（これでArrowエラーは100%消えます）
        df1_table = df1_table.astype(str) 
        
        # 最新の width='stretch' を使用しない　1.42まで
        st.dataframe(df1_table, 
                     # width="stretch" #Streramlitあげたら復活させる
                     )
        
    
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

if __name__ == '__main__':
    main()
    
