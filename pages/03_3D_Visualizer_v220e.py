
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:00:21 2023

@author: Toyoho Ishimura @Kyoto-U

2026/02/10 update 
"""


# --- Version info ---
version = "1.0.0" #v220_20260317

# ToDo




import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events
import math
import envgeo_utils  
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)



def main():
    
    # 注意書き
    st.header(f'3D Visualizer ({version})')
    

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
    # サイドバーここから　　df1フィルタリング　も一括で
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





    ##############################################################################
    # 3D-4Dでは，depthをマイナス表示にする場合あり，それ以外は後半の定義用
    ##############################################################################
        
    df1['lat'] = df1['Latitude_degN']
    df1['lon'] = df1['Longitude_degE']
    # df1['Depth_m'] = df1['Depth_m']*(-1)


    ##############################################################################
    # 図のスケール変更，使わない場合もあり
    ##############################################################################
   
    # #スペース入れる
    # st.sidebar.subheader(':blue[--- for fig scale only ---]')
    

    ##############################################################################
    # --- 図の種類選択　---　　 
    ##############################################################################

    fig_type_d18Osal = "d18O-Salinity relationship"   
    fig_type_TS = "Temperature–Salinity (T–S) diagram"


    plot_figure = st.radio(":blue[Select plot type:]", (fig_type_d18Osal, fig_type_TS), horizontal=True, args=[1, 0])
    


    ##############################################################################
    # キャッシュクリア
    ##############################################################################
        
    # キャッシュのクリア　サイドバーの一番下などに配置
    if st.sidebar.button("🔄 Clear cache"):
        envgeo_utils.clear_app_cache()
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
    

    
    ##########################################################################
    # --- 共通スタイル設定関数の定義 (すべての図にこれを適用) ---
    ##########################################################################
    def unify_plot_layout(fig, x_label, y_label, color_title):
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            
            # 1. 全体サイズを固定
            width=850,   # 右側のマージンを考慮して少し広めに設定
            height=600,
            autosize=False,
            
            xaxis=dict(
                title=x_label,
                showline=True, linewidth=1, linecolor='grey', mirror=True,
                showgrid=True, gridcolor='rgba(220, 220, 220, 0.4)', gridwidth=0.5,
                ticks="inside", ticklen=5
            ),
            yaxis=dict(
                title=y_label,
                showline=True, linewidth=1, linecolor='grey', mirror=True,
                showgrid=True, gridcolor='rgba(220, 220, 220, 0.4)', gridwidth=0.5,
                ticks="inside", ticklen=5
            ),
            
            # 2. カラーバーを「枠の外」に固定配置する設定
            coloraxis_colorbar=dict(
                title=color_title,
                thickness=15,
                len=0.8,
                x=1.02,           # グラフ枠のすぐ右外側に固定（1.0が枠の右端）
                xanchor='left',   # 左端を基準に配置
                y=0.5,
                yanchor='middle'
            ),
            
            # 3. マージンの設定
            # 右側(r)を150px程度確保することで、文字が長くても枠サイズに影響を与えない
            margin=dict(l=80, r=150, t=50, b=80), 
            
            font=dict(size=12)
        )
        
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        
        return fig
    



    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    
    # 区切り線
    st.divider()
    
    
    ###############################################################################################
    ############################################################################################### 
    # Salinity - Temperature
    ###############################################################################################
    ###############################################################################################
    
    if plot_figure == fig_type_TS:
    

        # --- 1. セッション状態の初期化 ---
        if 'ts_selected_indices' not in st.session_state:
            st.session_state.ts_selected_indices = []
        
        st.subheader('Temperature - Salinity Relationship & Auto-Zoom Map')
        
        # (ラジオボタンやスケール設定)
        col_map = {"d18O": "d18O", "Latitude": "Latitude_degN", "Longitude": "Longitude_degE", "Water Depth": "Depth_m"}
        sel_col = st.radio("Select color element:", list(col_map.keys()), horizontal=True, key="fig_TS_zoom")
        target_item = col_map[sel_col]
        
        # カラースケール，depthの時は反転
        c_scale_final = envgeo_utils.get_custom_colorscale(target_item)

        
        # --- 2. T-S図の描画とサイズ圧縮 ---
        
        # 【2026/03/11修正ポイント】選択された要素(target_item)にデータがない行を排除する
        # 同時に、T-S図の基本要素である Salinity と Temperature がない行も消しておく
        df_plot_ts = df1.dropna(subset=[target_item, "Salinity", "Temperature_degC"]).reset_index(drop=True)
    
    
        # 排除したサンプル数を計算（メッセージなどで使う用）
        excluded_count = len(df1) - len(df_plot_ts)
        if excluded_count > 0:
            st.caption(f":red[Note: {excluded_count:,} samples were excluded due to missing ({sel_col}, temperature, salinity) data.]")
    
        # 案内を出す
        # st.info("T-S図で範囲を選択すると、その範囲にズームします。")
        st.info("Use the Box or Lasso selection tools on the Salinity–Temperature plot to highlight the corresponding sampling locations on the map.")
        
        
    
        fig_fixed_TS = px.scatter(
            df_plot_ts, # リセット済みのデータを使用
            x="Salinity", y="Temperature_degC", 
            color=target_item, color_continuous_scale=c_scale_final,
            hover_data={
                "Salinity": True,
                "Temperature_degC": True,
                "lat": True,
                "lon": True,
                "d18O": True,
                "dD": True,
                "Year": True,
                "Month": True,
                "Day": True,
                "Cruise": True,
                "Station": True,
                "Depth_m": True,
                "reference": True
            }
        )
        
        
        
        # ① 共通レイアウト適用
        fig_fixed_TS = unify_plot_layout(fig_fixed_TS, "Salinity", "Temperature (C)", sel_col)
        
    
        
        # ② 【枠サイズを固定するための修正】
        fig_fixed_TS.update_layout(
            hovermode='closest', # 近くの点を探し回るのをやめる
            hoverdistance=5,     # 反応する距離を大幅に小さくする（初期値は20程度）
            width=800, 
            height=600, 
            margin=dict(l=80, r=200, t=50, b=80, autoexpand=False), 
            coloraxis_colorbar=dict(
                x=1.02, 
                xanchor='left',
                len=0.8,
                thickness=15,
            ),
            xaxis=dict(
                zeroline=False, zerolinewidth=1, zerolinecolor='grey',
                range=[df_plot_ts["Salinity"].min()*0.95, df_plot_ts["Salinity"].max()*1.05]
            ),
            yaxis=dict(
                zeroline=False, zerolinewidth=1, zerolinecolor='grey',
                range=[df_plot_ts["Temperature_degC"].min()-1, df_plot_ts["Temperature_degC"].max()+1]
            )
        )
    

        # ③ 表示枠（窓枠）の設定
        selected_points = plotly_events(
            fig_fixed_TS, 
            select_event=True, 
            key="ts_zoom_event",
            override_height=600, 
            override_width=850
        )
        
        # --- 【選択個数の処理】 ---
        if selected_points:
            st.session_state.ts_selected_indices = [p['pointIndex'] for p in selected_points]
            num_selected = len(selected_points)
            # 地図のすぐ上に個数を表示
            st.write(f"📊 **Number of selected points: {num_selected}**")
        else:
            # 何も選択されていない場合は全データ（初期状態）
            st.session_state.ts_selected_indices = []
            
        
        # --- 3. 地図の表示データ決定 ---
        # 選択されているか判定
        is_selected = len(st.session_state.ts_selected_indices) > 0
    
        if is_selected:
            # 選択されたデータのみ
            df_ts_map_display = df_plot_ts.iloc[st.session_state.ts_selected_indices]
        else:
            # 初期状態は全データ（または特定のデフォルト範囲）
            df_ts_map_display = df_plot_ts
    
        # --- 4. 範囲とズームの計算 ---
        lat_min, lat_max = df_ts_map_display["Latitude_degN"].min(), df_ts_map_display["Latitude_degN"].max()
        lon_min, lon_max = df_ts_map_display["Longitude_degE"].min(), df_ts_map_display["Longitude_degE"].max()
        
        
        # import math
        lat_diff = lat_max - lat_min if lat_max != lat_min else 0.5
        lon_diff = lon_max - lon_min if lon_max != lon_min else 0.5
        
        #　地図の中心を設定
        center_lat = df_ts_map_display['Latitude_degN'].mean()
        center_lon = df_ts_map_display['Longitude_degE'].mean()
        
        
        # 地図サイズに基づいたズーム計算
        map_width_px = 850
        map_height_px = 600
        zoom_lon = math.log2((map_width_px * 360) / (lon_diff * 256))
        zoom_lat = math.log2((map_height_px * 180) / (lat_diff * 256))
        
        # 選択時はズーム、初期状態は少し引き気味にするなどの調整
        auto_zoom = min(zoom_lon, zoom_lat) - (0.8 if is_selected else 1.5)
        auto_zoom = max(1, min(15, auto_zoom))
    
        # --- 5. 地図の描画 (常に実行) ---
        fig_ts_map = px.scatter_mapbox(
            df_ts_map_display, 
            lat="Latitude_degN", lon="Longitude_degE", 
            color=target_item, color_continuous_scale=c_scale_final,
            mapbox_style="carto-positron",
            hover_data=["d18O",'dD',"Salinity",'Temperature_degC','Year','Month','Day','Cruise','Station','Depth_m','reference'],
        )
        
        
    
        fig_ts_map = unify_plot_layout(fig_ts_map, "Lon", "Lat", sel_col)
        
        fig_ts_map.update_layout(
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=auto_zoom
                
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            width=750,
            height=500
        )
        

        
        # 地図背景の選択　envgeo_utilsｋara
        #  モード選択（keyをユニークにする）
        map_mode_ts = st.radio(
            "Map Style:", 
            ["Standard", "Satellite", "Bathymetry (Sea)", "Contour (GSI)"], 
            horizontal=True, key="ms_ts"
        )
    
        
        #  設定ファイルからスタイルを適用
        fig_ts_map = envgeo_utils.apply_map_style(fig_ts_map, map_mode_ts)
        

        
        # 地図を表示
        # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）　
        # マウスホイールでのズームが強制的に有効
        st.plotly_chart(
            fig_ts_map, 
            # width="stretch", # Streamlitあげた復活させる
            key="3d_visualizer_map_TS",
            config={'scrollZoom': True, 'displayModeBar': True} # ズームを有効化
        )
        
            
        # Sidebar-filtered datasetを読み出しデータフレームを作成
        envgeo_utils.display_isotope_table(df1)
        # Plotly-filtered datasetを読み出しデータフレームを作成
        envgeo_utils.display_isotope_table(df_ts_map_display,  title="Dataset from Box/Lasso selection (CSV)")
        
        
        
        #htmlで書き出す場合
        # fig_ts_map.write_html('filename.html')
        
    
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################


    
    ###############################################################################################
    ############################################################################################### 
    # Salinity - δ18O 
    ###############################################################################################
    ###############################################################################################
    elif  plot_figure == fig_type_d18Osal:
    
    
        # --- 1. セッション状態の初期化 ---
        if 'd18o_selected_indices' not in st.session_state:
            st.session_state.d18o_selected_indices = []
    
        st.subheader('δ18O - Salinity Relationship & Auto-Zoom Map')
    
        # カラー設定
        col_map = {"d18O": "d18O", "Latitude": "Latitude_degN", "Longitude": "Longitude_degE", "Water Depth": "Depth_m"}
        sel_col_d18o = st.radio("Select color element (δ18O plot):", list(col_map.keys()), horizontal=True, key="fig_d18O_zoom")
        target_item = col_map[sel_col_d18o]
        # カラースケール，depthの時は反転
        c_scale_final = envgeo_utils.get_custom_colorscale(target_item)

        
        
        # --- 2. Salinity - δ18O 図の描画 ---
    
        # 【2026/03/11修正ポイント】選択された要素(target_item)にデータがない行を排除する
        # 同時に、T-S図の基本要素である Salinity と Temperature がない行も消しておく
        df_plot_d18o = df1.dropna(subset=[target_item, "Salinity", "d18O"]).reset_index(drop=True)

    
        # 排除したサンプル数を計算（メッセージなどで使う用）
        excluded_count2 = len(df1) - len(df_plot_d18o)
        if excluded_count2 > 0:
            st.caption(f":red[Note: {excluded_count2:,} samples were excluded due to missing ({sel_col_d18o}, Salinity, d18O) data.]")
    
    
        # 案内を出す
        # st.info("T-S図で範囲を選択すると、その範囲にズーム")
        st.info("Use the Box or Lasso selection tools on the Salinity–δ18O plot to highlight the corresponding sampling locations on the map.")
    
    
    
        fig_d18O = px.scatter(
            df_plot_d18o, # リセット済みのデータを使用
            x="Salinity", y="d18O", 
            color=target_item, color_continuous_scale=c_scale_final,
            hover_data={
                "Salinity": True,
                "d18O": True,
                "lat": True,
                "lon": True,
                "dD": True,
                "Temperature_degC": True,
                "Year": True,
                "Month": True,
                "Day": True,
                "Cruise": True,
                "Station": True,
                "Depth_m": True,
                "reference": True
            }
        )
        
        fig_d18O = unify_plot_layout(fig_d18O, "Salinity", "δ18O (‰)", sel_col_d18o)
        
    
        # レイアウト固定設定
        fig_d18O.update_layout(
            hovermode='closest', # 近くの点を探し回るのをやめる
            hoverdistance=5,     # 反応する距離を大幅に小さくする（初期値は20程度）
            width=800, 
            margin=dict(l=80, r=200, t=50, b=80, autoexpand=False), 
            coloraxis_colorbar=dict(x=1.02, xanchor='left', len=0.8),
            xaxis=dict(
                zeroline=False, zerolinewidth=1, zerolinecolor='grey',
                showline=True, linewidth=1, linecolor='grey', mirror=True,
                range=[df_plot_d18o["Salinity"].min()*0.95, df_plot_d18o["Salinity"].max()*1.05]
            ),
            yaxis=dict(
                zeroline=False, zerolinewidth=1, zerolinecolor='grey',
                showline=True, linewidth=1, linecolor='grey', mirror=True,
                range=[df_plot_d18o["d18O"].min()-0.5, df_plot_d18o["d18O"].max()+0.5]
            )
        )
    
    
    
        selected_points_d18o = plotly_events(
            fig_d18O, select_event=True, key="d18o_zoom_event",
            override_height=600, override_width=850
        )
    
            
            # --- 【個数表示とセッション更新の処理】 ---
        if selected_points_d18o:
            st.session_state.d18o_selected_indices = [p['pointIndex'] for p in selected_points_d18o]
            num_selected_d18o = len(selected_points_d18o)
            # 地図のすぐ上に個数を太字で表示
            st.write(f"📊 **Number of selected points: {num_selected_d18o}**")
        else:
            st.session_state.d18o_selected_indices = []
            
    
        # --- 3. 地図の表示データ決定 ---
        is_selected = len(st.session_state.d18o_selected_indices) > 0
        df_map_d18o = df_plot_d18o.iloc[st.session_state.d18o_selected_indices] if is_selected else df_plot_d18o
    
        # --- 4. 範囲とズームの計算 (TypeError修正) ---
        lat_min, lat_max = df_map_d18o["Latitude_degN"].min(), df_map_d18o["Latitude_degN"].max()
        lon_min, lon_max = df_map_d18o["Longitude_degE"].min(), df_map_d18o["Longitude_degE"].max()
        
        
        #　地図の中心を設定
        center_lat = df_map_d18o['Latitude_degN'].mean()
        center_lon = df_map_d18o['Longitude_degE'].mean()
        
        
        lat_diff = max(lat_max - lat_min, 0.1)
        lon_diff = max(lon_max - lon_min, 0.1)
        
        # ズーム計算
        zoom_lon = math.log2((850 * 360) / (lon_diff * 256))
        zoom_lat = math.log2((600 * 180) / (lat_diff * 256))
        auto_zoom = min(zoom_lon, zoom_lat) - (0.8 if is_selected else 1.5)
        auto_zoom = max(1, min(15, auto_zoom))
    
        # --- 5. 地図の描画 ---
        fig_map_d18o = px.scatter_mapbox(
            df_map_d18o, lat="Latitude_degN", lon="Longitude_degE", 
            color=target_item, color_continuous_scale=c_scale_final,
            mapbox_style="carto-positron",
            hover_data=["d18O",'dD',"Salinity",'Temperature_degC','Year','Month','Day','Cruise','Station','Depth_m','reference'],
        )
        fig_map_d18o = unify_plot_layout(fig_map_d18o, "Lon", "Lat", sel_col_d18o)
        fig_map_d18o.update_layout(
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
            margin=dict(l=0, r=0, t=0, b=0), width=750, height=500
        )
        
        
        
        # 地図背景の選択　envgeo_utilsｋara
        #  モード選択（keyをユニークにする）
        map_mode_d18o = st.radio(
            "Map Style:", 
            ["Standard", "Satellite", "Bathymetry (Sea)", "Contour (GSI)"], 
            horizontal=True, key="ms_d18o"
        )
    
        
        #  設定ファイルからスタイルを適用
        fig_map_d18o = envgeo_utils.apply_map_style(fig_map_d18o, map_mode_d18o)
        
        
        
        # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）　
        # マウスホイールでのズームが強制的に有効
        st.plotly_chart(
            fig_map_d18o, 
            # width="stretch", # Streamlitあげた復活させる
            key="3d_visualizer_map_d18O",
            config={'scrollZoom': True, 'displayModeBar': True} # ズームを有効化
        )
        
        
        
        # Sidebar-filtered datasetを読み出しデータフレームを作成
        envgeo_utils.display_isotope_table(df1)
        # Plotly-filtered datasetを読み出しデータフレームを作成
        envgeo_utils.display_isotope_table(df_map_d18o,  title="Dataset from Box/Lasso selection (CSV)")
        
        
        #htmlで書き出す場合
        # fig_map_d18o.write_html('filename.html')
    

    
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
        
if __name__ == '__main__':
    main()
    








