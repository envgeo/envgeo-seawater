
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:00:21 2023

@author: Toyoho Ishimura @Kyoto-U

2026/02/10 update 
"""


# --- Version info ---
version = "1.3.0" #v220_20260317

# ToDo


COLOR_FILTERED_OPTIONS = {
    "d18O": "d18O",
    "dD": "dD",
    "d-excess": "d-excess",
    "Temperature": "Temperature_degC",
    "Salinity": "Salinity",
    "Water Depth": "Depth_m",
    "Latitude": "Latitude_degN",
    "Longitude": "Longitude_degE",
}


def available_color_filtered_options(df):
    """Return color options available in the current dataframe.

    現在のデータに存在する列だけを、色分け候補として表示します。
    """
    return {
        label: column
        for label, column in COLOR_FILTERED_OPTIONS.items()
        if column in df.columns
    }


def render_color_controls(df, key_prefix):
    """Render color-column and colormap controls side by side.

    色分けする列とカラーマップを横並びで選択します。
    """
    col_map = available_color_filtered_options(df)
    control_col, colormap_col = st.columns([1, 1])
    with control_col:
        selected_label = st.selectbox(
            "Color parameter",
            list(col_map.keys()),
            key=f"{key_prefix}_color_filtered",
            help=getattr(
                envgeo_utils,
                "COLOR_PARAMETER_HELP_TEXT",
                "Choose the variable used to color the plotted points or map markers.",
            ),
        )
    target_column = col_map[selected_label]
    colormap_options = envgeo_utils.get_plotly_colormap_options(target_column)
    colormap_labels = list(colormap_options.keys())
    default_index = (
        colormap_labels.index("EnvGeo variable default")
        if "EnvGeo variable default" in colormap_labels
        else 0
    )
    with colormap_col:
        selected_colormap_label = st.selectbox(
            "Colormap",
            colormap_labels,
            index=default_index,
            key=f"{key_prefix}_colormap",
            help=(
                "Choose the color palette used for the filtered plot and matching map. "
                "cmocean palettes are designed for oceanographic data."
            ),
        )
    colorscale = envgeo_utils.get_plotly_colormap(target_column, selected_colormap_label)
    return selected_label, target_column, colorscale


def add_regression_line(fig, df, x_col, y_col, line_name="Regression line"):
    """Add a simple least-squares regression line to a Plotly scatter figure.

    試験的な近似直線を追加します。点数不足や同じX値のみの場合は何もしません。
    """
    regression_df = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(regression_df) < 2 or regression_df[x_col].nunique() < 2:
        return None

    slope, intercept = np.polyfit(regression_df[x_col], regression_df[y_col], 1)
    x_min = float(regression_df[x_col].min())
    x_max = float(regression_df[x_col].max())
    x_vals = np.array([x_min, x_max])
    y_vals = slope * x_vals + intercept
    r_value = regression_df[x_col].corr(regression_df[y_col])
    stats_text = f"y = {slope:.3g}x + {intercept:.3g} | R = {r_value:.2f}"
    fig.add_scatter(
        x=x_vals,
        y=y_vals,
        mode="lines",
        line=dict(color="black", width=2),
        name=f"{line_name}: {stats_text}",
        hoverinfo="name",
    )
    return stats_text


def render_regression_stats(stats_text):
    """Render compact regression statistics beside the control."""
    if not stats_text:
        st.caption("Regression unavailable")
        return

    st.markdown(
        f"""
        <div style="
            margin-top: 0.18rem;
            padding: 0.38rem 0.55rem;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            background: #f9fafb;
            color: #374151;
            font-size: 0.82rem;
            line-height: 1.35;
            white-space: nowrap;
        ">
            <strong>Regression</strong>&nbsp;&nbsp;{stats_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def selected_point_indices(selected_points, max_len):
    """Return selected indices from the main scatter trace only.

    近似直線などの追加traceが混ざっても、元の散布点だけを地図連動に使います。
    """
    indices = []
    for point in selected_points or []:
        if point.get("curveNumber", 0) != 0:
            continue
        point_index = point.get("pointIndex")
        if point_index is None:
            continue
        if 0 <= point_index < max_len:
            indices.append(point_index)
    return indices


def show_selection_tip():
    """Show a compact guide below Plotly selection figures.

    Box/Lasso 選択の説明を、図の直下に見やすく表示します。
    """
    st.markdown(
        """
        <div style="
            margin: 0.25rem 0 0.8rem 0;
            padding: 0.55rem 0.75rem;
            border-left: 4px solid #3b82f6;
            background: #eef6ff;
            color: #1f2937;
            font-size: 0.92rem;
            line-height: 1.45;
        ">
            <strong>Tip:</strong> Use <strong>Box Select</strong> or <strong>Lasso Select</strong>
            to highlight matching sampling locations on the map.
        </div>
        """,
        unsafe_allow_html=True,
    )




import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events
import math
import envgeo_utils  
import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)



def main():
    
    # 注意書き
    st.header(f'3D Visualizer ({version})')
    st.caption("Use interactive Plotly selection to link T-S or salinity-δ18O plots with sampling locations.")
    

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
    # st.sidebar.subheader("Figure controls")
    

    ##############################################################################
    # --- 図の種類選択　---　　 
    ##############################################################################

    fig_type_d18Osal = "d18O-Salinity relationship"   
    fig_type_TS = "Temperature–Salinity (T–S) diagram"


    plot_figure = st.radio(
        "Plot type",
        (fig_type_d18Osal, fig_type_TS),
        horizontal=True,
        args=[1, 0],
        help="Choose the interactive Plotly view to display.",
    )
    


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
        
        st.subheader('Temperature-Salinity Diagram')
        
        sel_col, target_item, c_scale_final = render_color_controls(df1, "fig_TS_zoom")

        
        # --- 2. T-S図の描画とサイズ圧縮 ---
        
        # 【2026/03/11修正ポイント】選択された要素(target_item)にデータがない行を排除する
        # 同時に、T-S図の基本要素である Salinity と Temperature がない行も消しておく
        df_plot_ts = df1.dropna(subset=[target_item, "Salinity", "Temperature_degC"]).reset_index(drop=True)
    
    
        # 排除したサンプル数を計算（メッセージなどで使う用）
        excluded_count = len(df1) - len(df_plot_ts)
        if excluded_count > 0:
            st.caption(f":red[{excluded_count:,} rows excluded because {sel_col}, temperature, or salinity was missing.]")
    
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
        show_selection_tip()
        
        # --- 【選択個数の処理】 ---
        selected_indices = selected_point_indices(selected_points, len(df_plot_ts))
        if selected_indices:
            st.session_state.ts_selected_indices = selected_indices
            num_selected = len(selected_indices)
            # 地図のすぐ上に個数を表示
            st.write(f"**Selected points: {num_selected}**")
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
            mapbox_style="open-street-map",
            hover_data=["d18O",'dD',"Salinity",'Temperature_degC','Year','Month','Day','Cruise','Station','Depth_m','reference'],
        )
        
        
    
        fig_ts_map = unify_plot_layout(fig_ts_map, "Lon", "Lat", sel_col)
        
        fig_ts_map.update_layout(
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=auto_zoom
                
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            height=500,
            coloraxis_colorbar=dict(
                title=sel_col,
                x=0.98,
                xanchor="right",
                y=0.5,
                yanchor="middle",
                len=0.8,
                thickness=15,
            ),
        )
        

        
        # Keep map controls compact so the map remains visible after Streamlit reruns.
        # Streamlitの再実行後も地図が見つけやすいよう、地図設定をポップオーバーに集約する。
        with st.popover("Map controls", use_container_width=True):
            map_mode_ts = st.radio(
                "Map style", 
                envgeo_utils.MAP_MODE_OPTIONS, 
                horizontal=True,
                key="ms_ts",
                help=getattr(
                    envgeo_utils,
                    "MAP_STYLE_HELP_TEXT",
                    "Choose the background map style for the sampling-location map.",
                ),
            )
        st.caption(f"Map style: {map_mode_ts}")
    
        
        #  設定ファイルからスタイルを適用
        fig_ts_map = envgeo_utils.apply_map_style(fig_ts_map, map_mode_ts)
        

        
        # 地図を表示
        # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）　
        # マウスホイールでのズームが強制的に有効
        st.plotly_chart(
            fig_ts_map, 
            # width="stretch", # Streamlitあげた復活させる
            use_container_width=True,
            key="3d_visualizer_map_TS",
            config={'scrollZoom': True, 'displayModeBar': True} # ズームを有効化
        )
        
            
        # Sidebar-filtered datasetを読み出しデータフレームを作成
        envgeo_utils.display_isotope_table(df1)
        # Plotly-filtered datasetを読み出しデータフレームを作成
        envgeo_utils.display_isotope_table(df_ts_map_display,  title="Box/Lasso-selected dataset (CSV)")
        
        
        
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
    
        st.subheader('Salinity-δ18O Relationship')
    
        sel_col_d18o, target_item, c_scale_final = render_color_controls(df1, "fig_d18O_zoom")
        regression_control_col, regression_stats_col = st.columns([0.9, 2.1])
        with regression_control_col:
            show_regression_d18o = st.checkbox(
                "Regression line",
                value=False,
                key="fig_d18O_zoom_regression_line",
                help=getattr(
                    envgeo_utils,
                    "REGRESSION_HELP_TEXT",
                    "Add a simple least-squares regression line for quick visual reference.",
                ),
            )
        regression_stats_text_d18o = ""

        
        
        # --- 2. Salinity - δ18O 図の描画 ---
    
        # 【2026/03/11修正ポイント】選択された要素(target_item)にデータがない行を排除する
        # 同時に、T-S図の基本要素である Salinity と Temperature がない行も消しておく
        df_plot_d18o = df1.dropna(subset=[target_item, "Salinity", "d18O"]).reset_index(drop=True)

    
        # 排除したサンプル数を計算（メッセージなどで使う用）
        excluded_count2 = len(df1) - len(df_plot_d18o)
        if excluded_count2 > 0:
            st.caption(f":red[{excluded_count2:,} rows excluded because {sel_col_d18o}, salinity, or d18O was missing.]")
    
    
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
        if show_regression_d18o:
            regression_stats_text_d18o = add_regression_line(
                fig_d18O,
                df_plot_d18o,
                "Salinity",
                "d18O",
            )
        with regression_stats_col:
            if show_regression_d18o:
                render_regression_stats(regression_stats_text_d18o)
        
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
        show_selection_tip()
    
            
            # --- 【個数表示とセッション更新の処理】 ---
        selected_indices_d18o = selected_point_indices(selected_points_d18o, len(df_plot_d18o))
        if selected_indices_d18o:
            st.session_state.d18o_selected_indices = selected_indices_d18o
            num_selected_d18o = len(selected_indices_d18o)
            # 地図のすぐ上に個数を太字で表示
            st.write(f"**Selected points: {num_selected_d18o}**")
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
            mapbox_style="open-street-map",
            hover_data=["d18O",'dD',"Salinity",'Temperature_degC','Year','Month','Day','Cruise','Station','Depth_m','reference'],
        )
        fig_map_d18o = unify_plot_layout(fig_map_d18o, "Lon", "Lat", sel_col_d18o)
        fig_map_d18o.update_layout(
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=auto_zoom),
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            height=500,
            coloraxis_colorbar=dict(
                title=sel_col_d18o,
                x=0.98,
                xanchor="right",
                y=0.5,
                yanchor="middle",
                len=0.8,
                thickness=15,
            ),
        )
        
        
        
        # Keep map controls compact so the map remains visible after Streamlit reruns.
        # Streamlitの再実行後も地図が見つけやすいよう、地図設定をポップオーバーに集約する。
        with st.popover("Map controls", use_container_width=True):
            map_mode_d18o = st.radio(
                "Map style", 
                envgeo_utils.MAP_MODE_OPTIONS, 
                horizontal=True,
                key="ms_d18o",
                help=getattr(
                    envgeo_utils,
                    "MAP_STYLE_HELP_TEXT",
                    "Choose the background map style for the sampling-location map.",
                ),
            )
        st.caption(f"Map style: {map_mode_d18o}")
    
        
        #  設定ファイルからスタイルを適用
        fig_map_d18o = envgeo_utils.apply_map_style(fig_map_d18o, map_mode_d18o)
        
        
        
        # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）　
        # マウスホイールでのズームが強制的に有効
        st.plotly_chart(
            fig_map_d18o, 
            # width="stretch", # Streamlitあげた復活させる
            use_container_width=True,
            key="3d_visualizer_map_d18O",
            config={'scrollZoom': True, 'displayModeBar': True} # ズームを有効化
        )
        
        
        
        # Sidebar-filtered datasetを読み出しデータフレームを作成
        envgeo_utils.display_isotope_table(df1)
        # Plotly-filtered datasetを読み出しデータフレームを作成
        envgeo_utils.display_isotope_table(df_map_d18o,  title="Box/Lasso-selected dataset (CSV)")
        
        
        #htmlで書き出す場合
        # fig_map_d18o.write_html('filename.html')
    

    
    ###############################################################################################
    ############################################################################################### 
    ###############################################################################################
    ###############################################################################################
        
if __name__ == '__main__':
    main()
    
