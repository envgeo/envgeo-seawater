#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:00:21 2023

@author: Toyoho Ishimura @Kyoto-U

2026/03/18 update
"""

# --- Version info ---
version = "1.0.0" #v010a_20260317

# ToDo
# hover未調整



import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import envgeo_utils
pd.set_option('future.no_silent_downcasting', True)



st.header(f'3D/4D Visualizer Uploader ({version})')

st.text('Uploaded files are only temporarily cached in memory.')

# ファイルアップロード
uploaded_file = st.file_uploader("Please upload an Excel file (e.g., upload_data_tmp.xlsx).", type="xlsx")




# ファイルがアップロードされた場合
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    


    with st.sidebar.expander("Select axis", expanded=True): 
    
        # 行ラベル値を任意に設定する場合
        columns = list(df.columns)
        x_column = st.selectbox("Column to use for the X-axis", columns,index=0)
        y_column = st.selectbox("Column to use for the Y-axis", columns,index=1)
        z_column = st.selectbox("Column to use for the Z-axis", columns,index=2)
        index_column = st.selectbox("Column to use for the color scale", columns,index=3)
        
        
    with st.sidebar.expander("Uploaded data", expanded=False):
        st.write("Uploaded data preview:")
        st.write(df)
        
        
    # リロードボタン
    st.button('Reload')
    

    ##############################################################################
    # キャッシュクリア
    ##############################################################################
        
    # キャッシュのクリア　サイドバーの一番下などに配置
    if st.sidebar.button("🔄 Clear cache"):
        envgeo_utils.clear_app_cache()
        st.sidebar.success("Cache cleared! Reloading..")
        st.rerun() # アプリを再実行して最新のExcelを読み込ませる
    
    
    
    
    ##############################################################################
    # サイドバーでXYZ選んでもらう
    ##############################################################################
    

    st.markdown("##### :red[Select Axis Parameters (in Sidebar)]")
    st.caption(":red[(Automatically extracted from your file)]")

    

    
    ############################################################
    ###### Fig1 simple plot with coastline               #######
    ############################################################

    st.subheader('2.5D plot') # サブヘッダー
    
    # 1. 設定・カラー定義
    # 共通のカラースケール
    color_continuous_scale = ['darkblue', 'blue', 'lightblue', 'lightgreen', 'green', 'yellow', 'orange', 'red']
    
    # 2. 回帰直線（Trendline）の設定
    reg_line_plot = st.radio("Add regression line", ("No", "Yes"), horizontal=True)
    # "Yes" の場合は 'ols'、"No" の場合は None を指定
    current_trendline = 'ols' if reg_line_plot == "Yes" else None
    
    # 3. メインのプロット作成 (fig1)
    fig1 = px.scatter(
        df, 
        x=x_column, 
        y=y_column, 
        color=index_column, 
        trendline=current_trendline,  # ここで切り替え
        trendline_color_override='gray',
        width=700,
        height=600,
        color_continuous_scale=color_continuous_scale
    )
    
    # マーカーのスタイル更新
    fig1.update_traces(marker=dict(size=5))
    
    # 4. 海岸線（Coastline）のオーバーレイ設定
    coastline_plot = st.radio("Coastline overlay", ("None", "Japan", "World"), horizontal=True)
    
    # 選択に合わせてファイルを決定
    coastline_map = {
        "Japan": 'coastline/japan_coast_line.xlsx',
        "World": 'coastline/world_coastline_coordinates_50m.xlsx'
    }
    coastline_file = coastline_map.get(coastline_plot)
    
    # ファイルがある場合のみ描画
    if coastline_file:
        try:
            coastline_df = pd.read_excel(coastline_file)
            fig1.add_trace(go.Scatter(
                x=coastline_df['Longitude'], 
                y=coastline_df['Latitude'], 
                mode='lines',
                name='Coastline',
                line=dict(color='blue', width=0.8),
                showlegend=False
            ))
        except Exception as e:
            st.error(f"Error loading coastline file: {e}")
    
    # 5. 図の表示
    st.plotly_chart(
        fig1, 
        # width="stretch" #Streramlitあげたら復活させる
        )
        
    
    st.write(" ")

    ############################################################
    ###### Fig2 simple 4D plot                           #######
    ############################################################
    st.subheader('4D plot')
    
    #z軸を反転するラジオボタン
    z_inversion_fig2 = st.radio("z-axis inversion (4D plot) ", ("Yes", "No"), horizontal=True)
    
    if z_inversion_fig2 == "Yes":
        z_axis_config = dict(autorange="reversed")
    else:
        z_axis_config = dict()  # または空に設定
 
    # カラー定義
    color_continuous_scale = [
        'darkblue', 'blue', 'blue', 'blue', 
        'lightgray', 'lightgray', 'gray', 
        'lightgreen', 'lightgreen', 'green', 
        'yellow', 'orange', 'red'
    ]
    
    # プロット作成
    fig2 = px.scatter_3d(
        df, 
        x=x_column, 
        y=y_column, 
        z=z_column,
        color=index_column, 
        color_continuous_scale=color_continuous_scale,
        width=700,
        height=600
    )
    
    # マーカーのデザイン設定
    fig2.update_traces(
        mode='markers', 
        marker=dict(size=3)
    )
    
    # レイアウト・軸の一括設定
    # Z軸を反転させるかどうかの設定を反映
    z_axis_config = dict(autorange="reversed") if z_inversion_fig2 == "Yes" else dict()
    
    fig2.update_layout(
        scene=dict(
            xaxis_title=x_column,
            yaxis_title=y_column,
            zaxis_title=z_column,
            zaxis=z_axis_config,  # 水深反転設定
            # 必要に応じて範囲を固定する場合
            # xaxis=dict(range=[...]), 
            # yaxis=dict(range=[...]),
        ),
        margin=dict(r=20, l=10, b=10, t=10) # 余白を詰めると図が大きく見える
    )
    
    # 5. 表示
    st.plotly_chart(fig2, 
                    # width="stretch" #Streamlitあげたら復活させる
                    )


    
    st.write(" ")

    ############################################################
    ###### Fig3 4D with coastline (around JAPAN)         #######
    ############################################################
    
    st.subheader('4D Plot with coastline (Cross-referenced with Published Data)')
    
    

    # Z軸の設定（反転・範囲スライダー） 
    col1, col2 = st.columns([1, 2])
    with col1:
        z_inversion_fig3 = st.radio("Invert Z-axis", ("Yes", "No"), horizontal=True, key="inv_fig3")
    
    # スライダーの最大値を決定（自分のデータか、外部データか、大きい方を採用）
    max_depth_val = 10000 # デフォルト
    if not df.empty:
        max_depth_val = max(max_depth_val, int(df[z_column].max()))
    
    with col2:
        # Z軸の範囲を選択するスライダー
        z_range_fig3 = st.slider(
            "Z-axis Range (Depth)",
            min_value=0,
            max_value=10000,
            value=(0, 5000), # 初期表示範囲（例として5000m）
            step=100,
            key="range_fig3"
        )
    
    # スライダーの値と反転設定を組み合わせて config を作成
    if z_inversion_fig3 == "Yes":
        # [下端（深い）, 上端（浅い）]
        z_axis_config_fig3 = dict(range=[z_range_fig3[1], z_range_fig3[0]], autorange=False)
    else:
        # [上端（浅い）, 下端（深い）]
        z_axis_config_fig3 = dict(range=[z_range_fig3[0], z_range_fig3[1]], autorange=False)

    

        
    # メインの3Dプロット作成 
    fig3 = px.scatter_3d(
        df, x=x_column, y=y_column, z=z_column,
        color=index_column,
        color_continuous_scale=color_continuous_scale,
        hover_data={x_column: True, y_column: True, z_column: True, index_column: True}
    )
    fig3.update_traces(mode='markers', marker=dict(size=3))
        
        


    # 海岸線の追加 (3D空間の z=0 に描画) 
    coastline_plo_3D = st.radio("Draw coastline on 3D plot", ("Japan", "World", "None"), horizontal=True)
    
    coastline_map = {
        "Japan": 'coastline/japan_coast_line.xlsx',
        "World": 'coastline/world_coastline_coordinates_50m.xlsx'
    }
    coastline_file = coastline_map.get(coastline_plo_3D)
    
    if coastline_file:
        try:
            c_df = pd.read_excel(coastline_file)
            fig3.add_trace(go.Scatter3d(
                x=c_df['Longitude'], y=c_df['Latitude'], z=[0] * len(c_df),
                mode='lines', name='Coastline',
                line=dict(color='blue', width=0.8),
                showlegend=False
            ))
        except Exception as e:
            st.error(f"Error loading coastline: {e}")


 


    # --- 外部データセット の追加 ---
    data_sources = [
        "Do not plot", 
        envgeo_utils.data_source_JAPAN_SEA, 
        envgeo_utils.data_source_AROUND_JAPAN, 
        envgeo_utils.data_source_GLOBAL
    ]
    dataset_plot = st.radio("Plot reported dataset?", data_sources, horizontal=True)
    
    # 外部データの読み込み
    df_ref = envgeo_utils.load_isotope_data(dataset_plot)
    
    if not df_ref.empty:
        # 外部データをプロットに追加
        fig3.add_trace(go.Scatter3d(
            x=df_ref['Longitude_degE'], y=df_ref['Latitude_degN'], z=df_ref['Depth_m'],
            mode='markers', name='Reported Data',
            marker=dict(size=2, color='gray', opacity=0.5, symbol='circle')
        ))
        
        # # 深度スライダーの設定
        # max_ref_depth = int(math.ceil(df_ref['Depth_m'].max()))
        # fig_depth_min, fig_depth_max = st.slider(
        #     'Depth scale (Z-axis)', 0, 10000, (0, max_ref_depth), step=100
        # )
    
        # 範囲の決定
        x_range = None if dataset_plot == envgeo_utils.data_source_GLOBAL else [120, 160]
        y_range = None if dataset_plot == envgeo_utils.data_source_GLOBAL else [20, 60]
    
        # 共通レイアウトの適用（envgeo_utils内に関数がある場合）
        # 外部データの種類が不明なので，あえて以下は組み込まない。今後の地図専用のバージョン用にコメントアウトで残す
        # fig3 = envgeo_utils.apply_common_layout(
        #     fig3, dataset_plot, 
        #     # z_max=(), z_min=(),
        #     x_range=x_range, y_range=y_range
        # )
    
    # --- 5. 最終的なレイアウト微調整と表示 ---
    fig3.update_layout(
        scene=dict(
            xaxis_title=x_column,
            yaxis_title=y_column,
            zaxis_title=z_column,
            zaxis=z_axis_config_fig3
        ),
        width=700, height=600,
        margin=dict(r=20, l=10, b=10, t=10)
    )
    
    st.plotly_chart(fig3, 
                    # width="stretch" #Streramlitあげたら復活させる
                    )
    
    

# ユーザーファイルがアップロードされていない場合
else:  

    
    #　サンプルデータ出力。データフレームを作ってexcelで書き出し，ダウンロード
    st.write(":red[Sample files are available for download below.]")
        
    list1=[[1,11,21,"area1"], [2,12,22,'area2'], [3,13,23,'area3'],[4,14,24,'area4']]
    index1 = ["Row1", "Row2", "Row3","Row4"]
    columns1 =["x", "y", "z", "index"]
    df =     pd.DataFrame(data=list1, index=index1, columns=columns1)
    
    df.to_excel(buf := BytesIO(), index=False)
    
    st.download_button(
        "Blank Excel Template",
        buf.getvalue(),
        "upload_data_tmp.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    
    

    # サーバー上のデモ海岸線ファイルのパス
    file_path = "data/coastline_tmp.xlsx"  #
    
    # ファイルが存在するか確認
    try:
        with open(file_path, "rb") as file:
            excel_data = file.read()  # ファイルをバイナリデータとして読み込む
    
        # ダウンロードボタンを作成
        st.download_button(
            label="Download sample coastline data (Excel)",
            data=excel_data,
            file_name="coastline_tmp.xlsx",  # ダウンロード時のファイル名
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except FileNotFoundError:
        st.error(f"File '{file_path}' was not found.")
    
    
    # サーバー上のデモファイルのパス（データセット）
    file_path = "data/seawater_data_sample.xlsx"  
    
    # ファイルが存在するか確認
    try:
        with open(file_path, "rb") as file:
            excel_data = file.read()  # ファイルをバイナリデータとして読み込む
    
        # ダウンロードボタンを作成
        st.download_button(
            label="Excel Marine d18O data (sample dataset: from Kodama et al., 2024.)",
            data=excel_data,
            file_name="seawater_data_sample.xlsx",  # ダウンロード時のファイル名
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except FileNotFoundError:
        st.error(f"指定されたファイル '{file_path}' が見つかりません。")


