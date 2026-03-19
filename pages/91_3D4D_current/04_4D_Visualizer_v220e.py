
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
# 最後のマップのカラーバーの初期値を調整必要
# 最後に，各図を定義して，選ばれたときに個別に実行すれば軽くなるはず


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import envgeo_utils    
pd.set_option('future.no_silent_downcasting', True)


def main():
    
    st.header(f'4D Visualizer ({version})')

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
    # d-excessを計算
    ##############################################################################
    df1['d-excess'] = df1['dD'] - 8 * df1['d18O']


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


    ##############################################################################
    # 図のスケール変更
    ##############################################################################
   

   # サイドバーの中にコンテナを作成し、境界線（border）を有効にする
    with st.sidebar.container(border=True):
        st.subheader(':blue[--- for fig scale only ---]')

        fig_depth_min, fig_depth_max = st.slider(
            label='Depth scale',
            min_value=0,
            max_value=int(sld_depth_max + 100), # 整数化して小数点を防止
            value=(0, int(sld_depth_max)),
            step=50
        )
        

        # サイドバーにサイズ調整を追加
        marker_size = st.slider("Marker Size", 1, 10, 3)



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


    color_continuous_scale= ('darkblue', 'blue', 'lightgray', 'gray', 'lightgreen', 'lightgreen', 'green',  'yellow', 'orange', 'red','darkred','darkred')
   
    fig1=px.scatter_3d(df_fig1, x='Salinity', y='d18O', z='Depth_m',
                    color='Temperature_degC', 
                    #symbol='species'
                    width=700,
                    height=600,
                    color_continuous_scale=color_continuous_scale,
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
    
    
    
    # 2. カラースケールの定義
    color_continuous_scale= ('darkblue', 'darkblue', 'darkblue','darkblue', 'darkblue',  'blue', 'blue', 'blue','lightgray', 'lightgray', 'gray', 'lightgreen', 'lightgreen', 'green',  'yellow', 'orange', 'red','darkred')
   
    
   # 3. プロット作成
    fig2=px.scatter_3d(df_fig2, x='Salinity', y='Temperature_degC', z='Depth_m',
                    color='d18O', 
                    #symbol='species'
                    width=700,
                    height=600,
                    color_continuous_scale=color_continuous_scale,
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
    df_fig3 = df1.dropna(subset=['d18O', 'Depth_m'])

    # 消えた数を出力
    removed_num_fig3 = original_len_df1 - len(df_fig3)
    plotted_num_fig3 = original_len_df1 - removed_num_fig3
    # if removed_num_fig3 > 0:
    #     st.sidebar.info(f"{plotted_num_fig3} samples were plotted and {removed_num_fig3} samples were excluded due to no data.")


    # XYZC
    y = df_fig3['lat']
    x = df_fig3['lon']
    z = df_fig3['Depth_m']
    c = df_fig3['d18O']
    


    # --- envgeo_utils を使って読み込み ---
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    
    color_continuous_scale= ('darkblue', 'darkblue', 'darkblue','darkblue', 'darkblue',  'blue', 'blue', 'blue','lightgray', 'lightgray', 'gray', 'lightgreen', 'lightgreen', 'green',  'yellow', 'orange', 'red','darkred')

    

    fig3=px.scatter_3d(df_fig3, x='lon', y='lat', z='Depth_m',
                    color='d18O', 
                    #symbol='species'
                    width=700,
                    height=600,
                    color_continuous_scale=color_continuous_scale,
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
    if ref_data == data_source_GLOBAL:
        x_range = None
        y_range = None
    else:
        x_range = [120, 160] # デフォルトの日本近海範囲
        y_range = [20, 60]
    
    # 2. その後で共通レイアウトを呼び出す
    fig3 = envgeo_utils.apply_common_layout(
        fig3, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    
    
    # 海岸線を底面に追加する
    # データの最上部の場合
    fig3.add_traces(go.Scatter3d(x=coastline_x, y=coastline_y, z=[fig_depth_min] * len(coastline_x), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    #スケールの底面の場合
    fig3.add_traces(go.Scatter3d(x=coastline_x, y=coastline_y, z=[max(z)] * len(coastline_x), mode='lines',     marker = dict(size = 3),
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
    df_fig4 = df1.dropna(subset=['Temperature_degC', 'Depth_m'])
    
    # 消えた数を出力
    removed_num_fig4 = original_len_df1 - len(df_fig4)
    plotted_num_fig4 = original_len_df1 - removed_num_fig4
    # if removed_num_fig4 > 0:
    #     st.sidebar.info(f"{plotted_num_fig4} samples were plotted and {removed_num_fig4} samples were excluded due to no data.")



    # XYZC
    y = df_fig4['lat']
    x = df_fig4['lon']
    z = df_fig4['Depth_m']
    c = df_fig4['Temperature_degC']


    # --- envgeo_utils を使って読み込み ---
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)

    color_continuous_scale= ('darkblue', 'darkblue', 'darkblue','darkblue', 'darkblue',  'blue', 'blue', 'blue','lightgray', 'lightgray', 'gray', 'lightgreen', 'lightgreen', 'green',  'yellow', 'orange', 'red','darkred')


    
    fig4=px.scatter_3d(df_fig4, x='lon', y='lat', z='Depth_m',
                    color='Temperature_degC', 
                    #symbol='species'
                    width=700,
                    height=600,
                    color_continuous_scale=color_continuous_scale,
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
    if ref_data == data_source_GLOBAL:
        x_range = None
        y_range = None
    else:
        x_range = [120, 160] # デフォルトの日本近海範囲
        y_range = [20, 60]
    
    # 2. その後で共通レイアウトを呼び出す
    fig4 = envgeo_utils.apply_common_layout(
        fig4, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    
    
    
    # 海岸線を底面に追加する
    # データの最上部の場合
    fig4.add_traces(go.Scatter3d(x=coastline_x, y=coastline_y, z=[fig_depth_min] * len(coastline_x), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    
    #スケールの底面の場合
    fig4.add_traces(go.Scatter3d(x=coastline_x, y=coastline_y, z=[max(z)] * len(coastline_x), mode='lines',     marker = dict(size = 3),
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
    df_fig5 = df1.dropna(subset=['Salinity', 'Depth_m'])

    # 消えた数を出力
    removed_num_fig5 = original_len_df1 - len(df_fig5)
    plotted_num_fig5 = original_len_df1 - removed_num_fig5
    # if removed_num_fig5 > 0:
    #     st.sidebar.info(f"{plotted_num_fig5} samples were plotted and {removed_num_fig5} samples were excluded due to no data.")

     
    
    # XYZC
    y = df_fig5['lat']
    x = df_fig5['lon']
    z = df_fig5['Depth_m']
    c = df_fig5['Salinity']


    
    # --- envgeo_utils を使って読み込み ---
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)
    
    color_continuous_scale= ('darkblue', 'darkblue', 'darkblue', 'blue', 'blue', 'blue','lightgray', 'lightgray', 'gray',  'gray', 'lightgreen', 'lightgreen', 'green',  'green', 'yellow', 'orange', 'red','darkred')

    
    fig5=px.scatter_3d(df_fig5, x='lon', y='lat', z='Depth_m',
                    color='Salinity', 
                    #symbol='species'
                    width=700,
                    height=600,
                    color_continuous_scale=color_continuous_scale,
                    
                    #############################ポップアップ情報ここから##########################
                    hover_data={
                        "lat": True,  # 名前を表示
                        "lon": True,  # 値を表示
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
    if ref_data == data_source_GLOBAL:
        x_range = None
        y_range = None
    else:
        x_range = [120, 160] # デフォルトの日本近海範囲
        y_range = [20, 60]
    
    # 2. その後で共通レイアウトを呼び出す
    fig5 = envgeo_utils.apply_common_layout(
        fig5, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    
    
    
    # 海岸線を底面に追加する
    # データの最上部の場合
    fig5.add_traces(go.Scatter3d(x=coastline_x, y=coastline_y, z=[fig_depth_min] * len(coastline_x), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    
    #スケールの底面の場合
    fig5.add_traces(go.Scatter3d(x=coastline_x, y=coastline_y, z=[max(z)] * len(coastline_x), mode='lines',     marker = dict(size = 3),
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
    df_fig6 = df1.dropna(subset=['d-excess','Depth_m'])

    # 消えた数を出力
    removed_num_fig6 = original_len_df1 - len(df_fig6)
    plotted_num_fig6 = original_len_df1 - removed_num_fig6
    # if removed_num_fig6 > 0:
    #     st.sidebar.info(f" {plotted_num_fig6} samples were plotted and {removed_num_fig6} samples were excluded due to calculation errors.")

        
    
    # XYZC
    y = df_fig6['lat']
    x = df_fig6['lon']
    z = df_fig6['Depth_m']
    c = df_fig6['d-excess']
    

    
    # --- 海岸線の座標データをenvgeo_utils を使って読み込み ---
    coastline_x, coastline_y = envgeo_utils.load_coastline_data(ref_data)


    # --- カラーバー色指定 ---
    color_continuous_scale= ('darkblue', 'blue','lightgray', 'gray', 'lightgreen', 'lightgreen', 'green',  'yellow', 'orange', 'red','red','red','red')

    
    fig6=px.scatter_3d(df_fig6, x='lon', y='lat', z='Depth_m',
                    color='d-excess', 
                    #symbol='species'
                    width=700,
                    height=600,
                    color_continuous_scale=color_continuous_scale,
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
    if ref_data == data_source_GLOBAL:
        x_range = None
        y_range = None
    else:
        x_range = [120, 160] # デフォルトの日本近海範囲
        y_range = [20, 60]
    
    # 2. その後で共通レイアウトを呼び出す
    fig6 = envgeo_utils.apply_common_layout(
        fig6, 
        ref_data, 
        fig_depth_max, 
        fig_depth_min, 
        x_range=x_range, 
        y_range=y_range
    )
    
    
    # 海岸線を底面に追加する
    # データの最上部の場合
    fig6.add_traces(go.Scatter3d(x=coastline_x, y=coastline_y, z=[fig_depth_min] * len(coastline_x), mode='lines',     marker = dict(size = 3),
        # line = dict(width = 2), #color = 'Black',
        name='coastline', line=dict(color='blue', width=0.8),
        hoverinfo='none' # 海岸線にカーソルが当たっても邪魔しない
        ))
    #スケールの底面の場合
    fig6.add_traces(go.Scatter3d(x=coastline_x, y=coastline_y, z=[max(z)] * len(coastline_x), mode='lines',     marker = dict(size = 3),
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
        "4D salinity-δ18O-depth-temperature", 
        "4D salinity-temperature-depth-d18O", 
        "4D map-depth-d18O", 
        "4D map-depth-temperature", 
        "4D map-depth-salinity", 
        "4D map-depth-d-excess"
    ]

    # --- 3. ラジオボタンの設置 ---
    display_option = st.radio(
        "Choose data to display:",
        options
    )

    # --- 4. 選択された項目に応じて表示するfigとdfを決定する ---
    if display_option == options[0]:
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
        key="c3_slider"
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
    # --- 採取地点の地図表示 (Auto-Zoom & 幅広設定) ---
    st.divider()
    # st.subheader('Location Map (Auto-Zoom)')
    st.subheader("Geographical Distribution Map (Auto-Zoom)")
    import math

    # 地図背景の選択
    map_mode = st.radio(
        "Map Style:", 
        ["Standard", "Satellite", "Bathymetry (Sea)", "Contour (GSI)"], 
        horizontal=True,
        key="map_style_31_auto"
    )

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
    m_idx = options.index(display_option) if display_option in options else 0
    m_target = m_cols[m_idx]
    m_label = m_lbls[m_idx]
    
    # Map専用スライダーの作成（ここで r_map を定義）
    # st.write(f"### Map Scale Control ({m_label})")
    mv1, mv2 = float(df_map[m_target].min()), float(df_map[m_target].max())
    r_map = st.slider(
        f"Colorbar scale adjustment: {t_lbl} ", 
        float(math.floor(mv1*10)/10), float(math.ceil(mv2*10)/10), (mv1, mv2), 
        0.1, key="slider_map_unique"
    )
    
    # 地図作成 (color="d18O" 固定を解除)
    # 地図の作成
    c_scale_map = envgeo_utils.get_custom_colorscale(m_target)
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
        
        # d-excessの時だけd-excess追加
        if display_option == options[5]:
            df1_table = df_map[['reference','Cruise', 'Station', 'Date', 'Year', 'Month', 'Longitude_degE', 'Latitude_degN', 'Depth_m', 'Temperature_degC', 'Salinity', 'd18O', 'dD', 'd-excess']].copy()   
        else:
            df1_table = df_map[['reference','Cruise', 'Station', 'Date', 'Year', 'Month', 'Longitude_degE', 'Latitude_degN', 'Depth_m', 'Temperature_degC', 'Salinity', 'd18O', 'dD']].copy()   

      
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
    






