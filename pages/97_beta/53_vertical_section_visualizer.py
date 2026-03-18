#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:00:10 2026

@author: Toyoho Ishimura @Kyoto-U

2026/03/18 update
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import envgeo_utils



# --- 断面図作成用関数 ---
def create_section_plot(Z, xi, yi, df_points, x_col, target_col, z_min, z_max, plot_type="color"):
    fig = go.Figure()
    if plot_type == "color":
        fig.add_trace(go.Contour(
            z=Z, x=xi, y=yi, colorscale='Blues',
            zmin=z_min, zmax=z_max,
            colorbar=dict(title=target_col, orientation='h', y=-0.25, tickformat=".2f"),
            contours=dict(showlabels=True, coloring='heatmap', start=z_min, end=z_max),
            line_width=0, connectgaps=True
        ))
    else:
        fig.add_trace(go.Contour(
            z=Z, x=xi, y=yi,
            zmin=z_min, zmax=z_max,
            showscale=True,
            colorbar=dict(title=target_col, orientation='h', y=-0.25),
            contours=dict(
                coloring='none',
                showlabels=True,
                labelfont=dict(size=12, color='black'),
                start=z_min,
                end=z_max
            ),
            connectgaps=True
        ))
    
    fig.add_trace(go.Scatter(
        x=df_points[x_col], y=df_points['Depth_m'],
        mode='markers',
        marker=dict(size=6, color='black', opacity=0.6),
        name='Stations',
        hoverinfo='skip'
    ))

    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title="Depth (m)",
        yaxis=dict(autorange='reversed'),
        height=600,
        margin=dict(l=50, r=50, b=100, t=50)
    )
    return fig

# --- 起点マーク強調版 地図描画関数 (Mouse Zoom強化版) ---
def create_station_map(df_plot, x_axis_option):
    fig = go.Figure()

    if not df_plot.empty:
        # データの並び順から起点と終点を特定
        line_data = df_plot.sort_values(x_axis_option)
        start_pt = line_data.iloc[0]
        end_pt = line_data.iloc[-1]

        # 1. 全ての測点
        fig.add_trace(go.Scattermapbox(
            lat=df_plot["Latitude_degN"],
            lon=df_plot["Longitude_degE"],
            mode='markers',
            marker=go.scattermapbox.Marker(size=8, color='blue', opacity=0.6),
            text=df_plot["Station"] if "Station" in df_plot.columns else "",
            name="Stations"
        ))
        
        # 2. 断面ライン
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[start_pt["Longitude_degE"], end_pt["Longitude_degE"]],
            lat=[start_pt["Latitude_degN"], end_pt["Latitude_degN"]],
            line=dict(width=3, color="red"),
            name="Transect Line"
        ))
        
        # 3. 起点マーク (外枠)
        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=[start_pt["Longitude_degE"]],
            lat=[start_pt["Latitude_degN"]],
            marker=go.scattermapbox.Marker(size=20, color='orange', opacity=0.8),
            showlegend=False,
            hoverinfo='skip'
        ))

        # 4. 起点マーク (内側)
        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=[start_pt["Longitude_degE"]],
            lat=[start_pt["Latitude_degN"]],
            marker=go.scattermapbox.Marker(size=10, color='white'),
            name="START POINT (0km)",
            text="START POINT"
        ))

    # 地図の中心とズーム
    if not df_plot.empty:
        center_lat = df_plot["Latitude_degN"].mean()
        center_lon = df_plot["Longitude_degE"].mean()
        mapbox_dict = dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4
        )
    else:
        mapbox_dict = dict(style="open-street-map", center=dict(lat=35, lon=135), zoom=3)

    fig.update_layout(
        mapbox=mapbox_dict,
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600,
        showlegend=True,
        dragmode="pan"
    )
    return fig

def main():
    st.title("Vertical Section Visualizer")
    st.sidebar.title("Section Control Panel")
    
    # 1. データソース選択
    ref_data_source = st.radio("Select Data Source:", envgeo_utils.DATA_SOURCES)
    try:
        df_raw = envgeo_utils.load_isotope_data(ref_data_source)
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return

    st.sidebar.markdown("---")
    
    # 2. 詳細フィルタリング
    with st.sidebar.expander("🔍 Dataset Filters", expanded=True):
        df_f = df_raw.copy()
        
        # Transect フィルタ (追加)
        if 'Transect' in df_f.columns:
            transect_list = sorted(df_f['Transect'].dropna().unique().tolist())
            if transect_list:
                selected_transects = st.multiselect("Transect Line", transect_list, default=transect_list)
                df_f = df_f[df_f['Transect'].isin(selected_transects)]
            else:
                st.info("No Transect labels found.")

        # Reference フィルタ
        if 'Reference' in df_f.columns:
            ref_list = sorted(df_f['Reference'].dropna().unique().tolist())
            selected_refs = st.multiselect("Reference", ref_list, default=ref_list)
            df_f = df_f[df_f['Reference'].isin(selected_refs)]

        # Year / Month フィルタ

        if 'Year' in df_f.columns:
            years = sorted(df_f['Year'].dropna().unique().astype(int).tolist())
            sel_years = st.multiselect("Year", years, default=years)
            df_f = df_f[df_f['Year'].isin(sel_years)]

        if 'Month' in df_f.columns:
            months = sorted(df_f['Month'].dropna().unique().astype(int).tolist())
            sel_months = st.multiselect("Month", months, default=months)
            df_f = df_f[df_f['Month'].isin(sel_months)]

        # 緯度経度フィルタ
        if not df_f.empty:
            l_min, l_max = float(df_f['Latitude_degN'].min()), float(df_f['Latitude_degN'].max())
            ln_min, ln_max = float(df_f['Longitude_degE'].min()), float(df_f['Longitude_degE'].max())
            
            sel_lat = st.slider("Latitude Range", l_min, l_max, (l_min, l_max))
            sel_lon = st.slider("Longitude Range", ln_min, ln_max, (ln_min, ln_max))
            
            df_f = df_f[
                (df_f['Latitude_degN'] >= sel_lat[0]) & (df_f['Latitude_degN'] <= sel_lat[1]) &
                (df_f['Longitude_degE'] >= sel_lon[0]) & (df_f['Longitude_degE'] <= sel_lon[1])
            ]

    # パラメータ設定
    target_col = st.sidebar.radio("Target Parameter", ["d18O", "Salinity", "Temperature_degC", "dD"])
    x_axis_option = st.sidebar.selectbox("X-axis for Section", ["Longitude_degE", "Latitude_degN", "Distance_km"])
    
    # 距離計算
    if x_axis_option == "Distance_km" and not df_f.empty:
        # 起点を一意に定めるため、ソートしてから計算
        df_f = df_f.sort_values(['Latitude_degN', 'Longitude_degE'])
        b_lat, b_lon = df_f.iloc[0]['Latitude_degN'], df_f.iloc[0]['Longitude_degE']
        df_f['Distance_km'] = np.sqrt(
            ((df_f['Latitude_degN'] - b_lat) * 111.1)**2 +
            ((df_f['Longitude_degE'] - b_lon) * 111.1 * np.cos(np.radians(b_lat)))**2
        )

    df_plot = df_f.dropna(subset=[x_axis_option, 'Depth_m', target_col])

    # スケール設定
    valid_vals = df_plot[target_col]
    d_min = float(valid_vals.min()) if not valid_vals.empty else -10.0
    d_max = float(valid_vals.max()) if not valid_vals.empty else 35.0
    z_min, z_max = st.sidebar.slider(f"{target_col} Scale", -15.0, 40.0, (d_min, d_max))
    smoothness = st.sidebar.slider("Smoothing", 0.0, 5.0, 2.0)
    grid_res = st.sidebar.select_slider("Resolution", options=[10, 30, 50, 70, 90, 110, 130,150], value=70)


    st.write(f"Filtered Data: `{len(df_f)}` rows")
    
    if len(df_plot) > 5:
        xi = np.linspace(df_plot[x_axis_option].min(), df_plot[x_axis_option].max(), grid_res)
        yi = np.linspace(df_plot['Depth_m'].min(), df_plot['Depth_m'].max(), grid_res)
        X, Y = np.meshgrid(xi, yi)

        try:
            # Cubic補間 + 線形補間（外挿補完用）
            Z = griddata((df_plot[x_axis_option], df_plot['Depth_m']), df_plot[target_col], (X, Y), method='cubic')
            Z_lin = griddata((df_plot[x_axis_option], df_plot['Depth_m']), df_plot[target_col], (X, Y), method='linear')
            Z = np.where(np.isnan(Z), Z_lin, Z)
            
            if smoothness > 0:
                Z = gaussian_filter(Z, sigma=smoothness)
            
            t1, t2 = st.tabs(["Color", "Line"])
            with t1:
                st.plotly_chart(create_section_plot(Z, xi, yi, df_plot, x_axis_option, target_col, z_min, z_max, "color"), use_container_width=True)
            with t2:
                st.plotly_chart(create_section_plot(Z, xi, yi, df_plot, x_axis_option, target_col, z_min, z_max, "line"), use_container_width=True)
        except Exception as e:
            st.error(f"Interpolation error: {e}")
    else:
        st.warning("Insufficient points for the selected filters.")

    st.markdown("---")
    st.subheader("Station Map")
    if not df_plot.empty:
        # configでscrollZoomを明示的に有効化
        st.plotly_chart(
            create_station_map(df_plot, x_axis_option),
            use_container_width=True,
            config={'scrollZoom': True}
        )

if __name__ == "__main__":
    main()
