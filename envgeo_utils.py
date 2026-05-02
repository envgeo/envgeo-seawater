#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:15:03 2023
@author: Toyoho Ishimura @Kyoto-U
2026/05/01 update (Earthquake map)
"""

# --- App version / バージョン情報 ---
# version = "1.0.0a_stable_20260324" #2026/03/24
version = "1.0.4e_Earthquake_20260501" #2026/05/01


import pandas as pd
import streamlit as st
import numpy as np
import math
import json
from datetime import date, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import warnings # for M1/M2 Mac
# Shapely の内部計算（intersects, intersection, buffer等）から出る
# すべての RuntimeWarning を一括で非表示にする
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
warnings.filterwarnings("ignore", message="invalid value encountered in") # メッセージ指定でも念押し


"""
##############################################################################
# PANDAS CONFIGURATION: Optimized Memory Management
##############################################################################
"""

pd.options.mode.copy_on_write = True



"""
##############################################################################
# --- 0. Common definitions for dataset ---
##############################################################################
"""

# --- Radio button selections / ラジオボタン選択肢 ---

data_source_JAPAN_SEA    = "Kodama et al. (2024) [ECS - Japan Sea]"
data_source_AROUND_JAPAN = "with [Around Japan]"
data_source_GLOBAL       = "with [Global data sets]"
data_source_USGS_EARTHQUAKE = "USGS Earthquake Catalog [API]"


DATA_SOURCES = [
    data_source_JAPAN_SEA,
    data_source_AROUND_JAPAN,
    data_source_GLOBAL,
    data_source_USGS_EARTHQUAKE,
]



# DATA ATTRIBUTION & CITATIONS (For UI Display) / データ出典と引用表示

# --- Japan Sea: Samples analyzed by T. Ishimura using unified methods/standards ---
# Kodama et al.(2024) + upcoming reports
refs_JAPAN_SEA= ':blue[Data source:]  Kodama et al. (2024)' # To be updated

# --- Around Japan: Regional compilation ---
# Kodama et al.(2024) + around Japan 
refs_AROUND_JAPAN = ':blue[Data source:]Kodama et al. (2024), Yamamoto et al. (2001), Sakamoto et al. (2019), Kodaira et al. (2016), Horikawa et al. (2023).'

# --- Global: Comprehensive integration of international databases ---
# NASA GISS + CoralHydro2k + recent regional reports
refs_GLOBAL = ':blue[Data source:] Kodama et al. (2024), Yamamoto et al. (2001), Sakamoto et al. (2019), Kodaira et al. (2016), Horikawa et al. (2023).\
                Sakamoto et al. (2022).\
                :blue[Integrated with:] NASA GISS Global Seawater d18O Database (Jan 23, 2025)\
                :blue[and] CoralHydro2k d18O Database (Atwood et al., 2026; v1.0.0)'

refs_USGS_EARTHQUAKE = ':blue[Data source:] USGS Earthquake Catalog API (GeoJSON, eventtype=earthquake).'



"""
##############################################################################
# --- 1. Load main dataset ---
# Cache isotope data for rapid access (Japan/Global)
##############################################################################
"""
@st.cache_data
def load_isotope_data(ref_data, sheet_num=0): 
    """
    Load isotope datasets based on the selected reference source.
    Results are cached to ensure near-instantaneous retrieval on subsequent calls.
    Args:
        ref_data (str): Identifier for the data source (e.g., Japan Sea, Global).
        sheet_num (int): Index of the Excel sheet to load. Defaults to 0.
    Returns:
        pd.DataFrame: Loaded dataset.
    """

    
    # Select the source file by dataset / データセットに応じて読み込みファイルを選択
    #############################################################
    # Excel FIle
    #############################################################
    
    # ECS-Japan Sea
    file_01 = 'dataset/01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx'
    # around Japan
    file_02 = 'dataset/11_AROUND_JAPAN_PUB_20260305.xlsx'
    
    # Global
    file_03 = 'dataset/71_GLOBA_NASA_20260226.xlsx'
    file_04 = 'dataset/71_GLOBAL_Atwood_et_al_2026.xlsx' 
    file_05 = 'dataset/72_GLOBAL_RECENT_REPORTS_20260302.xlsx'
    
    # Reserved for unpublished user data / 未公表ユーザーデータ用
    file_unpub_91 = 'dataset/91_USER_UPLOAD_UNPUB.xlsx'
    # file_unpub_91 = 'dataset/91_AROUND_JAPAN_UNPUB_20260228.xlsx'
    # file_unpub_91 = 'dataset/91_AROUND_JAPAN_UNPUB_20260314_SGW.xlsx'
    
    
    #########################################################################
    # DATA INGESTION & CATEGORIZATION
    # Define 'Dataset' column for UI filtering.
    # Note: 'Dataset' refers to the UI category, not necessarily the original source.
    #########################################################################
    df1 = pd.read_excel(file_01)
    df1['Dataset'] = 'Around Japan'
    
    df2 = pd.read_excel(file_02)
    df2['Dataset'] = 'Around Japan'
    
    df3 = pd.read_excel(file_03)
    df3['Dataset'] = 'Global (NASA GISS)'
    
    df4 = pd.read_excel(file_04)
    df4['Dataset'] = 'Global (CoralHydro2k)'
    
    df5 = pd.read_excel(file_05)
    df5['Dataset'] = 'Global (other reports)'
    
    df_unpub_91 = pd.read_excel(file_unpub_91)
    df_unpub_91['Dataset'] = 'Unpublished dataset'




    #########################################################################
    # DATA SOURCE INTEGRATION: Toggle Public DB vs. User Data inclusion
    # This section manages the merging of standardized datasets with custom 
    # user inputs via manual configuration.
    #########################################################################

    """ public """

    # if ref_data == data_source_JAPAN_SEA:  
    #     df =  pd.concat([df1], ignore_index=True)
        
    # elif ref_data == data_source_AROUND_JAPAN:  
    #     df = pd.concat([df1, df2], ignore_index=True)
        
        
    # elif ref_data == data_source_GLOBAL:  
    #     df = pd.concat([df1, df2,df3,df4, df5], ignore_index=True)
        
    # else:
    #     return pd.DataFrame() # Return empty DF as fallback
  


    
    """  including user dataset """
    
    if ref_data == data_source_JAPAN_SEA:  
        df =  pd.concat([df1,df_unpub_91], ignore_index=True)
        
    elif ref_data == data_source_AROUND_JAPAN:  
        df = pd.concat([df1, df2, df_unpub_91], ignore_index=True)

        
    elif ref_data == data_source_GLOBAL: 
        df = pd.concat([df1, df2,df3,df4, df5, df_unpub_91], ignore_index=True)
        
    else:
        return pd.DataFrame()  # Return empty DF as fallback







    #########################################################################
    # DATA LOADING & CLEANING
    # Update (2026/02/23): Enhanced compatibility for depth profiles
    #########################################################################
    try:
        # Replace placeholders ('**') with NaN / プレースホルダ ('**') をNaNへ置換する
        df = df.replace('**', np.nan)
        
        # Enforce numeric conversion for safety (applies to all datasets) / 安全のため数値列を明示的に数値化する
        target_cols = ['d18O', 'dD', 'Longitude_degE', 'Latitude_degN', 
                       'Depth_m', 'Temperature_degC', 'Salinity']
        
        for col in target_cols:
            if col in df.columns:
                # Use errors='coerce' to turn non-numeric values (e.g., whitespace) into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize categorical columns as strings / カテゴリ列を文字列として標準化する
        str_cols = ['Station', 'Date', 'Cruise', "Transect", "reference"]
        for col in str_cols:
            if col in df.columns:
                # Using None instead of an empty string facilitates gap detection in plots
                df[col] = df[col].astype(str).replace('nan', None) 
                
        return df # Return full cleaned dataframe without dropping rows

    except Exception as e:
        # Display a detailed error message for troubleshooting / 詳細エラーを表示する
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()



"""
##############################################################################
# --- 2. LOAD COASTLINE DATA (Global or Japan) ---
# Fetch geographic boundaries for mapping based on the selected data source.
##############################################################################
"""
@st.cache_data
def load_coastline_data(ref_data):
    
    # 1. Select the source file (currently 50 m for all regions) / 1. 読み込みファイルを選択する（現状は全地域で50m解像度）
    if ref_data == data_source_GLOBAL:
        coastline_excel = 'coastline/world_coastline_coordinates_50m.xlsx'
    else:
        # Note: Regional settings (e.g., Japan Sea) currently utilize the global file　/ 注: 日本海など地域設定でも現状はグローバル海岸線ファイルを流用している
        # coastline_excel = 'coastline/japan_coast_line.xlsx'
        coastline_excel = 'coastline/world_coastline_coordinates_50m.xlsx'
        
    
    # 2. Process data loading / 2. 海岸線データを読み込む
    try:
        df_coast = pd.read_excel(coastline_excel)
        return df_coast['Longitude'].tolist(), df_coast['Latitude'].tolist()
    except Exception as e:
        st.error(f"Failed to load the file.: {coastline_excel} - {e}")
        return [], []


"""
##############################################################################
# --- 2b. LOAD USGS EARTHQUAKE CATALOG DATA ---
# Fetch hypocenter data from the USGS FDSN Event Web Service and normalize it
# into an EnvGeo-like dataframe for 4D mapping.
##############################################################################
"""

USGS_EARTHQUAKE_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def _format_usgs_datetime(value):
    """
    Convert date/datetime-like values to ISO8601 strings accepted by the USGS API.
    """
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")

    if isinstance(value, date):
        return value.isoformat()

    return str(value)


def _optional_usgs_param(params, key, value):
    """
    Add an API parameter only when the value is meaningful.
    """
    if value is None:
        return
    if isinstance(value, float) and math.isnan(value):
        return
    params[key] = value


def usgs_geojson_to_dataframe(payload):
    """
    Normalize USGS GeoJSON FeatureCollection records into a dataframe.

    USGS coordinates are ordered as [longitude, latitude, depth_km].
    """
    columns = [
        "EventID", "DateTime_UTC", "Time_UTC", "Updated_UTC",
        "Longitude_degE", "Latitude_degN", "Depth_km", "Depth_m",
        "Magnitude", "MagnitudeType", "Place", "Tsunami", "Alert",
        "Status", "URL", "DetailURL", "Dataset", "reference",
        "Year", "Month", "Day", "Hour", "Date", "lat", "lon",
    ]

    features = payload.get("features", []) if isinstance(payload, dict) else []
    if not features:
        return pd.DataFrame(columns=columns)

    records = []
    for feature in features:
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or [None, None, None]

        lon = coords[0] if len(coords) > 0 else None
        lat = coords[1] if len(coords) > 1 else None
        depth_km = coords[2] if len(coords) > 2 else None

        records.append({
            "EventID": feature.get("id"),
            "Time_ms": props.get("time"),
            "Updated_ms": props.get("updated"),
            "Longitude_degE": lon,
            "Latitude_degN": lat,
            "Depth_km": depth_km,
            "Magnitude": props.get("mag"),
            "MagnitudeType": props.get("magType"),
            "Place": props.get("place"),
            "Tsunami": props.get("tsunami"),
            "Alert": props.get("alert"),
            "Status": props.get("status"),
            "URL": props.get("url"),
            "DetailURL": props.get("detail"),
        })

    df = pd.DataFrame(records)

    numeric_cols = [
        "Longitude_degE", "Latitude_degN", "Depth_km", "Magnitude",
        "Tsunami", "Time_ms", "Updated_ms",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Depth_m"] = df["Depth_km"] * 1000.0
    df["Time_UTC"] = pd.to_datetime(df["Time_ms"], unit="ms", utc=True, errors="coerce")
    df["Updated_UTC"] = pd.to_datetime(df["Updated_ms"], unit="ms", utc=True, errors="coerce")
    df["DateTime_UTC"] = df["Time_UTC"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    df["Date"] = df["Time_UTC"].dt.strftime("%Y-%m-%d")
    df["Year"] = df["Time_UTC"].dt.year.astype("Int64")
    df["Month"] = df["Time_UTC"].dt.month.astype("Int64")
    df["Day"] = df["Time_UTC"].dt.day.astype("Int64")
    df["Hour"] = df["Time_UTC"].dt.hour.astype("Int64")
    df["lat"] = df["Latitude_degN"]
    df["lon"] = df["Longitude_degE"]
    df["Dataset"] = "USGS Earthquake Catalog"
    df["reference"] = "USGS Earthquake Catalog API"

    df = df.drop(columns=["Time_ms", "Updated_ms"], errors="ignore")
    df = df[columns]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_usgs_earthquake_data(
    starttime,
    endtime,
    minmagnitude=None,
    maxmagnitude=None,
    mindepth=None,
    maxdepth=None,
    minlatitude=None,
    maxlatitude=None,
    minlongitude=None,
    maxlongitude=None,
    limit=2000,
    orderby="time",
):
    """
    Fetch earthquake hypocenter data from the USGS API and return a dataframe.
    """
    params = {
        "format": "geojson",
        "eventtype": "earthquake",
        "starttime": _format_usgs_datetime(starttime),
        "endtime": _format_usgs_datetime(endtime),
        "orderby": orderby,
        "limit": int(limit),
    }

    optional_params = {
        "minmagnitude": minmagnitude,
        "maxmagnitude": maxmagnitude,
        "mindepth": mindepth,
        "maxdepth": maxdepth,
        "minlatitude": minlatitude,
        "maxlatitude": maxlatitude,
        "minlongitude": minlongitude,
        "maxlongitude": maxlongitude,
    }
    for key, value in optional_params.items():
        _optional_usgs_param(params, key, value)

    query_url = f"{USGS_EARTHQUAKE_QUERY_URL}?{urlencode(params)}"
    request = Request(
        query_url,
        headers={"User-Agent": "EnvGeoSeawater earthquake visualizer"},
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as e:
        raise RuntimeError(f"USGS API error ({e.code}): {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"USGS API connection error: {e.reason}") from e
    except TimeoutError as e:
        raise RuntimeError("USGS API request timed out.") from e

    df = usgs_geojson_to_dataframe(payload)
    df.attrs["query_url"] = query_url
    return df
    



"""
##############################################################################
# --- 3. UNIFIED LAYOUT CONFIGURATION ---
# Apply consistent styling and region-specific perspectives to Plotly figures.
##############################################################################
"""

def apply_common_layout(fig, ref_data, z_min, z_max, x_range=None, y_range=None):
    """
    Applies unified visual styling and dynamic camera perspectives 
    to all Plotly 3D plots based on the selected dataset.
    """
    is_Global = (ref_data == data_source_GLOBAL)
    
    # --- 1. Perspective & Aspect Ratio Configuration --- / 視点とアスペクト比の設定 ---
    if is_Global:
        # [Global View] High-altitude perspective overlooking Japan from the Pacific
        camera_setting = dict(
            eye=dict(x=1.2, y=-0.8, z=2.2), # High Z-value for a bird's-eye view
            center=dict(x=0, y=0, z=-0.1)
        )
        target_aspectratio = dict(x=2, y=1, z=0.5)
    else:
        # [Regional View] Lower-altitude perspective centered over Japan (Southwest tilt)
        camera_setting = dict(
            eye=dict(x=-0.8, y=-0.8, z=2.2), # Closer to vertical for detailed regional view
            center=dict(x=0, y=0, z=-0.1)
        )
        target_aspectratio = dict(x=1, y=1, z=1)

    # --- 2. Scene Definition ---
    scene_dict = dict(
        aspectmode='manual',
        aspectratio=target_aspectratio,
        zaxis=dict(range=[z_min, z_max]),
        xaxis_title='Longitude E',
        yaxis_title='Latitude N',
        zaxis_title='Water Depth',
        camera=camera_setting
    )
    
    # Apply axis constraints (if specific ranges are provided for regional filtering) / 明示的な範囲指定がある場合は軸範囲を適用する
    if x_range:
        scene_dict['xaxis'] = dict(range=x_range)
    if y_range:
        scene_dict['yaxis'] = dict(range=y_range)
        
    # --- 3. Final Layout Update ---
    fig.update_layout(
        scene=scene_dict,
        width=700,
        height=600,
        margin=dict(r=20, l=10, b=10, t=10)
    )
    
    return fig




"""
##############################################################################
# --- 4. DYNAMIC COLORSCALE SELECTION ---
# Automatically apply depth-optimized colorscales when depth-related 
# keywords (e.g., 'Depth_m') are detected in the data.
##############################################################################
"""



def get_custom_colorscale(selected_item):
    """
    Returns a custom colorscale tailored to the selected parameter.
    For depth-related items, the scale is optimized to highlight subtle 
    variations in shallow layers.
    """
    
    # 1. Standard colorscale (Default for non-depth parameters) / 1. 標準カラースケール（深度以外の既定値）
    standard_scale = [
        'darkblue', 'blue', 'lightblue', 'lightgreen', 
        'green', 'yellow', 'orange', 'red'
    ]

    # 2. Depth-optimized scale with enhanced resolution for shallow waters / 2. 浅海の変化を見やすくした深度用カラースケール
    # Colors are mapped to normalized values (0.0 to 1.0) / 色は0.0から1.0の正規化値に対応させる
    # The gradients are compressed between 0.0 and 0.3 to maximize / 0.0から0.3に勾配を圧縮して
    # visual contrast in the upper water column (shallow layers) / 表層から浅層のコントラストを強める
    
    depth_keywords = ['Depth_m', 'Water Depth', 'depth']
    
    if selected_item in depth_keywords:
        depth_scale = [
            [0.0, 'red'],         # Surface
            [0.02, 'pink'],
            [0.05, 'orange'],
            [0.1, 'yellow'],
            [0.15, 'lightgreen'],
            [0.3, 'lightblue'],
            [0.6, 'blue'],
            [1.0, 'darkblue']     # Deep water
        ]
        return depth_scale
    
    return standard_scale




"""
##############################################################################
# --- 5. MAP STYLE CONFIGURATION ---
# Apply background tile layers to the Plotly Mapbox figure.
##############################################################################
"""

def apply_map_style(fig, map_mode):
    """
    Apply the selected background tile layer to the Mapbox figure.
    All tile sources have been verified for web-use licensing.
    """
    
    fig.update_layout(mapbox_style="carto-positron")

    if map_mode == "Standard":
        fig.update_layout(mapbox_style="carto-positron")
        
    
    elif map_mode == "Satellite":
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[{
                "below": 'traces',
                "sourcetype": "raster",
                "source": [
                    # USGS
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"],
                "sourceattribution": "USGS"
            }]
        )
        
    elif map_mode == "Bathymetry (Sea)":
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "source": [
                    # Esri World Ocean Base 
                    "https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
                ],
                "sourceattribution": "Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri"
            }]
        )
        
    elif map_mode == "Contour (GSI)":
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[{
                "below": 'traces',
                "sourcetype": "raster",
                "source": [
                    # Geospatial Information Authority of Japan (GSI) tiles: 
                    # High-detail topographic data that scales dynamically.
                    "https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png"],
                "sourceattribution": "国土地理院 (GSI)"
            }]
        )
        
    
    # Unified layout settings for maximum map area
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    
    return fig



"""
##############################################################################
# --- 6. CACHE MANAGEMENT ---
# Utility functions to manage and reset Streamlit's data cache.
##############################################################################
"""

# 暫定版
# Provisional implementation
def clear_app_cache():
    """
    Clears all cached data across the application to ensure data consistency.
    """
    st.cache_data.clear()



"""
##############################################################################
# --- 7. DATA SEGMENTATION FOR DEPTH PROFILES ---
# Group data by coordinates and date, then insert NaN rows to break 
# line connections in 3D visualizations. CRITICAL STEP.
##############################################################################
"""

def insert_gap_rows(df):
    """
    Inserts blank (NaN-filled) rows at the boundaries of observation groups
    to prevent visual artifacts (unintended line connections) in plots.
    """
    # [Safety measure] Reset index to ensure sequential processing / [安全対策] 連続処理のためにインデックスを振り直す
    df = df.reset_index(drop=True)

    # Columns used to define a unique observation group / グループを識別する列
    check_cols = ['Latitude_degN', 'Longitude_degE', 'Year', 'Month', 'reference']
    

    # Create a temporary dataframe for boundary detection / 境界検出用の一時DataFrameを作成する
    tmp_check = df[check_cols].copy()
    
    # [Crucial] Convert NaN to strings consistently / [重要] NaNを一貫した文字列に変換する
    # This keeps grouping stable even for datasets with missing metadata (e.g., NASA) / メタデータ欠損を含むデータセットでも安定してグループ化できる
    for col in check_cols:
        tmp_check[col] = tmp_check[col].fillna('UNKNOWN').astype(str)

    # Detect row-to-row transitions: does the current row differ from the previous one? / 前の行と異なるかどうかでグループ境界を検出する
    is_new_group = tmp_check.ne(tmp_check.shift()).any(axis=1)

    # Get indices for group starts (excluding the first row) / 先頭行を除くグループ開始位置を取得する
    new_group_indices = df.index[is_new_group].tolist()
    if 0 in new_group_indices:
        new_group_indices.remove(0)


    # 空白行を挿入（インデックスに0.5を足して間に挟み込む）
    # Create placeholder rows with fractional indices to insert them in between
    # Using 0.5 offset ensures they are sorted correctly between original rows
    # Concatenate and sort to weave the blank rows into the dataset / 連結して並べ替え、空白行をデータ列の間に差し込む
    gap_indices = [i - 0.5 for i in new_group_indices]
    new_index = sorted(df.index.tolist() + gap_indices)
    
    df_final = df.reindex(new_index).reset_index(drop=True)
    
    return df_final

   


"""
##############################################################################
# --- 8. DATA TABLE VISUALIZATION ---
# Formats and displays the filtered dataframe within a Streamlit expander.
##############################################################################
"""


def display_isotope_table(df, title="Sidebar-filtered dataset (CSV)"):
    """
    Format and render the dataframe in a Streamlit expander.
    Includes integer conversion for dates and string-casting to prevent Arrow errors.
    """
    with st.expander(title, expanded=False):
        # Define priority columns (Safety check included for missing columns) / 欠損列に配慮しつつ優先列を定義する
        target_cols = [
            'reference','Cruise', 'Station', 'Date', 'Year', 'Month', 
            'Longitude_degE', 'Latitude_degN', 'Depth_m', 
            'Temperature_degC', 'Salinity', 'd18O', 'dD'
        ]
        
        # Extract only existing columns to avoid KeyError / KeyErrorを避けるため存在する列だけを抜き出す
        available_cols = [c for c in target_cols if c in df.columns]
        df_display = df[available_cols].copy()
        
        # 年と月を整数型に変換
        # Convert Year and Month to nullable integers
        for col in ['Year', 'Month']:
            if col in df_display.columns:
                # 数値化できないものはNaNにし、その上でInt64型へ
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce').astype('Int64')
        
        # Arrowエラー対策：全列を文字列化
        # [Arrow Serialization Fix] Cast all columns to strings to ensure UI stability
        df_display = df_display.astype(str)
        
        # Cleanup visual representation of missing values
        # df_display = df_display.replace('<NA>', '')
        df_display = df_display.replace(['<NA>', 'nan', 'None'], '')
        
        # Render table 
        st.dataframe(df_display,
                # use_container_width=True
                )






        # =============================================================================
        # USAGE EXAMPLES (External Module Calls)
        # =============================================================================
        # Import this utility via: import envgeo_utils as utils
        
        #
        # Note: Specialized visualizers (e.g., 4D or Depth Profile) may require 
        # custom handling for derived parameters like d-excess or gap rows.
        # 
        
        # 1. Standard dataset preview:
        # 各ファイルでの呼び出し例は以下
        # utils.display_isotope_table(df1)
        
        # 2. 4D Visualizer: 
        # Requires specific handling for derived parameters (e.g., d-excess calculation).
        # 4D visualizerは個別対応必要，d-exessを追加してあるので
        
        # 3. Depth Profile Visualizer: 
        # Note that gap rows (NaN rows) are utilized to prevent line connections, 
        # and empty rows may be filtered out before rendering.
        # depth_profileも個別対応必要。空いている行を削除しているので
        


"""
##############################################################################
# --- 9. DATA FILTERING & STATISTICAL SUMMARY ---
# Handle sidebar-driven data extraction and display selection metrics.
# Note: Visual styling for figures is managed in the main script.
# --- 図の調整はメインスクリプトに記載 ---
##############################################################################
"""
# ポイントは | df[col].isna() を加えることで、フィルタリング時に空白行を常に救い出す点
# Apply filters while exempting NaN rows (Gap Rows) to preserve data segmentation.


def sidebar_filter_and_display(df1, ref_data, data_source_JAPAN_SEA, data_source_AROUND_JAPAN):
    """
    サイドバーのフィルター設定、データ抽出、および選択データの統計表示を一括で行う関数。
    引数:
        df1: 元のDataFrame
        ref_data: 現在選択されているデータソース
        data_source_JAPAN_SEA: 日本海ソースの識別値
        data_source_AROUND_JAPAN: 日本周辺ソースの識別値
    戻り値:
        フィルタリング後のdf1, および地図・カラーバー用の各設定値
    """
    """
    Executes sidebar-based filtering, data extraction, and summary statistics.
    
    CRITICAL LOGIC: 
    Filter conditions include 'df[col].isna()' to preserve placeholder blank rows,
    ensuring depth profiles remain correctly segmented in 3D visualizations.

    Args:
        df1 (pd.DataFrame): The original dataset.
        ref_data (str): Current active data source identifier.
        data_source_JAPAN_SEA: Constant for Japan Sea dataset.
        data_source_AROUND_JAPAN: Constant for Around Japan dataset.

    Returns:
        tuple: (filtered_df, map_settings, colorscale_configs)
    """
    

    ##############################################################################
    # --- SIDEBAR CONFIGURATION AND INTEGRATED FILTERING / サイドバー設定と統合フィルタリング ---
    # Update (2026/03/06): switched to dynamic min-max acquisition directly / 最小値・最大値をDataFrameから動的取得する方式へ変更
    # from the dataframe to define filter ranges / フィルタ範囲をDataFrameから直接決める
    # Note: spatial coordinates (Lat/Lon) are kept in raw form to maintain precision / 注: 緯度経度は精度保持のため元の値で扱う
    ##############################################################################


    with st.sidebar.form("parameter", clear_on_submit=False):
        
        
        st.header(':blue[--- Data filtering ---]')
        
        st.form_submit_button(":red[submit]")

        # Two buttons can be placed at the top and bottom if needed / 必要ならsubmitボタンを上下に配置できる
        # st.form_submit_button(":red[submit (TOP)]")
        # submitted = st.form_submit_button(":red[submit (BOTTOM)]")

        #　一つだけの時は以下
        # submitted = st.form_submit_button(":red[submit]")
        
        

        ##########################
        # Dataset filtering
        ##########################
        # st.sidebar.subheader('航海区の範囲')dfから要素抽出
        
        # 1. 空欄（欠損値）を "no_name" に置き換える
        df1["Dataset"] = df1["Dataset"].fillna("no_name")
        
        # ※もし前の処理で 'nan' や 'None' という「文字列」になっている場合の念押し安全対策
        df1["Dataset"] = df1["Dataset"].replace({'nan': 'no_name', 'None': 'no_name', '': 'no_name'})

        # 2. 【変更】 .dropna() をしない、"no_name" もリストに含めるようにする
        Transect_list = df1["Dataset"].unique().tolist()
        # print(Transect_list, "<< Dataset list")
        
        
        # 3. マルチセレクトの作成
        with st.expander("Select sub-dataset", expanded=False):
            # st.sidebar.subheader('航海区の範囲')dfから要素抽出
            Transect_list = df1["Dataset"].dropna().unique().tolist()
            # print(Transect_list,"<<< Dataset list")
            
            selected_dataset = st.multiselect('Choose datasets', Transect_list,default=Transect_list)

            
        # datasetのフィルタリング　2026/03/09追加
        df1 = df1[df1["Dataset"].isin(selected_dataset)
                   | df1['Dataset'].isna()]  # ← 【修正】Datasetが空欄（または空白行）なら残す
        
        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()
            
        
        ##########################
        # Transect filtering
        ##########################
        # st.sidebar.subheader('航海区の範囲')dfから要素抽出
        
        # 1. 【追加】Transect列の空欄（欠損値）を "no_name" に置き換える
        df1["Transect"] = df1["Transect"].fillna("no_name")
        
        # ※もし前の処理で 'nan' や 'None' という「文字列」になっている場合の念押し安全対策
        df1["Transect"] = df1["Transect"].replace({'nan': 'no_name', 'None': 'no_name', '': 'no_name'})

        # 2. 【変更】 .dropna() をしない、"no_name" もリストに含めるようにする
        Transect_list = df1["Transect"].unique().tolist()
        # print(Transect_list, "AAA")
        
        
        # 3. マルチセレクトの作成
        with st.expander("Area / Transect", expanded=False):
            # st.sidebar.subheader('航海区の範囲')dfから要素抽出
            Transect_list = df1["Transect"].dropna().unique().tolist()
            # print(Transect_list,"<<< Transect list")
            
            selected_cruise = st.multiselect('Cruise / Area / Transect', Transect_list,default=Transect_list)
        

        # --- 航海区（Transect）の範囲 ---　2026/03/06修正済み
        # #streamlitのマルチ選択用
        df1 = df1[(df1['Transect'].isin(selected_cruise))
                   | df1['Transect'].isna()]  # ← 【修正】Transectが空欄（または空白行）なら残す
    
        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()
            
    
  
        ##########################
        # Map of Transects in Kodama et al (2024)
        ##########################
        with st.expander("Area map: Kodama et al.(2024)", expanded=False):
            st.write('Cruise tracks and study area (2015–2021)')
            st.caption('Click top right to expand.')
            st.image("data/sites_20230515.gif")


          
        
                
        ##########################
        # Station filtering
        ##########################
        # st.sidebar.subheader('航海区の範囲')dfから要素抽出
        
        # 1. Station列の空欄（欠損値）を "no_name" に置き換える
        df1["Station"] = df1["Station"].fillna("no_name")
        
        # ※もし前の処理で 'nan' や 'None' という「文字列」になっている場合の念押し安全対策
        df1["Station"] = df1["Station"].replace({'nan': 'no_name', 'None': 'no_name', '': 'no_name'})

        # 2. 【変更】 .dropna() をしない、"no_name" もリストに含めるようにする
        Station_list = df1["Station"].unique().tolist()
        # print(Station_list, "AAA")
        
        
        # 3. マルチセレクトの作成
        with st.expander("Station", expanded=False):
            # st.sidebar.subheader('Stationの範囲')dfから要素抽出
            Transect_list = df1["Station"].dropna().unique().tolist()
            # print(Station_list,"<<< Station list")
            
            selected_Station = st.multiselect('Station', Station_list,default=Station_list)
        

        # --- 地点（Station）の範囲 ---　2026/04/05 追加
        # #streamlitのマルチ選択用
        df1 = df1[(df1['Station'].isin(selected_Station))
                   | df1['Station'].isna()]  # ← 【修正】Stationが空欄（または空白行）なら残す
    
        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()
            


          
          
          

        ##########################
        # Year filtering
        ##########################
        # max_df_year = int(df1['Year'].max())
        # min_df_year = int(df1['Year'].min())
        # sld_year_min, sld_year_max = st.slider(label='Year',
        #                             min_value=min_df_year,
        #                             max_value=max_df_year,
        #                             value=(min_df_year, max_df_year),
        #                             )
        
        min_df_year = int(df1['Year'].dropna().min())
        max_df_year = int(df1['Year'].dropna().max())
        
        # 最小と最大が同じ場合、エラー回避のために範囲を広げる
        if min_df_year == max_df_year:
            slider_min = min_df_year - 1
            slider_max = max_df_year + 1
        else:
            slider_min = min_df_year
            slider_max = max_df_year
        
        sld_year_min, sld_year_max = st.slider(
            label='Year',
            min_value=slider_min,
            max_value=slider_max,
            value=(min_df_year, max_df_year) # 初期値は実際のデータ範囲にする
        )
        

        # --- 年の範囲 --- 修正済み
        df1 = df1[
            ((df1['Year'] >= sld_year_min) & (df1['Year'] <= sld_year_max))
            | df1['Year'].isna()
        ]
    
        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()

        ##########################
        # Month filtering ---multiselect---
        ##########################
        # selected_months = st.multiselect(
        #     label='Month',
        #     options=list(range(1, 13)),  # 1〜12の選択肢
        #     default=list(range(1, 13))   # 初期状態は全選択
        # )
        
        month_list = list(range(1, 13))
        
        # セグメントコントロールの設定
        selected_months = st.segmented_control(
            label="Month",
            options=month_list,
            selection_mode="multi",
            default=month_list  # 初期状態で全選択にする
        )
                
        
  
            
        # --- 月 (スライダー用) ---　2026/03/06修正済み
        # df1 = df1[(df1['Month'] == 'xxx')
                    
        #             |(df1['Month'] <= sld_month_max) & (df1['Month'] >= sld_month_min)
        #             | df1['Month'].isna()]  # ← 【修正】
          
        
        # --- 月 (multiselect用) ---　2026/03/06修正済み
        if selected_months:
            # isin で選ばれた月を抽出
            # | (または)
            # df1['Month'].isna() でMonthが空の行（挿入した空白行 ＋ 月が不明なNASAデータ）を抽出
            df1 = df1[df1['Month'].isin(selected_months) | df1['Month'].isna()]
        else:
            # 月が一つも選ばれていない場合でも、空白行や月不明データだけは残す
            df1 = df1[df1['Month'].isna()]
            
        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()
    
    

        ##########################
        # Longitude filtering
        ##########################
        
        # Safely compute the minimum and maximum, then floor/ceil them / 最小値・最大値を安全に取得し、切り下げ・切り上げする
        # 1. データの最小値・最大値を安全に取得し、切り下げ・切り上げを行う
        # 経度は範囲が広いため、整数(int)にしておくとユーザーが操作しやすくなる
        min_df_lon = int(math.floor(df1['Longitude_degE'].min()))
        max_df_lon = int(math.ceil(df1['Longitude_degE'].max()))
        
        # 2. スライダーの設定
        sld_lon_min, sld_lon_max = st.slider(
            label='Longitude',  # ラベルを少し自然に
            min_value=min_df_lon,
            max_value=max_df_lon,
            value=(min_df_lon, max_df_lon),
            step=1
            # formatは指定しないことでエラーを回避
        )

        # --- 経度(Longitude)の範囲 --- 修正済み
        df1 = df1[
            ((df1['Longitude_degE'] >= sld_lon_min) & (df1['Longitude_degE'] <= sld_lon_max))
            | df1['Longitude_degE'].isna()
        ]
    

        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()


        ##########################
        # Latitude filtering
        ##########################
        # 小数点以下を考慮して、最小値は切り下げ、最大値は切り上げる
        # Floor the minimum and ceil the maximum to keep the slider robust / スライダーを安定させるため、最小値は切り下げ、最大値は切り上げる
        
        max_df_lat = int(math.ceil(df1['Latitude_degN'].max()))
        min_df_lat = int(math.floor(df1['Latitude_degN'].min()))
        
        # 2. スライダーの設定
        sld_lat_min, sld_lat_max = st.slider(
            label='Latitude',
            min_value=min_df_lat,
            max_value=max_df_lat,
            value=(min_df_lat, max_df_lat),
            step=1  # 整数刻みに設定
        )
        

        # --- 緯度(Latitude)の範囲 --- 修正済み
        df1 = df1[
            ((df1['Latitude_degN'] >= sld_lat_min) & (df1['Latitude_degN'] <= sld_lat_max))
            | df1['Latitude_degN'].isna()
        ]

        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()
        

        ##########################
        # water depth filtering
        ##########################

        # Floor the minimum and ceil the maximum, then use integer steps / 最小値は切り下げ、最大値は切り上げた上で整数刻みにする
        # 水深は範囲が広いため、int型に変換してスッキリ
        min_depth = int(math.floor(df1['Depth_m'].min()))
        max_depth = int(math.ceil(df1['Depth_m'].max()))
        
        # 2. スライダーの設定
        if min_depth == max_depth:
            slider_max = max_depth + 1
        else:
            slider_max = max_depth
        
        if min_depth > 0:
            default_value = (min_depth, max_depth)
        else:
            default_value = (0, max_depth)
        
        sld_depth_min, sld_depth_max = st.slider(
            label='Water Depth (m)',
            min_value=min_depth,
            max_value=slider_max,
            value=default_value,
            step=10,
        )
                
        df1 = df1[
            ((df1['Depth_m'] >= sld_depth_min) & (df1['Depth_m'] <= sld_depth_max))
            | df1['Depth_m'].isna()
        ]
        
        
        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()






        ##########################
        # salinity filtering
        ##########################
        # Use integer bounds for a simpler salinity slider / 塩分スライダーを簡潔に保つため整数範囲を使う
        
        min_df_sal = int(math.floor(df1['Salinity'].min()))
        max_df_sal = int(math.ceil(df1['Salinity'].max()))

        
        # 2. スライダーの設定
        if min_df_sal == max_df_sal:
            slider_max_sal = max_df_sal + 1
        else:
            slider_max_sal = max_df_sal
        
        if min_df_sal > 0:
            default_sal = (min_df_sal, max_df_sal)
        else:
            default_sal = (0, max_df_sal)
        
        sld_sal_min, sld_sal_max = st.slider(
            label='Salinity',
            min_value=min_df_sal,
            max_value=slider_max_sal,
            value=default_sal,
            # step=0.1,
            # format="%0.1f"  # SyntaxErrorを避けるため %0.1f と書くか、不安ならformatを消す
        )
        
        df1 = df1[
            ((df1['Salinity'] >= sld_sal_min) & (df1['Salinity'] <= sld_sal_max))
            | df1['Salinity'].isna() # ← 【修正】Salinityが空欄なら残す
        ]


        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()

        
        ##########################
        # d18O filtering (2026/03/09 最終修正)
        ##########################
        
        # 1. データの型を強制的に「数値」に洗い直す (重要：空欄を本物のNaNに変換)
        df1['d18O'] = pd.to_numeric(df1['d18O'], errors='coerce')
        
        # 2. スライダー用の最小・最大値を取得 (NaNを除外して計算)
        d18o_data = df1['d18O'].dropna()
        if not d18o_data.empty:
            min_df_d18O = float(math.floor(d18o_data.min() * 10) / 10.0)
            max_df_d18O = float(math.ceil(d18o_data.max() * 10) / 10.0)
        else:
            min_df_d18O, max_df_d18O = -10.0, 10.0

        # 3. スライダーの作成
        sld_d18O_min, sld_d18O_max = st.slider(
            label='d18O (VSMOW)',
            min_value=float(min_df_d18O - 2.0),
            max_value=float(max_df_d18O + 2.0),
            value=(float(min_df_d18O), float(max_df_d18O)),
            step=0.1,
            format="%.1f"
        )

        # 4. フィルタリングの実行 (カッコの組み合わせを厳密に)
        # 「範囲内」か「欠損値」のどちらかであれば残す
        mask_d18O = (
            ((df1['d18O'] >= sld_d18O_min) & (df1['d18O'] <= sld_d18O_max))
            | (df1['d18O'].isna())
        )
        df1 = df1[mask_d18O]
        
        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()

        # ##########################
        # # temperature filtering
        # ##########################
        # min_df_temp = float(df1['Temperature_degC'].min())
        # max_df_temp = float(df1['Temperature_degC'].max())
        
        # sld_temp_min, sld_temp_max = st.slider(
        #     label='Temperature (C)',
        #     min_value=min_df_temp,
        #     max_value=max_df_temp,
        #     value=(min_df_temp, max_df_temp),
        #     format="%.1f", # 小数点第1位まで表示する場合
        #     step=0.1  # 1刻みにすることで整数のみの選択になる
        # )

        # # --- 水温の範囲 --- 修正済み
        # df1 = df1[
        #     ((df1['Temperature_degC'] >= sld_temp_min) & (df1['Temperature_degC'] <= sld_temp_max))
        #     | df1['Temperature_degC'].isna()
        # ]
        
        
        ##########################
        # Temperature filtering with integrated safety guards / 安全対策込みの水温フィルタリング
        # (Prevents errors from missing values or non-numeric entries)
        ##########################
        # 1. 念のため数値型に変換
        df1['Temperature_degC'] = pd.to_numeric(df1['Temperature_degC'], errors='coerce')

        # 2. 有効な数値データだけを取り出す
        temp_valid = df1['Temperature_degC'].dropna()

        # 3. データが存在するかチェックして最小・最大を決める
        if not temp_valid.empty:
            min_df_temp = float(temp_valid.min())
            max_df_temp = float(temp_valid.max())
        else:
            # データが1件もない場合のデフォルト値 (エラー回避用)
            min_df_temp, max_df_temp = 0.0, 40.0

        # 4. 万が一、minとmaxが同じ値（データが1種類だけ）だとスライダーが壊れるので微調整
        if min_df_temp == max_df_temp:
            min_df_temp -= 0.1
            max_df_temp += 0.1

        # 5. スライダー作成
        sld_temp_min, sld_temp_max = st.slider(
            label='Temperature (C)',
            min_value=min_df_temp,
            max_value=max_df_temp,
            value=(min_df_temp, max_df_temp),
            format="%.1f",
            step=0.1
        )

        # 6. フィルタリングの実行（NaNは救う）
        df1 = df1[
            ((df1['Temperature_degC'] >= sld_temp_min) & (df1['Temperature_degC'] <= sld_temp_max))
            | df1['Temperature_degC'].isna()
        ]
            
        
        if df1.empty:
            st.warning("⚠️ no data found.")
            st.stop()

 
        
        submitted = st.form_submit_button(":red[submit!]")
        
    # ----------------サイドバーここまで------------------------
    
    
    
    

    ##############################################################################
    # --- DATA SELECTION METRICS (Part 1) ---
    # 選択データ統計表示 1
    ##############################################################################
    # --- バリデーション ---
    #df1が空になっているかどうかを確認する
    df_empty = df1.empty

    # st.write(df_empty)
    data_found_num = str(len(df1["Dataset"]))

    
    # バリデーション処理
    if df_empty == 1:  #データが無かったとき
        st.warning('no data found')
        # 条件を満たないときは処理を停止する
        st.stop()
    elif df_empty == 0: #データがあったとき
        st.write(data_found_num,'data found')
        
        
        
        


    ##############################################################################
    # --- DATA SELECTION METRICS (Part 2) --- (simple)
    ##############################################################################
    # with st.expander("selected data", expanded=False):
    #     month_disp = ", ".join(map(str, sorted(selected_months))) if selected_months else "None"
    #     st.write(f':green[YEAR]:{sld_year_min}-{sld_year_max}, :green[MONTH]:[{month_disp}], '
    #              f':green[Lon]:{sld_lon_min}-{sld_lon_max}, :green[Lat]:{sld_lat_min}-{sld_lat_max}, '
    #              f':green[Depth]:{sld_depth_min}-{sld_depth_max}, :green[Sal]:{sld_sal_min}-{sld_sal_max}')
    #     st.write(':green[Selected Data (Cruise)]', list(selected_cruise))
    #     st.write(':green[Selected Data (detail)]', df1["Transect"].value_counts().to_dict())
        
    #     for lbl, col, dcm in [("d18O _ave", "d18O", 3), ("Sal_ave", "Salinity", 2), ("Temp_ave", "Temperature_degC", 2)]:
    #         c1, c2, c3, c4 = st.columns(4)
    #         with c2: st.write(f'{lbl}: {round(np.nanmean(df1[col]), dcm)}')
    #         with c3: st.write(f'stdev: ± {round(np.nanstd(df1[col]), dcm)}')



    ##############################################################################
    # --- DATA SELECTION METRICS (Part 3) --- (original)
    ##############################################################################

    selected_row = "Transect"

    #列の要素を表示
    d_select_add2 = df1[selected_row].value_counts().to_dict()
    # d_select_add2_sum = df1[selected_row].count().sum()
    # print('要素と出現数:', d_select_add2)
    # print('要素と出現数:', d_select_add2_sum)
    # print('---------------')
                        
    # with表記
    with st.expander("📊 Details and statistics of sidebar-filtered data", expanded=False):

    #選んだパラメーター表示

    # When month is slider / 月がスライダーの場合
        # st.write(':green[YEAR]:'+str(sld_year_min)+'-'+str(sld_year_max)+', '
        #           +':green[MONTH]:'+str(sld_month_min)+'-'+str(sld_month_max)+', '
        #           +':green[Longitude]:'+str(sld_lon_min)+'-'+str(sld_lon_max)+', '
        #           +':green[Latitude]:'+str(sld_lat_min)+'-'+str(sld_lat_max)+', '
        #           +':green[Water_depth]:'+str(sld_depth_min)+'-'+str(sld_depth_max)+', '
        #           +':green[Salinity]:'+str(sld_sal_min)+'-'+str(sld_sal_max))
        
    # 月がマルチセレクトの場合　　リストを文字列に変換（例: [1, 2] -> "1, 2"）
    # When month is selected via multiselect, convert the list to a display string / 月を複数選択した場合は表示用の文字列に変換する　　リストを文字列に変換（例: [1, 2] -> "1, 2"）
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
                
    ##############################################################################

    


    ##############################################################################
    # RETURN ALL PROCESSED VARIABLES
    # 最後にすべての変数を網羅して返す（呼び出し側との順序不整合に注意）
    ##############################################################################
    return (
        df1,                   # フィルタリング後のデータフレーム
        # --- フィルタリング条件（後で表示や計算に使う用） ---
        sld_year_min, sld_year_max, 
        selected_months, 
        sld_lon_min, sld_lon_max, 
        sld_lat_min, sld_lat_max, 
        sld_depth_min, sld_depth_max, 
        sld_sal_min, sld_sal_max, 
        sld_d18O_min, sld_d18O_max,      # ← フィルタリングで使ったd18O範囲
        sld_temp_min, sld_temp_max, 
        selected_cruise, 
        submitted                        # フォームの送信状態
    )




    """
    # -------------------------------------------------------------------------
    # MAIN SCRIPT IMPLEMENTATION: One-stop Filtering & Sidebar Execution
    # ---- メインの各ぺーじのスクリプトでは以下のコードで呼び出すだけ -----------
    # -------------------------------------------------------------------------
    """
    
    # Utilizing envgeo_utils for integrated filtering and sidebar UI generation.
    # envgeo_utils を使って一括フィルタリングとサイドバー生成
    # import envgeo_utils
    
    
    # Execute the function and unpack the filtered results.
    # The variables are returned in the exact order defined in the utility module.


    
    # # 関数の呼び出し
    # # すべての変数を順番通りに受け取り
    
    # (df1, 
    #  sld_year_min, sld_year_max, 
    #  selected_months, 
    #  sld_lon_min, sld_lon_max, 
    #  sld_lat_min, sld_lat_max, 
    #  sld_depth_min, sld_depth_max, 
    #  sld_sal_min, sld_sal_max, 
    #  sld_d18O_min, sld_d18O_max, 
    #  sld_temp_min, sld_temp_max, 
    #  selected_cruise,
    #  submitted) = envgeo_utils.sidebar_filter_and_display(df1, ref_data, data_source_JAPAN_SEA, data_source_AROUND_JAPAN)
