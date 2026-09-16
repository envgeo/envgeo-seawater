#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utility functions for EnvGeo-Seawater.

This module keeps common app behavior in one place: dataset choices, version
metadata, data loading, quality normalization, d-excess calculation, map
styling, and reusable Streamlit table display.

EnvGeo-Seawater 共通処理モジュールです。
データセット選択肢、バージョン情報、データ読み込み、品質チェック、
d-excess 計算、地図表示設定、共通テーブル表示をここに集約します。
"""

# --- App version / バージョン情報 ---
APP_VERSION = "1.3.0"
APP_VERSION_DATE = "2026-09-11"
APP_VERSION_LABEL = f"{APP_VERSION} ({APP_VERSION_DATE})"

# Backward-compatible alias used by older pages.
# 既存ページとの互換性を保つため、従来の version 変数も残します。
version = APP_VERSION

# Shared UI labels / 共通UIラベル
FIGURE_CONTROLS_LABEL = "Figure controls"
FIGURE_SCALE_SETTINGS_LABEL = "Figure scale settings"
MAP_DISPLAY_SETTINGS_LABEL = "Map display settings"
CUSTOM_PLOT_SETTINGS_LABEL = "Custom plot settings"
DATA_RANGE_SETTINGS_LABEL = "Data range settings"
DATA_FILTERING_LABEL = "Data filtering"
MAP_AREA_HELP_TEXT = "Map center, extent, colormap, and figure settings can be adjusted in the sidebar."
AREA_FILTER_MANUAL = "Manual / full data range"
AREA_FILTER_HELP_TEXT = (
    "Choose a preset to set the initial longitude and latitude range, "
    "then fine-tune the sliders if needed."
)
COLOR_PARAMETER_HELP_TEXT = (
    "Choose the variable used to color the plotted points or map markers."
)
COLORBAR_RANGE_HELP_TEXT = (
    "Set the displayed color range. Values outside the range use the end colors."
)
BACKGROUND_DATA_HELP_TEXT = (
    "Show the unfiltered dataset behind the currently filtered data for context."
)
MAP_STYLE_HELP_TEXT = (
    "Choose the background map style for the sampling-location map."
)
REGRESSION_HELP_TEXT = (
    "Add a simple least-squares regression line for quick visual reference."
)



import pandas as pd
import streamlit as st
import numpy as np
import math
import re
import unicodedata
from datetime import datetime

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
# --- Common filename helpers / ファイル名の共通整形 ---
##############################################################################
"""


def safe_filename_text(value, fallback="figure", max_length=180):
    """Convert figure titles and filter labels to safe ASCII filename text.

    図タイトルやフィルタ条件の文字列を、保存用ファイル名として安全な形に整えます。
    """
    text = "" if value is None else str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._+=()-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")

    if not text:
        text = fallback

    return text[:max_length]


def build_figure_filename(prefix, subtitle=None, extension="png"):
    """Build a consistent download filename for figures.

    各ページの図保存ファイル名を、同じルールで作成します。
    """
    safe_prefix = safe_filename_text(prefix, fallback="figure")
    safe_extension = safe_filename_text(str(extension).lstrip("."), fallback="png").lower()

    if subtitle is None or str(subtitle).strip() == "":
        return f"{safe_prefix}.{safe_extension}"

    safe_subtitle = safe_filename_text(subtitle, fallback="selection")
    return f"{safe_prefix}_{safe_subtitle}.{safe_extension}"



"""
##############################################################################
# --- 0. Common definitions for dataset ---
##############################################################################
"""

# --- Radio button selections / ラジオボタン選択肢 ---

data_source_JAPAN_SEA    = "Kodama et al. (2024) [ECS - Japan Sea]"
data_source_AROUND_JAPAN = "with [Around Japan]"
data_source_GLOBAL       = "with [Global data sets]"


DATA_SOURCES = [
    data_source_JAPAN_SEA,
    data_source_AROUND_JAPAN,
    data_source_GLOBAL,
]


STANDARD_UPLOAD_COLUMNS = [
    "Dataset",
    "reference",
    "Cruise",
    "Station",
    "Year",
    "Month",
    "Day",
    "Longitude_degE",
    "Latitude_degN",
    "Depth_m",
    "Temperature_degC",
    "Salinity",
    "d18O",
    "dD",
]


UPLOAD_COLUMN_ALIASES = {
    "Longitude_degE": [
        "longitude_dege",
        "longitude",
        "long",
        "lon",
        "x",
        "east_longitude",
        "longitude_e",
    ],
    "Latitude_degN": [
        "latitude_degn",
        "latitude",
        "lat",
        "y",
        "north_latitude",
        "latitude_n",
    ],
    "Depth_m": [
        "depth_m",
        "depth",
        "water_depth",
        "waterdepth",
        "sample_depth",
        "sampledepth",
        "depthmeter",
    ],
    "Temperature_degC": [
        "temperature_degc",
        "temperature",
        "temp",
        "temp_c",
        "temperature_c",
        "t",
    ],
    "Salinity": [
        "salinity",
        "sal",
        "psu",
        "s",
    ],
    "d18O": [
        "d18o",
        "delta18o",
        "delta_18o",
        "o18",
        "δ18o",
        "δ18O",
    ],
    "dD": [
        "dd",
        "d2h",
        "delta_d",
        "d_h",
        "deuterium",
        "δd",
        "δD",
    ],
    "d-excess": [
        "d-excess",
        "dexcess",
        "d_excess",
        "d excess",
    ],
}


def _normalize_column_label(label):
    """
    Normalize a column label for alias matching.

    列名の別名判定に使うため、空白・記号・大文字小文字の違いを吸収します。
    """
    text = str(label).strip()
    text = text.replace("δ", "delta")
    text = text.replace("Δ", "delta")
    text = text.replace("‰", "")
    normalized_chars = []
    for char in text.lower():
        if char.isalnum():
            normalized_chars.append(char)
        else:
            normalized_chars.append("_")
    normalized = "_".join(part for part in "".join(normalized_chars).split("_") if part)
    return normalized


def standardize_uploaded_column_names(df):
    """
    Rename common uploaded-data column aliases to EnvGeo-Seawater standard names.

    アップロードデータでよく使われる列名の別名を、EnvGeo-Seawaterの標準列名へ
    自動的に変換します。既に標準列がある場合は、その列を優先します。
    """
    df = df.copy()
    normalized_to_original = {
        _normalize_column_label(column): column for column in df.columns
    }
    rename_map = {}

    for standard_column, aliases in UPLOAD_COLUMN_ALIASES.items():
        if standard_column in df.columns:
            continue
        normalized_aliases = {_normalize_column_label(standard_column)}
        normalized_aliases.update(_normalize_column_label(alias) for alias in aliases)
        for normalized_alias in normalized_aliases:
            original_column = normalized_to_original.get(normalized_alias)
            if original_column is not None and original_column not in rename_map:
                rename_map[original_column] = standard_column
                break

    df = df.rename(columns=rename_map)
    df.attrs["standardized_column_renames"] = rename_map
    return df


MAP_REGION_AUTO = "Auto from filtered data"
MAP_REGION_PRESETS = {
    "Japan and surrounding area": {
        "bounds": (120.0, 155.0, 20.0, 50.0),
        "description_ja": "日本周辺を広めに表示します。",
    },
    "ECS - Japan Sea": {
        "bounds": (120.0, 145.0, 20.0, 47.0),
        "description_ja": "東シナ海から日本海を中心に表示します。",
    },
    "East China Sea": {
        "bounds": (118.0, 132.0, 23.0, 34.0),
        "description_ja": "東シナ海を中心に表示します。",
    },
    "Sea of Japan": {
        "bounds": (127.0, 143.0, 34.0, 48.0),
        "description_ja": "日本海を中心に表示します。",
    },
    "Kuroshio region": {
        "bounds": (120.0, 150.0, 20.0, 38.0),
        "description_ja": "黒潮および黒潮続流の西部域を表示します。",
    },
    "Oyashio region": {
        "bounds": (140.0, 165.0, 35.0, 55.0),
        "description_ja": "親潮から亜寒帯西部北太平洋を表示します。",
    },
    "Okhotsk Sea": {
        "bounds": (135.0, 165.0, 43.0, 62.0),
        "description_ja": "オホーツク海周辺を表示します。",
    },
    "Bering Sea": {
        "bounds": (160.0, 205.0, 50.0, 67.0),
        "description_ja": "ベーリング海周辺を表示します。",
    },
    "North Pacific": {
        "bounds": (120.0, 240.0, 0.0, 65.0),
        "description_ja": "日本から北太平洋東部までを表示します。",
    },
    "Western North Pacific": {
        "bounds": (115.0, 180.0, 0.0, 65.0),
        "description_ja": "日本周辺から西部北太平洋を表示します。",
    },
    "Tropical Pacific": {
        "bounds": (120.0, 290.0, -25.0, 25.0),
        "description_ja": "熱帯太平洋を広く表示します。",
    },
    "Equatorial Pacific": {
        "bounds": (120.0, 290.0, -10.0, 10.0),
        "description_ja": "赤道太平洋を中心に表示します。",
    },
    "South Pacific": {
        "bounds": (140.0, 290.0, -60.0, 0.0),
        "description_ja": "南太平洋を広く表示します。",
    },
    "Indo-Pacific": {
        "bounds": (90.0, 180.0, -45.0, 35.0),
        "description_ja": "インド洋東部から西部太平洋を表示します。",
    },
    "Indian Ocean": {
        "bounds": (20.0, 120.0, -45.0, 30.0),
        "description_ja": "インド洋全域を表示します。",
    },
    "Arabian Sea": {
        "bounds": (45.0, 80.0, 5.0, 30.0),
        "description_ja": "アラビア海を中心に表示します。",
    },
    "Bay of Bengal": {
        "bounds": (78.0, 100.0, 5.0, 25.0),
        "description_ja": "ベンガル湾を中心に表示します。",
    },
    "North Atlantic": {
        "bounds": (-85.0, 20.0, 0.0, 70.0),
        "description_ja": "北大西洋を広く表示します。",
    },
    "South Atlantic": {
        "bounds": (-70.0, 25.0, -60.0, 5.0),
        "description_ja": "南大西洋を広く表示します。",
    },
    "Equatorial Atlantic": {
        "bounds": (-60.0, 15.0, -15.0, 15.0),
        "description_ja": "赤道大西洋を中心に表示します。",
    },
    "Mediterranean Sea": {
        "bounds": (-6.0, 37.0, 30.0, 46.0),
        "description_ja": "地中海を中心に表示します。",
    },
    "Arctic Ocean": {
        "bounds": (-180.0, 180.0, 65.0, 90.0),
        "description_ja": "北極海を全球経度で表示します。",
    },
    "Nordic Seas": {
        "bounds": (-25.0, 25.0, 60.0, 82.0),
        "description_ja": "グリーンランド海・ノルウェー海周辺を表示します。",
    },
    "Southern Ocean": {
        "bounds": (-180.0, 180.0, -80.0, -40.0),
        "description_ja": "南大洋を全球経度で表示します。",
    },
    "Antarctic margin": {
        "bounds": (-180.0, 180.0, -78.0, -55.0),
        "description_ja": "南極周辺の海域を表示します。",
    },
    "Southern Ocean - Atlantic sector": {
        "bounds": (-70.0, 20.0, -80.0, -40.0),
        "description_ja": "南大洋の大西洋セクターを表示します。",
    },
    "Southern Ocean - Indian sector": {
        "bounds": (20.0, 150.0, -80.0, -40.0),
        "description_ja": "南大洋のインド洋セクターを表示します。",
    },
    "Southern Ocean - Pacific sector": {
        "bounds": (150.0, 290.0, -80.0, -40.0),
        "description_ja": "南大洋の太平洋セクターを表示します。",
    },
    "Global": {
        "bounds": (-180.0, 180.0, -90.0, 90.0),
        "description_ja": "全球を表示します。",
    },
}


def map_region_view(region_label):
    """
    Return map center and approximate zoom for a named lon/lat region preset.

    地図表示用の海域プリセット名から、中心座標と概略ズームを返します。
    Bounds are stored as (lon_min, lon_max, lat_min, lat_max).
    """
    preset = MAP_REGION_PRESETS[region_label]
    lon_min, lon_max, lat_min, lat_max = preset["bounds"]
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    if center_lon > 180:
        center_lon -= 360
    elif center_lon < -180:
        center_lon += 360

    lon_span = max(lon_max - lon_min, 0.1)
    lat_span = max(lat_max - lat_min, 0.1)
    zoom_lon = math.log2((1200 * 360) / (lon_span * 256))
    zoom_lat = math.log2((700 * 180) / (lat_span * 256))
    zoom = max(1, min(15, min(zoom_lon, zoom_lat) - 0.8))

    if lon_span >= 300:
        center_lon = 0.0
        zoom = 1.0 if lat_span > 40 else 1.8

    return center_lat, center_lon, zoom


def area_filter_bounds(region_label, lon_min, lon_max, lat_min, lat_max):
    """
    Return slider-friendly lon/lat bounds for a Data filtering area preset.

    Data filtering 用のエリアプリセットを、現在のデータ範囲に合わせて
    Longitude / Latitude スライダーの初期値として使える範囲に変換します。
    """
    lon_min = float(lon_min)
    lon_max = float(lon_max)
    lat_min = float(lat_min)
    lat_max = float(lat_max)

    if region_label == AREA_FILTER_MANUAL or region_label not in MAP_REGION_PRESETS:
        return lon_min, lon_max, lat_min, lat_max

    preset_lon_min, preset_lon_max, preset_lat_min, preset_lat_max = (
        MAP_REGION_PRESETS[region_label]["bounds"]
    )

    # Match the preset longitude frame to the dataset when possible.
    # 可能な範囲で、プリセット経度をデータ側の経度表現に合わせます。
    if lon_min >= 0 and lon_max > 180:
        if preset_lon_min < 0:
            preset_lon_min += 360
        if preset_lon_max < 0:
            preset_lon_max += 360
    elif lon_min < 0 and lon_max <= 180:
        if preset_lon_min > 180:
            preset_lon_min -= 360
        if preset_lon_max > 180:
            preset_lon_max -= 360

    preset_crosses_dateline = preset_lon_min > preset_lon_max
    if preset_crosses_dateline:
        selected_lon_min, selected_lon_max = lon_min, lon_max
    else:
        selected_lon_min = max(lon_min, float(preset_lon_min))
        selected_lon_max = min(lon_max, float(preset_lon_max))
        if selected_lon_min >= selected_lon_max:
            selected_lon_min, selected_lon_max = lon_min, lon_max

    selected_lat_min = max(lat_min, float(preset_lat_min))
    selected_lat_max = min(lat_max, float(preset_lat_max))
    if selected_lat_min >= selected_lat_max:
        selected_lat_min, selected_lat_max = lat_min, lat_max

    return selected_lon_min, selected_lon_max, selected_lat_min, selected_lat_max


QUALITY_FLAG_COLUMN = "Quality_Flags"
QUALITY_ORIGINAL_VALUE_COLUMN = "Quality_Original_Values"

QUALITY_VALUE_RULES = {
    "Depth_m": {
        "valid_range": (0, None),
        "flag": "Depth_m outside valid range; converted to NaN",
        "description_ja": "水深が0 m未満の場合は不適切値としてNaNに変換します。",
    },
    "Temperature_degC": {
        "valid_range": (-5, 45),
        "flag": "Temperature_degC outside valid range; converted to NaN",
        "description_ja": "水温が-5から45 degCの範囲外の場合はNaNに変換します。",
    },
    "Salinity": {
        "valid_range": (0, 50),
        "flag": "Salinity outside valid range; converted to NaN",
        "description_ja": "塩分が0から50の範囲外の場合はNaNに変換します。",
    },
}


def quality_flag_criteria_text():
    """
    Return compact quality-flag criteria for UI footnotes.

    品質フラグの判定基準を、ページ下部に置ける短い注記として返します。
    """
    return (
        "Quality flags: invalid values are converted to NaN "
        "(Depth_m < 0, Temperature_degC outside -5 to 45, Salinity outside 0 to 50)."
    )


def render_quality_flag_criteria_note():
    """
    Render a small Streamlit note explaining quality flag criteria.

    Sidebar-filtered datasetなどの下に置くための小さな注記です。
    """
    st.caption(quality_flag_criteria_text())


def normalize_quality_values(df):
    """
    Convert known invalid or sentinel values to NaN while preserving the
    original values in quality-report columns.

    既知の不適切値・欠損コードをNaNに変換し、元の値と理由を
    `Quality_Flags` と `Quality_Original_Values` に残します。
    これにより、可視化では異常値を除きつつ、元データ精査用の
    手がかりを失わないようにします。
    """
    df = df.copy()
    df[QUALITY_FLAG_COLUMN] = ""
    df[QUALITY_ORIGINAL_VALUE_COLUMN] = ""

    empty_series = pd.Series(index=df.index, dtype="float64")
    quality_rules = {
        "Depth_m": (
            df.get("Depth_m", empty_series) < 0,
            QUALITY_VALUE_RULES["Depth_m"]["flag"],
        ),
        "Temperature_degC": (
            ~df.get("Temperature_degC", empty_series).between(-5, 45)
            & df.get("Temperature_degC", empty_series).notna(),
            QUALITY_VALUE_RULES["Temperature_degC"]["flag"],
        ),
        "Salinity": (
            ~df.get("Salinity", empty_series).between(0, 50)
            & df.get("Salinity", empty_series).notna(),
            QUALITY_VALUE_RULES["Salinity"]["flag"],
        ),
    }

    for col, (mask, message) in quality_rules.items():
        if col not in df.columns:
            continue

        mask = mask.fillna(False)
        if not mask.any():
            continue

        original_values = df.loc[mask, col].astype(str)
        additions = message + " (original=" + original_values + ")"
        existing_flags = df.loc[mask, QUALITY_FLAG_COLUMN].astype(str)
        existing_values = df.loc[mask, QUALITY_ORIGINAL_VALUE_COLUMN].astype(str)

        df.loc[mask, QUALITY_FLAG_COLUMN] = [
            f"{old}; {message}" if old else message for old in existing_flags
        ]
        df.loc[mask, QUALITY_ORIGINAL_VALUE_COLUMN] = [
            f"{old}; {new}" if old else new for old, new in zip(existing_values, additions)
        ]
        df.loc[mask, col] = np.nan

    return df


def add_d_excess(df, output_col="d-excess", d18o_col="d18O", dd_col="dD"):
    """
    Add d-excess, defined as dD - 8 * d18O, using numeric isotope columns.

    d-excess (= dD - 8 * d18O) を共通計算列として追加します。
    同位体列がない場合や数値化できない場合は、誤った値を入れずNaNにします。
    """
    df = df.copy()
    if d18o_col not in df.columns or dd_col not in df.columns:
        df[output_col] = np.nan
        return df

    d18o = pd.to_numeric(df[d18o_col], errors="coerce")
    dd = pd.to_numeric(df[dd_col], errors="coerce")
    df[output_col] = dd - 8 * d18o
    return df



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
        df = df.mask(df.eq('**'), np.nan)
        
        # Enforce numeric conversion for safety (applies to all datasets) / 安全のため数値列を明示的に数値化する
        target_cols = ['d18O', 'dD', 'Longitude_degE', 'Latitude_degN', 
                       'Depth_m', 'Temperature_degC', 'Salinity']
        
        for col in target_cols:
            if col in df.columns:
                # Use errors='coerce' to turn non-numeric values (e.g., whitespace) into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = normalize_quality_values(df)
        df = add_d_excess(df)
        
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


CMOCEAN_COLORMAP_VARIABLE_SUGGESTIONS = {
    "Temperature_degC": "cmocean thermal",
    "Temperature": "cmocean thermal",
    "Temp": "cmocean thermal",
    "Salinity": "cmocean haline",
    "Depth_m": "cmocean deep",
    "Water Depth": "cmocean deep",
    "depth": "cmocean deep",
    "d18O": "cmocean balance",
    "dD": "cmocean balance",
    "d-excess": "cmocean delta",
}


def get_plotly_colormap_options(selected_item=None):
    """
    Return reusable Plotly colorscale choices for EnvGeo-Seawater pages.

    EnvGeo-Seawaterの各ページで使い回すPlotly用カラースケール候補を返します。
    cmocean系の名前はPlotlyに組み込まれているため、追加依存なしで利用できます。
    """
    return {
        "EnvGeo variable default": get_custom_colorscale(selected_item),
        "cmocean thermal": "thermal",
        "cmocean haline": "haline",
        "cmocean deep": "deep",
        "cmocean dense": "dense",
        "cmocean balance": "balance",
        "cmocean delta": "delta",
        "cmocean oxy": "oxy",
        "cmocean matter": "matter",
        "Jet": "jet",
        "Turbo": "Turbo",
        "Viridis": "Viridis",
        "Plasma": "Plasma",
        "Cividis": "Cividis",
        "RdYlBu": "RdYlBu_r",
    }


def recommended_plotly_colormap_label(selected_item):
    """
    Return the default colormap label for a selected variable.

    既存図との連続性を保つため、初期値はEnvGeo標準にします。
    cmocean系は選択肢として残し、ユーザーが必要に応じて切り替えます。
    """
    return "EnvGeo variable default"


def suggested_cmocean_colormap_label(selected_item):
    """
    Return a cmocean suggestion for a selected oceanographic variable.

    選択された海洋データ変数に対して、試用候補となるcmocean名を返します。
    """
    return CMOCEAN_COLORMAP_VARIABLE_SUGGESTIONS.get(selected_item)


def get_plotly_colormap(selected_item=None, colormap_label=None):
    """
    Resolve a selected colormap label to a Plotly colorscale.

    選択されたカラーマップ名を、Plotlyで使えるcolorscaleへ変換します。
    """
    options = get_plotly_colormap_options(selected_item)
    if colormap_label is None:
        colormap_label = recommended_plotly_colormap_label(selected_item)
    return options.get(colormap_label, options["EnvGeo variable default"])


def get_matplotlib_colormap(selected_item=None, colormap_label=None):
    """
    Resolve a selected colormap label to a Matplotlib colormap.

    選択されたカラーマップ名を、Matplotlibで使えるcolormapへ変換します。
    cmoceanが使えない環境では、EnvGeo標準カラーマップへ戻します。
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if colormap_label is None:
        colormap_label = recommended_plotly_colormap_label(selected_item)

    envgeo_default = mcolors.LinearSegmentedColormap.from_list(
        f"envgeo_{selected_item or 'default'}",
        get_custom_colorscale(selected_item),
    )
    if colormap_label == "EnvGeo variable default":
        return envgeo_default

    cmocean_names = {
        "cmocean thermal": "thermal",
        "cmocean haline": "haline",
        "cmocean deep": "deep",
        "cmocean dense": "dense",
        "cmocean balance": "balance",
        "cmocean delta": "delta",
        "cmocean oxy": "oxy",
        "cmocean matter": "matter",
    }
    if colormap_label in cmocean_names:
        try:
            import cmocean

            return getattr(cmocean.cm, cmocean_names[colormap_label])
        except Exception:
            return envgeo_default

    matplotlib_names = {
        "Jet": "jet",
        "Turbo": "turbo",
        "Viridis": "viridis",
        "Plasma": "plasma",
        "Cividis": "cividis",
        "RdYlBu": "RdYlBu_r",
    }
    try:
        return plt.get_cmap(matplotlib_names.get(colormap_label, colormap_label))
    except Exception:
        return envgeo_default




"""
##############################################################################
# --- 5. MAP STYLE CONFIGURATION ---
# Apply background tile layers to the Plotly Mapbox figure.
##############################################################################
"""

MAP_MODE_OPTIONS = ["Standard", "Satellite", "Bathymetry (Sea)", "Contour (GSI)"]
MAP_MODE_DESCRIPTIONS_JA = {
    "Standard": "APIキー不要のOpenStreetMap背景です。",
    "Satellite": "USGSの衛星画像タイルを使います。",
    "Bathymetry (Sea)": "Esri World Ocean Baseの海底地形背景を使います。",
    "Contour (GSI)": "国土地理院の標準地図タイルを使います。",
}


def apply_map_style(fig, map_mode):
    """
    Apply the selected background tile layer to the Mapbox figure.
    All tile sources have been verified for web-use licensing.

    Note:
        CARTO Positron previously worked without a key, but CARTO basemaps now
        require an API key. The Standard mode therefore uses OpenStreetMap.
    """
    
    fig.update_layout(mapbox_style="open-street-map")

    if map_mode == "Standard":
        fig.update_layout(mapbox_style="open-street-map")
        
    
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

   

def summarize_filtered_data(df):
    """
    Return compact statistics for sidebar-filtered data.

    サイドバーで抽出されたデータの概要統計を返します。
    Streamlit表示から切り離しておくことで、pytestで確認しやすくします。
    """
    stats_columns = ["d18O", "dD", "d-excess", "Salinity", "Temperature_degC", "Depth_m"]
    summary = {}

    for col in stats_columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce").dropna()
        summary[col] = {
            "count": int(values.count()),
            "mean": np.nan if values.empty else float(values.mean()),
            "std": np.nan if values.empty else float(values.std(ddof=0)),
            "min": np.nan if values.empty else float(values.min()),
            "max": np.nan if values.empty else float(values.max()),
        }

    if QUALITY_FLAG_COLUMN in df.columns:
        flags = df[QUALITY_FLAG_COLUMN].fillna("").astype(str)
        quality_flag_count = int((flags != "").sum())
    else:
        quality_flag_count = 0

    return {
        "row_count": int(len(df)),
        "quality_flag_count": quality_flag_count,
        "statistics": summary,
    }


def filtered_statistics_dataframe(summary):
    """
    Convert sidebar-filtered summary statistics into a table.

    サイドバー抽出データの統計情報を、表示・CSV/PDF出力しやすい表へ変換します。
    """
    statistic_rows = []
    for col, values in summary["statistics"].items():
        statistic_rows.append(
            {
                "Parameter": col,
                "Count": values["count"],
                "Mean": values["mean"],
                "Stdev": values["std"],
                "Min": values["min"],
                "Max": values["max"],
            }
        )

    if not statistic_rows:
        return pd.DataFrame(columns=["Parameter", "Count", "Mean", "Stdev", "Min", "Max"])

    return pd.DataFrame(statistic_rows)


def build_filtered_report_tables(df, filter_conditions=None, selected_counts=None):
    """
    Build report tables for sidebar-filtered data.

    フィルタ済みデータのCSV/PDFレポートに使う表を作ります。
    """
    summary = summarize_filtered_data(df)
    overview_df = pd.DataFrame(
        [
            {"Item": "Report generated", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Item": "Rows", "Value": summary["row_count"]},
            {"Item": "Quality flags", "Value": summary["quality_flag_count"]},
        ]
    )

    if filter_conditions is None:
        filter_conditions = {}
    filter_df = pd.DataFrame(
        [{"Item": key, "Value": value} for key, value in filter_conditions.items()]
    )

    if selected_counts is None:
        selected_counts = {}
    selected_counts_df = pd.DataFrame(
        [{"Item": key, "Value": value} for key, value in selected_counts.items()]
    )

    statistics_df = filtered_statistics_dataframe(summary)
    return overview_df, filter_df, selected_counts_df, statistics_df


def build_filtered_report_csv(df, filter_conditions=None, selected_counts=None):
    """
    Return a UTF-8 BOM CSV report for sidebar-filtered data.

    Excelで開きやすいようにUTF-8 BOM付きCSVとして返します。
    """
    overview_df, filter_df, selected_counts_df, statistics_df = build_filtered_report_tables(
        df,
        filter_conditions,
        selected_counts,
    )

    overview_export = overview_df.copy()
    overview_export.insert(0, "Section", "Overview")
    filter_export = filter_df.copy()
    filter_export.insert(0, "Section", "Filter Conditions")
    selected_counts_export = selected_counts_df.copy()
    selected_counts_export.insert(0, "Section", "Filtered Data Counts")
    statistics_export = statistics_df.rename(columns={"Parameter": "Item"}).copy()
    statistics_export.insert(0, "Section", "Statistics")

    return pd.concat(
        [overview_export, filter_export, selected_counts_export, statistics_export],
        ignore_index=True,
        sort=False,
    ).to_csv(index=False).encode("utf-8-sig")


def render_filtered_report_download(
    df,
    filter_conditions=None,
    selected_counts=None,
    key_prefix="filtered_data",
):
    """
    Render a CSV download button for sidebar-filtered data.

    各ページで再利用できる、フィルタ済みデータ概要CSVのダウンロードボタンです。
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    st.download_button(
        "Download filtered-data summary CSV",
        data=build_filtered_report_csv(df, filter_conditions, selected_counts),
        file_name=f"envgeo_filtered_data_summary_{timestamp}.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_filtered_summary_csv",
        help="Export filter conditions, row counts, quality-flag counts, and summary statistics as CSV.",
    )



"""
##############################################################################
# --- 8. DATA TABLE VISUALIZATION ---
# Formats and displays the filtered dataframe within a Streamlit expander.
##############################################################################
"""


def display_isotope_table(df, title="Filtered dataset (CSV)"):
    """
    Format and render the dataframe in a Streamlit expander.
    Includes integer conversion for dates and string-casting to prevent Arrow errors.
    """
    with st.expander(title, expanded=False):
        # Define priority columns (Safety check included for missing columns) / 欠損列に配慮しつつ優先列を定義する
        target_cols = [
            'reference','Cruise', 'Station', 'Date', 'Year', 'Month', 
            'Longitude_degE', 'Latitude_degN', 'Depth_m', 
            'Temperature_degC', 'Salinity', 'd18O', 'dD', 'd-excess',
            QUALITY_FLAG_COLUMN, QUALITY_ORIGINAL_VALUE_COLUMN
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
        render_quality_flag_criteria_note()






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


def sidebar_filter_and_display(
    df1,
    ref_data,
    data_source_JAPAN_SEA,
    data_source_AROUND_JAPAN,
):
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
        
        
        st.header(DATA_FILTERING_LABEL)
        st.caption("Change filters, then click **Apply settings** to update the figures.")
        
        submit_top = st.form_submit_button("Apply settings", use_container_width=True)

        # Two buttons can be placed at the top and bottom if needed / 必要ならsubmitボタンを上下に配置できる
        # In Streamlit 1.42, form submit buttons do not support key, so labels must be unique.
        # Streamlit 1.42 では form submit button に key が使えないため、ラベルを変えて重複を避ける。

        #　一つだけの時は以下
        # submitted = st.form_submit_button("Apply settings")
        
        

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
        max_df_lat = int(math.ceil(df1['Latitude_degN'].max()))
        min_df_lat = int(math.floor(df1['Latitude_degN'].min()))

        area_filter_preset = st.selectbox(
            "Area filter preset",
            [AREA_FILTER_MANUAL] + list(MAP_REGION_PRESETS),
            help=AREA_FILTER_HELP_TEXT,
        )
        (
            default_lon_min,
            default_lon_max,
            default_lat_min,
            default_lat_max,
        ) = area_filter_bounds(
            area_filter_preset,
            min_df_lon,
            max_df_lon,
            min_df_lat,
            max_df_lat,
        )
        default_lon_min = int(math.floor(default_lon_min))
        default_lon_max = int(math.ceil(default_lon_max))
        default_lat_min = int(math.floor(default_lat_min))
        default_lat_max = int(math.ceil(default_lat_max))
        
        # 2. スライダーの設定
        sld_lon_min, sld_lon_max = st.slider(
            label='Longitude',  # ラベルを少し自然に
            min_value=min_df_lon,
            max_value=max_df_lon,
            value=(default_lon_min, default_lon_max),
            step=1,
            key=f"filter_longitude::{area_filter_preset}",
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
        
        # 2. スライダーの設定
        sld_lat_min, sld_lat_max = st.slider(
            label='Latitude',
            min_value=min_df_lat,
            max_value=max_df_lat,
            value=(default_lat_min, default_lat_max),
            step=1,  # 整数刻みに設定
            key=f"filter_latitude::{area_filter_preset}",
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

 
        
        submit_bottom = st.form_submit_button("Apply settings!", use_container_width=True)
        submitted = submit_top or submit_bottom
        
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
                        
    with st.expander("📊 Details and statistics of filtered data", expanded=False):
        # 月を複数選択した場合は、表示・レポート・図タイトルで使いやすい文字列へ整形する
        # Format selected months for display, reporting, and figure titles.
        month_display = ", ".join(map(str, sorted(selected_months))) if selected_months else "None"
        selected_cruise_indicate =str(list(selected_cruise[:]))

        filter_conditions = {
            "Year": f"{sld_year_min}-{sld_year_max}",
            "Month": f"[{month_display}]",
            "Longitude_degE": f"{sld_lon_min}-{sld_lon_max}",
            "Latitude_degN": f"{sld_lat_min}-{sld_lat_max}",
            "Depth_m": f"{sld_depth_min}-{sld_depth_max}",
            "Salinity": f"{sld_sal_min}-{sld_sal_max}",
            "d18O": f"{sld_d18O_min}-{sld_d18O_max}",
            "Temperature_degC": f"{sld_temp_min}-{sld_temp_max}",
            "Filtered data (Cruise, papers)": selected_cruise_indicate,
        }
        summary = summarize_filtered_data(df1)
        metric_cols = st.columns(2)
        metric_cols[0].metric("Rows", f"{summary['row_count']:,}")
        metric_cols[1].metric("Quality flags", f"{summary['quality_flag_count']:,}")

        st.markdown("**Filter conditions**")
        st.dataframe(
            pd.DataFrame(
                [{"Item": key, "Value": value} for key, value in filter_conditions.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Filtered data counts by dataset**")
        df_selected_counts = pd.DataFrame(
            [{"Dataset": key, "Rows": value} for key, value in d_select_add2.items()]
        )
        if not df_selected_counts.empty:
            st.dataframe(df_selected_counts, use_container_width=True, hide_index=True)

        st.markdown("**Summary statistics**")
        df_stats = filtered_statistics_dataframe(summary)
        if not df_stats.empty:
            numeric_cols = ["Mean", "Stdev", "Min", "Max"]
            df_stats[numeric_cols] = df_stats[numeric_cols].round(3)
            st.dataframe(df_stats, use_container_width=True)
        render_filtered_report_download(
            df1,
            filter_conditions,
            selected_counts=d_select_add2,
            key_prefix="sidebar_filtered_data",
        )
                
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
