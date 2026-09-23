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

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

# --- App version / バージョン情報 ---
APP_VERSION = "1.3.3"
APP_VERSION_DATE = "2026-09-23"
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
MANUAL_FILTER_APPLY_NOTE = (
    ":red[Change filters, then click **Apply settings** to update the figures.]"
)
AUTO_APPLY_NOTE = ":blue[Changes in this section are applied automatically.]"
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



import os
import socket
import time
from functools import lru_cache

import pandas as pd
import streamlit as st
import numpy as np
import inspect
import math
import re
import unicodedata
from pathlib import Path
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

def _major_version(version_text):
    """Return the leading numeric component of a package version."""
    match = re.match(r"(\d+)", str(version_text))
    return int(match.group(1)) if match else 0


def configure_pandas_compatibility():
    """Enable future Pandas behavior only where the options are still needed.

    Pandas 3 uses these behaviors by default and warns when the former opt-in
    options are set. Pandas 2 still benefits from explicitly enabling them.
    """
    if _major_version(pd.__version__) < 3:
        pd.options.mode.copy_on_write = True
        pd.set_option("future.no_silent_downcasting", True)


def stretch_width_kwargs(widget):
    """Return full-width arguments compatible with old and new Streamlit APIs.

    Streamlit 1.42 uses ``use_container_width=True``. Newer releases use
    ``width="stretch"``. Inspection also handles widgets such as ``dataframe``
    whose older API already had a numeric ``width`` argument.
    """
    width_parameter = inspect.signature(widget).parameters.get("width")
    supports_stretch = width_parameter is not None and (
        "Width" in str(width_parameter.annotation)
        or width_parameter.default in {"stretch", "content"}
    )
    if supports_stretch:
        return {"width": "stretch"}
    return {"use_container_width": True}


def arrow_display_dataframe(df):
    """Return an Arrow-safe copy for Streamlit table display.

    Uploaded station, sample, or cruise identifiers commonly mix numeric and
    text values in one spreadsheet column (for example ``14`` and ``14_5``).
    PyArrow cannot serialize that mixed object column as a single type.  This
    helper converts only object columns in a display copy to pandas strings;
    the original dataframe remains unchanged for downloads and calculations.
    """
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_object_dtype(display_df[column]):
            display_df[column] = display_df[column].astype("string")
    return display_df


def render_earthquake_tab_style():
    """Render the shared blue card-style tabs used by Earthquake Advanced.

    Streamlit 1.63 changed tabs from BaseWeb controls to React Aria controls.
    Keep both selector families here so public pages retain the same appearance
    in every supported Streamlit release (1.42--1.63).
    """
    tab_css = """
        <style>
        div[data-baseweb="tab-list"] { gap: 0.25rem; flex-wrap: wrap; }
        div[data-baseweb="tab-list"] button[role="tab"] {
            background: rgba(248, 249, 250, 0.95); color: #1f2937;
            border: 1px solid rgba(49, 51, 63, 0.22); border-radius: 6px 6px 0 0;
            padding: 0.35rem 0.65rem; min-height: 2.1rem; white-space: nowrap;
            font-weight: 600;
        }
        div[data-baseweb="tab-list"] button[role="tab"] p { margin: 0; color: inherit; }
        div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #e8f2ff 0%, #ddeaff 100%);
            border-color: #4a90e2; color: #0b3e75;
            box-shadow: inset 0 0 0 1px rgba(74, 144, 226, 0.35);
        }
        html[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"],
        body[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"] {
            background: rgba(44, 49, 61, 0.96); color: rgba(245, 247, 250, 0.95);
            border-color: rgba(240, 244, 250, 0.26);
        }
        html[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"],
        body[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #204061 0%, #1a314a 100%);
            color: #e9f2ff; border-color: #76adff;
            box-shadow: inset 0 0 0 1px rgba(118, 173, 255, 0.42);
        }
        @media (prefers-color-scheme: dark) {
            div[data-baseweb="tab-list"] button[role="tab"] {
                background: rgba(44, 49, 61, 0.96); color: rgba(245, 247, 250, 0.95);
                border-color: rgba(240, 244, 250, 0.26);
            }
            div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
                background: linear-gradient(180deg, #204061 0%, #1a314a 100%);
                color: #e9f2ff; border-color: #76adff;
                box-shadow: inset 0 0 0 1px rgba(118, 173, 255, 0.42);
            }
        }
        @media (max-width: 900px) {
            div[data-baseweb="tab-list"] button[role="tab"] { font-size: 0.86rem; padding: 0.30rem 0.52rem; }
        }
        /* Streamlit 1.63 uses data-baseweb="tab" directly.  Keep the
           role selector above for older Streamlit releases, then apply this
           more specific override for current releases. */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.25rem !important;
            flex-wrap: wrap !important;
            border-bottom: 1px solid rgba(49, 51, 63, 0.18) !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"],
        [data-testid="stTabs"] button[role="tab"] {
            background-color: rgba(248, 249, 250, 0.95) !important;
            background-image: none !important;
            color: #1f2937 !important;
            border: 1px solid rgba(49, 51, 63, 0.22) !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 0.35rem 0.65rem !important;
            min-height: 2.1rem !important;
            white-space: nowrap !important;
            font-weight: 600 !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] p,
        [data-testid="stTabs"] button[role="tab"] p { color: inherit !important; }
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background-color: #ddeaff !important;
            background-image: linear-gradient(180deg, #e8f2ff 0%, #ddeaff 100%) !important;
            border-color: #4a90e2 !important;
            border-bottom-color: #ddeaff !important;
            color: #0b3e75 !important;
            box-shadow: inset 0 0 0 1px rgba(74, 144, 226, 0.35) !important;
        }
        @media (prefers-color-scheme: dark) {
            [data-testid="stTabs"] [data-baseweb="tab"],
            [data-testid="stTabs"] button[role="tab"] {
                background-color: rgba(44, 49, 61, 0.96) !important;
                color: rgba(245, 247, 250, 0.95) !important;
                border-color: rgba(240, 244, 250, 0.26) !important;
            }
            [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
            [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
                background-color: #1a314a !important;
                background-image: linear-gradient(180deg, #204061 0%, #1a314a 100%) !important;
                color: #e9f2ff !important;
                border-color: #76adff !important;
            }
        }
        /* Streamlit may render the tab buttons and its primary-colour
           highlight outside the expected wrapper.  These global selectors
           intentionally override that 1.63 structure as well. */
        [data-baseweb="tab"] {
            background-color: rgba(248, 249, 250, 0.95) !important;
            background-image: none !important;
            color: #1f2937 !important;
            border: 1px solid rgba(49, 51, 63, 0.22) !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 0.35rem 0.65rem !important;
            min-height: 2.1rem !important;
            font-weight: 600 !important;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background-color: #ddeaff !important;
            background-image: linear-gradient(180deg, #e8f2ff 0%, #ddeaff 100%) !important;
            border-color: #4a90e2 !important;
            color: #0b3e75 !important;
            box-shadow: inset 0 0 0 1px rgba(74, 144, 226, 0.35) !important;
        }
        [data-baseweb="tab-highlight"] {
            display: none !important;
            background-color: transparent !important;
        }
        /* Streamlit 1.63+: tabs are React Aria controls, not BaseWeb tabs.
           The selected state is expressed by data-selected, and the default
           red/primary underline is a nested SelectionIndicator element. */
        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.25rem !important;
            flex-wrap: wrap !important;
            border-bottom: 1px solid rgba(49, 51, 63, 0.18) !important;
            padding-bottom: 0 !important;
        }
        [data-testid="stTabs"] [role="tablist"]::after {
            background-color: transparent !important;
            height: 0 !important;
        }
        [data-testid="stTabs"] [data-testid="stTab"] {
            height: auto !important;
            min-height: 2.1rem !important;
            padding: 0.35rem 0.65rem !important;
            background: rgba(248, 249, 250, 0.95) !important;
            color: #1f2937 !important;
            border: 1px solid rgba(49, 51, 63, 0.22) !important;
            border-radius: 6px 6px 0 0 !important;
            font-weight: 600 !important;
        }
        [data-testid="stTabs"] [data-testid="stTab"] p {
            margin: 0 !important;
            color: inherit !important;
        }
        [data-testid="stTabs"] [data-testid="stTab"] .react-aria-SelectionIndicator {
            height: 0 !important;
            background-color: transparent !important;
        }
        [data-testid="stTabs"] [data-testid="stTab"][data-selected] {
            background: linear-gradient(180deg, #e8f2ff 0%, #ddeaff 100%) !important;
            border-color: #4a90e2 !important;
            color: #0b3e75 !important;
            box-shadow: inset 0 0 0 1px rgba(74, 144, 226, 0.35) !important;
        }
        @media (prefers-color-scheme: dark) {
            [data-testid="stTabs"] [data-testid="stTab"] {
                background: rgba(44, 49, 61, 0.96) !important;
                color: rgba(245, 247, 250, 0.95) !important;
                border-color: rgba(240, 244, 250, 0.26) !important;
            }
            [data-testid="stTabs"] [data-testid="stTab"][data-selected] {
                background: linear-gradient(180deg, #204061 0%, #1a314a 100%) !important;
                color: #e9f2ff !important;
                border-color: #76adff !important;
            }
        }
        </style>
        """
    # The original Earthquake page injects page-wide styles through Markdown.
    # This also works in both supported Streamlit lines, whereas st.html()
    # may isolate the style node from sibling elements in newer releases.
    st.markdown(tab_css, unsafe_allow_html=True)


def bounded_container(max_width=850):
    """Return a container capped on desktop and fluid on narrow screens.

    Streamlit 1.63 accepts an integer container width and automatically caps it
    to the parent width. Older supported versions fall back to a normal
    container.

    PCでは指定幅を上限とし、狭い画面では親幅まで縮むコンテナを返します。
    """
    if "width" in inspect.signature(st.container).parameters:
        return st.container(width=max_width)
    return st.container()


configure_pandas_compatibility()



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

UPLOADED_DATA_LABEL = "Uploaded data"
USER_EXCEL_DATA_LABEL = "User Excel data"
UPLOAD_NUMERIC_COLUMNS = [
    "Longitude_degE",
    "Latitude_degN",
    "Depth_m",
    "Temperature_degC",
    "Salinity",
    "d18O",
    "dD",
]
UPLOAD_SESSION_DATA_KEY = "envgeo_uploaded_data"
UPLOAD_SESSION_FILENAME_KEY = "envgeo_uploaded_filename"
LOCAL_USER_DATA_PATH_ENV = "ENVGEO_LOCAL_USER_DATA_PATH"
LOCAL_USER_DATA_DISABLE_ENV = "ENVGEO_DISABLE_LOCAL_USER_DATA"
DEFAULT_LOCAL_USER_DATA_PATH = Path(__file__).resolve().parent / "local_data" / "user_data.xlsx"
INTEGRATED_EMBEDDED_PAGE_KEY = "envgeo_integrated_embedded_page"


UPLOAD_COLUMN_ALIASES = {
    "Month": [
        "month",
        "sampling_month",
        "sample_month",
        "月",
    ],
    "Longitude_degE": [
        "longitude_dege",
        "longitude",
        "long",
        "lon",
        "x",
        "east_longitude",
        "longitude_e",
        "経度",
        "東経",
    ],
    "Latitude_degN": [
        "latitude_degn",
        "latitude",
        "lat",
        "y",
        "north_latitude",
        "latitude_n",
        "緯度",
        "北緯",
    ],
    "Depth_m": [
        "depth_m",
        "depth",
        "water_depth",
        "waterdepth",
        "sample_depth",
        "sampledepth",
        "depthmeter",
        "水深",
        "水深_m",
    ],
    "Temperature_degC": [
        "temperature_degc",
        "temperature",
        "temp",
        "temp_c",
        "temperature_c",
        "t",
        "水温",
        "水温_c",
    ],
    "Salinity": [
        "salinity",
        "sal",
        "psu",
        "s",
        "塩分",
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


def read_uploaded_table(uploaded_file):
    """Read an uploaded CSV or Excel table without saving it to disk."""
    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload a CSV, XLSX, or XLS file.")


def resolve_local_user_data_path(environ=None, default_path=None):
    """Return the configured optional local user-data path.

    ``ENVGEO_LOCAL_USER_DATA_PATH`` may point to a CSV, XLSX, or XLS file. If
    it is unset, ``local_data/user_data.xlsx`` is used only when that file
    exists. Relative environment paths are resolved from the application root.
    """
    environ = os.environ if environ is None else environ
    disabled = str(environ.get(LOCAL_USER_DATA_DISABLE_ENV, "")).strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return None
    app_root = Path(__file__).resolve().parent
    configured = str(environ.get(LOCAL_USER_DATA_PATH_ENV, "")).strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else app_root / path

    path = Path(default_path) if default_path is not None else DEFAULT_LOCAL_USER_DATA_PATH
    return path if path.exists() else None


def read_local_user_table(path):
    """Read an optional local CSV or Excel table without modifying it."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Local user-data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Local user data must be a CSV, XLSX, or XLS file.")


def load_local_user_data(path, dataset_label=USER_EXCEL_DATA_LABEL):
    """Load local user data and assign its dataset category explicitly."""
    prepared = prepare_uploaded_data(read_local_user_table(path))
    prepared["Dataset"] = dataset_label
    return prepared


def build_upload_template_csv():
    """Return a small seawater upload template for spreadsheet applications."""
    template = pd.DataFrame(
        [
            {
                "Dataset": UPLOADED_DATA_LABEL,
                "reference": "Your reference",
                "Cruise": "Cruise ID",
                "Station": "Station ID",
                "Year": 2026,
                "Month": 1,
                "Day": 1,
                "Longitude_degE": 135.0,
                "Latitude_degN": 35.0,
                "Depth_m": 10.0,
                "Temperature_degC": 20.0,
                "Salinity": 34.5,
                "d18O": 0.0,
                "dD": 0.0,
            }
        ],
        columns=STANDARD_UPLOAD_COLUMNS,
    )
    return template.to_csv(index=False).encode("utf-8-sig")


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
        # 198.0 は -180..180 表記での -162.0 (162W) を 0..360 側に伸ばした
        # 表現。実際の抽出は (lon >= 160E) OR (lon <= -162W) の OR 条件で
        # 行うため、北大西洋など無関係な海域が緯度だけで混入しない。
        # 198.0 is -162.0 (162W) expressed on the 0..360 side. The actual
        # row filter applies an OR condition — (lon >= 160E) OR
        # (lon <= -162W) — so unrelated basins like the North Atlantic are
        # not swept in by latitude alone.
        "bounds": (160.0, 198.0, 51.0, 66.0),
        "description_ja": "ベーリング海周辺（東経160度以東または西経162度以西、北緯51-66度）を表示します。",
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


def normalize_longitude_deg(lon):
    """
    経度を -180..180 と 0..360 のどちらの表記で受け取っても、
    (-180, 180] の正準表現へ正規化する共通ヘルパー。

    Normalize longitude value(s) into the canonical (-180, 180] range,
    regardless of whether the input uses the -180..180 or 0..360
    convention. Works with a plain number, a numpy array, or a pandas
    Series (preserving its index), so the same helper can normalize both
    an area-filter preset's scalar bounds and a DataFrame's
    Longitude_degE column before comparing them.
    """
    lon_numeric = lon.astype(float) if hasattr(lon, "astype") else float(lon)
    return ((lon_numeric + 180.0) % 360.0) - 180.0


def region_preset_crosses_dateline(region_label):
    """
    指定したエリアプリセットの経度範囲が日付変更線(180度/-180度)を
    またぐかどうかを返す。

    Return whether a named MAP_REGION_PRESETS entry's longitude span
    crosses the antimeridian. Such presets are stored with lon_max > 180
    (e.g. Bering Sea's 198.0 stands for -162.0) to signal that their real
    extent wraps past 180/-180, and must be matched with an OR condition
    rather than a simple min/max range.
    """
    if region_label not in MAP_REGION_PRESETS:
        return False
    _, lon_max, _, _ = MAP_REGION_PRESETS[region_label]["bounds"]
    return lon_max > 180.0


def region_preset_longitude_mask(lon_values, region_label):
    """
    エリアプリセットの経度範囲に基づく行マスクを返す。

    日付変更線をまたぐプリセット（例: Bering Sea）は、単一の
    Longitude range スライダーでは表現できないため、東側の弧
    (経度 >= 東端) OR 西側の弧 (経度 <= 西端) の OR 条件で判定する。
    normalize_longitude_deg() で正規化してから比較するため、データが
    -180..180 と 0..360 のどちらの表記でも正しく同じ海域を抽出できる。
    日付変更線をまたがない通常のプリセットは、単純な範囲判定のまま
    (既存のスライダー挙動を変えない)。

    Return a boolean mask selecting rows inside an area-filter preset's
    longitude span. Presets that cross the antimeridian (e.g. Bering Sea)
    cannot be expressed by a single Longitude range slider, so they are
    matched with an OR condition — the eastern arm (lon >= east bound) OR
    the western arm (lon <= west bound) — after normalizing both the data
    and the preset bounds with normalize_longitude_deg(), so the same
    real-world sea area is extracted whether the data uses -180..180 or
    0..360 longitude. Ordinary (non-crossing) presets keep the simple
    min/max range check, matching the existing slider-based behavior.
    """
    lon_norm = normalize_longitude_deg(pd.to_numeric(lon_values, errors="coerce"))

    if region_label not in MAP_REGION_PRESETS:
        return pd.Series(True, index=lon_norm.index) if hasattr(lon_norm, "index") else np.ones_like(lon_norm, dtype=bool)

    lon_min, lon_max, _, _ = MAP_REGION_PRESETS[region_label]["bounds"]

    if region_preset_crosses_dateline(region_label):
        east_bound = normalize_longitude_deg(lon_min)
        west_bound = normalize_longitude_deg(lon_max)
        return (lon_norm >= east_bound) | (lon_norm <= west_bound)

    return (lon_norm >= normalize_longitude_deg(lon_min)) & (lon_norm <= normalize_longitude_deg(lon_max))


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

    d18o = coerce_numeric_values(df[d18o_col])
    dd = coerce_numeric_values(df[dd_col])
    df[output_col] = dd - 8 * d18o
    return df


def coerce_numeric_values(values):
    """Convert spreadsheet values after removing invisible Unicode spaces.

    Excel cells copied from formatted sources can contain non-breaking spaces
    such as U+00A0. They look numeric in the sheet but ``pd.to_numeric`` would
    otherwise coerce them to NaN.
    """
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.to_numeric(values, errors="coerce")
    cleaned = values.astype("string")
    # Use literal replacements instead of ``\u`` escapes in a regex. Pandas
    # may store strings with PyArrow, whose regex engine rejects those escapes.
    for whitespace in (" ", "\t", "\r", "\n", "\u00A0", "\u202F", "\u3000"):
        cleaned = cleaned.str.replace(whitespace, "", regex=False)
    cleaned = cleaned.str.replace("\u2212", "-", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_uploaded_data(df, dataset_label=UPLOADED_DATA_LABEL):
    """Prepare uploaded seawater data for quality review and plotting.

    File reading is generic. This composition deliberately contains the
    seawater-specific numeric columns, quality rules, and d-excess calculation.
    """
    prepared = standardize_uploaded_column_names(df)
    rename_map = prepared.attrs.get("standardized_column_renames", {}).copy()

    if "Dataset" not in prepared.columns:
        prepared["Dataset"] = dataset_label

    # Replace '**' placeholders with NaN (same convention as main data loading)
    # プレースホルダ '**' をNaNへ置換する（メインデータの読み込みと同じ処理）
    prepared = prepared.mask(prepared.eq('**'), np.nan)

    # Convert Year and Month to nullable integers to prevent Arrow serialization errors
    # Year・Month を nullable 整数型に変換してArrowシリアライズエラーを防ぐ
    for col in ('Year', 'Month'):
        if col in prepared.columns:
            prepared[col] = coerce_numeric_values(prepared[col]).astype('Int64')

    for column in UPLOAD_NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = coerce_numeric_values(prepared[column])

    prepared = normalize_quality_values(prepared)
    prepared = add_d_excess(prepared)
    prepared.attrs["standardized_column_renames"] = rename_map
    return prepared


def apply_uploaded_column_mapping(df, column_mapping):
    """Copy user-selected source columns into standard roles and revalidate.

    ``column_mapping`` uses standard EnvGeo column names as keys and uploaded
    source-column names as values. Source columns are retained so experimental
    parameters remain available for later custom plots.
    """
    mapped = df.copy()
    existing_flags = mapped.get(
        QUALITY_FLAG_COLUMN,
        pd.Series("", index=mapped.index, dtype="object"),
    ).fillna("").astype(str)
    existing_original_values = mapped.get(
        QUALITY_ORIGINAL_VALUE_COLUMN,
        pd.Series("", index=mapped.index, dtype="object"),
    ).fillna("").astype(str)
    applied_mapping = {}

    for target_column, source_column in column_mapping.items():
        if not source_column:
            continue
        if source_column not in mapped.columns:
            raise KeyError(f"Uploaded column not found: {source_column}")
        if source_column != target_column:
            mapped[target_column] = mapped[source_column]
        applied_mapping[target_column] = source_column

    mapped = prepare_uploaded_data(mapped)
    mapped[QUALITY_FLAG_COLUMN] = [
        _merge_quality_text(old, new)
        for old, new in zip(existing_flags, mapped[QUALITY_FLAG_COLUMN])
    ]
    mapped[QUALITY_ORIGINAL_VALUE_COLUMN] = [
        _merge_quality_text(old, new)
        for old, new in zip(
            existing_original_values,
            mapped[QUALITY_ORIGINAL_VALUE_COLUMN],
        )
    ]
    mapped.attrs["manual_column_mapping"] = applied_mapping
    return mapped


def _merge_quality_text(existing, new):
    """Combine quality messages while avoiding exact repeated content."""
    existing = str(existing).strip()
    new = str(new).strip()
    if not existing:
        return new
    if not new or new in existing:
        return existing
    if existing in new:
        return new
    return f"{existing}; {new}"


def get_quality_rows(df):
    """Return only rows carrying one or more quality flags."""
    if df is None or df.empty or QUALITY_FLAG_COLUMN not in df.columns:
        columns = getattr(df, "columns", None)
        return pd.DataFrame(columns=columns)
    flags = df[QUALITY_FLAG_COLUMN].fillna("").astype(str).str.strip()
    return df.loc[flags.ne("")].copy()


def store_uploaded_data(df, filename=None, state=None):
    """Keep prepared upload data in the current Streamlit session only."""
    state = st.session_state if state is None else state
    state[UPLOAD_SESSION_DATA_KEY] = df.copy()
    state[UPLOAD_SESSION_FILENAME_KEY] = filename


def get_uploaded_data(state=None):
    """Return a copy of upload data shared by EnvGeo pages in this session."""
    state = st.session_state if state is None else state
    uploaded_df = state.get(UPLOAD_SESSION_DATA_KEY)
    if not isinstance(uploaded_df, pd.DataFrame):
        return pd.DataFrame()
    return uploaded_df.copy()


def get_uploaded_filename(state=None):
    """Return the source filename recorded for the current session upload."""
    state = st.session_state if state is None else state
    return state.get(UPLOAD_SESSION_FILENAME_KEY)


def clear_uploaded_data(state=None):
    """Remove shared upload data from the current Streamlit session."""
    state = st.session_state if state is None else state
    state.pop(UPLOAD_SESSION_DATA_KEY, None)
    state.pop(UPLOAD_SESSION_FILENAME_KEY, None)



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

    # Optional always-loaded user table for local operation.
    # ローカル実行時に常時読み込む任意のユーザー表
    file_user_excel = resolve_local_user_data_path()
    
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

    df_user_excel = pd.DataFrame()
    if file_user_excel is not None:
        try:
            # Match the former local-workbook structure: read the table beside
            # the bundled sources, assign its Dataset category, concatenate it
            # below, then apply the shared cleaning pipeline to the full frame.
            df_user_excel = read_local_user_table(file_user_excel)
            df_user_excel['Dataset'] = USER_EXCEL_DATA_LABEL
        except Exception as exc:
            # A local configuration error must not prevent public data loading.
            st.warning(f"Local user Excel data could not be loaded: {exc}")

    # Browser uploads remain session-only. The optional local user table is an
    # always-loaded dataset, matching the former local-workbook workflow.
    if ref_data == data_source_JAPAN_SEA:  
        df = pd.concat([df1, df_user_excel], ignore_index=True)
        
    elif ref_data == data_source_AROUND_JAPAN:  
        df = pd.concat([df1, df2, df_user_excel], ignore_index=True)

        
    elif ref_data == data_source_GLOBAL: 
        df = pd.concat(
            [df1, df2, df3, df4, df5, df_user_excel],
            ignore_index=True,
        )
        
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
                # Remove invisible spreadsheet spaces before numeric conversion.
                df[col] = coerce_numeric_values(df[col])

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
def load_coastline_data(ref_data, resolution="50m"):
    """Load shared Natural Earth coastline coordinates from CSV.

    ``ref_data`` remains in the signature for compatibility with existing pages.
    All regions currently use the same global coastline coordinate file.
    """
    _ = ref_data
    coastline_files = {
        "50m": "world_coastline_coordinates_50m.csv",
        "110m": "world_coastline_coordinates_110m.csv",
    }
    if resolution not in coastline_files:
        st.error(
            f"Unsupported coastline resolution: {resolution}. "
            "Choose '50m' or '110m'."
        )
        return [], []

    coastline_path = (
        Path(__file__).resolve().parent
        / "coastline"
        / coastline_files[resolution]
    )

    try:
        df_coast = pd.read_csv(coastline_path)
        return df_coast['Longitude'].tolist(), df_coast['Latitude'].tolist()
    except Exception as e:
        st.error(f"Failed to load the coastline file: {coastline_path.name} - {e}")
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

# Offline mode is listed first; pages must pass index=MAP_MODE_DEFAULT_INDEX to the widget.
# オフラインを先頭に置きつつ、既定値は "Standard" を維持する。
MAP_MODE_OPTIONS = [
    "Coastline (offline)",
    "Standard",
    "Satellite",
    "Bathymetry (Sea)",
    "Contour (GSI)",
]
MAP_MODE_DEFAULT = "Standard"
MAP_MODE_DEFAULT_INDEX = MAP_MODE_OPTIONS.index(MAP_MODE_DEFAULT)  # = 1

MAP_MODE_DESCRIPTIONS_JA = {
    "Coastline (offline)": (
        "同梱50m海岸線CSVと緯経線のみで表示します。通信不要です。 / "
        "Offline: bundled 50 m coastline CSV only, no external tiles."
    ),
    "Standard": "APIキー不要のOpenStreetMap背景です。",
    "Satellite": "USGSの衛星画像タイルを使います。",
    "Bathymetry (Sea)": "Esri World Ocean Baseの海底地形背景を使います。",
    "Contour (GSI)": "国土地理院の標準地図タイルを使います。",
}

# --- Connectivity check and offline fallback ---
# 通信確認とオフライン縮退

_CONNECTIVITY_CACHE_INTERVAL_S = 60  # re-check at most once per minute / 1分以内に重複確認しない
_CONNECTIVITY_TIMEOUT_S = 3

# Per-mode tile hosts used for connectivity checks.
# モードごとの接続確認先ホスト（モードに対応するタイルサーバー）。
_MODE_TILE_HOSTS: dict = {
    "Standard":          ("tile.openstreetmap.org",                  443),
    "Satellite":         ("basemap.nationalmap.gov",                 443),
    "Bathymetry (Sea)":  ("services.arcgisonline.com",               443),
    "Contour (GSI)":     ("cyberjapandata.gsi.go.jp",                443),
}
_DEFAULT_TILE_HOST = ("tile.openstreetmap.org", 443)


@lru_cache(maxsize=64)
def _check_connectivity_cached(bucket: int, host: str, port: int) -> bool:
    """Inner check keyed to (time-bucket, host, port). Use check_online_connectivity() instead.
    (時間バケット, ホスト, ポート)をキーにキャッシュ。直接呼ばず check_online_connectivity() を使うこと。
    Sockets are closed immediately after the connection test.
    接続確認後にソケットを即座にクローズする。
    """
    try:
        conn = socket.create_connection((host, port), timeout=_CONNECTIVITY_TIMEOUT_S)
        conn.close()
        return True
    except OSError:
        return False


def check_online_connectivity(mode: str = "Standard") -> bool:
    """Return True if the tile server for *mode* is reachable. Result cached for ~60 s.

    Uses the mode-specific tile host (e.g. Satellite → basemap.nationalmap.gov).
    Falls back to the OSM host for unknown modes.

    モードに対応するタイルサーバーに接続できる場合 True を返す。
    約60秒キャッシュ。未知のモードはOSMホストで代替確認する。
    """
    host, port = _MODE_TILE_HOSTS.get(mode, _DEFAULT_TILE_HOST)
    bucket = int(time.time()) // _CONNECTIVITY_CACHE_INTERVAL_S
    return _check_connectivity_cached(bucket, host, port)


def resolve_map_mode(map_mode: str):
    """Resolve effective map mode, falling back to offline if needed.

    Returns (effective_mode, fell_back). fell_back=True means an online mode
    was requested but connectivity to *that mode's* tile server is unavailable;
    pages should show OFFLINE_FALLBACK_WARNING to the user in that case.

    オンラインモードが選ばれたが、そのモードのタイルサーバーに通信できない場合、
    ("Coastline (offline)", True) を返す。ページ側で警告を表示すること。
    """
    if map_mode == "Coastline (offline)":
        return map_mode, False
    if not check_online_connectivity(map_mode):
        return "Coastline (offline)", True
    return map_mode, False


# Warning message displayed when falling back to offline mode.
# オフライン縮退時に表示する警告文。
OFFLINE_FALLBACK_WARNING = (
    "⚠️ Online map tiles are unavailable. Showing the local coastline map."
)


def add_coastline_overlay(fig) -> bool:
    """Overlay the bundled 50 m coastline CSV as a Scattermapbox trace.

    Idempotent: if a trace named ``_coastline_overlay`` already exists on
    *fig*, the function returns True without adding a duplicate.
    冪等: すでに _coastline_overlay トレースが存在する場合は追加せず True を返す。

    Applied to all map modes so that coastlines and observation points remain
    visible even when tile loading fails in the browser.
    全地図モードに重ね、タイル取得がブラウザ側で失敗しても海岸線が残るようにする。

    Returns True if the overlay is present (added now or already existed),
    False if coastline data is unavailable.
    データが利用できなかった場合は False を返す（例外は発生しない）。
    """
    import plotly.graph_objects as go  # avoid circular import at module load time

    # Idempotency guard — do not add a second trace if one already exists.
    if any(getattr(t, "name", None) == "_coastline_overlay" for t in fig.data):
        return True

    lon, lat = load_coastline_data(None, resolution="50m")
    if not lon:
        return False
    fig.add_trace(
        go.Scattermapbox(
            lon=lon,
            lat=lat,
            mode="lines",
            line=dict(width=0.8, color="rgba(70,70,70,0.55)"),
            showlegend=False,
            hoverinfo="none",
            name="_coastline_overlay",
        )
    )
    return True


def plot_bundled_coastline(ax, *, transform=None, zorder=None, resolution="50m", **kwargs):
    """Plot the bundled coastline CSV onto a Cartopy Axes (or any Axes-like object).

    Retrieves coordinate data via :func:`load_coastline_data` and calls
    ``ax.plot(lon, lat, ...)`` with the supplied keyword arguments.
    Does **not** invoke Cartopy's shapefile downloader.
    バンドル済み海岸線CSVをMatplotlib/Cartopy Axesに描画する。
    Cartopyのダウンローダーは呼び出さない。

    Parameters
    ----------
    ax:
        Matplotlib/Cartopy Axes (or any duck-typed object with a ``plot`` method).
    transform:
        Cartopy coordinate reference system (e.g. ``ccrs.PlateCarree()``).
    zorder:
        Drawing order passed to ``ax.plot``.
    resolution:
        Coastline CSV resolution tag forwarded to :func:`load_coastline_data`.
    **kwargs:
        Any additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    bool
        True if the coastline was plotted, False if no data was available.
    """
    lon, lat = load_coastline_data(None, resolution=resolution)
    if not lon:
        return False
    plot_kwargs = dict(kwargs)
    if transform is not None:
        plot_kwargs["transform"] = transform
    if zorder is not None:
        plot_kwargs["zorder"] = zorder
    ax.plot(lon, lat, **plot_kwargs)
    return True


# ---------------------------------------------------------------------------
# Self-contained Plotly HTML export
# ---------------------------------------------------------------------------

_PLOTLY_HTML_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "responsive": True,
}


def figure_to_self_contained_html(fig) -> bytes:
    """Return self-contained HTML bytes for *fig* with Plotly.js embedded inline.

    Uses ``include_plotlyjs=True`` so the saved file works offline without a CDN.
    The Plotly toolbar and scroll-zoom are always enabled in the saved file.

    Plotly.jsを埋め込んだ自己完結HTML（バイト列）を返す。
    保存後もCDN不要でオフライン閲覧・操作が可能。

    Parameters
    ----------
    fig:
        A Plotly figure object (``plotly.graph_objects.Figure`` or compatible).

    Returns
    -------
    bytes
        UTF-8-encoded HTML with Plotly.js inlined (~4–5 MB).
    """
    html_str = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        config=_PLOTLY_HTML_CONFIG,
    )
    return html_str.encode("utf-8")


def add_graticule_overlay(fig, lat_step: int = 30, lon_step: int = 30) -> None:
    """Add lat/lon graticule lines to the Plotly Mapbox figure.

    Intended for offline (white-bg) mode where no tile background is present.
    Each parallel and meridian is drawn as an independent line segment group,
    using None separators, to avoid artefacts across the antimeridian.

    オフライン地図（白背景）用の経緯線オーバーレイ。
    日付変更線をまたぐ描画アーティファクトを避けるため、
    None区切りで各線を独立した線分として描く。
    """
    import plotly.graph_objects as go  # avoid circular import at module load time

    lons_g: list = []
    lats_g: list = []

    # Parallels — horizontal lines at each lat_step degree
    for lat in range(-90, 91, lat_step):
        for lon in range(-180, 181):
            lons_g.append(float(lon))
            lats_g.append(float(lat))
        lons_g.append(None)
        lats_g.append(None)

    # Meridians — vertical lines at each lon_step degree
    # Split at the poles to keep each meridian as a tidy segment.
    for lon in range(-180, 181, lon_step):
        for lat in range(-90, 91):
            lons_g.append(float(lon))
            lats_g.append(float(lat))
        lons_g.append(None)
        lats_g.append(None)

    fig.add_trace(
        go.Scattermapbox(
            lon=lons_g,
            lat=lats_g,
            mode="lines",
            line=dict(width=0.4, color="rgba(150,150,150,0.35)"),
            showlegend=False,
            hoverinfo="none",
            name="_graticule_overlay",
        )
    )


def apply_map_style(fig, map_mode):
    """Apply the selected background tile layer to the Mapbox figure.

    Handles offline fallback internally via resolve_map_mode().  Pages should
    call resolve_map_mode() first and show OFFLINE_FALLBACK_WARNING when
    fell_back is True.  All tile sources have been verified for web-use
    licensing.  Note: CARTO basemaps now require an API key; Standard uses OSM.

    resolve_map_mode() でオフライン縮退を処理する。ページ側は先に
    resolve_map_mode() を呼び、fell_back=True なら警告を表示すること。
    """
    effective_mode, _ = resolve_map_mode(map_mode)

    if effective_mode == "Coastline (offline)":
        # Offline: white background only — no external tile URL at all.
        # オフライン: 白背景のみ。外部タイルURLは一切設定しない。
        fig.update_layout(mapbox_style="white-bg")

    elif effective_mode == "Standard":
        fig.update_layout(mapbox_style="open-street-map")

    elif effective_mode == "Satellite":
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "source": [
                    # USGS
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"],
                "sourceattribution": "USGS",
            }]
        )

    elif effective_mode == "Bathymetry (Sea)":
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "source": [
                    # Esri World Ocean Base
                    "https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
                ],
                "sourceattribution": "Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri",
            }]
        )

    elif effective_mode == "Contour (GSI)":
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "source": [
                    # Geospatial Information Authority of Japan (GSI) tiles
                    "https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png"],
                "sourceattribution": "国土地理院 (GSI)",
            }]
        )

    # Unified layout settings for maximum map area
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # Always overlay the bundled coastline so position reference is visible even
    # when online tile fetching fails in the browser.  Idempotent — safe to call
    # even if the page already added the overlay.
    add_coastline_overlay(fig)

    return fig


def apply_standard_map_layout(fig, height=480):
    """Apply the reusable full-width Plotly Mapbox layout used by map pages.

    Keep the legend inside the map so it remains readable without causing
    Plotly to reserve an external margin and shrink the geographic viewport.
    凡例を地図内に重ね、外側余白による地図領域の縮小を防ぐ。
    """
    fig.update_layout(
        mapbox=dict(domain=dict(x=[0.0, 1.0], y=[0.0, 1.0])),
        margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        autosize=True,
        height=height,
        legend=dict(
            x=0.01,
            y=0.01,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(150,150,150,0.5)",
            borderwidth=1,
        ),
    )
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


def _uploaded_filter_state_key(filter_key):
    """Return the session-state key used by a page's uploaded-data filter."""
    return f"uploaded_data_filter::{filter_key}"


def uploaded_dataset_selected(filter_key, state=None):
    """Return whether the shared Dataset chooser includes uploaded rows."""
    state = st.session_state if state is None else state
    return bool(
        state.get(_uploaded_filter_state_key(filter_key), {}).get(
            "include_uploaded_dataset", False
        )
    )


def combine_reference_and_uploaded_for_filtering(reference_df, uploaded_df):
    """Build a page-local dataframe for one common filtering workflow.

    This does not modify the loaded reference data or the session upload.  It
    simply lets the Dataset selector and range controls operate on both row
    sources during the current page run.
    """
    reference_copy = reference_df.copy()
    if uploaded_df is None or uploaded_df.empty:
        return reference_copy
    return pd.concat(
        [reference_copy, uploaded_df.copy()], ignore_index=True, sort=False
    )


def split_uploaded_rows(filtered_df, dataset_label=UPLOADED_DATA_LABEL):
    """Return reference rows and selected uploaded rows from a filtered frame."""
    if filtered_df is None or filtered_df.empty:
        empty = pd.DataFrame(columns=getattr(filtered_df, "columns", None))
        return empty, empty.copy()
    upload_mask = filtered_df["Dataset"].eq(dataset_label)
    return (
        filtered_df.loc[~upload_mask].copy(),
        filtered_df.loc[upload_mask].copy(),
    )


def filter_uploaded_data_for_sidebar(
    uploaded_df, filter_key, state=None, respect_visibility=True
):
    """Apply the common Data filtering choices to an uploaded-data copy.

    Uploaded rows remain independent from the reference dataframe: this helper
    only determines which uploaded rows are rendered as an overlay.  A missing
    uploaded column is deliberately ignored, so a partially mapped upload can
    still be used by pages that do not need that particular measurement.
    """
    if uploaded_df is None:
        return pd.DataFrame()

    result = uploaded_df.copy()
    state = st.session_state if state is None else state
    settings = state.get(_uploaded_filter_state_key(filter_key), {})
    if (
        settings.get("uploaded_dataset_mode", False)
        and not settings.get("include_uploaded_dataset", False)
    ):
        return result.iloc[0:0].copy()
    if respect_visibility and not settings.get("show", True):
        return result.iloc[0:0].copy()
    include_uploaded_dataset = settings.get("include_uploaded_dataset", False)
    if not settings.get("apply_reference_filters", False) and not include_uploaded_dataset:
        return result

    # Dataset and Transect choices identify reference cruises.  Uploads are
    # normally labelled "Uploaded data", so applying those choices would hide
    # every overlay by default.  Month is a shared physical field and is safe
    # to apply when it is supplied by the upload.
    categorical_filters = {"Month": settings.get("selected_months")}
    for column, selected_values in categorical_filters.items():
        if column in result.columns and selected_values is not None:
            result = result[result[column].isin(selected_values)].copy()

    numeric_filters = {
        "Year": settings.get("year_range"),
        "Longitude_degE": settings.get("longitude_range"),
        "Latitude_degN": settings.get("latitude_range"),
        "Depth_m": settings.get("depth_range"),
        "Salinity": settings.get("salinity_range"),
        "d18O": settings.get("d18o_range"),
        "Temperature_degC": settings.get("temperature_range"),
    }
    for column, value_range in numeric_filters.items():
        if column not in result.columns or value_range is None:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        result = result[values.between(value_range[0], value_range[1])].copy()
    return result


def sidebar_filter_and_display(
    df1,
    ref_data,
    data_source_JAPAN_SEA,
    data_source_AROUND_JAPAN,
    uploaded_df=None,
    uploaded_filter_key=None,
    uploaded_dataset_label=None,
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
        uploaded_df: Optional separately managed uploaded rows for an overlay.
        uploaded_filter_key: Page-specific key for the uploaded overlay state.
        uploaded_dataset_label: Optional Dataset chooser label for uploaded rows.

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
        st.caption(MANUAL_FILTER_APPLY_NOTE)
        
        submit_top = st.form_submit_button(
            "Apply settings", **stretch_width_kwargs(st.form_submit_button)
        )

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
            if (
                uploaded_dataset_label is not None
                and uploaded_dataset_label not in Transect_list
            ):
                Transect_list.append(uploaded_dataset_label)
            # print(Transect_list,"<<< Dataset list")
            
            selected_dataset = st.multiselect('Choose datasets', Transect_list,default=Transect_list)
            user_excel_count = int(
                df1["Dataset"].eq(USER_EXCEL_DATA_LABEL).sum()
            )
            if user_excel_count:
                st.caption(
                    f"{USER_EXCEL_DATA_LABEL}: {user_excel_count:,} rows loaded "
                    "from the always-loaded local table."
                )

            
        # datasetのフィルタリング　2026/03/09追加
        # When a page supplies uploaded rows in its local filter dataframe,
        # Uploaded data behaves exactly like any other sub-dataset.
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
        
        year_values = pd.to_numeric(df1['Year'], errors='coerce').dropna()
        if year_values.empty:
            min_df_year, max_df_year = 0, 0
        else:
            min_df_year = int(year_values.min())
            max_df_year = int(year_values.max())
        
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
        lon_values = pd.to_numeric(df1['Longitude_degE'], errors='coerce').dropna()
        lat_values = pd.to_numeric(df1['Latitude_degN'], errors='coerce').dropna()
        if lon_values.empty:
            min_df_lon, max_df_lon = -180, 180
        else:
            min_df_lon = int(math.floor(lon_values.min()))
            max_df_lon = int(math.ceil(lon_values.max()))
        if lat_values.empty:
            min_df_lat, max_df_lat = -90, 90
        else:
            max_df_lat = int(math.ceil(lat_values.max()))
            min_df_lat = int(math.floor(lat_values.min()))

        # Prevent st.slider crash when all values share the same coordinate
        if min_df_lon == max_df_lon:
            max_df_lon += 1
        if min_df_lat == max_df_lat:
            max_df_lat += 1

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
        if region_preset_crosses_dateline(area_filter_preset):
            # 単一の Longitude range スライダーでは日付変更線をまたぐ範囲を
            # 表現できないため、このプリセットではスライダー値ではなく
            # プリセット自身の経度範囲を OR 条件で直接適用する
            # (緯度は下のスライダーで通常通り絞り込む)。
            # A single Longitude range slider cannot express a span that
            # crosses the antimeridian, so for these presets longitude is
            # matched by the preset's own east/west arms (OR condition)
            # instead of the slider values above; latitude still narrows
            # normally via its own slider below.
            st.caption(
                f"'{area_filter_preset}' spans the antimeridian (dateline); "
                "longitude is matched by the preset's own east/west arms "
                "rather than the slider above."
            )
            df1 = df1[
                region_preset_longitude_mask(df1['Longitude_degE'], area_filter_preset)
                | df1['Longitude_degE'].isna()
            ]
        else:
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
        depth_values = pd.to_numeric(df1['Depth_m'], errors='coerce').dropna()
        if depth_values.empty:
            min_depth, max_depth = 0, 0
        else:
            min_depth = int(math.floor(depth_values.min()))
            max_depth = int(math.ceil(depth_values.max()))
        
        # 2. スライダーの設定
        if min_depth == max_depth:
            slider_max = max_depth + 1
        else:
            slider_max = max_depth
        
        if min_depth == max_depth:
            default_value = (min_depth, slider_max)
        elif min_depth > 0:
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
        
        salinity_values = pd.to_numeric(df1['Salinity'], errors='coerce').dropna()
        if salinity_values.empty:
            min_df_sal, max_df_sal = 0, 0
        else:
            min_df_sal = int(math.floor(salinity_values.min()))
            max_df_sal = int(math.ceil(salinity_values.max()))

        
        # 2. スライダーの設定
        if min_df_sal == max_df_sal:
            slider_max_sal = max_df_sal + 1
        else:
            slider_max_sal = max_df_sal
        
        if min_df_sal == max_df_sal:
            default_sal = (min_df_sal, slider_max_sal)
        elif min_df_sal > 0:
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

        # The upload is never merged into df1.  It can nevertheless follow the
        # same filter choices for display, which keeps large uploads from
        # burdening map/section selectors unnecessarily.
        # Pages that expose uploads as a normal sub-dataset (currently the
        # vertical section workflow) use that single chooser for both display
        # and calculation.  Keep the separate overlay controls only for the
        # other native-overlay pages, where no Dataset chooser is available.
        show_uploaded_data = True
        apply_reference_filters = False
        if uploaded_filter_key is not None and uploaded_dataset_label is None:
            uploaded_count = 0 if uploaded_df is None else len(uploaded_df)
            with st.expander("Uploaded data", expanded=False):
                show_uploaded_data = st.checkbox(
                    "Show uploaded data",
                    value=True,
                    key=f"{uploaded_filter_key}::show_uploaded_data",
                    disabled=uploaded_count == 0,
                )
                apply_reference_filters = st.checkbox(
                    "Apply current data filters to uploaded data",
                    value=False,
                    key=f"{uploaded_filter_key}::apply_uploaded_filters",
                    disabled=uploaded_count == 0,
                    help=(
                        "Filters only the upload overlay. Uploaded rows are not "
                        "added to interpolation, statistics, or reference data."
                    ),
                )
                if uploaded_count:
                    st.caption(
                        f"{uploaded_count:,} uploaded rows available for the overlay."
                    )
                else:
                    st.caption("No uploaded data is currently available.")

        if uploaded_filter_key is not None:
            st.session_state[_uploaded_filter_state_key(uploaded_filter_key)] = {
                "show": show_uploaded_data,
                "apply_reference_filters": apply_reference_filters,
                "uploaded_dataset_mode": uploaded_dataset_label is not None,
                "include_uploaded_dataset": (
                    uploaded_dataset_label in selected_dataset
                    if uploaded_dataset_label is not None
                    else False
                ),
                "selected_dataset": list(selected_dataset),
                "selected_cruise": list(selected_cruise),
                "selected_months": list(selected_months) if selected_months else [],
                "year_range": (sld_year_min, sld_year_max),
                "longitude_range": (sld_lon_min, sld_lon_max),
                "latitude_range": (sld_lat_min, sld_lat_max),
                "depth_range": (sld_depth_min, sld_depth_max),
                "salinity_range": (sld_sal_min, sld_sal_max),
                "d18o_range": (sld_d18O_min, sld_d18O_max),
                "temperature_range": (sld_temp_min, sld_temp_max),
            }

 
        
        submit_bottom = st.form_submit_button(
            "Apply settings!", **stretch_width_kwargs(st.form_submit_button)
        )
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
    uploaded_filtered_count = 0
    if uploaded_dataset_label is not None and "Dataset" in df1.columns:
        uploaded_filtered_count = int(
            df1["Dataset"].eq(uploaded_dataset_label).sum()
        )

    
    # バリデーション処理
    if df_empty:  #データが無かったとき
        st.warning('no data found')
        # 条件を満たないときは処理を停止する
        st.stop()
    else: #データがあったとき
        if uploaded_dataset_label is not None:
            st.write(
                f"{data_found_num} data found "
                f"(Uploaded data: {uploaded_filtered_count})"
            )
        else:
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
            hide_index=True,
            **stretch_width_kwargs(st.dataframe),
        )

        st.markdown("**Filtered data counts by dataset**")
        df_selected_counts = pd.DataFrame(
            [{"Dataset": key, "Rows": value} for key, value in d_select_add2.items()]
        )
        if not df_selected_counts.empty:
            st.dataframe(
                df_selected_counts,
                hide_index=True,
                **stretch_width_kwargs(st.dataframe),
            )

        st.markdown("**Summary statistics**")
        df_stats = filtered_statistics_dataframe(summary)
        if not df_stats.empty:
            numeric_cols = ["Mean", "Stdev", "Min", "Max"]
            df_stats[numeric_cols] = df_stats[numeric_cols].round(3)
            st.dataframe(df_stats, **stretch_width_kwargs(st.dataframe))
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
