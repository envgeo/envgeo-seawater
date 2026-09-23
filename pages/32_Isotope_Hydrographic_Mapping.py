#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isotope and hydrographic mapping visualizer for EnvGeo-Seawater data.

Created: 2023-04-22
Author: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""




# --- Version info ---
version = "1.3.3"  # 2026-09-23

# ToDo
# このバージョンは補完計算の調整が必要



fig_title = "envgeo-seawater-database"  # 2026/02/12


import streamlit as st
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import plotly.express as px
import math
import envgeo_utils
import envgeo_user_data
from scipy.interpolate import griddata # コンターマップ用
import cartopy.io.shapereader as shapereader  # ローカル NE land shapefile 読み込み用
import io # ファイル処理用
import pathlib
import warnings


MAP_PARAMETER_LABELS = {
    "d18O": r"$\delta^{18}$O (VSMOW)",
    "dD": r"$\delta$D (VSMOW)",
    "d-excess": "d-excess",
    "Salinity": "Salinity",
    "Temperature_degC": "Temperature (degC)",
}

MAP_PARAMETER_PLOTLY_LABELS = {
    "d18O": "δ18O (‰)",
    "dD": "δD (‰)",
    "d-excess": "d-excess (‰)",
    "Salinity": "Salinity",
    "Temperature_degC": "Temperature (degC)",
}


def safe_cartopy_extent(lon_min, lon_max, lat_min, lat_max, lon_domain_min, lon_domain_max):
    """
    Return a Cartopy-safe map extent.

    Cartopy may produce NaN axis limits when the longitude extent exactly matches
    the full projection domain, especially on Streamlit Cloud with 0-360 maps.
    This keeps the requested view visually unchanged while nudging the bounds
    slightly inside the projection seam.
    """
    lon_min = max(float(lon_domain_min), float(lon_min))
    lon_max = min(float(lon_domain_max), float(lon_max))
    lat_min = max(-90.0, float(lat_min))
    lat_max = min(90.0, float(lat_max))

    epsilon = 0.001
    lon_domain_span = float(lon_domain_max) - float(lon_domain_min)
    if (lon_max - lon_min) >= (lon_domain_span - epsilon):
        lon_min = float(lon_domain_min) + epsilon
        lon_max = float(lon_domain_max) - epsilon

    if lon_min >= lon_max:
        lon_min = float(lon_domain_min) + epsilon
        lon_max = float(lon_domain_max) - epsilon

    if lat_min >= lat_max:
        lat_min = max(-90.0, lat_min - epsilon)
        lat_max = min(90.0, lat_max + epsilon)

    return lon_min, lon_max, lat_min, lat_max


def set_cartopy_extent_safely(ax, extent):
    """
    Apply a map extent without crashing on Cartopy projection seam issues.

    Streamlit Cloud can raise ``ValueError: Axis limits cannot be NaN or Inf``
    for nearly global PlateCarree extents. If that happens, fall back to a
    global view so the figure still renders.
    """
    try:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        return True
    except ValueError:
        ax.set_global()
        return False


def map_display_preset_bounds(region_label, lon_center):
    """
    Convert a map-region preset to the longitude frame used by the map center.

    Map display presets change only the figure view. They do not change the
    sidebar data-filtering result.
    """
    if region_label not in envgeo_utils.MAP_REGION_PRESETS:
        return None

    lon_min, lon_max, lat_min, lat_max = envgeo_utils.MAP_REGION_PRESETS[region_label]["bounds"]
    if lon_center == 0:
        if lon_min > 180:
            lon_min -= 360
        if lon_max > 180:
            lon_max -= 360
        if lon_min > lon_max:
            lon_min, lon_max = -180.0, 180.0
    else:
        if lon_min < 0:
            lon_min += 360
        if lon_max < 0:
            lon_max += 360

    return (float(lon_min), float(lon_max)), (float(lat_min), float(lat_max))


def get_parameter_color_range_defaults(parameter, ref_data, data_source_global):
    """
    Return slider limits, default color range, and step for each map parameter.

    地図で色分けするパラメーターごとに、カラーバー範囲の初期値を返します。
    """
    if parameter == "d18O":
        default_range = (-5.0, 2.0) if ref_data == data_source_global else (-1.5, 1.0)
        return -20.0, 5.0, default_range, 0.1
    if parameter == "dD":
        default_range = (-50.0, 20.0) if ref_data == data_source_global else (-20.0, 10.0)
        return -200.0, 100.0, default_range, 1.0
    if parameter == "d-excess":
        return -30.0, 40.0, (-10.0, 25.0), 0.5
    if parameter == "Salinity":
        default_range = (0.0, 40.0) if ref_data == data_source_global else (30.0, 36.0)
        return 0.0, 50.0, default_range, 0.1
    if parameter == "Temperature_degC":
        default_range = (-5.0, 35.0) if ref_data == data_source_global else (-2.0, 30.0)
        return -5.0, 45.0, default_range, 0.5

    return -20.0, 20.0, (-5.0, 5.0), 0.1


# ── Local Natural Earth 50m Land shapefile ────────────────────────────────────
# The shapefile is bundled in the repository so that the land mask can be drawn
# without any external Natural Earth download, even in offline environments.
# Path is resolved relative to this script's directory so it works regardless
# of the working directory when Streamlit launches the page.
_NE50M_LAND_DIR = (
    pathlib.Path(__file__).parent.parent / "coastline" / "natural_earth_50m_land"
)
_NE50M_LAND_SHP = _NE50M_LAND_DIR / "ne_50m_land.shp"
_NE50M_LAND_REQUIRED_EXTS = (".shp", ".shx", ".dbf")


def _load_ne50m_land_geometries():
    """Return a list of Shapely geometries from the bundled NE 50m land shapefile.

    Returns an empty list (with a Streamlit warning) if any required component
    is missing.  Never fetches from the network.

    Rendering with an empty list degrades gracefully: the land mask is skipped,
    but coastlines and data points are still drawn.
    """
    missing = [
        _NE50M_LAND_DIR / ("ne_50m_land" + ext)
        for ext in _NE50M_LAND_REQUIRED_EXTS
        if not (_NE50M_LAND_DIR / ("ne_50m_land" + ext)).exists()
    ]
    if missing:
        st.warning(
            "Land mask (Natural Earth 50m) could not be drawn: "
            f"missing bundled file(s): {[p.name for p in missing]}. "
            "Coastlines and data points are still displayed."
        )
        return []
    try:
        reader = shapereader.Reader(str(_NE50M_LAND_SHP))
        return list(reader.geometries())
    except Exception as exc:
        st.warning(
            f"Land mask (Natural Earth 50m) could not be loaded: {exc}. "
            "Coastlines and data points are still displayed."
        )
        return []


def main():
    
        
    # タイトル
    st.header(f'Isotope & Hydrographic Mapping ({version})')
    st.caption(
        "Map d18O, dD, d-excess, salinity, temperature, and related seawater parameters."
    )
  
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
    ref_data = st.radio("Data source (see Home > About):", (data_source_JAPAN_SEA, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True)



    ##############################################################################
    # 選択したデータセットの文献表示
    ##############################################################################
    
    if ref_data == data_source_JAPAN_SEA:
        st.write(envgeo_utils.refs_JAPAN_SEA)
        
    elif ref_data == data_source_AROUND_JAPAN:
        # st.text('including data from previous reports')
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
    # アップロードデータUI（Integrated埋め込み時はファイルアップロードを省略）
    ##############################################################################
    embedded_in_integrated = (
        st.session_state.get(envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY)
        == "32_Isotope_Hydrographic_Mapping.py"
    )
    if embedded_in_integrated:
        uploaded_df = envgeo_utils.get_uploaded_data()
    else:
        uploaded_df = envgeo_user_data.render_upload_panel(
            "iso_map",
            "Longitude and latitude columns are required to plot uploaded "
            "locations on the map. The currently selected map parameter "
            "(d18O, dD, etc.) is used for color when available.",
        )
    uploaded_df = envgeo_user_data.render_column_controls(
        uploaded_df,
        {
            "Longitude column": "Longitude_degE",
            "Latitude column": "Latitude_degN",
        },
        "iso_map",
    )
    uploaded_style = envgeo_user_data.render_marker_style_controls(
        uploaded_df, "iso_map"
    )

    ##############################################################################
    # サイドバーここから　　df1フィルタリング　も一括で
    #　2026/03/06　Min-Maxをdfから取得に変更
    #  緯度経度などは型変換をせず、そのまま最小・最大を取得
    ##############################################################################
    
    # 関数の呼び出し
    # すべての変数を順番通りに受け取ります
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
     submitted) = envgeo_utils.sidebar_filter_and_display(
         envgeo_utils.combine_reference_and_uploaded_for_filtering(
             df1, uploaded_df
         ),
         ref_data, data_source_JAPAN_SEA, data_source_AROUND_JAPAN,
         uploaded_df=uploaded_df, uploaded_filter_key="isotope_mapping",
         uploaded_dataset_label=envgeo_utils.UPLOADED_DATA_LABEL,
     )
    # Use the selected, integrated table for both scatter and contour
    # calculations.  Keep a separate uploaded subset only to redraw it above
    # the calculated layer with the user-selected marker style.
    filtered_integrated_df = df1.copy()
    _, uploaded_df = envgeo_utils.split_uploaded_rows(
        filtered_integrated_df, envgeo_utils.UPLOADED_DATA_LABEL
    )
    df1 = filtered_integrated_df

    map_parameter_candidates = [
        "d18O",
        "dD",
        "d-excess",
        "Salinity",
        "Temperature_degC",
    ]
    map_parameter_options = [
        item for item in map_parameter_candidates
        if item in df1.columns
    ]
    if not map_parameter_options:
        st.warning("No mappable isotope or hydrographic parameters are available.")
        return

    st.subheader("Map")
    parameter_col, map_type_col = st.columns([1, 1])
    with parameter_col:
        selected_parameter = st.selectbox(
            "Mapped parameter",
            map_parameter_options,
            index=0,
            help=(
                "Choose the seawater parameter plotted on the map. "
                "Rows without values for the selected parameter are excluded from the map."
            ),
        )
    with map_type_col:
        map_type = st.radio(
            "Map type",
            ("Scatter Map", "Contour Map"),
            horizontal=True,
        )
    parameter_label = MAP_PARAMETER_LABELS.get(selected_parameter, selected_parameter)
    parameter_plotly_label = MAP_PARAMETER_PLOTLY_LABELS.get(selected_parameter, selected_parameter)




    ##############################################################################
    # 図の中心とスケール変更
    ##############################################################################

    # サイドバーの中にコンテナを作成し、境界線（border）を有効にする
    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "MAP_DISPLAY_SETTINGS_LABEL", "Map display settings"))
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        
        center_option = st.radio(
            ":blue[Map Center:]",
            ("Atlantic (0°)", "Pacific (180°)"),
            horizontal=True
        )
        
        lon_center = 0 if "Atlantic" in center_option else 180

        colormap_options = envgeo_utils.get_plotly_colormap_options(selected_parameter)
        selected_colormap_label = st.selectbox(
            "Colormap",
            list(colormap_options.keys()),
            index=list(colormap_options.keys()).index("Jet"),
            help=(
                "Choose the color palette used for parameter maps. "
                "cmocean palettes are designed for oceanographic data."
            ),
        )
        map_plotly_colorscale = envgeo_utils.get_plotly_colormap(
            selected_parameter,
            selected_colormap_label,
        )
        map_matplotlib_colormap = envgeo_utils.get_matplotlib_colormap(
            selected_parameter,
            selected_colormap_label,
        )

        # Re-map longitudes into the visible 360-degree window of the selected map center.
        # 選択した地図中心で見えている 360 度の範囲に経度を並べ替える。
        def normalize_lon_to_center(lon, center):
            return ((np.asarray(lon) - (center - 180)) % 360) + (center - 180)

        # Pacific-centered maps use 0-360 so regional windows such as 120-240E remain selectable.
        # 太平洋中心では 0-360 を使い、120-240E のような範囲をそのまま選べるようにする。
        lon_slider_min = -180 if lon_center == 0 else 0
        lon_slider_max = 180 if lon_center == 0 else 360
        
        # --- 図のスケール設定 (表示は整数、内部計算は微小オフセットあり) ---
        if ref_data == data_source_JAPAN_SEA:
            # 地図の描画範囲（日本海）
            map_lon_default = (120, 145)
            map_lat_default = (20, 45)
            lat_slider_min, lat_slider_max = 0, 70
    
        elif ref_data == data_source_AROUND_JAPAN:
            # 地図の描画範囲（日本周辺）
            map_lon_default = (120, 180) if lon_center == 0 else (120, 240)
            map_lat_default = (0, 55)
            lat_slider_min, lat_slider_max = -70, 70
    
        else:
            # 地図の描画範囲（全体）
            map_lon_default = (-180, 180) if lon_center == 0 else (0, 360)
            map_lat_default = (-90, 90)
            lat_slider_min, lat_slider_max = -90, 90

        dataset_lon_default = map_lon_default
        dataset_lat_default = map_lat_default

        region_preset_options = ["Dataset default"] + list(envgeo_utils.MAP_REGION_PRESETS)
        region_preset = st.selectbox(
            "Region preset",
            region_preset_options,
            index=0,
            help=(
                "Set the map display extent from a common ocean-region preset. "
                "This changes only the figure view, not the sidebar-filtered data."
            ),
        )
        preset_bounds = map_display_preset_bounds(region_preset, lon_center)
        if preset_bounds is not None:
            preset_lon_default, preset_lat_default = preset_bounds
            candidate_lon_default = (
                max(lon_slider_min, int(math.floor(preset_lon_default[0]))),
                min(lon_slider_max, int(math.ceil(preset_lon_default[1]))),
            )
            candidate_lat_default = (
                max(lat_slider_min, int(math.floor(preset_lat_default[0]))),
                min(lat_slider_max, int(math.ceil(preset_lat_default[1]))),
            )
            if (
                candidate_lon_default[0] < candidate_lon_default[1]
                and candidate_lat_default[0] < candidate_lat_default[1]
            ):
                map_lon_default = candidate_lon_default
                map_lat_default = candidate_lat_default
            else:
                map_lon_default = dataset_lon_default
                map_lat_default = dataset_lat_default

        map_state_key = f"map_display::{ref_data}::{lon_center}::{region_preset}"

        color_range_min, color_range_max, color_range_default, color_range_step = (
            get_parameter_color_range_defaults(
                selected_parameter,
                ref_data,
                data_source_GLOBAL,
            )
        )

        map_lon_raw = st.slider(
            "Map Longitude",
            lon_slider_min,
            lon_slider_max,
            map_lon_default,
            step=1,
            key=f"{map_state_key}::longitude",
        )
        map_lat_raw = st.slider(
            "Map Latitude",
            lat_slider_min,
            lat_slider_max,
            map_lat_default,
            step=1,
            key=f"{map_state_key}::latitude",
        )

        selected_color_range = st.slider(
            label=f"{selected_parameter} range for colorbar",
            min_value=color_range_min,
            max_value=color_range_max,
            value=color_range_default,
            step=color_range_step,
            key=f"{map_state_key}::color_range::{selected_parameter}",
        )
        colorbar_thickness_value = st.slider(
            "Colorbar thickness",
            min_value=2,
            max_value=10,
            value=4,
            step=1,
            key=f"{map_state_key}::colorbar_thickness",
            help=(
                "Adjust the thickness of the horizontal parameter colorbar "
                "in the Matplotlib scatter and contour maps."
            ),
        )
        colorbar_length_value = st.slider(
            "Colorbar length",
            min_value=40,
            max_value=100,
            value=90,
            step=5,
            key=f"{map_state_key}::colorbar_length",
            help=(
                "Adjust the displayed length of the horizontal parameter colorbar. "
                "100 uses the full available width."
            ),
        )
        colorbar_font_size = st.number_input(
            "Colorbar font size",
            min_value=8,
            max_value=20,
            value=12,
            step=1,
            key=f"{map_state_key}::colorbar_font_size",
            help="Adjust the label and tick font size of the parameter colorbar.",
        )

        # アップロードデータのうちカラーバー要素が無いポイントの表示切替
        show_nodata_uploaded = st.checkbox(
            f"Show uploaded points without {selected_parameter} values",
            value=True,
            key=f"{map_state_key}::show_nodata_uploaded",
            help="Show or hide uploaded map points that have no value for the mapped parameter.",
        )

        parameter_min, parameter_max = selected_color_range
        colorbar_thickness = colorbar_thickness_value / 100
        colorbar_length = colorbar_length_value / 100

        # 内部計算用に0.001のオフセットを適用
        map_lon_min, map_lon_max = map_lon_raw[0] - 0.001, map_lon_raw[1] + 0.001
        map_lat_min, map_lat_max = map_lat_raw[0] - 0.001, map_lat_raw[1] + 0.001

   



    ##############################################################################
    #  ここから図の設定と描画
    ##############################################################################


    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    st.caption(getattr(envgeo_utils, "MAP_AREA_HELP_TEXT", "Map center, extent, colormap, and figure settings can be adjusted in the sidebar."))



    
    ###############################################################################################
    ###############################################################################################
    # 計算できない，もしくはカラーバー用のデータが無い場合に除外

    # 1. 計算する
    original_len_df1 = len(df1)
    # 2. 【追加】計算できなかった行（null）をその場で除外する
    df1 = df1.dropna(subset=[selected_parameter])
   
    # 消えた数を出力
    removed_len_df1 = original_len_df1 - len(df1)
    plotted_len_df1 = original_len_df1 - removed_len_df1

    if removed_len_df1 > 0:
        st.caption(
            f":red[Note: {plotted_len_df1} samples were plotted and "
            f"{removed_len_df1} samples were excluded due to no {selected_parameter} data.]"
        )

    if df1.empty:
        st.warning(f"No valid {selected_parameter} data are available for the selected conditions.")
        return
       
    ###############################################################################################
    ###############################################################################################
    
    

  


    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    
    
    #ファイル名用の項目
    
    #全体のタイトル名
    main_title = f"{fig_title} - {selected_parameter}"
    
    # --- 月 (スライダー用) ---
    # sub_title = 'Lon:'+str(sld_lon_min)+'-'+str(sld_lon_max)+', Lat:'+str(sld_lat_min)+'-'+str(sld_lat_max)+', Y:'+str(sld_year_min)+'-'+str(sld_year_max)+', M:'+str(sld_month_min)+'-'+str(sld_month_max)+', S:'+str(sld_sal_min)+'-'+str(sld_sal_max)+', D:'+str(sld_depth_min)+'-'+str(sld_depth_max)+'m'
    # --- 月 (multiselect用) ---
    ### もし「月が多すぎてサブタイトルが長くなる」のが嫌な場合
   # 月の表示ロジック
    if len(selected_months) == 12:
        month_display = "All"
    elif len(selected_months) == 0:
        month_display = "None"
    else:
        # 標準機能だけで「1-3」のように短縮するロジック
        sorted_m = sorted(list(set(selected_months)))
        ranges = []
        if sorted_m:
            start = sorted_m[0]
            for i in range(len(sorted_m)):
                # 次の要素が連続していない、または最後の要素の場合に書き出し
                if i + 1 == len(sorted_m) or sorted_m[i+1] != sorted_m[i] + 1:
                    end = sorted_m[i]
                    ranges.append(f"{start}-{end}" if start != end else str(start))
                    if i + 1 < len(sorted_m):
                        start = sorted_m[i+1]
        month_display = ", ".join(ranges)
        
        

    sub_title = f"Lon:{sld_lon_min}-{sld_lon_max}, Lat:{sld_lat_min}-{sld_lat_max}, Y:{sld_year_min}-{sld_year_max}, M:{month_display}, S:{sld_sal_min}-{sld_sal_max}, D:{sld_depth_min}-{sld_depth_max}m"


    main_title2 = sub_title
    
    sub_title2 = ''
    
    title_head = str(main_title+'\n'+sub_title+'\n'+sub_title2)
    title_head2 = title_head.replace('_', ' ') #図のタイトル表示用

    # File names for downloaded figures / 図保存用のファイル名
    safe_parameter_name = envgeo_utils.safe_filename_text(selected_parameter)

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################


    ##############################################################################
    # アップロードデータ前処理と品質チェック
    ##############################################################################
    uploaded_map_valid = pd.DataFrame()
    if not uploaded_df.empty:
        _has_position = {
            "Longitude_degE", "Latitude_degN"
        }.issubset(uploaded_df.columns)
        if _has_position:
            _udf = uploaded_df.copy()
            _udf["Longitude_degE"] = pd.to_numeric(
                _udf["Longitude_degE"], errors="coerce"
            )
            _udf["Latitude_degN"] = pd.to_numeric(
                _udf["Latitude_degN"], errors="coerce"
            )
            uploaded_map_valid = (
                _udf.dropna(subset=["Longitude_degE", "Latitude_degN"])
                .loc[lambda d: d["Latitude_degN"].between(-90, 90)]
                .copy()
            )
        _u_total = len(uploaded_df)
        _u_plotted = len(uploaded_map_valid)
        if (
            _has_position
            and uploaded_style["color_mode"] == "Use current colorbar when possible"
            and selected_parameter in uploaded_map_valid.columns
            and not show_nodata_uploaded
        ):
            _u_plotted = int(
                pd.to_numeric(
                    uploaded_map_valid[selected_parameter], errors="coerce"
                ).notna().sum()
            )
        _u_excluded = _u_total - _u_plotted
        if _has_position:
            st.caption(
                f":blue[Uploaded overlay: {_u_plotted:,} / {_u_total:,} samples "
                f"plotted ({_u_excluded:,} excluded due to missing or invalid "
                "longitude/latitude).]"
            )
        else:
            st.info(
                "Uploaded data is not shown because longitude and latitude "
                "columns are not assigned. "
                "Use \"Uploaded data columns\" in the sidebar."
            )
        _uploaded_quality_df = envgeo_utils.get_quality_rows(uploaded_df)
        with st.expander("Uploaded data quality check", expanded=False):
            envgeo_utils.render_quality_flag_criteria_note()
            st.write(
                f"Quality-flagged rows: {len(_uploaded_quality_df):,} / "
                f"{_u_total:,}"
            )
            if _uploaded_quality_df.empty:
                st.success("No uploaded rows triggered the current quality rules.")
            else:
                st.dataframe(
                    _uploaded_quality_df,
                    **envgeo_utils.stretch_width_kwargs(st.dataframe),
                )

    plt.rcParams["font.size"] = 15

    # Keep the requested extent inside the current longitude domain.
    # Exact full-globe bounds can fail in Cartopy on Streamlit Cloud, so the
    # helper nudges them slightly inside the projection seam.
    map_lon_min, map_lon_max, map_lat_min, map_lat_max = safe_cartopy_extent(
        map_lon_min,
        map_lon_max,
        map_lat_min,
        map_lat_max,
        lon_slider_min,
        lon_slider_max,
    )
    
    if map_type == "Scatter Map":
        #############################################################
        # Scatter Map
        #############################################################
        fig = plt.figure(figsize=(12, 8), facecolor="white", dpi=150)
        
        ax = fig.add_subplot(
            1, 1, 1,
            projection=ccrs.PlateCarree(central_longitude=lon_center)
        )
        
        set_cartopy_extent_safely(
            ax,
            [map_lon_min, map_lon_max, map_lat_min, map_lat_max],
        )
        
        # Draw land mask from bundled NE 50m shapefile (no network access).
        # Drawing order: contours (zorder=1) → land mask (zorder=2) →
        # coastlines (zorder=3) → gridlines (zorder=4) → data points (zorder≥5).
        _ne_geoms_sc = _load_ne50m_land_geometries()
        if _ne_geoms_sc:
            ax.add_geometries(
                _ne_geoms_sc,
                crs=ccrs.PlateCarree(),
                facecolor="white",
                edgecolor="none",
                zorder=2,
            )
        envgeo_utils.plot_bundled_coastline(
            ax,
            transform=ccrs.PlateCarree(),
            zorder=3,
            color="0.35",
            linewidth=0.6,
        )
        ax.gridlines(draw_labels=True, zorder=4)
        
        # Plot sample points in the same longitude frame as the selected map extent.
        # 観測点も、選択した表示範囲と同じ経度系に変換してから描画する。
        lon_wrapped = normalize_lon_to_center(df1["Longitude_degE"].values, lon_center)
        
        ax_scatter = ax.scatter(
            lon_wrapped,
            df1["Latitude_degN"],
            c=df1[selected_parameter],
            cmap=map_matplotlib_colormap,
            s=10,
            alpha=0.7,
            vmin=parameter_min,
            vmax=parameter_max,
            transform=ccrs.PlateCarree(),
            zorder=1
        )
        
        # ---- Scatter Map用のカラーバーを追加 ----
        cbar_scatter = fig.colorbar(
            ax_scatter,
            ax=ax,
            orientation="horizontal", # 横向き
            pad=0.08,                  # 地図との隙間
            fraction=colorbar_thickness, # カラーバーの太さ
            shrink=colorbar_length,    # カラーバーの長さ
            aspect=25,                 # カラーバーの細長さ
            extend="neither"           # 【重要】ここを "neither" にすると両端が□になります
        )
        cbar_scatter.set_label(parameter_label, fontsize=colorbar_font_size)
        cbar_scatter.ax.tick_params(labelsize=colorbar_font_size)

        # --- Uploaded data overlay (Scatter Map / 最前面) ---
        if not uploaded_map_valid.empty:
            _lon_up_sc = normalize_lon_to_center(
                uploaded_map_valid["Longitude_degE"].values, lon_center
            )
            _use_color_sc = (
                uploaded_style["color_mode"] == "Use current colorbar when possible"
                and selected_parameter in uploaded_map_valid.columns
            )
            if _use_color_sc:
                _up_cv_sc = pd.to_numeric(
                    uploaded_map_valid[selected_parameter], errors="coerce"
                )
                _up_valid_sc = _up_cv_sc.notna()
                if _up_valid_sc.any():
                    ax.scatter(
                        _lon_up_sc[_up_valid_sc.values],
                        uploaded_map_valid.loc[_up_valid_sc, "Latitude_degN"],
                        c=_up_cv_sc[_up_valid_sc],
                        cmap=map_matplotlib_colormap,
                        s=uploaded_style["size"],
                        alpha=uploaded_style["alpha"],
                        vmin=parameter_min,
                        vmax=parameter_max,
                        marker=uploaded_style["marker"],
                        linewidths=uploaded_style["outline_width"],
                        edgecolors=uploaded_style["outline_color"],
                        transform=ccrs.PlateCarree(),
                        zorder=10,
                    )
                if (~_up_valid_sc).any() and show_nodata_uploaded:
                    ax.scatter(
                        _lon_up_sc[~_up_valid_sc.values],
                        uploaded_map_valid.loc[~_up_valid_sc, "Latitude_degN"],
                        c=uploaded_style["color"],
                        s=uploaded_style["size"],
                        alpha=uploaded_style["alpha"],
                        marker=uploaded_style["marker"],
                        linewidths=uploaded_style["outline_width"],
                        edgecolors=uploaded_style["outline_color"],
                        transform=ccrs.PlateCarree(),
                        zorder=10,
                    )
            else:
                ax.scatter(
                    _lon_up_sc,
                    uploaded_map_valid["Latitude_degN"],
                    c=uploaded_style["color"],
                    s=uploaded_style["size"],
                    alpha=uploaded_style["alpha"],
                    marker=uploaded_style["marker"],
                    linewidths=uploaded_style["outline_width"],
                    edgecolors=uploaded_style["outline_color"],
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                )

        ax.set_title(title_head2,fontsize=15)
        
        # PNG保存（Scatter）
        img_scatter = io.BytesIO()
        fig.savefig(img_scatter, format="png", dpi=300, bbox_inches="tight")
        img_scatter.seek(0)

        st.pyplot(fig)
        
        st.download_button(
            "Download Scatter Map",
            img_scatter,
            envgeo_utils.build_figure_filename(
                f"Fig_{safe_parameter_name}_scatter",
                f"{sub_title}_center{lon_center}"
            ),
            "image/png"
        )
    
    else:
        #############################################################
        # Contour Map
        #############################################################
        # ``linear`` interpolation needs at least three non-collinear points.
        # Uploaded-only selections can legitimately contain fewer points, so
        # fall back to nearest-neighbour interpolation rather than erroring.
        contour_df = df1.loc[:, [
            "Longitude_degE", "Latitude_degN", selected_parameter
        ]].copy()
        for _column in contour_df.columns:
            contour_df[_column] = pd.to_numeric(
                contour_df[_column], errors="coerce"
            )
        contour_df = contour_df.dropna().loc[
            lambda data: data["Latitude_degN"].between(-90, 90)
        ]
        if contour_df.empty:
            st.warning(
                f"No valid longitude, latitude, and {selected_parameter} values "
                "are available for the contour map."
            )
            return

        lon_original = contour_df["Longitude_degE"].values
        # Interpolation also needs the center-adjusted longitude frame to match the displayed window.
        # 補間計算でも、表示中のウィンドウと同じ経度系を使う必要がある。
        lon_for_interp = normalize_lon_to_center(lon_original, lon_center)
        # Build the interpolation grid in the same longitude domain as the slider and set_extent.
        # 補間グリッドも slider / set_extent と同じ経度範囲で作る。
        grid_lon = np.linspace(lon_slider_min, lon_slider_max, 360)
        lat_vals     = contour_df["Latitude_degN"].values
        val          = contour_df[selected_parameter].values
        
        # ---- グリッド ----
        grid_lat = np.linspace(map_lat_min, map_lat_max, 250)
        X, Y = np.meshgrid(grid_lon, grid_lat)
        
        # ---- 補間 ----
        Z = None
        if len(contour_df) >= 3:
            try:
                Z = griddata(
                    (lon_for_interp, lat_vals), val, (X, Y), method="linear"
                )
            except Exception:
                # Collinear or duplicate locations are valid uploads but are
                # not a valid Delaunay triangulation for linear interpolation.
                Z = None
        if Z is None or not np.isfinite(Z).any():
            try:
                Z = griddata(
                    (lon_for_interp, lat_vals), val, (X, Y), method="nearest"
                )
            except Exception:
                st.warning(
                    "The selected locations could not be interpolated for the contour map."
                )
                return
        Z_plot = np.ma.masked_invalid(Z)
        lon_plot = grid_lon
        
        fig_contour = plt.figure(figsize=(12, 8), facecolor="white", dpi=150)
        ax2 = fig_contour.add_subplot(
            1, 1, 1,
            projection=ccrs.PlateCarree(central_longitude=lon_center)
        )
        
        set_cartopy_extent_safely(
            ax2,
            [map_lon_min, map_lon_max, map_lat_min, map_lat_max],
        )
        
        levels = np.linspace(parameter_min, parameter_max, 51)
        ax_cntr = ax2.contourf(
            lon_plot,
            grid_lat,
            Z_plot,
            levels=levels,
            cmap=map_matplotlib_colormap,
            transform=ccrs.PlateCarree(),
            alpha=0.85,
            zorder=1
        )
        
        # Draw land mask from bundled NE 50m shapefile (no network access).
        # Drawing order: contours (zorder=1) → land mask (zorder=2) →
        # coastlines (zorder=3) → gridlines (zorder=4) → data points (zorder≥5).
        _ne_geoms_cn = _load_ne50m_land_geometries()
        if _ne_geoms_cn:
            ax2.add_geometries(
                _ne_geoms_cn,
                crs=ccrs.PlateCarree(),
                facecolor="white",
                edgecolor="none",
                zorder=2,
            )
        envgeo_utils.plot_bundled_coastline(
            ax2,
            transform=ccrs.PlateCarree(),
            zorder=3,
            color="0.35",
            linewidth=0.6,
        )
        ax2.gridlines(draw_labels=True, zorder=4)
        
        lon_wrapped = normalize_lon_to_center(lon_original, lon_center)
        ax2.scatter(
            lon_wrapped,
            lat_vals,
            c="black",
            s=2,
            alpha=0.3,
            transform=ccrs.PlateCarree(),
            zorder=5
        )
        ax2.set_title(title_head2,fontsize=15)
        
        num_ticks = 6
        raw_ticks = np.linspace(parameter_min, parameter_max, num_ticks)
        fixed_ticks = [round(t, 1) for t in raw_ticks]
        
        cbar = fig_contour.colorbar(
            ax_cntr,
            ax=ax2,
            orientation="horizontal",
            pad=0.08,
            fraction=colorbar_thickness,
            shrink=colorbar_length,
            ticks=fixed_ticks,
            extend="neither"
        )
        cbar.ax.set_xticklabels([f"{t:.1f}" for t in fixed_ticks])
        cbar.set_label(parameter_label, fontsize=colorbar_font_size)
        cbar.ax.tick_params(labelsize=colorbar_font_size)

        # --- Uploaded data overlay (Contour Map / 最前面) ---
        if not uploaded_map_valid.empty:
            _lon_up_ct = normalize_lon_to_center(
                uploaded_map_valid["Longitude_degE"].values, lon_center
            )
            _use_color_ct = (
                uploaded_style["color_mode"] == "Use current colorbar when possible"
                and selected_parameter in uploaded_map_valid.columns
            )
            if _use_color_ct:
                _up_cv_ct = pd.to_numeric(
                    uploaded_map_valid[selected_parameter], errors="coerce"
                )
                _up_valid_ct = _up_cv_ct.notna()
                if _up_valid_ct.any():
                    ax2.scatter(
                        _lon_up_ct[_up_valid_ct.values],
                        uploaded_map_valid.loc[_up_valid_ct, "Latitude_degN"],
                        c=_up_cv_ct[_up_valid_ct],
                        cmap=map_matplotlib_colormap,
                        s=uploaded_style["size"],
                        alpha=uploaded_style["alpha"],
                        vmin=parameter_min,
                        vmax=parameter_max,
                        marker=uploaded_style["marker"],
                        linewidths=uploaded_style["outline_width"],
                        edgecolors=uploaded_style["outline_color"],
                        transform=ccrs.PlateCarree(),
                        zorder=10,
                    )
                if (~_up_valid_ct).any() and show_nodata_uploaded:
                    ax2.scatter(
                        _lon_up_ct[~_up_valid_ct.values],
                        uploaded_map_valid.loc[~_up_valid_ct, "Latitude_degN"],
                        c=uploaded_style["color"],
                        s=uploaded_style["size"],
                        alpha=uploaded_style["alpha"],
                        marker=uploaded_style["marker"],
                        linewidths=uploaded_style["outline_width"],
                        edgecolors=uploaded_style["outline_color"],
                        transform=ccrs.PlateCarree(),
                        zorder=10,
                    )
            else:
                ax2.scatter(
                    _lon_up_ct,
                    uploaded_map_valid["Latitude_degN"],
                    c=uploaded_style["color"],
                    s=uploaded_style["size"],
                    alpha=uploaded_style["alpha"],
                    marker=uploaded_style["marker"],
                    linewidths=uploaded_style["outline_width"],
                    edgecolors=uploaded_style["outline_color"],
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                )

        img_contour = io.BytesIO()
        fig_contour.savefig(img_contour, format="png", dpi=300, bbox_inches="tight")
        img_contour.seek(0)

        st.pyplot(fig_contour)
        
        st.download_button(
            "Download Contour Map",
            img_contour,
            envgeo_utils.build_figure_filename(
                f"Fig_{safe_parameter_name}_contour",
                f"{sub_title}_center{lon_center}"
            ),
            "image/png"
        )


    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    
    # Map section
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################


    # 選択されたデータの地点プロット
    # --- Location map / 採取地点の地図表示 ---
    st.divider()
    st.subheader('Sampling Location Map')



    # Keep map controls compact so the map remains visible after Streamlit reruns.
    # Streamlitの再実行後も地図が見つけやすいよう、地図設定をポップオーバーに集約する。
    with st.popover(
        "Map controls", **envgeo_utils.stretch_width_kwargs(st.popover)
    ):
        map_mode = st.radio(
            "Map style",
            envgeo_utils.MAP_MODE_OPTIONS,
            index=envgeo_utils.MAP_MODE_DEFAULT_INDEX,
            horizontal=True,
            key="map_style_32_auto",
            help=getattr(envgeo_utils, "MAP_STYLE_HELP_TEXT", "Choose the background map style for the sampling-location map."),
        )
    st.caption(f"Map style: {map_mode}")

    # 2. データの範囲から中心座標とズームレベルを計算（アップロード地点を含む）
    _map_extent_srcs = [df1[["Longitude_degE", "Latitude_degN"]]]
    if not uploaded_map_valid.empty:
        _map_extent_srcs.append(
            uploaded_map_valid[["Longitude_degE", "Latitude_degN"]]
        )
    _map_extent_df = pd.concat(_map_extent_srcs, ignore_index=True)
    lat_min, lat_max = (
        _map_extent_df["Latitude_degN"].min(),
        _map_extent_df["Latitude_degN"].max(),
    )
    lon_min, lon_max = (
        _map_extent_df["Longitude_degE"].min(),
        _map_extent_df["Longitude_degE"].max(),
    )

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
    
  


    hover_columns = [
        "Latitude_degN",
        "Longitude_degE",
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
        "Depth_m",
        "reference",
    ]
    hover_data = {column: True for column in hover_columns if column in df1.columns}

    # 3. 地図の作成 (px.scatter_mapbox内ではwidthを指定しない)
    fig_map = px.scatter_mapbox(
        df1,
        lat="Latitude_degN",
        lon="Longitude_degE",
        color=selected_parameter,
        color_continuous_scale=map_plotly_colorscale,
        hover_data=hover_data,
        opacity=0.6,
        height=500  # 高さはここで固定
    )

    # 4. 背景スタイルの適用
    fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)

    # --- Uploaded overlay (Plotly Sampling Location Map) ---
    _plotly_color_range = (
        (parameter_min, parameter_max)
        if parameter_min < parameter_max
        else None
    )
    fig_map, _uploaded_plotly_count = envgeo_user_data.add_uploaded_map_overlay(
        fig_map,
        uploaded_map_valid,
        uploaded_style,
        color_column=selected_parameter,
        colorscale=map_plotly_colorscale,
        color_range=_plotly_color_range,
        show_nodata=show_nodata_uploaded,
    )
    if not uploaded_df.empty:
        if _uploaded_plotly_count:
            st.caption(
                f":blue[Uploaded locations: {_uploaded_plotly_count:,} / "
                f"{len(uploaded_df):,} plotted on the sampling location map.]"
            )
        else:
            st.caption(
                ":gray[Uploaded locations are not shown on the sampling location "
                "map because valid longitude and latitude columns are unavailable.]"
            )

    # 5. レイアウト設定 (ここが幅を広げる決め手)
    # カラーバーと凡例を地図内オーバーレイにして、外側余白で地図が圧縮されないようにする。
    # x=1.0 は Plotly が余白を自動追加して地図を圧縮するため使わない。
    # autoexpand=False で凡例による余白自動拡張を抑制し、
    # mapbox.domain で地図がフル幅を使うよう明示する。
    fig_map.update_layout(
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=auto_zoom,
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        ),
        margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        autosize=True,
        coloraxis_colorbar=dict(
            title=parameter_plotly_label,
            x=0.98,          # 地図内右端にオーバーレイ（1.0 にすると外側扱いで余白が生じる）
            xanchor='right',
            bgcolor='rgba(255,255,255,0.75)',
            bordercolor='rgba(150,150,150,0.5)',
            borderwidth=1,
        ),
        # 凡例をツールバー（右上）と重ならないよう左下に配置
        legend=dict(
            x=0.01,
            y=0.01,
            xanchor='left',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='rgba(150,150,150,0.5)',
            borderwidth=1,
        ),
        coloraxis_cmin=parameter_min,
        coloraxis_cmax=parameter_max,
    )
    

    # 6. 表示 (use_container_width=True を確実に使う)
    # ID重複を割けるために，Keyを追加。　修正後（一意のキーを追加）
    # マウスホイールでのズームが強制的に有効
    st.plotly_chart(
        fig_map,
        key=f"parameter_map_{selected_parameter}",
        config={'scrollZoom': True, 'displayModeBar': True}, # ズームを有効化
        **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
    )






    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

        
    # Sidebar-filtered datasetを読み出し
    envgeo_utils.display_isotope_table(df1)
    
  
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

if __name__ == '__main__':
    main()
    
    
