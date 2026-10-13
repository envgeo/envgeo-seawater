#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isotope and hydrographic mapping visualizer for EnvGeo-Seawater data.

EnvGeo-Seawater データの同位体・水理パラメーター地図表示ページです。

Author: Toyoho Ishimura, Kyoto University
Last reviewed: 2026-09-30
"""

# =============================================================================
# Page configuration / ページ設定
# =============================================================================
version = "1.3.4"
fig_title = "envgeo-seawater-database"

import io
import math

import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.interpolate import griddata

import envgeo_assets
import envgeo_user_data
import envgeo_utils

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

# =============================================================================
# Static map projections / 静的地図の投影法
# =============================================================================
STATIC_MAP_PROJECTION_OPTIONS = (
    "Plate Carrée (longitude–latitude)",
    "Mercator",
    "Robinson",
    "Azimuthal Equidistant",
    "Orthographic (map centre)",
    "Orthographic (South Polar)",
    "Orthographic (North Polar)",
)

STATIC_MAP_PROJECTION_DISPLAY_NAMES = {
    "Plate Carrée (longitude–latitude)": "Plate Carrée (longitude–latitude) / 正距円筒図法（経緯度）",
    "Mercator": "Mercator / メルカトル図法",
    "Robinson": "Robinson / ロビンソン図法",
    "Azimuthal Equidistant": "Azimuthal Equidistant / 正距方位図法",
    "Orthographic (map centre)": "Orthographic (map centre) / 正射図法（地図中心）",
    "Orthographic (South Polar)": "Orthographic (South Polar) / 正射図法（南極中心）",
    "Orthographic (North Polar)": "Orthographic (North Polar) / 正射図法（北極中心）",
}


def build_static_map_projection(projection_label, central_longitude, central_latitude):
    """Return the Cartopy projection selected for static Mapping figures.

    Input coordinates always remain longitude/latitude in Plate Carrée. The
    selected projection affects only the static Scatter Map and Contour Map,
    not the interactive Plotly sampling-location map.

    静的Scatter／Contour地図に使うCartopy投影を返す。入力座標は常に
    Plate Carréeの経緯度として扱い、インタラクティブな採水地点地図には影響しない。
    """
    if projection_label == "Plate Carrée (longitude–latitude)":
        return ccrs.PlateCarree(central_longitude=central_longitude)
    if projection_label == "Mercator":
        return ccrs.Mercator(central_longitude=central_longitude)
    if projection_label == "Robinson":
        return ccrs.Robinson(central_longitude=central_longitude)
    if projection_label == "Azimuthal Equidistant":
        return ccrs.AzimuthalEquidistant(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
        )
    if projection_label in {
        "Orthographic (map centre)",
        "Orthographic (South Polar)",
        "Orthographic (North Polar)",
    }:
        return ccrs.Orthographic(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
        )
    raise ValueError(f"Unknown static map projection: {projection_label}")


def static_projection_center(extent, projection_label):
    """Return a suitable map centre for the selected static projection.

    Full-longitude polar presets are centred on their respective pole for an
    Azimuthal Equidistant view. Named orthographic polar views are always
    centred on their selected pole, while ``Orthographic (map centre)`` uses
    the midpoint of the displayed extent. This retains the requested regional
    latitude band rather than accidentally treating a Southern or Arctic Ocean
    view as a global map merely because it spans all longitudes.

    正距方位図法では、全経度を含む極域プリセットを対応する極に中心合わせする。
    南北極中心の正射図法は、選択した極を常に中心にする。一方、地図中心の
    正射図法は表示範囲の中点を中心にする。南極海・北極海を、経度幅だけを理由に
    全球表示として扱わない。
    """
    lon_min, lon_max, lat_min, lat_max = (float(value) for value in extent)
    central_longitude = (lon_min + lon_max) / 2
    central_latitude = (lat_min + lat_max) / 2
    if projection_label == "Orthographic (South Polar)":
        central_latitude = -90.0
    elif projection_label == "Orthographic (North Polar)":
        central_latitude = 90.0
    elif projection_label == "Azimuthal Equidistant":
        if lat_max <= -45.0:
            central_latitude = -90.0
        elif lat_min >= 45.0:
            central_latitude = 90.0
    return central_longitude, central_latitude


def safe_static_projection_extent(extent, projection_label):
    """Return a drawable extent and whether a Mercator latitude clamp was applied.

    Mercator cannot represent the poles. Its displayed latitude is therefore
    restricted to ±84.5°; choose Azimuthal Equidistant for a polar-centred view.

    Mercatorでは極を描画できないため、表示緯度を±84.5°へ制限する。
    極域中心の表示には正距方位図法を使う。
    """
    lon_min, lon_max, lat_min, lat_max = (float(value) for value in extent)
    if projection_label != "Mercator":
        return (lon_min, lon_max, lat_min, lat_max), False

    limit = 84.5
    clamped_lat_min = max(-limit, lat_min)
    clamped_lat_max = min(limit, lat_max)
    was_clamped = (clamped_lat_min != lat_min) or (clamped_lat_max != lat_max)
    if clamped_lat_min >= clamped_lat_max:
        clamped_lat_min, clamped_lat_max = -limit, limit
        was_clamped = True
    return (lon_min, lon_max, clamped_lat_min, clamped_lat_max), was_clamped


def add_static_map_gridlines(ax, projection_label):
    """Add graticules, with labels only where Cartopy can place them reliably.

    Robinson, azimuthal, and orthographic projections have curved boundaries,
    so unlabeled graticules avoid misleading or overlapping edge labels.

    Robinson、方位図法、正射図法は境界が曲線のため、重なりや誤解を招く端部ラベルは
    表示せず、グリッド線だけを描く。
    """
    draw_labels = projection_label in {
        "Plate Carrée (longitude–latitude)",
        "Mercator",
    }
    return ax.gridlines(draw_labels=draw_labels, zorder=4)


def safe_cartopy_extent(lon_min, lon_max, lat_min, lat_max, lon_domain_min, lon_domain_max):
    """
    Return a Cartopy-safe map extent.

    Cartopy may produce NaN axis limits when the longitude extent exactly matches
    the full projection domain, especially on Streamlit Cloud with 0-360 maps.
    This keeps the requested view visually unchanged while nudging the bounds
    slightly inside the projection seam.

    投影境界に触れる範囲をわずかに内側へ調整し、CartopyのNaN軸範囲を防ぎます。
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

def is_near_global_map_extent(extent):
    """Return whether a displayed extent represents the bundled global domain.

    The bundled global data span approximately 168° in latitude (−78° to 90°),
    rather than a mathematically complete 180°. This is nevertheless a global
    selection. The threshold remains far wider than the Arctic and Southern
    Ocean presets, which retain their regional extent.

    同梱データの全球範囲は約168度（−78〜90度）で、数学的な180度ではないが、
    全球選択として扱う。北極海・南極海の地域プリセットより十分に広い閾値にする。
    """
    lon_min, lon_max, lat_min, lat_max = (float(value) for value in extent)
    return (lon_max - lon_min) >= 359.0 and (lat_max - lat_min) >= 160.0


def uses_global_circular_extent(extent, projection_label):
    """Return whether a curved projection should use Cartopy's global globe."""
    is_circular_projection = projection_label in {
        "Azimuthal Equidistant",
        "Orthographic (map centre)",
        "Orthographic (South Polar)",
        "Orthographic (North Polar)",
    }
    return is_near_global_map_extent(extent) and is_circular_projection


def periodic_contour_interpolation_points(longitudes, latitudes, values, is_global):
    """Return contour inputs with periodic longitude copies for global maps.

    ``scipy.interpolate.griddata`` does not recognise that −180° and +180°
    are adjacent. Copies at ±360° avoid an artificial date-line gap in the
    exploratory global contours; neither the source dataframe nor its values
    are changed. Regional inputs remain untouched.

    ``griddata``は−180°と+180°が隣接することを認識しない。全球の探索的
    コンターでは±360°のコピーを補間にのみ加え、日付変更線の人工的な切れ目を
    避ける。元データとその値は変更せず、地域表示の入力はそのまま扱う。
    """
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    value = np.asarray(values, dtype=float)
    if not is_global:
        return lon, lat, value
    return (
        np.concatenate((lon - 360.0, lon, lon + 360.0)),
        np.concatenate((lat, lat, lat)),
        np.concatenate((value, value, value)),
    )


def resolve_static_map_type_for_extent(extent, requested_map_type, projection_label):
    """Return the safe static-map type for the selected geographic extent.

    Global exploratory contours are available for the cylindrical Plate Carrée
    and Mercator projections, with periodic handling at the date line. The
    curved Robinson boundary and globe-style azimuthal or orthographic views
    can break or obscure a global rectangular contour mesh, so they show
    observed samples instead; regional contours remain available.

    全球の探索的コンターは円筒図法であるPlate CarréeとMercatorで利用でき、日付変更線を
    周期境界として扱う。曲線境界のRobinson、円形の正距方位図法・正射図法では、全球の
    矩形補間格子が描画破綻または隠れた領域を生むため、観測点を表示する。地域的なコンター
    はすべての投影法で利用できる。
    """
    globe_style_projections = {
        "Robinson",
        "Azimuthal Equidistant",
        "Orthographic (map centre)",
        "Orthographic (South Polar)",
        "Orthographic (North Polar)",
    }
    if (
        requested_map_type == "Contour Map"
        and is_near_global_map_extent(extent)
        and projection_label in globe_style_projections
    ):
        return "Scatter Map"
    return requested_map_type


def set_cartopy_extent_safely(ax, extent, projection_label=None):
    """
    Apply a map extent without crashing on Cartopy projection seam issues.

    Streamlit Cloud can raise ``ValueError: Axis limits cannot be NaN or Inf``
    for nearly global PlateCarree extents. If that happens, fall back to a
    global view so the figure still renders. Near-global azimuthal and
    orthographic maps use Cartopy's global circular view: applying a
    360-degree rectangular extent can collapse the drawing. Full-longitude
    polar regions are still regional views.

    投影境界の問題で失敗した場合は、全球表示にフォールバックする。方位図法・正射図法の
    全球表示も、360度の矩形範囲で描画が崩れるため、円形の全球表示にする。ただし、
    全経度を含む極域は地域表示として扱う。
    """
    if uses_global_circular_extent(extent, projection_label):
        ax.set_global()
        return True
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

    地域プリセットは地図の表示範囲だけを変え、サイドバーの絞り込み結果は変えません。
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

# =============================================================================
# Bundled Natural Earth land mask / 同梱Natural Earth陸域マスク
# =============================================================================
# The shapefile is bundled in the repository so that the land mask can be drawn
# without any external Natural Earth download, even in offline environments.
# Path is resolved relative to this script's directory so it works regardless
# of the working directory when Streamlit launches the page.
_NE50M_LAND_DIR = envgeo_assets.asset_path(
    "coastline/natural_earth_50m_land", required=False
)
_NE50M_LAND_SHP = _NE50M_LAND_DIR / "ne_50m_land.shp"
_NE50M_LAND_REQUIRED_EXTS = (".shp", ".shx", ".dbf")


# Minimal test stubs do not need Streamlit's process cache.  In the running
# application, this resolves to ``st.cache_resource`` and retains the immutable
# geometries across widget reruns.
# 最小のテスト用st stubではStreamlitのprocess cacheを必要としない。実行時は
# ``st.cache_resource``を使い、不変のgeometryをウィジェット再実行後も保持する。
_cache_land_geometries = getattr(
    st, "cache_resource", lambda *args, **kwargs: lambda function: function
)


@_cache_land_geometries(show_spinner=False)
def _read_ne50m_land_geometries():
    """Read and retain bundled land geometries once per Streamlit process.

    The shapefile reader holds several operating-system file descriptors. The
    geometries are immutable for this application, so caching them prevents a
    new reader being opened on every widget rerun. The reader is explicitly
    closed after its geometries have been materialised.

    shapefile readerは複数のOSファイル記述子を使う。地形ジオメトリはこの
    アプリでは不変なので、ウィジェット操作ごとにreaderを開かないよう
    キャッシュする。ジオメトリを読み切った後、readerを明示的に閉じる。
    """
    reader = shapereader.Reader(str(_NE50M_LAND_SHP))
    try:
        return tuple(reader.geometries())
    finally:
        close_reader = getattr(reader, "close", None)
        if callable(close_reader):
            close_reader()


def _load_ne50m_land_geometries():
    """Return cached Shapely geometries from the bundled NE 50m land shapefile.

    Returns an empty list (with a Streamlit warning) if any required component
    is missing.  Never fetches from the network.

    Rendering with an empty list degrades gracefully: the land mask is skipped,
    but coastlines and data points are still drawn.

    必要な同梱ファイルがない場合は陸域マスクだけを省略し、海岸線と観測点は表示します。
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
        # Keep the long-standing public helper return type stable for callers
        # and tests; the cached immutable tuple remains internal.
        # 既存の呼出し元・テストとの互換性のため、公開helperはlistを返す。
        # cache内部では不変のtupleを保持する。
        return list(_read_ne50m_land_geometries())
    except Exception as exc:
        st.warning(
            f"Land mask (Natural Earth 50m) could not be loaded: {exc}. "
            "Coastlines and data points are still displayed."
        )
        return []

def main():
    # =============================================================================
    # Page header / ページ見出し
    # =============================================================================
    st.header(f'Isotope & Hydrographic Mapping ({version})')
    st.caption(
        "Map d18O, dD, d-excess, salinity, temperature, and related seawater parameters."
    )
    st.button('Reload')

    # =============================================================================
    # Data-source selection / データソースの選択
    # =============================================================================
    data_source_ENVGEO = envgeo_utils.data_source_ENVGEO
    data_source_AROUND_JAPAN = envgeo_utils.data_source_AROUND_JAPAN
    data_source_GLOBAL = envgeo_utils.data_source_GLOBAL

    ref_data = st.radio("Data source (see Home > About):", (data_source_ENVGEO, data_source_AROUND_JAPAN, data_source_GLOBAL), horizontal=True)

    # -----------------------------------------------------------------------------
    # Attribution / 出典表示
    # -----------------------------------------------------------------------------
    if ref_data == data_source_ENVGEO:
        st.write(envgeo_utils.refs_ENVGEO)
        
    elif ref_data == data_source_AROUND_JAPAN:
        st.write(envgeo_utils.refs_AROUND_JAPAN)

        
    elif ref_data == data_source_GLOBAL:
        st.write(envgeo_utils.refs_GLOBAL)
        
    else:
        st.warning("Invalid data source selection.")

    # =============================================================================
    # Data loading and uploaded overlay / データ読込とアップロード重ね表示
    # =============================================================================
    df1 = envgeo_utils.load_isotope_data(ref_data)
   
    if df1.empty:
        st.warning("No data available for the selected conditions.")
        return

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

    # =============================================================================
    # Shared sidebar filtering / 共通サイドバーによるデータ絞り込み
    # =============================================================================
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
         ref_data, data_source_ENVGEO, data_source_AROUND_JAPAN,
         uploaded_df=uploaded_df, uploaded_filter_key="isotope_mapping",
         uploaded_dataset_label=envgeo_utils.UPLOADED_DATA_LABEL,
     )
    # Use the selected integrated table for map and contour calculations.
    # アップロード行は、選択したマーカー書式で前景に重ね描きするために別途保持する。
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

    # =============================================================================
    # Map parameter and type / 地図パラメーターと種類
    # =============================================================================
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

    # =============================================================================
    # Map display controls / 地図表示の設定
    # =============================================================================
    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "MAP_DISPLAY_SETTINGS_LABEL", "Map display settings"))
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        
        center_option = st.radio(
            ":blue[Map Center:]",
            ("Atlantic (0°)", "Pacific (180°)"),
            horizontal=True
        )
        
        lon_center = 0 if "Atlantic" in center_option else 180

        static_map_projection_label = st.selectbox(
            "Map projection / 投影法",
            STATIC_MAP_PROJECTION_OPTIONS,
            index=0,
            format_func=STATIC_MAP_PROJECTION_DISPLAY_NAMES.get,
            help=(
                "Applies only to the downloadable Cartopy Scatter Map and Contour Map. "
                "The interactive sampling-location map keeps its web-map style. "
                "Mercator is limited to ±84.5°. Use Azimuthal Equidistant for "
                "regional distance-and-direction views, or Orthographic for a globe-style "
                "view. Global exploratory contours are available with Plate Carrée or "
                "Mercator; Robinson and globe-style projections show observations."
            ),
        )

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
        if ref_data == data_source_ENVGEO:
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

        # Uploaded points without the mapped value / 色分け値がないアップロード地点
        show_nodata_uploaded = st.checkbox(
            f"Show uploaded points without {selected_parameter} values",
            value=True,
            key=f"{map_state_key}::show_nodata_uploaded",
            help="Show or hide uploaded map points that have no value for the mapped parameter.",
        )

        parameter_min, parameter_max = selected_color_range
        colorbar_thickness = colorbar_thickness_value / 100
        colorbar_length = colorbar_length_value / 100

        # A small offset keeps full-domain Cartopy extents away from the projection seam.
        # 微小オフセットにより、Cartopy投影の境界線と完全一致する範囲を避ける。
        map_lon_min, map_lon_max = map_lon_raw[0] - 0.001, map_lon_raw[1] + 0.001
        map_lat_min, map_lat_max = map_lat_raw[0] - 0.001, map_lat_raw[1] + 0.001

    # =============================================================================
    # Map preparation and drawing / 地図の準備と描画
    # =============================================================================

    st.caption(getattr(envgeo_utils, "MAP_AREA_HELP_TEXT", "Map center, extent, colormap, and figure settings can be adjusted in the sidebar."))

    # -----------------------------------------------------------------------------
    # Parameter availability / 選択パラメーターの有効値
    # -----------------------------------------------------------------------------
    original_len_df1 = len(df1)
    df1 = df1.dropna(subset=[selected_parameter])
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
    # -----------------------------------------------------------------------------
    # Figure title and file name / 図題とファイル名
    # -----------------------------------------------------------------------------
    main_title = f"{fig_title} - {selected_parameter}"
    
    # Compact month-range text for the figure title. / 月範囲を図題用に短縮する。
    if len(selected_months) == 12:
        month_display = "All"
    elif len(selected_months) == 0:
        month_display = "None"
    else:
        sorted_m = sorted(list(set(selected_months)))
        ranges = []
        if sorted_m:
            start = sorted_m[0]
            for i in range(len(sorted_m)):
                if i + 1 == len(sorted_m) or sorted_m[i+1] != sorted_m[i] + 1:
                    end = sorted_m[i]
                    ranges.append(f"{start}-{end}" if start != end else str(start))
                    if i + 1 < len(sorted_m):
                        start = sorted_m[i+1]
        month_display = ", ".join(ranges)
        
        

    sub_title = f"Lon:{sld_lon_min}-{sld_lon_max}, Lat:{sld_lat_min}-{sld_lat_max}, Y:{sld_year_min}-{sld_year_max}, M:{month_display}, S:{sld_sal_min}-{sld_sal_max}, D:{sld_depth_min}-{sld_depth_max}m"

    main_title2 = sub_title
    title_head = f"{main_title}\n{sub_title}"
    title_head2 = title_head.replace('_', ' ')

    # File names for downloaded figures / 図保存用のファイル名
    safe_parameter_name = envgeo_utils.safe_filename_text(selected_parameter)

    # -----------------------------------------------------------------------------
    # Uploaded-data preparation and quality check / アップロードデータの前処理と品質確認
    # -----------------------------------------------------------------------------
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
    # 完全な全球範囲によるCartopyの投影境界エラーを、安全な範囲へ調整して防ぐ。
    map_lon_min, map_lon_max, map_lat_min, map_lat_max = safe_cartopy_extent(
        map_lon_min,
        map_lon_max,
        map_lat_min,
        map_lat_max,
        lon_slider_min,
        lon_slider_max,
    )
    static_map_extent, mercator_extent_was_clamped = safe_static_projection_extent(
        (map_lon_min, map_lon_max, map_lat_min, map_lat_max),
        static_map_projection_label,
    )
    map_lon_min, map_lon_max, map_lat_min, map_lat_max = static_map_extent
    effective_map_type = resolve_static_map_type_for_extent(
        static_map_extent, map_type, static_map_projection_label
    )
    if effective_map_type != map_type:
        st.warning(
            "Global Contour Maps are available only with Plate Carrée or Mercator. "
            "Robinson's curved boundary and globe-style projections can break or obscure "
            "a global rectangular contour mesh. Observed samples are shown as a Scatter "
            "Map instead. Choose Plate Carrée or Mercator for a global exploratory contour."
        )
    projection_centre_lon, projection_centre_lat = static_projection_center(
        static_map_extent, static_map_projection_label
    )
    static_projection = build_static_map_projection(
        static_map_projection_label,
        central_longitude=projection_centre_lon,
        central_latitude=projection_centre_lat,
    )
    if mercator_extent_was_clamped:
        st.info(
            "Mercator maps are limited to ±84.5°; the static map extent was adjusted. "
            "Choose Azimuthal Equidistant for a polar-centred view."
        )
    global_circular_view = uses_global_circular_extent(
        static_map_extent, static_map_projection_label
    )
    if global_circular_view:
        st.info(
            f"Global {static_map_projection_label} uses Cartopy's circular globe view. "
            "Use a regional extent for a closer view."
        )
    
    # =============================================================================
    # Scatter or contour map / 散布図またはコンター地図
    # =============================================================================
    if effective_map_type == "Scatter Map":
        # -------------------------------------------------------------------------
        # Scatter map / 散布図
        # -------------------------------------------------------------------------
        fig = plt.figure(figsize=(12, 8), facecolor="white", dpi=150)
        
        ax = fig.add_subplot(
            1, 1, 1,
            projection=static_projection,
        )
        
        set_cartopy_extent_safely(
            ax,
            [map_lon_min, map_lon_max, map_lat_min, map_lat_max],
            static_map_projection_label,
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
        add_static_map_gridlines(ax, static_map_projection_label)
        
        # Plot points in the longitude frame used by the selected map extent.
        # 選択した表示範囲と同じ経度系に変換して観測点を描画する。
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
        
        # Parameter colorbar / パラメーターのカラーバー
        cbar_scatter = fig.colorbar(
            ax_scatter,
            ax=ax,
            orientation="horizontal",
            pad=0.08,
            fraction=colorbar_thickness,
            shrink=colorbar_length,
            aspect=25,
            extend="neither",
        )
        cbar_scatter.set_label(parameter_label, fontsize=colorbar_font_size)
        cbar_scatter.ax.tick_params(labelsize=colorbar_font_size)

        # Uploaded overlay, drawn in the foreground / 前景に描くアップロードデータ
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
        # -------------------------------------------------------------------------
        # Contour map / コンター地図
        # -------------------------------------------------------------------------
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
        # Interpolation uses the longitude frame of the displayed window.
        # 補間計算にも表示ウィンドウと同じ経度系を用いる。
        lon_for_interp = normalize_lon_to_center(lon_original, lon_center)
        # Build the interpolation grid in the longitude domain used by the slider.
        # 補間グリッドもsliderと同じ経度範囲で作る。
        grid_lon = np.linspace(lon_slider_min, lon_slider_max, 360)
        lat_vals     = contour_df["Latitude_degN"].values
        val          = contour_df[selected_parameter].values
        
        # Interpolation grid / 補間グリッド
        grid_lat = np.linspace(map_lat_min, map_lat_max, 250)
        X, Y = np.meshgrid(grid_lon, grid_lat)
        
        # Interpolation / 補間
        # Global rectangular projections use periodic longitude copies only here.
        # 全球の矩形投影では、補間時だけ経度の周期コピーを用いる。
        interp_lon, interp_lat, interp_values = periodic_contour_interpolation_points(
            lon_for_interp,
            lat_vals,
            val,
            is_near_global_map_extent(static_map_extent),
        )
        Z = None
        if len(contour_df) >= 3:
            try:
                Z = griddata(
                    (interp_lon, interp_lat), interp_values, (X, Y), method="linear"
                )
            except Exception:
                # Collinear or duplicate locations are valid uploads but are
                # not a valid Delaunay triangulation for linear interpolation.
                Z = None
        if Z is None or not np.isfinite(Z).any():
            try:
                Z = griddata(
                    (interp_lon, interp_lat), interp_values, (X, Y), method="nearest"
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
            projection=static_projection,
        )
        
        set_cartopy_extent_safely(
            ax2,
            [map_lon_min, map_lon_max, map_lat_min, map_lat_max],
            static_map_projection_label,
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
        add_static_map_gridlines(ax2, static_map_projection_label)
        
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

        # Uploaded overlay, drawn in the foreground / 前景に描くアップロードデータ
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

    
    # =============================================================================
    # Sampling-location map / 採取地点の地図
    # =============================================================================
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

    # -----------------------------------------------------------------------------
    # Automatic map extent / 地図範囲の自動計算
    # -----------------------------------------------------------------------------
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

    # Japan-centered fallback when no valid coordinates exist. / 有効座標がない場合の日本中心設定。
    default_lat, default_lon, default_zoom = 36.0, 138.0, 4.0

    if pd.isna(lat_min) or pd.isna(lon_min):
        center_lat, center_lon, auto_zoom = default_lat, default_lon, default_zoom
    else:
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        lat_diff = max(lat_max - lat_min, 0.1)
        lon_diff = max(lon_max - lon_min, 0.1)
        
        # Pixel dimensions estimate a zoom level that includes the data extent.
        # ピクセル寸法を用いて、データ範囲を収めるズームを見積もる。
        map_width_px, map_height_px = 1200, 700
        zoom_lon = math.log2((map_width_px * 360) / (lon_diff * 256))
        zoom_lat = math.log2((map_height_px * 180) / (lat_diff * 256))
        
        # Leave margin around the selected extent. / 選択範囲の周囲に余白を確保する。
        auto_zoom = min(zoom_lon, zoom_lat) - 2.0
        auto_zoom = max(1, min(15, auto_zoom))

        # Use a wide Japan-centered view for near-global longitude spans.
        # 経度範囲がほぼ全球の場合は、日本中心の広域表示にする。
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

    # -----------------------------------------------------------------------------
    # Plotly location map / Plotly採取地点地図
    # -----------------------------------------------------------------------------
    fig_map = px.scatter_mapbox(
        df1,
        lat="Latitude_degN",
        lon="Longitude_degE",
        color=selected_parameter,
        color_continuous_scale=map_plotly_colorscale,
        hover_data=hover_data,
        opacity=0.6,
        height=500,
    )

    # Apply the selected background style. / 選択した背景スタイルを適用する。
    fig_map = envgeo_utils.apply_map_style(fig_map, map_mode)

    # Uploaded overlay / アップロードデータの重ね表示
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

    # Keep the colorbar and legend inside the map, preserving map width.
    # カラーバーと凡例を地図内に置き、外側余白による圧縮を避ける。
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
            x=0.98,
            xanchor='right',
            bgcolor='rgba(255,255,255,0.75)',
            bordercolor='rgba(150,150,150,0.5)',
            borderwidth=1,
        ),
        # Put the legend at lower left to avoid the upper-right toolbar.
        # 凡例は右上のツールバーを避けて左下に置く。
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
    

    # The unique key prevents Streamlit element-ID collisions; wheel zoom is enabled.
    # 一意のkeyでStreamlit要素IDの重複を避け、ホイールズームを有効にする。
    st.plotly_chart(
        fig_map,
        key=f"parameter_map_{selected_parameter}",
        config={'scrollZoom': True, 'displayModeBar': True},
        **envgeo_utils.stretch_width_kwargs(st.plotly_chart),
    )

    # =============================================================================
    # Filtered-data table / 絞り込みデータ表
    # =============================================================================
    envgeo_utils.display_isotope_table(df1)
    
  
    

if __name__ == '__main__':
    main()
    
    
