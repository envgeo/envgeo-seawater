#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:00:10 2026

@author: Toyoho Ishimura @Kyoto-U

2026/03/30 update

This file was developed with support from CODEX.
このファイルのみ CODEX の支援を受けて作成・改良してみました。
すごいですね，びっくりしました。。。

This file uses a lightweight bathymetry grid derived from the GEBCO 2025 Grid.
このファイルでは GEBCO 2025 Grid をもとに軽量化した海底地形グリッドを使用しています。
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter
from scipy.io import netcdf_file

import envgeo_utils

pd.set_option('future.no_silent_downcasting', True)


DEFAULT_GEBCO_PATH = "data_beta/GEBCO_2025_6min.nc"
MAX_MAP_POINTS = 50000
DEFAULT_MAX_ROWS_FOR_SECTION_PLOT = 3000
GEBCO_ATTRIBUTION = (
    "Bathymetry data source: GEBCO Compilation Group (2025), GEBCO 2025 Grid. "
    "GEBCO Grid data are in the public domain and may be used free of charge. "
    "The GEBCO Grid should not be used for navigation or any purpose involving safety at sea."
)
MAP_MODE_OPTIONS = ["Standard", "Satellite", "Bathymetry (Sea)", "Contour (GSI)"]


def sample_points_for_map(df_points, max_points=MAX_MAP_POINTS):
    # 地図描画が重くなりすぎないよう、表示点数を抑える
    # Limit point count for maps so the interactive view stays responsive.
    if df_points.empty or len(df_points) <= max_points:
        return df_points
    return df_points.sample(max_points, random_state=42).sort_index()


def maybe_sample_points(df_points, max_points=None):
    # max_points が未指定なら全点をそのまま返す
    # Return all points when no display cap is requested.
    if max_points is None:
        return df_points
    return sample_points_for_map(df_points, max_points=max_points)


def lonlat_to_local_km(lon, lat, lon0, lat0, lat_ref):
    # 緯度経度を、基準点まわりのローカル直交座標(km)へ近似変換する
    # Approximate lon/lat as local Cartesian coordinates (km) around a reference point.
    x = (np.asarray(lon, dtype=float) - lon0) * 111.32 * np.cos(np.radians(lat_ref))
    y = (np.asarray(lat, dtype=float) - lat0) * 111.32
    return x, y


def normalize_section_vertices(section_vertices):
    # 測線頂点を [lat, lon] 形式へ正規化し、重複する連続点は落とす
    # Normalize section vertices to [lat, lon] pairs and remove repeated consecutive points.
    normalized = []
    for vertex in section_vertices:
        if len(vertex) < 2:
            continue
        lat = float(vertex[0])
        lon = float(vertex[1])
        if not normalized or (lat, lon) != tuple(normalized[-1]):
            normalized.append([lat, lon])
    return normalized


def build_section_polyline(section_vertices):
    # 折れ線測線をローカル km 座標へ変換し、各頂点までの累積距離を計算する
    # Convert the section polyline to local km coordinates and compute cumulative along-track distance.
    vertices = normalize_section_vertices(section_vertices)
    if len(vertices) < 2:
        return None

    lat0, lon0 = vertices[0]
    lat_ref = float(np.mean([lat for lat, _ in vertices]))
    xs, ys = lonlat_to_local_km(
        [lon for _, lon in vertices],
        [lat for lat, _ in vertices],
        lon0,
        lat0,
        lat_ref,
    )
    xy = np.column_stack([xs, ys])
    diffs = np.diff(xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    valid_seg = seg_lengths > 1.0e-9
    if not np.any(valid_seg):
        return None

    keep_idx = np.concatenate([[True], valid_seg])
    xy = xy[keep_idx]
    vertices = [vertices[i] for i, keep in enumerate(keep_idx) if keep]
    diffs = np.diff(xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    return {
        "vertices_latlon": vertices,
        "xy_km": xy,
        "lat_ref": lat_ref,
        "origin_lon": lon0,
        "origin_lat": lat0,
        "seg_lengths": seg_lengths,
        "cumulative_km": cumulative,
        "length_km": float(cumulative[-1]),
    }


def section_vertices_from_ab(a_lat, a_lon, b_lat, b_lon):
    # 直線測線も内部的には2点の折れ線として扱う
    # Treat a straight A-B section as a two-vertex polyline internally.
    return [[float(a_lat), float(a_lon)], [float(b_lat), float(b_lon)]]


def suggest_default_section_vertices(df):
    # 初期 A-B 測線は経度端どうしではなく、測点群の主軸に沿って提案する
    # Suggest the initial A-B line along the dominant spatial axis of the stations, not just min/max longitude.
    df_points = df.dropna(subset=["Longitude_degE", "Latitude_degN"]).copy()
    if df_points.empty:
        return None

    df_points = df_points[["Latitude_degN", "Longitude_degE"]].drop_duplicates()
    if len(df_points) < 2:
        point = df_points.iloc[0]
        return section_vertices_from_ab(point["Latitude_degN"], point["Longitude_degE"], point["Latitude_degN"], point["Longitude_degE"])

    lat0 = float(df_points["Latitude_degN"].mean())
    lon0 = float(df_points["Longitude_degE"].mean())
    lat_ref = lat0
    x, y = lonlat_to_local_km(
        df_points["Longitude_degE"].to_numpy(),
        df_points["Latitude_degN"].to_numpy(),
        lon0,
        lat0,
        lat_ref,
    )
    coords = np.column_stack([x, y])
    centered = coords - coords.mean(axis=0, keepdims=True)

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0])

    if not np.isfinite(axis).all() or np.hypot(axis[0], axis[1]) < 1.0e-9:
        axis = np.array([1.0, 0.0])

    scores = centered @ axis
    start_xy = coords.mean(axis=0) + axis * float(scores.min())
    end_xy = coords.mean(axis=0) + axis * float(scores.max())

    cos_lat = np.cos(np.radians(lat_ref))
    if abs(cos_lat) < 1.0e-9:
        cos_lat = 1.0e-9
    start_lon = lon0 + start_xy[0] / (111.32 * cos_lat)
    start_lat = lat0 + start_xy[1] / 111.32
    end_lon = lon0 + end_xy[0] / (111.32 * cos_lat)
    end_lat = lat0 + end_xy[1] / 111.32
    return section_vertices_from_ab(start_lat, start_lon, end_lat, end_lon)


def project_points_to_polyline(df, section_vertices, corridor_km):
    # 観測点を折れ線測線へ投影し、最も近い線分で along/cross-track を決める
    # Project observations onto a polyline section and assign along/cross-track from the nearest segment.
    if df.empty:
        return df.copy(), 0.0, None

    polyline = build_section_polyline(section_vertices)
    if polyline is None:
        return pd.DataFrame(), 0.0, None

    px, py = lonlat_to_local_km(
        df["Longitude_degE"],
        df["Latitude_degN"],
        polyline["origin_lon"],
        polyline["origin_lat"],
        polyline["lat_ref"],
    )
    points = np.column_stack([px, py])

    best_dist = np.full(len(points), np.inf, dtype=float)
    best_cross = np.zeros(len(points), dtype=float)
    best_along = np.zeros(len(points), dtype=float)

    for i, seg_len in enumerate(polyline["seg_lengths"]):
        a = polyline["xy_km"][i]
        b = polyline["xy_km"][i + 1]
        ab = b - a
        ap = points - a
        t = np.clip((ap @ ab) / (seg_len ** 2), 0.0, 1.0)
        proj = a + t[:, None] * ab
        delta = points - proj
        dist = np.hypot(delta[:, 0], delta[:, 1])
        cross_signed = (ap[:, 0] * ab[1] - ap[:, 1] * ab[0]) / seg_len
        along = polyline["cumulative_km"][i] + t * seg_len

        better = dist < best_dist
        best_dist[better] = dist[better]
        best_cross[better] = cross_signed[better]
        best_along[better] = along[better]

    projected = df.copy()
    projected["SectionDistance_km"] = best_along
    projected["CrossTrack_km"] = best_cross
    projected["DistanceFromA_km"] = best_along

    projected = projected[
        projected["SectionDistance_km"].between(0.0, polyline["length_km"])
        & (projected["CrossTrack_km"].abs() <= corridor_km)
    ].copy()

    return projected, polyline["length_km"], polyline


def densify_section_line(section_vertices, n_points=200):
    # 折れ線測線を高密度化して、地図表示や地形サンプリングに使う
    # Densify a polyline section for map display and bathymetry sampling.
    polyline = build_section_polyline(section_vertices)
    if polyline is None:
        return np.array([]), np.array([])

    target_dist = np.linspace(0.0, polyline["length_km"], n_points)
    cum = polyline["cumulative_km"]
    xy = polyline["xy_km"]
    sample_x = np.interp(target_dist, cum, xy[:, 0])
    sample_y = np.interp(target_dist, cum, xy[:, 1])

    lon = polyline["origin_lon"] + sample_x / (111.32 * np.cos(np.radians(polyline["lat_ref"])))
    lat = polyline["origin_lat"] + sample_y / 111.32 # 緯度一度当たりの距離（km）
    return lon, lat


def extract_section_vertices_from_draw_result(draw_result):
    # Folium/streamlit-folium の描画結果から最後に引いた LineString の全頂点を取り出す
    # Extract all vertices of the latest drawn LineString from the Folium/streamlit-folium draw payload.
    if not draw_result:
        return None

    candidates = []
    if isinstance(draw_result, dict):
        for key in ("last_active_drawing", "last_drawn", "last_drawing"):
            value = draw_result.get(key)
            if value:
                candidates.append(value)

        drawings = draw_result.get("all_drawings")
        if isinstance(drawings, list) and drawings:
            candidates.extend(reversed(drawings))

    for item in candidates:
        geometry = item.get("geometry", item) if isinstance(item, dict) else None
        if not isinstance(geometry, dict):
            continue
        if geometry.get("type") != "LineString":
            continue
        coords = geometry.get("coordinates", [])
        if isinstance(coords, list) and len(coords) >= 2:
            vertices = []
            for coord in coords:
                if len(coord) >= 2:
                    vertices.append([float(coord[1]), float(coord[0])])
            if len(vertices) >= 2:
                return normalize_section_vertices(vertices)
    return None


def render_ab_selector_map(df_points, map_mode):
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium

    # 観測点群の中心を初期表示位置にして、線引き用の対話地図を作る
    # Build an interactive map centered on the observation cloud for drawing a section line.
    center_lat = float(df_points["Latitude_degN"].mean()) if not df_points.empty else 35.0
    center_lon = float(df_points["Longitude_degE"].mean()) if not df_points.empty else 135.0

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles=None)

    if map_mode == "Standard":
        folium.TileLayer("CartoDB positron", name="Standard").add_to(fmap)
    elif map_mode == "Satellite":
        folium.TileLayer(
            tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
            attr="USGS",
            name="Satellite",
        ).add_to(fmap)
    elif map_mode == "Bathymetry (Sea)":
        folium.TileLayer(
            tiles="https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Bathymetry (Sea)",
        ).add_to(fmap)
    elif map_mode == "Contour (GSI)":
        folium.TileLayer(
            tiles="https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
            attr="国土地理院 (GSI)",
            name="Contour (GSI)",
        ).add_to(fmap)

    # 既存の測点を薄い青点で表示し、線を引く目安にする
    # Show observation points as faint blue markers to help the user place the section line.
    map_points = sample_points_for_map(df_points[["Latitude_degN", "Longitude_degE"]].dropna())
    for _, row in map_points.iterrows():
        folium.CircleMarker(
            location=[float(row["Latitude_degN"]), float(row["Longitude_degE"])],
            radius=3,
            color="royalblue",
            weight=1,
            fill=True,
            fill_opacity=0.5,
        ).add_to(fmap)

    # ユーザーには polyline だけ描かせ、断面線以外の図形は無効化する
    # Allow only polyline drawing so the user defines just a section line.
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": {
                "shapeOptions": {
                    "color": "red",
                    "weight": 4,
                }
            },
            "polygon": False,
            "rectangle": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)

    return st_folium(fmap, width=None, height=420, key="ab_selector_map")


def load_bathymetry_table(uploaded_file):
    # ユーザーアップロードの簡易地形ファイル(CSV/Excel)を DataFrame に正規化する
    # Normalize a user-uploaded simple bathymetry file (CSV/Excel) into a standard DataFrame.
    if uploaded_file is None:
        return None, None

    suffix = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    bio = io.BytesIO(raw)

    if suffix.endswith(".csv") or suffix.endswith(".txt"):
        df = pd.read_csv(bio)
    elif suffix.endswith(".xlsx") or suffix.endswith(".xls"):
        df = pd.read_excel(bio)
    else:
        return None, "Bathymetry file must be CSV or Excel."

    col_map = {str(col).strip().lower(): col for col in df.columns}

    def pick_column(candidates):
        # 列名のゆらぎに対応するため、候補名の中から最初に一致したものを使う
        # Handle column-name variations by taking the first matching candidate.
        for candidate in candidates:
            if candidate in col_map:
                return col_map[candidate]
        return None

    lon_col = pick_column(["longitude_dege", "longitude", "lon", "x"])
    lat_col = pick_column(["latitude_degn", "latitude", "lat", "y"])
    depth_col = pick_column(["depth_m", "water_depth", "bathymetry_m", "bathymetry", "elevation_m", "z"])

    if not all([lon_col, lat_col, depth_col]):
        return None, "Bathymetry file needs lon/lat/depth columns."

    df_bathy = df[[lon_col, lat_col, depth_col]].copy()
    df_bathy.columns = ["Longitude_degE", "Latitude_degN", "BathymetryRaw"]
    df_bathy = df_bathy.apply(pd.to_numeric, errors="coerce").dropna()

    if df_bathy.empty:
        return None, "Bathymetry file has no usable numeric rows."

    # 負値の標高は海底深度へ反転し、正の値はすでに depth とみなしてそのまま使う
    # Convert negative elevations to positive water depth; keep positive values as already-downward depth.
    df_bathy["Bathymetry_m"] = np.where(
        df_bathy["BathymetryRaw"] < 0,
        -df_bathy["BathymetryRaw"],
        df_bathy["BathymetryRaw"],
    )
    return df_bathy[["Longitude_degE", "Latitude_degN", "Bathymetry_m"]], None


@st.cache_data(show_spinner=False)
def load_gebco_grid(nc_path):
    # GEBCO NetCDF をキャッシュして、再描画時の読み込み負荷を下げる
    # Cache the GEBCO NetCDF arrays to avoid repeated heavy disk reads.
    with netcdf_file(nc_path, "r", mmap=False) as nc:
        lon = np.array(nc.variables["lon"][:], dtype=float)
        lat = np.array(nc.variables["lat"][:], dtype=float)
        if "Height" in nc.variables:
            height_name = "Height"
        elif "elevation" in nc.variables:
            height_name = "elevation"
        else:
            raise KeyError("Bathymetry variable 'Height' or 'elevation' was not found in the NetCDF file.")
        height = np.array(nc.variables[height_name][:], dtype=float)
    return lon, lat, height


def crop_gebco_to_section(lon, lat, height, section_vertices, pad_deg=1.0):
    # 全球格子のまま補間せず、測線まわりだけを切り出して軽くする
    # Crop the global grid around the section before interpolation to reduce workload.
    lats = [lat for lat, _ in section_vertices]
    lons = [lon for _, lon in section_vertices]
    lon_min = min(lons) - pad_deg
    lon_max = max(lons) + pad_deg
    lat_min = min(lats) - pad_deg
    lat_max = max(lats) + pad_deg

    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lat_mask = (lat >= lat_min) & (lat <= lat_max)

    if lon_mask.sum() < 2 or lat_mask.sum() < 2:
        return lon, lat, height

    return lon[lon_mask], lat[lat_mask], height[np.ix_(lat_mask, lon_mask)]


def sample_netcdf_bathymetry_along_section(nc_path, section_vertices, xi):
    # GEBCO 格子を A-B 線上に補間して、海底断面の深度プロファイルを作る
    # Interpolate the GEBCO grid along the A-B line to build a bathymetric section profile.
    lon, lat, height = load_gebco_grid(nc_path)
    lon, lat, height = crop_gebco_to_section(lon, lat, height, section_vertices)
    interp = RegularGridInterpolator(
        (lat, lon),
        height,
        bounds_error=False,
        fill_value=np.nan,
    )

    sample_lon, sample_lat = densify_section_line(section_vertices, len(xi))
    values = interp(np.column_stack([sample_lat, sample_lon]))
    return np.where(np.isnan(values), np.nan, np.where(values < 0, -values, 0.0))


def sample_bathymetry_along_section(df_bathy, section_vertices, xi, corridor_km):
    # 点群の地形データを測線へ投影し、測線中央(cross-track=0)で深度を内挿する
    # Project point bathymetry onto the section and interpolate depth at cross-track = 0.
    if df_bathy is None or df_bathy.empty:
        return None

    bathy_projected, _, _ = project_points_to_polyline(df_bathy, section_vertices, corridor_km)
    if len(bathy_projected) < 3:
        return None

    points = (bathy_projected["SectionDistance_km"], bathy_projected["CrossTrack_km"])
    target = (xi, np.zeros_like(xi))
    z_linear = griddata(points, bathy_projected["Bathymetry_m"], target, method="linear")
    z_nearest = griddata(points, bathy_projected["Bathymetry_m"], target, method="nearest")
    return np.where(np.isnan(z_linear), z_nearest, z_linear)


def estimate_plot_depth_slider_max(
    df_section,
    section_mode,
    bathy_source,
    section_vertices,
    section_length_km,
    corridor_km,
    df_bathy_loaded,
):
    # サイドバーの縦軸上限を、海底地形の最大深度 + 200 m を基準に見積もる
    # Estimate the sidebar depth-axis upper bound from max seafloor depth + 200 m.
    if df_section.empty:
        return 100.0

    data_depth_max = float(df_section["Depth_m"].max())
    preview_xi = np.linspace(
        float(df_section["SectionDistance_km"].min()),
        float(df_section["SectionDistance_km"].max()),
        200,
    )

    bottom_profile = None
    if section_mode == "A-B section":
        if bathy_source == "Built-in GEBCO" and section_vertices is not None and section_length_km > 0:
            try:
                bottom_profile = sample_netcdf_bathymetry_along_section(
                    DEFAULT_GEBCO_PATH,
                    section_vertices,
                    preview_xi,
                )
            except Exception:
                bottom_profile = None
        elif bathy_source == "Upload CSV/Excel" and df_bathy_loaded is not None:
            bottom_profile = sample_bathymetry_along_section(
                df_bathy_loaded,
                section_vertices,
                preview_xi,
                corridor_km,
            )

    if bottom_profile is None:
        bottom_profile = build_bottom_profile_generic(df_section, "SectionDistance_km", preview_xi)
    if bottom_profile is None:
        bottom_profile = build_bottom_profile_from_observations(df_section, preview_xi)

    if bottom_profile is not None and np.isfinite(bottom_profile).any():
        return max(data_depth_max, float(np.nanmax(bottom_profile)) + 200.0)

    return max(100.0, data_depth_max + 200.0)


def build_bottom_profile_from_observations(df_points, xi):
    # 観測点の最深サンプルをつないで、簡易的な海底線を作る
    # Build a simple seafloor profile by linking the deepest observation at each along-track position.
    if df_points.empty:
        return None

    bottom_df = (
        df_points[["SectionDistance_km", "Depth_m"]]
        .dropna()
        .groupby("SectionDistance_km", as_index=False)["Depth_m"]
        .max()
        .sort_values("SectionDistance_km")
    )

    if len(bottom_df) < 2:
        return None

    return np.interp(
        xi,
        bottom_df["SectionDistance_km"].to_numpy(dtype=float),
        bottom_df["Depth_m"].to_numpy(dtype=float),
    )


def build_bottom_profile_generic(df_points, x_col, xi):
    # 任意の x 軸列に対して、各 x での最深観測深度から海底線を作る
    # Build a bottom profile from the deepest observed depth at each value of a generic x-axis column.
    if df_points.empty:
        return None

    bottom_df = (
        df_points[[x_col, "Depth_m"]]
        .dropna()
        .groupby(x_col, as_index=False)["Depth_m"]
        .max()
        .sort_values(x_col)
    )

    if len(bottom_df) < 2:
        return None

    return np.interp(
        xi,
        bottom_df[x_col].to_numpy(dtype=float),
        bottom_df["Depth_m"].to_numpy(dtype=float),
    )


def mask_below_bottom(z_grid, yi, bottom_profile):
    # 海底より下のグリッドを NaN にして、海中だけを描画対象にする
    # Mask grid cells below the seafloor so only the water column remains visible.
    if bottom_profile is None:
        return z_grid

    masked = z_grid.copy()
    depth_grid = np.tile(yi.reshape(-1, 1), (1, len(bottom_profile)))
    bottom_grid = np.tile(bottom_profile, (len(yi), 1))
    masked[depth_grid > bottom_grid] = np.nan
    return masked


def extend_section_toward_bottom(z_grid, yi, bottom_profile, max_fill_gap_m=250.0):
    # 最深の有効値を海底方向へ少しだけ延長し、深部の大きなデータ空白は白のまま残す
    # Extend the deepest valid value a limited distance toward the seafloor, while keeping large deep gaps white.
    if bottom_profile is None or z_grid.size == 0:
        return z_grid

    extended = z_grid.copy()
    yi = np.asarray(yi, dtype=float)
    bottom_profile = np.asarray(bottom_profile, dtype=float)

    for col in range(extended.shape[1]):
        column = extended[:, col]
        valid_idx = np.flatnonzero(np.isfinite(column))
        if valid_idx.size == 0:
            continue

        bottom_depth = bottom_profile[col] if col < len(bottom_profile) else np.nan
        if not np.isfinite(bottom_depth):
            continue

        deepest_idx = int(valid_idx[-1])
        deepest_depth = yi[deepest_idx]
        gap_to_bottom = bottom_depth - deepest_depth
        if gap_to_bottom <= 0:
            continue

        fill_limit_depth = min(bottom_depth, deepest_depth + max_fill_gap_m)
        fill_mask = (yi > deepest_depth) & (yi <= fill_limit_depth)
        if not np.any(fill_mask):
            continue

        # 直下の未観測帯だけを最深有効値で埋める
        # Fill only the immediate no-data band below the deepest valid sample.
        column[fill_mask & ~np.isfinite(column)] = column[deepest_idx]
        extended[:, col] = column

    return extended


def interpolate_section_grid(df_section, target_col, x_grid, y_grid):
    # 補間点が退化している場合に備え、cubic 失敗時は linear / nearest へ安全にフォールバックする
    # Safely fall back to linear/nearest when cubic interpolation fails on degenerate point geometry.
    x_vals = df_section["SectionDistance_km"].to_numpy(dtype=float)
    y_vals = df_section["Depth_m"].to_numpy(dtype=float)
    z_vals = df_section[target_col].to_numpy(dtype=float)
    points = (x_vals, y_vals)

    x_span = np.ptp(x_vals) if len(x_vals) else 0.0
    y_span = np.ptp(y_vals) if len(y_vals) else 0.0

    z_grid = None
    if x_span > 1.0e-9 and y_span > 1.0e-9 and len(df_section) >= 4:
        try:
            z_grid = griddata(points, z_vals, (x_grid, y_grid), method="cubic")
        except Exception:
            z_grid = None

    try:
        z_linear = griddata(points, z_vals, (x_grid, y_grid), method="linear")
    except Exception:
        z_linear = None

    z_nearest = griddata(points, z_vals, (x_grid, y_grid), method="nearest")

    if z_grid is None:
        z_grid = z_linear
    elif z_linear is not None:
        z_grid = np.where(np.isnan(z_grid), z_linear, z_grid)

    if z_grid is None:
        z_grid = z_nearest
    else:
        z_grid = np.where(np.isnan(z_grid), z_nearest, z_grid)

    return z_grid


def create_section_plot(
    z_grid,
    xi,
    yi,
    df_points,
    target_col,
    z_min,
    z_max,
    plot_type="color",
    bottom_profile=None,
    display_depth_max=None,
    xaxis_title="Distance along A-B (km)",
    hover_mode="ab",
):
    # 断面のコンター図と測点、必要に応じて海底線・海底塗りつぶしを重ねる
    # Draw the section contours, sample markers, and optionally the seafloor line/fill.
    fig = go.Figure()

    contour_kwargs = dict(
        z=z_grid,
        x=xi,
        y=yi,
        zmin=z_min,
        zmax=z_max,
        connectgaps=False,
        colorbar=dict(
            title=target_col,
            orientation="h",
            y=-0.30,
            x=0.5,
            xanchor="center",
            len=0.7,
            tickformat=".2f",
        ),
    )

    if plot_type == "color":
        # カラー断面表示
        # Filled color section.
        fig.add_trace(
            go.Contour(
                **contour_kwargs,
                colorscale="Blues",
                contours=dict(showlabels=True, coloring="heatmap", start=z_min, end=z_max),
                line_width=0,
            )
        )
    else:
        # 等値線のみの表示
        # Contour-line-only display.
        fig.add_trace(
            go.Contour(
                **contour_kwargs,
                contours=dict(
                    coloring="none",
                    showlabels=True,
                    labelfont=dict(size=12, color="black"),
                    start=z_min,
                    end=z_max,
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df_points["SectionDistance_km"],
            y=df_points["Depth_m"],
            mode="markers",
            marker=dict(size=6, color="black", opacity=0.65),
            name="Samples",
            customdata=np.column_stack([
                df_points["Longitude_degE"],
                df_points["Latitude_degN"],
                df_points.get("CrossTrack_km", pd.Series(np.zeros(len(df_points)))),
            ]),
            hovertemplate=(
                "Along: %{x:.2f}<br>"
                "Depth: %{y:.1f} m<br>"
                "Lon: %{customdata[0]:.4f}<br>"
                "Lat: %{customdata[1]:.4f}<br>"
                + ("Offset: %{customdata[2]:.2f} km<br>" if hover_mode == "ab" else "")
                + "<extra></extra>"
            ),
        )
    )

    # 海面付近を 0 m に合わせ、必要に応じて海底塗りつぶし分だけ描画下端を伸ばす
    # Anchor the top near 0 m and extend the lower limit when a seafloor fill is drawn.
    y_top = min(0.0, float(np.nanmin(yi)))
    y_bottom = float(display_depth_max) if display_depth_max is not None else float(np.nanmax(yi))

    if bottom_profile is not None and np.isfinite(bottom_profile).any():
        valid = np.isfinite(bottom_profile)
        clipped_bottom = np.minimum(bottom_profile, y_bottom)

        # 海底線そのもの
        # Seafloor line itself.
        fig.add_trace(
            go.Scatter(
                x=xi[valid],
                y=clipped_bottom[valid],
                mode="lines",
                line=dict(color="#4a4036", width=2),
                name="Seafloor",
            )
        )

        # 海底下を茶色で塗りつぶし、断面として見やすくする
        # Fill the subsurface area with brown shading for easier section interpretation.
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([xi[valid], xi[valid][::-1]]),
                y=np.concatenate(
                    [
                        clipped_bottom[valid],
                        np.full(valid.sum(), y_bottom, dtype=float)[::-1],
                    ]
                ),
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                fill="toself",
                fillcolor="rgba(110, 102, 89, 0.65)",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis_title=xaxis_title,
        xaxis=dict(title_standoff=18),
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed", range=[y_bottom, y_top]),
        height=620,
        margin=dict(l=50, r=50, b=150, t=40),
    )
    return fig


def create_station_map(
    df_points,
    section_vertices,
    sample_name="Samples in corridor",
    line_name="A-B Section",
    df_background=None,
    background_name="Filtered stations",
    max_background_points=None,
    max_foreground_points=None,
    map_mode="Standard",
):
    # 測点と A-B 線を同じ地図上に描き、どの観測点が断面に使われたか確認できるようにする
    # Plot samples and the A-B line together so the user can verify which points feed the section.
    fig = go.Figure()

    if df_background is not None and not df_background.empty:
        background_points = maybe_sample_points(df_background, max_background_points)
        fig.add_trace(
            go.Scattermapbox(
                lat=background_points["Latitude_degN"],
                lon=background_points["Longitude_degE"],
                mode="markers",
                marker=go.scattermapbox.Marker(size=6, color="lightgray", opacity=0.35),
                text=background_points["Station"] if "Station" in background_points.columns else "",
                name=background_name,
            )
        )

    if not df_points.empty:
        map_points = maybe_sample_points(df_points, max_foreground_points)
        fig.add_trace(
            go.Scattermapbox(
                lat=map_points["Latitude_degN"],
                lon=map_points["Longitude_degE"],
                mode="markers",
                marker=go.scattermapbox.Marker(size=9, color="blue", opacity=0.8),
                text=map_points["Station"] if "Station" in map_points.columns else "",
                name=sample_name,
            )
        )

    line_lon, line_lat = densify_section_line(section_vertices)
    fig.add_trace(
        go.Scattermapbox(
            lon=line_lon,
            lat=line_lat,
            mode="lines",
            line=dict(width=3, color="red"),
            name=line_name,
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lon=[section_vertices[0][1], section_vertices[-1][1]],
            lat=[section_vertices[0][0], section_vertices[-1][0]],
            mode="markers+text",
            marker=go.scattermapbox.Marker(size=12, color=["orange", "green"]),
            text=["A", "B"],
            textposition="top right",
            name="Section endpoints",
        )
    )

    lats = [lat for lat, _ in section_vertices]
    lons = [lon for _, lon in section_vertices]
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))
    lon_span = max(lons) - min(lons)
    lat_span = max(lats) - min(lats)
    span = max(lon_span, lat_span, 0.2)
    zoom = 7 if span < 0.5 else 6 if span < 1.5 else 5 if span < 4 else 4

    fig.update_layout(
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=520,
        showlegend=True,
        dragmode="pan",
    )
    return envgeo_utils.apply_map_style(fig, map_mode)


def prepare_axis_section(df_f, target_col, x_axis_option):
    # 元コード互換の Axis-based モード用に、選択軸を断面の x 軸へ整形する
    # Prepare the original axis-based section mode by turning the selected axis into the section x-axis.
    df_axis = df_f.copy()
    if x_axis_option == "Distance_km" and not df_axis.empty:
        # 元コード同様、左下側の基準点から近似距離を計算する
        # As in the original code, compute approximate distance from the sorted reference point.
        df_axis = df_axis.sort_values(["Latitude_degN", "Longitude_degE"]).copy()
        b_lat = df_axis.iloc[0]["Latitude_degN"]
        b_lon = df_axis.iloc[0]["Longitude_degE"]
        df_axis["Distance_km"] = np.sqrt(
            ((df_axis["Latitude_degN"] - b_lat) * 111.1) ** 2
            + ((df_axis["Longitude_degE"] - b_lon) * 111.1 * np.cos(np.radians(b_lat))) ** 2
        )

    df_plot = df_axis.dropna(subset=[x_axis_option, "Depth_m", target_col]).copy()
    if df_plot.empty:
        return df_plot, 0.0

    df_plot["SectionDistance_km"] = df_plot[x_axis_option].astype(float)
    df_plot["CrossTrack_km"] = 0.0
    length = float(df_plot["SectionDistance_km"].max() - df_plot["SectionDistance_km"].min())
    return df_plot, length


def main():
    # アプリ本体。フィルタ、断面条件、補間、地図表示を順にまとめる
    # Main app body: filters, section settings, interpolation, and map visualization.
    st.title("Vertical Section Visualizer")
    st.sidebar.title("Section Control Panel")

    ref_data_source = st.radio("Select Data Source:", envgeo_utils.DATA_SOURCES)
    try:
        df_raw = envgeo_utils.load_isotope_data(ref_data_source)
    except Exception as exc:
        st.error(f"Data loading error: {exc}")
        return

    st.sidebar.markdown("---")

    with st.sidebar.expander("Dataset Filters", expanded=False):
        # 元コードのデータ絞り込み操作はできるだけ維持する
        # Preserve the original dataset filtering workflow as much as possible.
        df_f = df_raw.copy()

        if "Dataset" in df_f.columns:
            dataset_list = sorted(df_f["Dataset"].dropna().unique().tolist())
            if dataset_list:
                selected_datasets = st.multiselect("Dataset", dataset_list, default=dataset_list)
                df_f = df_f[df_f["Dataset"].isin(selected_datasets)]

        if "Transect" in df_f.columns:
            transect_list = sorted(df_f["Transect"].dropna().unique().tolist())
            if transect_list:
                selected_transects = st.multiselect("Transect Line", transect_list, default=transect_list)
                df_f = df_f[df_f["Transect"].isin(selected_transects)]

        if "reference" in df_f.columns:
            ref_list = sorted(df_f["reference"].dropna().unique().tolist())
            if ref_list:
                selected_refs = st.multiselect("Reference", ref_list, default=ref_list)
                df_f = df_f[df_f["reference"].isin(selected_refs)]

        if "Year" in df_f.columns:
            years = sorted(df_f["Year"].dropna().unique().astype(int).tolist())
            if years:
                sel_years = st.multiselect("Year", years, default=years)
                df_f = df_f[df_f["Year"].isin(sel_years)]

        if "Month" in df_f.columns:
            months = sorted(df_f["Month"].dropna().unique().astype(int).tolist())
            if months:
                sel_months = st.multiselect("Month", months, default=months)
                df_f = df_f[df_f["Month"].isin(sel_months)]

        if not df_f.empty:
            lat_min = float(df_f["Latitude_degN"].min())
            lat_max = float(df_f["Latitude_degN"].max())
            lon_min = float(df_f["Longitude_degE"].min())
            lon_max = float(df_f["Longitude_degE"].max())

            sel_lat = st.slider("Latitude Range", lat_min, lat_max, (lat_min, lat_max))
            sel_lon = st.slider("Longitude Range", lon_min, lon_max, (lon_min, lon_max))

            df_f = df_f[
                df_f["Latitude_degN"].between(sel_lat[0], sel_lat[1])
                & df_f["Longitude_degE"].between(sel_lon[0], sel_lon[1])
            ]

    target_col = st.sidebar.radio("Target Parameter", ["d18O", "Salinity", "Temperature_degC", "dD"])
    section_mode = st.sidebar.radio("Section Mode", ["Axis-based", "A-B section"], index=1)
    x_axis_option = None
    if section_mode == "Axis-based":
        x_axis_option = st.sidebar.selectbox("X-axis for Section", ["Longitude_degE", "Latitude_degN", "Distance_km"])
    if ref_data_source == envgeo_utils.data_source_JAPAN_SEA:
        corridor_default = 30.0
    elif ref_data_source == envgeo_utils.data_source_AROUND_JAPAN:
        corridor_default = 100.0
    else:
        corridor_default = 150.0
    corridor_km = st.sidebar.slider("Half-width of section corridor (km)", 1.0, 300.0, corridor_default, 1.0)
    grid_res = st.sidebar.select_slider("Resolution", options=[30, 50, 70, 90, 110, 130, 150, 180], value=70)
    smoothness = st.sidebar.slider("Smoothing (visual only)", 0.0, 5.0, 1.0)
    show_seafloor = st.sidebar.checkbox("Show Seafloor", value=True)
    bottom_fill_limit_m = st.sidebar.slider(
        "Bottom fill limit below deepest data (m)",
        0.0,
        1000.0,
        250.0,
        10.0,
    )
    bathy_source = st.sidebar.selectbox(
        "Bathymetry Source",
        ["Observed deepest samples", "Built-in GEBCO", "Upload CSV/Excel"],
        index=1,
    )
    map_mode = st.sidebar.selectbox("Map mode", MAP_MODE_OPTIONS, index=0)
    max_rows_for_section_plot = int(
        st.sidebar.number_input(
            "Max valid rows for section plotting",
            min_value=100,
            max_value=50000,
            value=DEFAULT_MAX_ROWS_FOR_SECTION_PLOT,
            step=100,
        )
    )

    if bathy_source == "Built-in GEBCO":
        st.caption(GEBCO_ATTRIBUTION)

    st.sidebar.markdown("---")

    if df_f.empty:
        st.warning("No data remain after filtering.")
        return

    # 初期フィルタ後の有効データ件数が多すぎる場合は、Cloud での極端な重さを避ける
    # Avoid extremely heavy section rendering on Streamlit Cloud when too many valid rows remain.
    df_section_source = df_f.dropna(subset=["Longitude_degE", "Latitude_degN", "Depth_m", target_col]).copy()
    section_plot_allowed = len(df_section_source) <= max_rows_for_section_plot
    section_plot_blocked_message = (
        f"Section plotting is disabled while more than {max_rows_for_section_plot} valid rows remain "
        f"after filtering. Current valid rows: {len(df_section_source)}. "
        f"Please narrow Dataset / Transect / Year / Month / Lat-Lon filters."
    )

    suggested_vertices = suggest_default_section_vertices(df_f)
    if suggested_vertices is None:
        default_a = df_f.sort_values(["Longitude_degE", "Latitude_degN"]).iloc[0]
        default_b = df_f.sort_values(["Longitude_degE", "Latitude_degN"]).iloc[-1]
    else:
        default_a = pd.Series(
            {
                "Latitude_degN": float(suggested_vertices[0][0]),
                "Longitude_degE": float(suggested_vertices[0][1]),
            }
        )
        default_b = pd.Series(
            {
                "Latitude_degN": float(suggested_vertices[-1][0]),
                "Longitude_degE": float(suggested_vertices[-1][1]),
            }
        )
    section_vertices = None

    submitted_vertices_key = "v003_submitted_section_vertices"
    a_lat = a_lon = b_lat = b_lon = None
    section_ready_for_plot = True
    if section_mode == "A-B section":
        # A-B モードでは手入力と地図描画入力の両方を残す
        # In A-B mode, keep both manual endpoint entry and map-based drawing.
        endpoint_mode = st.sidebar.radio("A-B input", ["Manual", "Draw on map"], index=1, horizontal=False)
        st.sidebar.caption("Section endpoints")
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            a_lat = st.number_input("A lat", value=float(default_a["Latitude_degN"]), format="%.4f")
            a_lon = st.number_input("A lon", value=float(default_a["Longitude_degE"]), format="%.4f")
        with col_b:
            b_lat = st.number_input("B lat", value=float(default_b["Latitude_degN"]), format="%.4f")
            b_lon = st.number_input("B lon", value=float(default_b["Longitude_degE"]), format="%.4f")
        section_vertices = section_vertices_from_ab(a_lat, a_lon, b_lat, b_lon)

        if endpoint_mode == "Draw on map":
            # 地図上で引いた最後の線を A-B として採用する
            # Use the most recently drawn map line as the active A-B section.
            if section_plot_allowed:
                st.caption("Draw a single line on the map below. The first point becomes A and the last point becomes B.")
                try:
                    draw_result = render_ab_selector_map(df_f, map_mode)
                    drawn_vertices = extract_section_vertices_from_draw_result(draw_result)
                    if drawn_vertices is not None:
                        st.caption(
                            f"Detected drawn line with {len(drawn_vertices)} vertices. "
                            "Click Submit to use this line for section plotting."
                        )
                    submit_col, clear_col = st.columns(2)
                    with submit_col:
                        submitted_draw_line = st.button(
                            "Submit drawn A-B line",
                            disabled=drawn_vertices is None,
                            use_container_width=True,
                        )
                    with clear_col:
                        clear_drawn_line = st.button(
                            "Clear submitted line",
                            use_container_width=True,
                        )

                    if clear_drawn_line:
                        st.session_state.pop(submitted_vertices_key, None)

                    if submitted_draw_line and drawn_vertices is not None:
                        st.session_state[submitted_vertices_key] = normalize_section_vertices(drawn_vertices)

                    submitted_vertices = st.session_state.get(submitted_vertices_key)
                    if submitted_vertices is not None and len(submitted_vertices) >= 2:
                        section_vertices = normalize_section_vertices(submitted_vertices)
                        a_lat, a_lon = section_vertices[0]
                        b_lat, b_lon = section_vertices[-1]
                        st.success(
                            f"Using submitted section: A=({a_lat:.4f}, {a_lon:.4f}), "
                            f"B=({b_lat:.4f}, {b_lon:.4f})"
                        )
                        st.caption(f"Vertices in submitted section: {len(section_vertices)}")
                    else:
                        section_ready_for_plot = False
                        st.info("Draw a line and click 'Submit drawn A-B line' to run the section plot.")
                except Exception as exc:
                    st.warning(f"Interactive line drawing is unavailable here: {exc}")
            else:
                st.warning(section_plot_blocked_message)

    uploaded_bathy = None
    if bathy_source == "Upload CSV/Excel":
        uploaded_bathy = st.sidebar.file_uploader(
            "Bathymetry CSV/Excel",
            type=["csv", "txt", "xlsx", "xls"],
            help="Columns like Longitude_degE, Latitude_degN, Depth_m are supported.",
        )

    if section_mode == "A-B section":
        # A-B 線に沿って点群を投影し、断面用データを作る
        # Project observations onto the A-B line to create section-ready data.
        df_section, section_length_km, section_polyline = project_points_to_polyline(
            df_section_source,
            section_vertices,
            corridor_km,
        )
        status_label = "Samples inside A-B corridor"
        xaxis_title = "Distance along A-B (km)"
        hover_mode = "ab"
    else:
        # 従来の axis-based 断面をそのまま選べるようにしておく
        # Keep the legacy axis-based section workflow available.
        df_section, section_length_km = prepare_axis_section(df_f, target_col, x_axis_option)
        status_label = "Samples used for axis section"
        xaxis_title = x_axis_option
        hover_mode = "axis"

    st.write(f"Filtered Data: `{len(df_f)}` rows")
    st.write(f"Valid rows for section plotting: `{len(df_section_source)}` rows")
    st.write(f"{status_label}: `{len(df_section)}` rows")
    st.write(f"Map background points: `{len(df_f.dropna(subset=['Longitude_degE', 'Latitude_degN']))}` rows")
    if section_mode == "A-B section":
        st.write(f"Section length: `{section_length_km:.2f} km`")

    if not section_plot_allowed:
        st.warning(section_plot_blocked_message)
        return

    if section_mode == "A-B section" and not section_ready_for_plot:
        return

    df_bathy_loaded = None
    bathy_message = None
    if show_seafloor and section_mode == "A-B section" and bathy_source == "Upload CSV/Excel":
        # ユーザー地形ファイルを使う場合
        # Case 1: use user-uploaded bathymetry.
        df_bathy_loaded, bathy_error = load_bathymetry_table(uploaded_bathy)
        if bathy_error:
            bathy_message = bathy_error

    if not df_section.empty:
        slider_depth_max = estimate_plot_depth_slider_max(
            df_section,
            section_mode,
            bathy_source,
            section_vertices,
            section_length_km,
            corridor_km,
            df_bathy_loaded,
        )
        default_depth_max = min(slider_depth_max, max(10.0, float(df_section["Depth_m"].max())))
        plot_depth_max = st.sidebar.slider(
            "Max plotting depth (m)",
            min_value=10.0,
            max_value=max(10.0, slider_depth_max),
            value=max(10.0, default_depth_max),
            step=10.0,
        )
    else:
        plot_depth_max = 100.0

    if section_mode == "A-B section" and section_length_km == 0:
        st.error("A and B are identical. Please change the endpoints.")
        return

    if len(df_section) > 5:
        # 測線方向距離 x 深度 の2次元格子を作って、そこへ観測値を補間する
        # Create a 2D grid of along-section distance and depth, then interpolate observations onto it.
        x_min = float(df_section["SectionDistance_km"].min())
        x_max = float(df_section["SectionDistance_km"].max())
        xi = np.linspace(x_min, x_max, grid_res)
        yi = np.linspace(0.0, plot_depth_max, grid_res)
        x_grid, y_grid = np.meshgrid(xi, yi)

        try:
            # 点配置に応じて安全な補間法へ切り替える
            # Choose a safe interpolation path based on the geometry of the sampled points.
            z_grid = interpolate_section_grid(df_section, target_col, x_grid, y_grid)

            if show_seafloor:
                # 海底線データは GEBCO -> upload -> 観測最深点 の順で優先する
                # Prioritize seafloor sources in this order: GEBCO -> upload -> deepest observations.
                if bathy_source == "Built-in GEBCO" and section_mode == "A-B section":
                    try:
                        bottom_profile = sample_netcdf_bathymetry_along_section(
                            DEFAULT_GEBCO_PATH,
                            section_vertices,
                            xi,
                        )
                        if bottom_profile is not None and not np.isfinite(bottom_profile).any():
                            bottom_profile = None
                    except Exception:
                        bathy_message = "Built-in GEBCO could not be read, so observed maximum depth is being used instead."
                        bottom_profile = None
                elif df_bathy_loaded is not None and section_mode == "A-B section":
                    bottom_profile = sample_bathymetry_along_section(
                        df_bathy_loaded,
                        section_vertices,
                        xi,
                        corridor_km,
                    )
                    if bottom_profile is not None and not np.isfinite(bottom_profile).any():
                        bottom_profile = None
                else:
                    bottom_profile = build_bottom_profile_generic(df_section, "SectionDistance_km", xi)

                if bottom_profile is None:
                    if bathy_source == "Built-in GEBCO" and section_mode == "A-B section" and bathy_message is None:
                        bathy_message = "Built-in GEBCO returned no valid profile here, so observed maximum depth is being used instead."
                    bottom_profile = build_bottom_profile_from_observations(df_section, xi)
            else:
                bottom_profile = None

            z_grid = mask_below_bottom(z_grid, yi, bottom_profile)
            if show_seafloor and bottom_fill_limit_m > 0:
                z_grid = extend_section_toward_bottom(
                    z_grid,
                    yi,
                    bottom_profile,
                    max_fill_gap_m=bottom_fill_limit_m,
                )

            if smoothness > 0:
                # 平滑化後にも再度マスクして、海底下に値がにじまないようにする
                # Re-apply the mask after smoothing so values do not bleed below the seafloor.
                z_grid = gaussian_filter(z_grid, sigma=smoothness)
                z_grid = mask_below_bottom(z_grid, yi, bottom_profile)
                if show_seafloor and bottom_fill_limit_m > 0:
                    z_grid = extend_section_toward_bottom(
                        z_grid,
                        yi,
                        bottom_profile,
                        max_fill_gap_m=bottom_fill_limit_m,
                    )

            valid_vals = df_section[target_col].dropna()
            d_min = float(valid_vals.min()) if not valid_vals.empty else -10.0
            d_max = float(valid_vals.max()) if not valid_vals.empty else 35.0
            z_min, z_max = st.sidebar.slider(f"{target_col} Scale", -15.0, 40.0, (d_min, d_max))

            tab_color, tab_line = st.tabs(["Color", "Line"])
            with tab_color:
                st.plotly_chart(
                    create_section_plot(
                        z_grid,
                        xi,
                        yi,
                        df_section,
                        target_col,
                        z_min,
                        z_max,
                        "color",
                        bottom_profile,
                        plot_depth_max,
                        xaxis_title,
                        hover_mode,
                    ),
                    use_container_width=True,
                )
            with tab_line:
                st.plotly_chart(
                    create_section_plot(
                        z_grid,
                        xi,
                        yi,
                        df_section,
                        target_col,
                        z_min,
                        z_max,
                        "line",
                        bottom_profile,
                        plot_depth_max,
                        xaxis_title,
                        hover_mode,
                    ),
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"Interpolation error: {exc}")
    else:
        if section_mode == "A-B section":
            st.warning(
                f"Insufficient points inside the A-B corridor for interpolation. "
                f"Current corridor half-width: {corridor_km:.0f} km. "
                f"Please widen the corridor, redraw the section, or use Axis-based mode."
            )
        else:
            st.warning("Insufficient points for the selected filters.")

    if bathy_message:
        st.info(bathy_message)

    st.markdown("---")
    st.subheader("Section Map")
    df_map_background = df_f.dropna(subset=["Longitude_degE", "Latitude_degN"]).copy()
    if section_mode == "A-B section":
        map_fig = create_station_map(
            df_section,
            section_vertices,
            sample_name="Samples used for section",
            line_name="A-B Section",
            df_background=df_map_background,
            background_name="Filtered stations",
            max_background_points=None,
            max_foreground_points=None,
            map_mode=map_mode,
        )
    else:
        line_data = df_section.sort_values("SectionDistance_km")
        start_pt = line_data.iloc[0]
        end_pt = line_data.iloc[-1]
        map_fig = create_station_map(
            df_section,
            section_vertices_from_ab(
                float(start_pt["Latitude_degN"]),
                float(start_pt["Longitude_degE"]),
                float(end_pt["Latitude_degN"]),
                float(end_pt["Longitude_degE"]),
            ),
            sample_name="Samples used for section",
            line_name="Transect Line",
            df_background=df_map_background,
            background_name="Filtered stations",
            max_background_points=None,
            max_foreground_points=None,
            map_mode=map_mode,
        )

    st.plotly_chart(
        map_fig,
        use_container_width=True,
        config={"scrollZoom": True},
    )

    with st.expander("selected dataset (CSV)", expanded=False):
        # 断面描画に実際に使ったデータだけを、見やすい列順で表示する
        # Show only the dataset actually used for plotting, with a readable column order.
        preferred_cols = [
            "reference",
            "Cruise",
            "Station",
            "Date",
            "Year",
            "Month",
            "Longitude_degE",
            "Latitude_degN",
            "Depth_m",
            "SectionDistance_km",
            "CrossTrack_km",
            "Temperature_degC",
            "Salinity",
            "d18O",
            "dD",
        ]
        selected_cols = [col for col in preferred_cols if col in df_section.columns]
        if not selected_cols:
            selected_cols = df_section.columns.tolist()

        df_section_table = df_section[selected_cols].copy()
        df_section_table = df_section_table.dropna(how="all")
        df_section_table = df_section_table.astype(str)
        st.dataframe(df_section_table)


if __name__ == "__main__":
    main()
