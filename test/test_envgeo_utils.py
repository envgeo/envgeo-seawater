#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for EnvGeo-Seawater shared utility functions.

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

import io
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import envgeo_utils


REQUIRED_COLUMNS = {
    "Dataset",
    "Latitude_degN",
    "Longitude_degE",
    "Depth_m",
    "Salinity",
    "d18O",
}


# Verifies that app version metadata is kept in one reusable place.
# アプリのバージョン情報が、使い回せる共通定数として管理されていることを確認する。
def test_app_version_metadata_is_available():
    assert envgeo_utils.APP_VERSION == "1.3.3"
    assert envgeo_utils.APP_VERSION_DATE == "2026-09-23"
    assert envgeo_utils.APP_VERSION in envgeo_utils.APP_VERSION_LABEL
    assert envgeo_utils.APP_VERSION_DATE in envgeo_utils.APP_VERSION_LABEL
    assert envgeo_utils.version == envgeo_utils.APP_VERSION


def test_apply_standard_map_layout_keeps_map_and_legend_inside_full_figure():
    fig = envgeo_utils.apply_standard_map_layout(go.Figure(), height=480)

    assert tuple(fig.layout.mapbox.domain.x) == (0.0, 1.0)
    assert tuple(fig.layout.mapbox.domain.y) == (0.0, 1.0)
    assert fig.layout.height == 480
    assert fig.layout.margin.autoexpand is False
    assert fig.layout.legend.x == 0.01
    assert fig.layout.legend.y == 0.01
    assert fig.layout.legend.bgcolor == "rgba(255,255,255,0.85)"


# Verifies that quality rules document both machine-readable ranges and Japanese explanations.
# 品質チェック基準に、機械的に使える範囲と日本語説明の両方が含まれていることを確認する。
def test_quality_value_rules_are_documented():
    rules = envgeo_utils.QUALITY_VALUE_RULES

    assert rules["Depth_m"]["valid_range"] == (0, None)
    assert rules["Temperature_degC"]["valid_range"] == (-5, 45)
    assert rules["Salinity"]["valid_range"] == (0, 50)

    for rule in rules.values():
        assert rule["flag"]
        assert rule["description_ja"]


# Verifies compact UI text for quality-flag criteria.
# 品質フラグ基準をページ下部などに短く表示できることを確認する。
def test_quality_flag_criteria_text_mentions_core_rules():
    text = envgeo_utils.quality_flag_criteria_text()

    assert "Quality flags" in text
    assert "Depth_m" in text
    assert "Temperature_degC" in text
    assert "Salinity" in text
    assert "NaN" in text


# Verifies that figure download filenames are built safely in one shared function.
# 図の保存ファイル名が、共通関数で安全に整形されることを確認する。
def test_build_figure_filename_removes_unsafe_characters():
    filename = envgeo_utils.build_figure_filename(
        "Fig_T-S_SW",
        'Lon:120-140, Lat:30-40, Name:"test/data"',
    )

    assert filename.endswith(".png")
    for unsafe_character in [":", ",", " ", '"', "/"]:
        assert unsafe_character not in filename
    assert filename.startswith("Fig_T-S_SW_")


# Verifies fallback names for empty titles.
# 空のタイトルでも保存可能なファイル名が作られることを確認する。
def test_build_figure_filename_uses_fallback_for_empty_text():
    assert envgeo_utils.build_figure_filename("", "", "png") == "figure.png"


# Verifies reusable map-region presets for integrated and future pages.
# 統合ページや将来の共通地図設定で使う海域プリセットが利用できることを確認する。
def test_map_region_presets_include_expected_ocean_regions():
    presets = envgeo_utils.MAP_REGION_PRESETS

    for region in [
        "Japan and surrounding area",
        "East China Sea",
        "Kuroshio region",
        "North Pacific",
        "Tropical Pacific",
        "Indian Ocean",
        "North Atlantic",
        "Mediterranean Sea",
        "Arctic Ocean",
        "Southern Ocean",
        "Southern Ocean - Pacific sector",
        "Antarctic margin",
        "Global",
    ]:
        assert region in presets
        center_lat, center_lon, zoom = envgeo_utils.map_region_view(region)
        assert -90 <= center_lat <= 90
        assert -180 <= center_lon <= 180
        assert 1 <= zoom <= 15


# Verifies that area presets can initialize Data filtering lon/lat sliders.
# Data filtering のエリアプリセットが、緯度経度スライダーの初期範囲として使えることを確認する。
def test_area_filter_bounds_uses_region_presets_within_data_extent():
    lon_min, lon_max, lat_min, lat_max = envgeo_utils.area_filter_bounds(
        "Japan and surrounding area",
        100,
        180,
        0,
        70,
    )

    assert (lon_min, lon_max, lat_min, lat_max) == (120.0, 155.0, 20.0, 50.0)


# Verifies that Manual keeps the full current data extent.
# Manual 選択時は、現在のデータ範囲全体を保つことを確認する。
def test_area_filter_bounds_manual_keeps_full_data_extent():
    assert envgeo_utils.area_filter_bounds(
        envgeo_utils.AREA_FILTER_MANUAL,
        110,
        150,
        10,
        60,
    ) == (110.0, 150.0, 10.0, 60.0)


# Verifies the shared longitude-normalization helper, regardless of
# whether the input already uses -180..180 or 0..360.
# -180..180 と 0..360 のどちらの表記でも、共通ヘルパーが同じ正準値に
# 正規化することを確認する。
def test_normalize_longitude_deg_handles_both_conventions():
    assert envgeo_utils.normalize_longitude_deg(-162.0) == pytest.approx(-162.0)
    assert envgeo_utils.normalize_longitude_deg(198.0) == pytest.approx(-162.0)
    assert envgeo_utils.normalize_longitude_deg(179.0) == pytest.approx(179.0)
    assert envgeo_utils.normalize_longitude_deg(-179.0) == pytest.approx(-179.0)
    assert envgeo_utils.normalize_longitude_deg(181.0) == pytest.approx(-179.0)

    series_360 = pd.Series([160.0, 200.0, 330.0])
    normalized = envgeo_utils.normalize_longitude_deg(series_360)
    assert list(normalized.round(1)) == [160.0, -160.0, -30.0]


# Verifies which area-filter presets are flagged as antimeridian-crossing.
# どのエリアプリセットが日付変更線をまたぐと判定されるかを確認する。
def test_region_preset_crosses_dateline_flags_only_wrap_around_presets():
    assert envgeo_utils.region_preset_crosses_dateline("Bering Sea") is True
    assert envgeo_utils.region_preset_crosses_dateline("North Pacific") is True
    assert envgeo_utils.region_preset_crosses_dateline("Mediterranean Sea") is False
    assert envgeo_utils.region_preset_crosses_dateline("Sea of Japan") is False
    assert envgeo_utils.region_preset_crosses_dateline(envgeo_utils.AREA_FILTER_MANUAL) is False


# Regression test for the Bering Sea Data-filtering bug: a single
# Longitude range slider cannot express a range crossing the antimeridian,
# so the old area_filter_bounds()-based slider silently fell back to the
# full data longitude range and let unrelated basins (e.g. the North
# Atlantic) through on latitude alone. The OR-based mask must keep points
# on either arm of the Bering Sea and reject points east of it (mainland
# North America) or on the wrong ocean (North Atlantic) at the same
# latitude — and must do so identically whether longitude is expressed as
# -180..180 or 0..360.
# Bering Sea の Data filtering 不具合の回帰テスト: 単一の Longitude range
# スライダーでは日付変更線をまたぐ範囲を表現できず、従来の
# area_filter_bounds() ベースのスライダーはデータの全経度範囲へ静かに
# フォールバックし、緯度だけで北大西洋のような無関係な海域まで通してし
# まっていた。OR条件のマスクは、ベーリング海の両端（東経160度以東・
# 西経162度以西）の点を残し、その東側（北米大陸）や別の海域（北大西洋）
# の同緯度の点を除外しなければならず、経度が -180..180 と 0..360 の
# どちらの表記でも同じ結果になる必要がある。
@pytest.mark.parametrize(
    ("lon_included", "lon_excluded"),
    [
        # -180..180 convention.
        ((160.0, 179.0, -179.0, -162.0), (-150.0, -30.0)),
        # 0..360 convention for the same real-world points.
        ((160.0, 179.0, 181.0, 198.0), (210.0, 330.0)),
    ],
)
def test_bering_sea_longitude_mask_uses_or_condition_across_conventions(
    lon_included, lon_excluded
):
    lon_values = pd.Series(list(lon_included) + list(lon_excluded))
    mask = envgeo_utils.region_preset_longitude_mask(lon_values, "Bering Sea")

    assert mask.iloc[: len(lon_included)].all(), (
        "Points on either arm of the Bering Sea (160E.. / ..162W) must be kept."
    )
    assert not mask.iloc[len(lon_included):].any(), (
        "Points east of the Bering Sea (mainland North America) or in an "
        "unrelated basin (North Atlantic) at the same latitude must be excluded."
    )


# Verifies that the Japan Sea dataset can be loaded successfully and is not empty.
# 日本海データセットが正常に読み込まれ、空でないことを確認する。
def test_load_isotope_data_japan_sea_not_empty():
    df = envgeo_utils.load_isotope_data(envgeo_utils.data_source_JAPAN_SEA)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


# Verifies that the global dataset can be loaded successfully and is not empty.
# グローバルデータセットが正常に読み込まれ、空でないことを確認する。
def test_load_isotope_data_global_not_empty():
    df = envgeo_utils.load_isotope_data(envgeo_utils.data_source_GLOBAL)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


# Verifies that the loaded dataframe contains the core columns required by the application.
# 読み込まれたデータフレームに、アプリの主要処理に必要な基本列が含まれていることを確認する。
def test_load_isotope_data_has_required_columns():
    df = envgeo_utils.load_isotope_data(envgeo_utils.data_source_AROUND_JAPAN)
    assert REQUIRED_COLUMNS.issubset(df.columns)


# Verifies that the main analytical columns are converted to numeric types during data loading.
# 主要な解析用列がデータ読み込み時に数値型へ正しく変換されていることを確認する。
def test_load_isotope_data_numeric_columns_are_numeric():
    df = envgeo_utils.load_isotope_data(envgeo_utils.data_source_GLOBAL)

    numeric_cols = [
        "d18O",
        "dD",
        "Longitude_degE",
        "Latitude_degN",
        "Depth_m",
        "Temperature_degC",
        "Salinity",
    ]

    for col in numeric_cols:
        assert col in df.columns
        assert is_numeric_dtype(df[col]), f"{col} is not numeric"


# Verifies that a blank separator row is inserted when observation groups change.
# 観測グループが切り替わる位置で、区切り用の空白行が挿入されることを確認する。
def test_insert_gap_rows_inserts_blank_row_between_groups():
    df = pd.DataFrame(
        {
            "Latitude_degN": [35.0, 35.0, 36.0],
            "Longitude_degE": [135.0, 135.0, 136.0],
            "Year": [2020, 2020, 2020],
            "Month": [1, 1, 2],
            "reference": ["A", "A", "B"],
            "d18O": [0.1, 0.2, 0.3],
        }
    )

    out = envgeo_utils.insert_gap_rows(df)

    assert len(out) == len(df) + 1
    gap_rows = out[out.isna().all(axis=1)]
    assert len(gap_rows) == 1


# Verifies that inserting gap rows does not disturb the original order of non-empty observations.
# 空白行を挿入しても、元の観測データの順序が維持されていることを確認する。
def test_insert_gap_rows_preserves_non_gap_row_order():
    df = pd.DataFrame(
        {
            "Latitude_degN": [35.0, 35.0, 36.0],
            "Longitude_degE": [135.0, 135.0, 136.0],
            "Year": [2020, 2020, 2020],
            "Month": [1, 1, 2],
            "reference": ["A", "A", "B"],
            "d18O": [0.1, 0.2, 0.3],
        }
    )

    out = envgeo_utils.insert_gap_rows(df)
    non_gap = out[out["d18O"].notna()].reset_index(drop=True)

    pd.testing.assert_series_equal(non_gap["d18O"], df["d18O"], check_names=False)
    pd.testing.assert_series_equal(non_gap["reference"], df["reference"], check_names=False)


# Verifies that no blank row is inserted when all observations belong to the same group.
# すべて同じ観測グループの場合、余計な空白行が入らないことを確認する。
def test_insert_gap_rows_does_not_insert_gap_within_same_group():
    df = pd.DataFrame(
        {
            "Latitude_degN": [35.0, 35.0],
            "Longitude_degE": [135.0, 135.0],
            "Year": [2020, 2020],
            "Month": [1, 1],
            "reference": ["A", "A"],
            "d18O": [0.1, 0.2],
        }
    )

    out = envgeo_utils.insert_gap_rows(df)

    assert len(out) == len(df)
    assert out.isna().all(axis=1).sum() == 0


# Verifies that missing metadata can be handled consistently during group checks.
# メタデータに欠損があっても、グループ判定が安定して動くことを確認する。
def test_insert_gap_rows_handles_missing_group_metadata():
    df = pd.DataFrame(
        {
            "Latitude_degN": [35.0, 35.0, 36.0],
            "Longitude_degE": [135.0, 135.0, 136.0],
            "Year": [2020, 2020, 2020],
            "Month": [1, 1, 1],
            "reference": [None, None, None],
            "d18O": [0.1, 0.2, 0.3],
        }
    )

    out = envgeo_utils.insert_gap_rows(df)

    assert len(out) == len(df) + 1
    assert out.isna().all(axis=1).sum() == 1


# Verifies that depth-related variables use the dedicated depth colorscale.
# 深度に関する変数に対して、専用の深度カラースケールが返されることを確認する。
def test_get_custom_colorscale_returns_depth_scale_for_depth():
    scale = envgeo_utils.get_custom_colorscale("Depth_m")
    assert isinstance(scale, list)
    assert scale[0][1] == "red"
    assert scale[-1][1] == "darkblue"


# Verifies that non-depth variables use the standard colorscale.
# 深度以外の変数に対して、標準カラースケールが返されることを確認する。
def test_get_custom_colorscale_returns_standard_scale_for_non_depth():
    scale = envgeo_utils.get_custom_colorscale("d18O")
    assert isinstance(scale, list)
    assert "darkblue" in scale
    assert "red" in scale


# Verifies shared Plotly colormap options, including cmocean choices for oceanographic data.
# 海洋データ向けのcmocean候補を含む、共通Plotlyカラーマップ選択肢を確認する。
def test_plotly_colormap_options_include_cmocean_choices():
    options = envgeo_utils.get_plotly_colormap_options("Salinity")

    assert options["cmocean thermal"] == "thermal"
    assert options["cmocean haline"] == "haline"
    assert options["cmocean deep"] == "deep"
    assert options["cmocean balance"] == "balance"
    assert options["cmocean delta"] == "delta"
    assert options["Jet"] == "jet"


# Verifies that EnvGeo remains the default while cmocean stays available as a suggestion.
# 初期値はEnvGeo標準のままにし、cmoceanは候補として利用できることを確認する。
def test_colormap_default_is_envgeo_with_cmocean_suggestions():
    assert envgeo_utils.recommended_plotly_colormap_label("Temperature_degC") == "EnvGeo variable default"
    assert envgeo_utils.recommended_plotly_colormap_label("Salinity") == "EnvGeo variable default"
    assert envgeo_utils.recommended_plotly_colormap_label("Depth_m") == "EnvGeo variable default"
    assert envgeo_utils.recommended_plotly_colormap_label("d18O") == "EnvGeo variable default"
    assert envgeo_utils.recommended_plotly_colormap_label("d-excess") == "EnvGeo variable default"

    assert envgeo_utils.suggested_cmocean_colormap_label("Temperature_degC") == "cmocean thermal"
    assert envgeo_utils.suggested_cmocean_colormap_label("Salinity") == "cmocean haline"
    assert envgeo_utils.suggested_cmocean_colormap_label("Depth_m") == "cmocean deep"
    assert envgeo_utils.suggested_cmocean_colormap_label("d18O") == "cmocean balance"
    assert envgeo_utils.suggested_cmocean_colormap_label("d-excess") == "cmocean delta"


# Verifies that Matplotlib figures can use the shared EnvGeo/cmocean colormap selector.
# Matplotlib図でも共通カラーマップ選択を使えることを確認する。
def test_get_matplotlib_colormap_returns_colormap_object():
    colormap = envgeo_utils.get_matplotlib_colormap("d18O", "EnvGeo variable default")

    assert hasattr(colormap, "__call__")
    assert hasattr(colormap, "name")


# Verifies that the legacy Jet palette remains available for pages that used it originally.
# 以前からJetを使っていたページ向けに、Jetカラーマップが引き続き使えることを確認する。
def test_get_matplotlib_colormap_supports_jet():
    colormap = envgeo_utils.get_matplotlib_colormap("d18O", "Jet")

    assert colormap.name == "jet"


# Verifies that common uploaded-data labels can be mapped to standard EnvGeo columns.
# アップロードデータでよくある列名の別名を、標準列名へ自動変換できることを確認する。
def test_standardize_uploaded_column_names_maps_common_aliases():
    df = pd.DataFrame(
        {
            "lon": [135.0],
            "lat": [35.0],
            "sampling_month": [4],
            "Depth": [10.0],
            "Temp": [20.0],
            "S": [34.5],
            "δ18O": [0.1],
            "delta_D": [1.2],
        }
    )

    out = envgeo_utils.standardize_uploaded_column_names(df)

    for column in [
        "Longitude_degE",
        "Latitude_degN",
        "Month",
        "Depth_m",
        "Temperature_degC",
        "Salinity",
        "d18O",
        "dD",
    ]:
        assert column in out.columns
    assert out.attrs["standardized_column_renames"]["lon"] == "Longitude_degE"
    assert out.attrs["standardized_column_renames"]["δ18O"] == "d18O"


# Verifies that existing standard columns are not overwritten by alias columns.
# 既に標準列がある場合は、別名列より標準列を優先することを確認する。
def test_standardize_uploaded_column_names_preserves_existing_standard_columns():
    df = pd.DataFrame(
        {
            "Longitude_degE": [135.0],
            "lon": [140.0],
            "Latitude_degN": [35.0],
        }
    )

    out = envgeo_utils.standardize_uploaded_column_names(df)

    assert out["Longitude_degE"].tolist() == [135.0]
    assert "lon" in out.columns
    assert "lon" not in out.attrs["standardized_column_renames"]


# Verifies that unambiguous Japanese labels are recognized without guessing unknown labels.
# 明確な日本語列名のみを標準列名へ変換できることを確認する。
def test_standardize_uploaded_column_names_maps_japanese_aliases():
    df = pd.DataFrame(
        {"経度": [135.0], "緯度": [35.0], "月": [4], "水深": [10], "水温": [20], "塩分": [34.5]}
    )

    out = envgeo_utils.standardize_uploaded_column_names(df)

    assert {
        "Longitude_degE",
        "Latitude_degN",
        "Month",
        "Depth_m",
        "Temperature_degC",
        "Salinity",
    }.issubset(out.columns)


class NamedBytesIO(io.BytesIO):
    """In-memory upload stand-in carrying the filename used by Streamlit."""

    def __init__(self, content, name):
        super().__init__(content)
        self.name = name


# Verifies that CSV uploads are read directly from memory.
# CSVアップロードをファイル保存なしで読み込めることを確認する。
def test_read_uploaded_table_reads_csv_from_memory():
    uploaded_file = NamedBytesIO(b"Salinity,Temperature_degC\n34.5,20.0\n", "sample.csv")

    out = envgeo_utils.read_uploaded_table(uploaded_file)

    assert out.to_dict("records") == [{"Salinity": 34.5, "Temperature_degC": 20.0}]


# Verifies the complete upload preparation contract used by app pages.
# 数値化、品質判定、d-excess計算の一連の前処理を確認する。
def test_prepare_uploaded_data_applies_numeric_quality_and_d_excess():
    df = pd.DataFrame(
        {
            "Depth": ["10", "-1"],
            "Temp": ["20", "46"],
            "S": ["34.5", "55"],
            "delta18o": ["0.5", "1.0"],
            "d2h": ["4", "10"],
        }
    )

    out = envgeo_utils.prepare_uploaded_data(df)

    assert out["Dataset"].tolist() == ["Uploaded data", "Uploaded data"]
    assert out.loc[0, "d-excess"] == 0.0
    assert pd.isna(out.loc[1, "Depth_m"])
    assert pd.isna(out.loc[1, "Temperature_degC"])
    assert pd.isna(out.loc[1, "Salinity"])
    assert "Depth_m outside valid range" in out.loc[1, envgeo_utils.QUALITY_FLAG_COLUMN]
    assert "Temperature_degC outside valid range" in out.loc[1, envgeo_utils.QUALITY_FLAG_COLUMN]
    assert "Salinity outside valid range" in out.loc[1, envgeo_utils.QUALITY_FLAG_COLUMN]


def test_spreadsheet_numeric_values_accept_invisible_unicode_spaces():
    values = pd.Series(["34.620\u00a0", " 141.103 ", "\u22120.30", None])

    converted = envgeo_utils.coerce_numeric_values(values)

    assert converted.iloc[:3].tolist() == [34.62, 141.103, -0.3]
    assert pd.isna(converted.iloc[3])


def test_spreadsheet_numeric_values_accept_pyarrow_strings():
    pytest.importorskip("pyarrow")
    values = pd.Series(
        ["34.620\u00a0", "\u3000141.103", "\u22120.30"],
        dtype="string[pyarrow]",
    )

    converted = envgeo_utils.coerce_numeric_values(values)

    assert converted.tolist() == [34.62, 141.103, -0.3]


# Verifies that d-excess remains unknown when dD is not supplied.
# dDが無い場合にd-excessを推測せずNaNとすることを確認する。
def test_prepare_uploaded_data_keeps_d_excess_nan_without_dd():
    out = envgeo_utils.prepare_uploaded_data(
        pd.DataFrame({"Salinity": [34.5], "Temperature_degC": [20], "d18O": [0.5]})
    )

    assert out["d-excess"].isna().all()


def test_arrow_display_dataframe_coerces_mixed_identifier_columns_to_strings():
    """Mixed spreadsheet identifiers must not trigger Streamlit Arrow errors."""
    source = pd.DataFrame({"Station": [14, "14_5", None], "Salinity": [34.1, 34.2, 34.3]})

    displayed = envgeo_utils.arrow_display_dataframe(source)

    assert str(displayed["Station"].dtype) == "string"
    assert displayed["Station"].tolist() == ["14", "14_5", pd.NA]
    assert source["Station"].tolist() == [14, "14_5", None]


def test_filter_uploaded_data_for_sidebar_keeps_overlay_separate_and_filterable():
    uploaded = pd.DataFrame(
        {
            "Year": [2020, 2021, 2021],
            "Month": [1, 2, 3],
            "Longitude_degE": [135.0, 140.0, 145.0],
            "Latitude_degN": [35.0, 36.0, 37.0],
            "Depth_m": [10.0, 20.0, 30.0],
            "Salinity": [34.0, 35.0, 36.0],
        }
    )
    state = {
        envgeo_utils._uploaded_filter_state_key("test"): {
            "show": True,
            "apply_reference_filters": True,
            "selected_dataset": [],
            "selected_cruise": [],
            "selected_months": [2, 3],
            "year_range": (2021, 2021),
            "longitude_range": (139, 146),
            "latitude_range": (0, 90),
            "depth_range": (0, 100),
            "salinity_range": (0, 100),
            "d18o_range": (-10, 10),
            "temperature_range": (0, 40),
        }
    }

    filtered = envgeo_utils.filter_uploaded_data_for_sidebar(
        uploaded, "test", state=state
    )

    assert filtered.index.tolist() == [1, 2]
    state[envgeo_utils._uploaded_filter_state_key("test")]["show"] = False
    assert envgeo_utils.filter_uploaded_data_for_sidebar(
        uploaded, "test", state=state
    ).empty


# Verifies that unknown labels can be assigned manually without losing source columns.
# 未知の列名を手動対応させても、元の列を残したまま品質判定できることを確認する。
def test_apply_uploaded_column_mapping_copies_and_validates_selected_columns():
    source = pd.DataFrame(
        {
            "Water property A": [20.0, 50.0],
            "Water property B": [34.5, 60.0],
            "New element": [1.2, 1.5],
        }
    )

    out = envgeo_utils.apply_uploaded_column_mapping(
        source,
        {
            "Temperature_degC": "Water property A",
            "Salinity": "Water property B",
        },
    )

    assert out.loc[0, "Temperature_degC"] == 20.0
    assert out.loc[0, "Salinity"] == 34.5
    assert pd.isna(out.loc[1, "Temperature_degC"])
    assert pd.isna(out.loc[1, "Salinity"])
    assert out["New element"].tolist() == [1.2, 1.5]
    assert out.attrs["manual_column_mapping"] == {
        "Temperature_degC": "Water property A",
        "Salinity": "Water property B",
    }


# Verifies that a stale manual mapping cannot silently select a missing column.
# 古い手動対応で存在しない列を黙って採用しないことを確認する。
def test_apply_uploaded_column_mapping_rejects_missing_source_column():
    source = pd.DataFrame({"Unknown temperature": [20.0]})

    try:
        envgeo_utils.apply_uploaded_column_mapping(
            source,
            {"Temperature_degC": "Missing column"},
        )
    except KeyError as exc:
        assert "Missing column" in str(exc)
    else:
        raise AssertionError("Missing source column should raise KeyError")


# Verifies that automatic column confirmation does not erase existing quality flags.
# 自動認識済み列の確認で、既存の品質フラグが消えないことを確認する。
def test_apply_uploaded_column_mapping_preserves_existing_quality_flags():
    prepared = envgeo_utils.prepare_uploaded_data(
        pd.DataFrame({"Temperature_degC": [50.0], "Salinity": [34.5]})
    )

    mapped = envgeo_utils.apply_uploaded_column_mapping(
        prepared,
        {
            "Temperature_degC": "Temperature_degC",
            "Salinity": "Salinity",
        },
    )

    quality_rows = envgeo_utils.get_quality_rows(mapped)
    assert len(quality_rows) == 1
    assert "Temperature_degC outside valid range" in quality_rows.iloc[0][
        envgeo_utils.QUALITY_FLAG_COLUMN
    ]
    assert "original=50.0" in quality_rows.iloc[0][
        envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN
    ]


# Verifies that manual assignments add new flags without removing earlier ones.
# 手動列対応で新しい品質フラグを追加しても、以前のフラグを保持することを確認する。
def test_apply_uploaded_column_mapping_merges_existing_and_new_quality_flags():
    prepared = envgeo_utils.prepare_uploaded_data(
        pd.DataFrame({"Depth_m": [-1.0], "Unknown temperature": [50.0], "S": [34.5]})
    )

    mapped = envgeo_utils.apply_uploaded_column_mapping(
        prepared,
        {
            "Temperature_degC": "Unknown temperature",
            "Salinity": "Salinity",
        },
    )

    flag = mapped.loc[0, envgeo_utils.QUALITY_FLAG_COLUMN]
    assert "Depth_m outside valid range" in flag
    assert "Temperature_degC outside valid range" in flag


# Verifies that shared page state is memory-only and can be cleared explicitly.
# ページ間で共有するメモリ上のデータを保持・取得・消去できることを確認する。
def test_uploaded_data_session_helpers_round_trip_and_clear():
    state = {}
    source = pd.DataFrame({"Salinity": [34.5]})

    envgeo_utils.store_uploaded_data(source, "sample.csv", state=state)
    loaded = envgeo_utils.get_uploaded_data(state=state)

    assert loaded.equals(source)
    assert envgeo_utils.get_uploaded_filename(state=state) == "sample.csv"
    loaded.loc[0, "Salinity"] = 0
    assert envgeo_utils.get_uploaded_data(state=state).loc[0, "Salinity"] == 34.5

    envgeo_utils.clear_uploaded_data(state=state)
    assert envgeo_utils.get_uploaded_data(state=state).empty
    assert envgeo_utils.get_uploaded_filename(state=state) is None


def test_local_user_data_is_labeled_as_user_excel_data(tmp_path):
    local_file = tmp_path / "researcher_data.csv"
    local_file.write_text(
        "Longitude,Latitude,Depth,delta18o\n135.0,35.0,10,0.5\n",
        encoding="utf-8",
    )
    loaded = envgeo_utils.load_local_user_data(local_file)

    assert loaded["Dataset"].tolist() == ["User Excel data"]
    assert loaded["Longitude_degE"].tolist() == [135.0]


def test_missing_local_user_data_file_reports_clear_error(tmp_path):
    missing_file = tmp_path / "missing.xlsx"

    with pytest.raises(FileNotFoundError, match="not found"):
        envgeo_utils.load_local_user_data(missing_file)


def test_reference_loader_merges_configured_user_excel_data(tmp_path, monkeypatch):
    local_file = tmp_path / "user_data.xlsx"
    pd.DataFrame(
        {
            "Longitude_degE": [135.0],
            "Latitude_degN": [35.0],
            "Depth_m": [10.0],
            "d18O": [0.5],
        }
    ).to_excel(local_file, index=False)
    monkeypatch.delenv(envgeo_utils.LOCAL_USER_DATA_DISABLE_ENV, raising=False)
    monkeypatch.setenv(envgeo_utils.LOCAL_USER_DATA_PATH_ENV, str(local_file))
    envgeo_utils.load_isotope_data.clear()

    try:
        loaded = envgeo_utils.load_isotope_data(envgeo_utils.data_source_GLOBAL)
        user_rows = loaded.loc[loaded["Dataset"] == envgeo_utils.USER_EXCEL_DATA_LABEL]
        assert len(user_rows) == 1
        assert user_rows.iloc[0]["d18O"] == 0.5
    finally:
        envgeo_utils.load_isotope_data.clear()


# Verifies that coastline loading returns valid longitude and latitude lists of equal length.
# 海岸線データ読み込み結果として、有効な経度・緯度リストが同じ長さで返ることを確認する。
def test_load_coastline_data_returns_same_length_coordinate_lists():
    lon, lat = envgeo_utils.load_coastline_data(envgeo_utils.data_source_GLOBAL)
    assert isinstance(lon, list)
    assert isinstance(lat, list)
    assert len(lon) > 0
    assert len(lon) == len(lat)


# Verifies that the smaller Natural Earth CSV can be selected explicitly.
# 軽量なNatural Earth 110m版をCSVから明示的に読み込めることを確認する。
def test_load_coastline_data_supports_110m_csv():
    lon_50m, lat_50m = envgeo_utils.load_coastline_data(
        envgeo_utils.data_source_GLOBAL,
        resolution="50m",
    )
    lon_110m, lat_110m = envgeo_utils.load_coastline_data(
        envgeo_utils.data_source_GLOBAL,
        resolution="110m",
    )

    assert len(lon_110m) == len(lat_110m)
    assert 0 < len(lon_110m) < len(lon_50m)


def test_plot_bundled_coastline_uses_csv_without_cartopy_downloader(monkeypatch):
    """The Matplotlib helper must plot bundled coordinates without Cartopy I/O."""
    calls = []

    class DummyAxes:
        def plot(self, lon, lat, **kwargs):
            calls.append((lon, lat, kwargs))

    marker_transform = object()
    monkeypatch.setattr(
        envgeo_utils,
        "load_coastline_data",
        lambda ref_data, resolution="50m": ([130.0, 131.0, None], [35.0, 36.0, None]),
    )

    assert envgeo_utils.plot_bundled_coastline(
        DummyAxes(), transform=marker_transform, zorder=7
    )
    assert len(calls) == 1
    lon, lat, kwargs = calls[0]
    assert lon == [130.0, 131.0, None]
    assert lat == [35.0, 36.0, None]
    assert kwargs["transform"] is marker_transform
    assert kwargs["zorder"] == 7


# Verifies that the public dataset choices remain available for app pages.
# アプリページで使う公開データセット選択肢が維持されていることを確認する。
def test_data_sources_include_expected_public_choices():
    assert envgeo_utils.data_source_JAPAN_SEA in envgeo_utils.DATA_SOURCES
    assert envgeo_utils.data_source_AROUND_JAPAN in envgeo_utils.DATA_SOURCES
    assert envgeo_utils.data_source_GLOBAL in envgeo_utils.DATA_SOURCES


# Verifies the standard d-excess calculation: d-excess = dD - 8 * d18O.
# 標準的な d-excess 計算式 dD - 8 * d18O が正しく適用されることを確認する。
def test_add_d_excess_calculates_expected_values():
    df = pd.DataFrame({"d18O": [0.5, -1.0], "dD": [4.0, -5.0]})

    out = envgeo_utils.add_d_excess(df)

    assert out["d-excess"].tolist() == [0.0, 3.0]


# Verifies compact statistics used by the shared sidebar summary.
# 共通サイドバー統計表示で使う概要統計が正しく計算されることを確認する。
def test_summarize_filtered_data_includes_core_statistics_and_quality_count():
    df = pd.DataFrame(
        {
            "d18O": [0.0, 1.0, None],
            "dD": [0.0, 10.0, None],
            "d-excess": [0.0, 2.0, None],
            "Salinity": [34.0, 35.0, None],
            "Temperature_degC": [10.0, 12.0, None],
            "Depth_m": [0.0, 100.0, None],
            envgeo_utils.QUALITY_FLAG_COLUMN: ["", "Depth_m outside valid range", ""],
        }
    )

    summary = envgeo_utils.summarize_filtered_data(df)

    assert summary["row_count"] == 3
    assert summary["quality_flag_count"] == 1
    assert summary["statistics"]["d18O"]["count"] == 2
    assert summary["statistics"]["d18O"]["mean"] == 0.5
    assert summary["statistics"]["d18O"]["min"] == 0.0
    assert summary["statistics"]["d18O"]["max"] == 1.0
    assert summary["statistics"]["d-excess"]["mean"] == 1.0


# Verifies CSV export data for sidebar-filtered reports.
# サイドバー抽出データの概要レポートをCSVとして書き出せることを確認する。
def test_build_filtered_report_csv_includes_filters_and_statistics():
    df = pd.DataFrame(
        {
            "d18O": [0.0, 1.0, None],
            "Salinity": [34.0, 35.0, None],
            envgeo_utils.QUALITY_FLAG_COLUMN: ["", "Salinity outside valid range", ""],
        }
    )

    csv_text = envgeo_utils.build_filtered_report_csv(
        df,
        {"Year": "2020-2026", "Month": "[1, 2]"},
        {"ECS2021": 10},
    ).decode("utf-8-sig")
    header = csv_text.splitlines()[0]

    assert "Overview" in csv_text
    assert "Filter Conditions" in csv_text
    assert "Filtered Data Counts" in csv_text
    assert "Statistics" in csv_text
    assert header == "Section,Item,Value,Count,Mean,Stdev,Min,Max"
    assert "Quality flags" in csv_text
    assert "2020-2026" in csv_text
    assert "ECS2021" in csv_text
    assert "d18O" in csv_text
    assert "Salinity" in csv_text


# Verifies that missing isotope values produce NaN instead of an incorrect number.
# 同位体値が欠損している場合、誤った数値ではなくNaNになることを確認する。
def test_add_d_excess_handles_missing_values():
    df = pd.DataFrame({"d18O": [0.5, None], "dD": [None, -5.0]})

    out = envgeo_utils.add_d_excess(df)

    assert out["d-excess"].isna().all()


# Verifies that loaded app data already include d-excess for reuse by pages.
# 読み込み済みデータに d-excess が含まれ、各ページで使い回せることを確認する。
def test_load_isotope_data_includes_d_excess_column():
    df = envgeo_utils.load_isotope_data(envgeo_utils.data_source_GLOBAL)

    assert "d-excess" in df.columns
