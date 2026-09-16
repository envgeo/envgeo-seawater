import os
import sys
from pathlib import Path

import pandas as pd
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
    assert envgeo_utils.APP_VERSION == "1.3.0"
    assert envgeo_utils.APP_VERSION_DATE == "2026-09-11"
    assert envgeo_utils.APP_VERSION in envgeo_utils.APP_VERSION_LABEL
    assert envgeo_utils.APP_VERSION_DATE in envgeo_utils.APP_VERSION_LABEL
    assert envgeo_utils.version == envgeo_utils.APP_VERSION


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


# Verifies that coastline loading returns valid longitude and latitude lists of equal length.
# 海岸線データ読み込み結果として、有効な経度・緯度リストが同じ長さで返ることを確認する。
def test_load_coastline_data_returns_same_length_coordinate_lists():
    lon, lat = envgeo_utils.load_coastline_data(envgeo_utils.data_source_GLOBAL)
    assert isinstance(lon, list)
    assert isinstance(lat, list)
    assert len(lon) > 0
    assert len(lon) == len(lat)


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
