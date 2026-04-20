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


# Verifies that coastline loading returns valid longitude and latitude lists of equal length.
# 海岸線データ読み込み結果として、有効な経度・緯度リストが同じ長さで返ることを確認する。
def test_load_coastline_data_returns_same_length_coordinate_lists():
    lon, lat = envgeo_utils.load_coastline_data(envgeo_utils.data_source_GLOBAL)
    assert isinstance(lon, list)
    assert isinstance(lat, list)
    assert len(lon) > 0
    assert len(lon) == len(lat)
