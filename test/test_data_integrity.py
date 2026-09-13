import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import envgeo_utils


REQUIRED_COLUMNS = {
    "Dataset",
    "Latitude_degN",
    "Longitude_degE",
    "Depth_m",
    "Temperature_degC",
    "Salinity",
    "d18O",
    "dD",
    "Year",
    "Month",
    "reference",
}

NUMERIC_COLUMNS = [
    "Latitude_degN",
    "Longitude_degE",
    "Depth_m",
    "Temperature_degC",
    "Salinity",
    "d18O",
    "dD",
]


# The public dataset choices should all load into non-empty dataframes.
# 公開データセットの選択肢が、すべて空でないDataFrameとして読み込めることを確認する。
def test_all_public_data_sources_load_non_empty_dataframes():
    for data_source in envgeo_utils.DATA_SOURCES:
        df = envgeo_utils.load_isotope_data(data_source)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, f"{data_source} loaded an empty dataframe"


# Important columns are the contract between the datasets and every visualizer.
# 主要列はデータセットと各可視化ページをつなぐ約束なので、欠けていないことを確認する。
def test_loaded_datasets_have_required_columns():
    for data_source in envgeo_utils.DATA_SOURCES:
        df = envgeo_utils.load_isotope_data(data_source)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        assert not missing, f"{data_source} is missing columns: {sorted(missing)}"


# Placeholder strings such as "**" should be cleaned before plotting or analysis.
# "**" のような欠損プレースホルダーが、解析用数値列に残っていないことを確認する。
def test_numeric_columns_do_not_keep_placeholder_strings():
    for data_source in envgeo_utils.DATA_SOURCES:
        df = envgeo_utils.load_isotope_data(data_source)
        for col in NUMERIC_COLUMNS:
            assert "**" not in set(df[col].dropna().astype(str)), f"{col} still has '**'"


# Geographic and hydrographic values should stay within broad sanity-check ranges.
# 緯度・経度・水深・塩分が、かなり広めの健全性チェック範囲に収まることを確認する。
def test_loaded_data_values_are_within_broad_physical_ranges():
    for data_source in envgeo_utils.DATA_SOURCES:
        df = envgeo_utils.load_isotope_data(data_source)

        lat = df["Latitude_degN"].dropna()
        lon = df["Longitude_degE"].dropna()
        depth = df["Depth_m"].dropna()
        salinity = df["Salinity"].dropna()

        assert lat.between(-90, 90).all(), f"{data_source} has out-of-range latitude"
        assert lon.between(-360, 360).all(), f"{data_source} has out-of-range longitude"
        temp = df["Temperature_degC"].dropna()

        assert (depth >= 0).all(), f"{data_source} has negative depth"
        assert salinity.between(0, 50).all(), f"{data_source} has out-of-range salinity"
        assert temp.between(-5, 45).all(), f"{data_source} has out-of-range temperature"


# Known invalid values should be converted to NaN, but their original values should remain visible.
# 既知の不適切値はNaNへ変換しつつ、元の値は品質情報として確認できることをテストする。
def test_known_invalid_values_are_flagged_before_display():
    df = envgeo_utils.load_isotope_data(envgeo_utils.data_source_GLOBAL)

    assert envgeo_utils.QUALITY_FLAG_COLUMN in df.columns
    assert envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN in df.columns

    quality_rows = df[df[envgeo_utils.QUALITY_FLAG_COLUMN].astype(str) != ""]
    assert len(quality_rows) == 8

    original_values = " ".join(quality_rows[envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN].astype(str))
    for value in ["-999.0", "-11.82", "-13.0", "99.9"]:
        assert value in original_values


# Invalid source names should fail safely by returning an empty dataframe.
# 不正なデータソース名では、例外ではなく空のDataFrameで安全に戻ることを確認する。
def test_invalid_data_source_returns_empty_dataframe():
    df = envgeo_utils.load_isotope_data("not a real data source")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
