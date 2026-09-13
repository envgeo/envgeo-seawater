import py_compile
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _markdown_image_paths(markdown_text: str):
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown_text)


# Every stable Streamlit page should at least be valid Python syntax.
# 安定版ページがPython構文として壊れていないことを確認する。
def test_stable_streamlit_pages_compile():
    python_files = [ROOT / "home.py", ROOT / "envgeo_utils.py"]
    python_files.extend(sorted((ROOT / "pages").glob("*.py")))
    python_files.extend(sorted((ROOT / "tools").glob("*.py")))

    for path in python_files:
        py_compile.compile(str(path), doraise=True)


# README images should point to files that exist in the repository.
# README内の画像リンクが、実際に存在するファイルを指していることを確認する。
def test_readme_image_links_point_to_existing_files():
    for readme_name in ["README.md", "README_Japanese.md"]:
        readme_path = ROOT / readme_name
        text = readme_path.read_text(encoding="utf-8")

        for image_path in _markdown_image_paths(text):
            if image_path.startswith(("http://", "https://", "data:")):
                continue

            resolved = ROOT / image_path
            assert resolved.exists(), f"{readme_name} image does not exist: {image_path}"


# Key project documents should be present before a public release.
# 公開リリースに必要な基本文書が存在することを確認する。
def test_key_project_documents_exist():
    for filename in [
        "README.md",
        "README_Japanese.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "requirements.txt",
        "runtime.txt",
        "data_text/update_log.md",
        "data_text/update_log_Japanese.md",
    ]:
        assert (ROOT / filename).exists(), f"{filename} is missing"


# The 4D visualizer has its own selected-data table, so it must include quality columns explicitly.
# 4D Visualizerは独自の選択データ表を持つため、品質情報列を明示的に含める必要がある。
def test_4d_visualizer_selected_table_includes_quality_columns():
    page_text = (ROOT / "pages" / "04_4D_Visualizer.py").read_text(encoding="utf-8")

    assert "QUALITY_FLAG_COLUMN" in page_text
    assert "QUALITY_ORIGINAL_VALUE_COLUMN" in page_text


# The integrated beta page should support uploaded user data without replacing stable pages.
# 統合betaページが、安定版ページを置き換えずにユーザーデータのアップロード比較に対応していることを確認する。
def test_integrated_beta_page_includes_upload_overlay_workflow():
    page_text = (ROOT / "pages" / "90_Integrated_Visualizer_beta.py").read_text(
        encoding="utf-8"
    )

    assert "Uploaded data overlay" in page_text
    assert "st.file_uploader" in page_text
    assert "render_uploaded_plots" in page_text
    assert "render_uploaded_marker_style_controls" in page_text
    assert "Marker size multiplier" in page_text
    assert "Marker color" in page_text
    assert "Marker color mode" in page_text
    assert "Use current colorbar when possible" in page_text
    assert "Marker shape" in page_text
    assert "Marker outline width" in page_text
    assert "Controls the black outline around uploaded data markers" in page_text
    assert "Map marker offset" in page_text
    assert "Moves only the displayed uploaded markers" in page_text
    assert "Uploaded data outline" in page_text
    assert 'below=""' in page_text
    assert "Fixed marker color" in page_text
    assert "_merge_reference_and_uploaded" in page_text
    assert "load_isotope_data_with_upload" in page_text
    assert "Correlation Overview" in page_text
    assert "Vertical Section" in page_text
    assert "Download CSV template" in page_text
    assert "go.Scattergl" in page_text
    assert "_can_use_current_colorbar" in page_text
    assert "_apply_shared_coloraxis_range" in page_text
    assert "_numeric_first_color_options" in page_text
    assert "_plotly_color_scale_args" in page_text
    assert "_select_plotly_colormap" in page_text
    assert "get_plotly_colormap_options" in page_text
    assert "recommended_plotly_colormap_label" in page_text
    assert "standardize_uploaded_column_names" in page_text
    assert "Standardized column names" in page_text
    assert "Uploaded Data Quality Check" in page_text
    assert "Uploaded quality flags" in page_text
    assert "Download uploaded quality-flagged rows" in page_text
    assert "upload_tab_download_uploaded_quality_flags" in page_text
    assert "quality_tab_download_uploaded_quality_flags" in page_text
    assert "quality_tab_download_reference_quality_flags" in page_text


# Dense Plotly Express scatter plots may use WebGL traces, so uploaded overlays should use
# Scattergl too. This keeps user data visible above the reference dataset.
# 点数の多いPlotly Expressの散布図ではWebGL描画になるため、アップロード点もScatterglで重ねる。
def test_integrated_beta_upload_overlays_use_scattergl_for_2d_plotly_views():
    page_text = (ROOT / "pages" / "90_Integrated_Visualizer_beta.py").read_text(
        encoding="utf-8"
    )
    ts_block = page_text.split("def render_ts_diagram", 1)[1].split(
        "def render_salinity_d18o", 1
    )[0]
    salinity_d18o_block = page_text.split("def render_salinity_d18o", 1)[1].split(
        "def render_uploaded_plots", 1
    )[0]

    assert "go.Scattergl" in ts_block
    assert "Uploaded data outline" in ts_block
    assert "Uploaded data" in ts_block
    assert "go.Scattergl" in salinity_d18o_block
    assert "Uploaded data outline" in salinity_d18o_block
    assert "Uploaded data" in salinity_d18o_block


# The integrated map should use shared ocean-region presets.
# 統合ページの地図が、共通定義された海域プリセットを使うことを確認する。
def test_integrated_beta_map_uses_shared_region_presets():
    page_text = (ROOT / "pages" / "90_Integrated_Visualizer_beta.py").read_text(
        encoding="utf-8"
    )

    map_block = page_text.split("def render_map", 1)[0]

    assert "@st.fragment" in map_block
    assert "MAP_REGION_PRESETS" in page_text
    assert "Region preset" in page_text
    assert "map_region_view" in page_text
    assert "left_controls, right_controls = st.columns" in page_text
    assert "color_controls = st.columns" in page_text


# Filtered-data summaries should be exportable from shared and integrated views.
# 抽出データ概要は、個別ページ共通処理と統合ページの両方からCSV出力できる。
def test_filtered_data_summary_csv_export_is_available():
    utils_text = (ROOT / "envgeo_utils.py").read_text(encoding="utf-8")
    integrated_text = (ROOT / "pages" / "90_Integrated_Visualizer_beta.py").read_text(
        encoding="utf-8"
    )

    assert "build_filtered_report_csv" in utils_text
    assert "render_filtered_report_download" in utils_text
    assert "Download filtered-data summary CSV" in utils_text
    assert "render_filtered_report_download" in integrated_text


# The sidebar-filtered data table should explain quality-flag criteria near the table.
# Sidebar-filtered datasetの近くで品質フラグ基準を確認できることを守る。
def test_sidebar_filtered_table_mentions_quality_flag_criteria():
    utils_text = (ROOT / "envgeo_utils.py").read_text(encoding="utf-8")

    table_block = utils_text.split("def display_isotope_table", 1)[1].split(
        "##############################################################################\n# --- 9.",
        1,
    )[0]
    assert "render_quality_flag_criteria_note()" in table_block
    assert "quality_flag_criteria_text" in utils_text


# The integrated Quality Flags tab should show criteria before any tables.
# 統合ページのQuality Flagsタブでは、表を見る前に品質フラグ基準が見えるようにする。
def test_integrated_quality_tab_mentions_quality_flag_criteria():
    page_text = (ROOT / "pages" / "90_Integrated_Visualizer_beta.py").read_text(
        encoding="utf-8"
    )
    quality_block = page_text.split("def render_quality_table", 1)[1].split(
        "def main",
        1,
    )[0]

    assert "st.subheader(\"Quality Flags\")" in quality_block
    assert "render_quality_flag_criteria_note()" in quality_block


# The shared-filter beta tabs should use compact icon labels like the earthquake Advanced page.
# Shared-filter betaのタブはearthquake Advancedに近い、見分けやすい短いラベルにする。
def test_integrated_shared_filter_tabs_use_readable_icon_labels():
    page_text = (ROOT / "pages" / "90_Integrated_Visualizer_beta.py").read_text(
        encoding="utf-8"
    )

    assert "def render_tab_style" in page_text
    assert 'div[data-baseweb="tab-list"]' in page_text
    assert 'button[role="tab"][aria-selected="true"]' in page_text
    assert "Select a tab to switch visualization views." in page_text

    for label in [
        "📌 Summary",
        "🗺️ Map",
        "🌡️ T-S",
        "💧 Salinity-d18O",
        "📤 Upload",
        "🗂️ Data(CSV)",
        "✅ Quality Flags",
    ]:
        assert label in page_text


# Standard maps should avoid CARTO tiles because CARTO basemaps now require API keys.
# CARTO背景はAPIキー必須化の影響を受けるため、標準地図はOpenStreetMapにする。
def test_standard_map_style_uses_openstreetmap_not_carto():
    utils_text = (ROOT / "envgeo_utils.py").read_text(encoding="utf-8")

    assert 'mapbox_style="open-street-map"' in utils_text

    active_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "pages").glob("*.py"))
    )
    assert "carto-positron" not in active_text
    assert "CartoDB positron" not in active_text


# The parameter mapping page should use shared colormap helpers for both Plotly and Matplotlib maps.
# parameter mappingページでは、Plotly地図とMatplotlib地図の両方で共通カラーマップ関数を使う。
def test_parameter_mapping_uses_shared_colormap_helpers():
    page_text = (ROOT / "pages" / "32_d18O_mapping.py").read_text(encoding="utf-8")

    assert "Seawater Parameter Mapping" in page_text
    assert '"Map parameter"' in page_text
    assert '"d-excess"' in page_text
    assert "get_plotly_colormap_options" in page_text
    assert "get_plotly_colormap(" in page_text
    assert "get_matplotlib_colormap(" in page_text
    assert '.index("Jet")' in page_text
    assert "cmocean palettes are designed for oceanographic data" in page_text


# The standalone uploader has its own upload-first workflow and should not be loaded as a full-page workflow.
# 独立アップロードページは共通フィルタと流れが異なるため、統合betaの既存ページ選択肢には含めない。
def test_integrated_beta_excludes_standalone_uploader_from_full_page_workflows():
    page_text = (ROOT / "pages" / "90_Integrated_Visualizer_beta.py").read_text(
        encoding="utf-8"
    )
    workflow_block = page_text.split("FULL_PAGE_WORKFLOWS = {", 1)[1].split("}", 1)[0]

    assert "05_3D4D_Visualizer_Uploader.py" not in workflow_block
    assert "3D/4D Uploader" not in workflow_block


# Uploaded user files should stay in memory during the Streamlit session.
# ユーザーのアップロードファイルを、ローカル/サーバーへ保存しない方針を確認する。
def test_integrated_beta_uploads_are_memory_only():
    page_text = (ROOT / "pages" / "90_Integrated_Visualizer_beta.py").read_text(
        encoding="utf-8"
    )

    forbidden_write_patterns = [
        ".to_excel(",
        ".to_parquet(",
        ".to_pickle(",
        ".to_feather(",
        "open(uploaded_file",
        "uploaded_file.write",
    ]

    for pattern in forbidden_write_patterns:
        assert pattern not in page_text

    assert "are not saved to local or server storage" in page_text
