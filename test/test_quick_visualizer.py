"""
pytest tests for pages/05_User_Data_Check_Quick_Visualizer.py
Verifies the _EnvGeoDataOrigin column logic and hover header behavior.
No real network, no real data files, no Streamlit server.

Test isolation: stubs inserted into sys.modules are fully removed after
this module's tests finish, so subsequent test files (e.g. test_offline_map.py)
run against the real modules.
"""
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------

def _make_streamlit_stub():
    st = types.ModuleType("streamlit")
    for attr in (
        "set_page_config", "title", "caption", "subheader", "write", "info",
        "warning", "error", "radio", "selectbox", "number_input", "checkbox",
        "sidebar", "tabs", "expander", "plotly_chart", "download_button",
        "dataframe", "segmented_control", "container",
    ):
        setattr(st, attr, MagicMock(return_value=None))
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    st.sidebar = MagicMock()
    st.sidebar.container = MagicMock(return_value=cm)
    st.tabs = MagicMock(return_value=[cm] * 6)
    st.expander = MagicMock(return_value=cm)
    st.session_state = {}
    return st


def _make_envgeo_utils_stub():
    eu = types.ModuleType("envgeo_utils")
    eu.QUALITY_FLAG_COLUMN = "_QualityFlag"
    eu.QUALITY_ORIGINAL_VALUE_COLUMN = "_QualityOriginalValue"
    eu.USER_EXCEL_DATA_LABEL = "User Excel data"
    eu.UPLOADED_DATA_LABEL = "Uploaded"
    eu.data_source_JAPAN_SEA = "Japan Sea"
    eu.data_source_AROUND_JAPAN = "Around Japan"
    eu.data_source_GLOBAL = "Global"
    eu.MAP_MODE_OPTIONS = ["Standard"]
    eu.MAP_MODE_DEFAULT_INDEX = 0
    eu.MAP_REGION_PRESETS = {}
    eu.AUTO_APPLY_NOTE = ""
    eu.OFFLINE_FALLBACK_WARNING = "fallback"
    eu.load_isotope_data = MagicMock(return_value=pd.DataFrame())
    eu.combine_reference_and_uploaded_for_filtering = MagicMock(
        side_effect=lambda ref, upl: pd.concat(
            [df for df in [ref, upl] if not df.empty], ignore_index=True
        ) if not (ref.empty and upl.empty) else pd.DataFrame()
    )
    eu.split_uploaded_rows = MagicMock(return_value=(pd.DataFrame(), pd.DataFrame()))
    eu.sidebar_filter_and_display = MagicMock(side_effect=lambda df, *a, **kw: (df,))
    eu.get_quality_rows = MagicMock(return_value=pd.DataFrame())
    eu.render_quality_flag_criteria_note = MagicMock()
    eu.arrow_display_dataframe = MagicMock(side_effect=lambda df: df)
    eu.stretch_width_kwargs = MagicMock(return_value={})
    eu.get_plotly_colormap_options = MagicMock(return_value={"Viridis": "viridis"})
    eu.recommended_plotly_colormap_label = MagicMock(return_value="Viridis")
    eu.build_upload_template_csv = MagicMock(return_value=b"")
    eu.build_figure_filename = MagicMock(return_value="fig.html")
    eu.render_earthquake_tab_style = MagicMock()
    eu.resolve_map_mode = MagicMock(return_value=("Standard", False))
    eu.add_coastline_overlay = MagicMock()
    eu.add_graticule_overlay = MagicMock()
    return eu


# Names of every stub module we will inject
_STUB_NAMES = [
    "streamlit",
    "envgeo_utils",
    "envgeo_user_data",
    "envgeo_map_data",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
    "scipy",
    "scipy.stats",
    "PIL",
    "PIL.Image",
]


@pytest.fixture(scope="module")
def p05():
    """Load p05 with stubs; restore sys.modules fully after all tests in this module."""
    # Save the state of sys.modules for every stub name *before* installing stubs
    saved = {name: sys.modules.get(name, None) for name in _STUB_NAMES}
    # Also save the p05 module name itself in case it was already imported
    saved["p05_quick_viz"] = sys.modules.get("p05_quick_viz", None)

    # Install named stubs
    sys.modules["streamlit"] = _make_streamlit_stub()
    sys.modules["envgeo_utils"] = _make_envgeo_utils_stub()
    for name in _STUB_NAMES:
        if name not in ("streamlit", "envgeo_utils"):
            sys.modules[name] = MagicMock()

    # Locate and load the actual page file
    page_path = (
        Path(__file__).parent.parent / "pages" / "05_User_Data_Check_Quick_Visualizer.py"
    )
    if not page_path.exists():
        # Restore before skipping so other tests aren't affected
        _restore(saved)
        pytest.skip(f"Page file not found: {page_path}")

    spec = importlib.util.spec_from_file_location("p05_quick_viz", page_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p05_quick_viz"] = mod
    spec.loader.exec_module(mod)

    yield mod  # tests run here

    # ---- Teardown: restore sys.modules to pre-stub state ----
    _restore(saved)


def _restore(saved: dict):
    """Remove or restore sys.modules entries to their pre-stub state."""
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ref_df():
    return pd.DataFrame({
        "Salinity": [34.0, 34.5],
        "d18O": [-0.5, -0.3],
        "Temperature_degC": [5.0, 6.0],
    })


@pytest.fixture()
def excel_ref_df():
    return pd.DataFrame({
        "Salinity": [33.0],
        "Dataset": ["User Excel data"],
    })


@pytest.fixture()
def uploaded_df():
    return pd.DataFrame({
        "Salinity": [30.0],
        "d18O": [0.1],
        "Dataset": ["MyCustomDataset"],
    })


# ---------------------------------------------------------------------------
# Test 1: reference rows → hover header "Reference data"
# ---------------------------------------------------------------------------

def test_reference_row_hover_header(p05, ref_df):
    df = ref_df.copy()
    df[p05._ORIGIN_COL] = "Reference data"
    texts = p05.rich_hover_text(df)
    assert texts[0].startswith("<b>Reference data</b>"), texts[0]


# ---------------------------------------------------------------------------
# Test 2: uploaded rows → hover header "User-uploaded data"
# ---------------------------------------------------------------------------

def test_uploaded_row_hover_header(p05, uploaded_df):
    df = uploaded_df.copy()
    df[p05._ORIGIN_COL] = "User-uploaded data"
    texts = p05.rich_hover_text(df)
    assert texts[0].startswith("<b>User-uploaded data</b>"), texts[0]


# ---------------------------------------------------------------------------
# Test 3: custom Dataset value does NOT change origin label
# ---------------------------------------------------------------------------

def test_custom_dataset_stays_uploaded(p05):
    df = pd.DataFrame({
        "Salinity": [28.0],
        "Dataset": ["ArbitraryName"],
        p05._ORIGIN_COL: ["User-uploaded data"],
    })
    texts = p05.rich_hover_text(df)
    assert texts[0].startswith("<b>User-uploaded data</b>"), texts[0]


# ---------------------------------------------------------------------------
# Test 4: USER_EXCEL_DATA_LABEL rows → hover header "User Excel data"
# ---------------------------------------------------------------------------

def test_user_excel_row_hover_header(p05, excel_ref_df):
    _USER_EXCEL_LABEL = "User Excel data"
    df = excel_ref_df.copy()
    df[p05._ORIGIN_COL] = df["Dataset"].apply(
        lambda d: "User Excel data" if d == _USER_EXCEL_LABEL else "Reference data"
    )
    texts = p05.rich_hover_text(df)
    assert texts[0].startswith("<b>User Excel data</b>"), texts[0]


# ---------------------------------------------------------------------------
# Test 5: _ORIGIN_COL absent from _user_facing_df() output
# ---------------------------------------------------------------------------

def test_origin_col_absent_from_user_facing_df(p05):
    df = pd.DataFrame({
        "Salinity": [34.0, 33.0],
        p05._ORIGIN_COL: ["Reference data", "User-uploaded data"],
        "d18O": [-0.5, 0.1],
    })
    out = p05._user_facing_df(df)
    assert p05._ORIGIN_COL not in out.columns, (
        f"_ORIGIN_COL leaked into user-facing df: {list(out.columns)}"
    )
    assert "Salinity" in out.columns
    assert "d18O" in out.columns


# ---------------------------------------------------------------------------
# Test 6: _ORIGIN_COL absent from numeric_columns() output
# ---------------------------------------------------------------------------

def test_origin_col_absent_from_numeric_columns(p05):
    df = pd.DataFrame({
        "Salinity": [34.0],
        "d18O": [-0.5],
        p05._ORIGIN_COL: ["Reference data"],
    })
    cols = p05.numeric_columns(df)
    assert p05._ORIGIN_COL not in cols, (
        f"_ORIGIN_COL appeared in numeric_columns: {cols}"
    )
    assert "Salinity" in cols


# ---------------------------------------------------------------------------
# Test 7: _ORIGIN_COL not printed in hover body (no double-display)
# ---------------------------------------------------------------------------

def test_origin_col_not_in_hover_body(p05):
    df = pd.DataFrame({
        "Salinity": [34.0],
        p05._ORIGIN_COL: ["Reference data"],
    })
    texts = p05.rich_hover_text(df)
    assert p05._ORIGIN_COL not in texts[0], (
        f"_ORIGIN_COL column name visible in hover text: {texts[0]}"
    )
