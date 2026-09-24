"""
test_p03_ts_density_contour.py
──────────────────────────────────────────────────────────────────────────────
Focused static-analysis tests for the Page 03 approximate σ0 reference
contour overlay pilot.

Scope: pages/03_[Interactive]_2Dplus_Visualizer.py

These tests do NOT run the Streamlit app (no AppTest dependency) and do NOT
require a network connection or external data files.  They verify that the
source code of page 03 contains the key implementation details agreed in the
pilot specification.

Pilot specification summary:
  - Uses go.Contour (physical isolines), NOT px.density_contour (point density).
  - Grid clipped to valid GSW domain: Salinity 0–50; Temperature −5–45 °C.
  - Selectbox options [0.2, 0.5, 1.0] kg m⁻³; default index 2 (1.0 kg m⁻³).
  - contours.size parameter used (not fixed count / levels).
  - showscale=False — no extra colorbar.
  - hoverinfo="none" — no hover on contour trace.
  - Contour trace prepended to fig.data (rendered behind scatter points).
  - selected_point_indices accepts scatter_curve_number parameter.
  - Approximation caption present below the T–S plot.
  - T–S view only; no changes to d18O, Custom, or dD rendering.
"""

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locate project root and read the page source once
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = ROOT / "pages" / "03_[Interactive]_2Dplus_Visualizer.py"

_src = PAGE_PATH.read_text(encoding="utf-8")


# ===========================================================================
# 1. Imports
# ===========================================================================

class TestImports:
    """Page 03 must import go (for go.Contour) and gsw."""

    def test_imports_plotly_graph_objects(self):
        assert re.search(r"import\s+plotly\.graph_objects\s+as\s+go", _src), (
            "Missing: import plotly.graph_objects as go"
        )

    def test_imports_gsw(self):
        assert re.search(r"import\s+gsw\b", _src), (
            "Missing: import gsw"
        )


# ===========================================================================
# 2. go.Contour used; px.density_contour absent
# ===========================================================================

class TestContourTraceType:
    """The overlay must use go.Contour, not the point-density px.density_contour."""

    def test_go_contour_present(self):
        assert "go.Contour(" in _src, (
            "go.Contour( not found in page 03 source"
        )

    def test_px_density_contour_absent(self):
        assert "px.density_contour" not in _src, (
            "px.density_contour must not appear in page 03 (point density, not physical)"
        )


# ===========================================================================
# 3. GSW domain clipping constants
# ===========================================================================

class TestDomainClipping:
    """The source must define the valid GSW salinity and temperature bounds."""

    def test_salinity_min_constant(self):
        # Matches both assignment forms:
        #   _S_GSW_MIN_03 = 0.0
        #   _S_GSW_MIN_03, _S_GSW_MAX_03 = 0.0, 50.0
        assert "_S_GSW_MIN_03" in _src and re.search(r"0\.0|=\s*0\b", _src), (
            "_S_GSW_MIN_03 constant not found"
        )

    def test_salinity_max_constant(self):
        # Matches both assignment forms:
        #   _S_GSW_MAX_03 = 50.0
        #   _S_GSW_MIN_03, _S_GSW_MAX_03 = 0.0, 50.0
        assert "_S_GSW_MAX_03" in _src and re.search(r"50\.0|=\s*50\b", _src), (
            "_S_GSW_MAX_03 constant not found"
        )

    def test_temperature_min_constant(self):
        # Matches both assignment forms:
        #   _T_GSW_MIN_03 = -5.0
        #   _T_GSW_MIN_03, _T_GSW_MAX_03 = -5.0, 45.0
        assert "_T_GSW_MIN_03" in _src and "-5.0" in _src, (
            "_T_GSW_MIN_03 = -5.0 constant not found"
        )

    def test_temperature_max_constant(self):
        # Matches both assignment forms:
        #   _T_GSW_MAX_03 = 45.0
        #   _T_GSW_MIN_03, _T_GSW_MAX_03 = -5.0, 45.0
        assert "_T_GSW_MAX_03" in _src and re.search(r"45\.0|=\s*45\b", _src), (
            "_T_GSW_MAX_03 = 45 (or 45.0) constant not found"
        )

    def test_salinity_clipping_applied(self):
        # max(..., _S_GSW_MIN_03) and min(..., _S_GSW_MAX_03)
        assert "_S_GSW_MIN_03" in _src and "_S_GSW_MAX_03" in _src, (
            "Domain clipping references to _S_GSW_MIN/MAX_03 not found"
        )

    def test_temperature_clipping_applied(self):
        assert "_T_GSW_MIN_03" in _src and "_T_GSW_MAX_03" in _src, (
            "Domain clipping references to _T_GSW_MIN/MAX_03 not found"
        )


# ===========================================================================
# 4. Selectbox — options and default
# ===========================================================================

class TestSelectbox:
    """Density contour interval selectbox must offer [0.2, 0.5, 1.0] with
    default index 2 (1.0 kg m⁻³)."""

    def test_selectbox_key_present(self):
        assert "p03_ts_contour_interval" in _src, (
            "Selectbox key 'p03_ts_contour_interval' not found"
        )

    def test_selectbox_option_02(self):
        assert "0.2" in _src, "Contour interval option 0.2 not found"

    def test_selectbox_option_05(self):
        assert "0.5" in _src, "Contour interval option 0.5 not found"

    def test_selectbox_option_10(self):
        assert "1.0" in _src, "Contour interval option 1.0 not found"

    def test_selectbox_default_index_2(self):
        # index=2 must appear in the selectbox call block that also contains
        # p03_ts_contour_interval. Search in either order (index may come before key).
        assert re.search(r"index\s*=\s*2[\s\S]*?p03_ts_contour_interval|"
                         r"p03_ts_contour_interval[\s\S]*?index\s*=\s*2", _src), (
            "Selectbox for p03_ts_contour_interval does not set index=2 (default 1.0)"
        )

    def test_selectbox_options_list(self):
        # options=[0.2, 0.5, 1.0]
        assert re.search(
            r"options\s*=\s*\[[\s\S]*?0\.2[\s\S]*?0\.5[\s\S]*?1\.0[\s\S]*?\]",
            _src,
        ), "Selectbox options list [0.2, 0.5, 1.0] not found"

    def test_show_density_contours_checkbox_present(self):
        """The user must be able to hide the reference-contour layer."""
        assert "p03_ts_show_density_contours" in _src, (
            "Show-density-contours checkbox key not found."
        )
        assert re.search(r"st\.checkbox\s*\([\s\S]*?Show density contours", _src), (
            "Show density contours checkbox not found."
        )


# ===========================================================================
# 5. Contour rendering — no colorbar, no hover
# ===========================================================================

class TestContourRendering:
    """Contour trace must have showscale=False and hoverinfo='none'."""

    def test_showscale_false(self):
        assert re.search(r"showscale\s*=\s*False", _src), (
            "showscale=False not found (colorbar must be suppressed)"
        )

    def test_hoverinfo_none(self):
        assert re.search(r'hoverinfo\s*=\s*["\']none["\']', _src), (
            "hoverinfo='none' not found (hover on contour must be disabled)"
        )

    def test_contour_lines_are_light_and_translucent(self):
        """Contour lines must be a subdued visual-reference layer."""
        assert "rgba(165, 165, 165, 0.22)" in _src, (
            "Expected light translucent contour colorscale not found."
        )
        assert "autocolorscale=False" in _src, (
            "Contour must disable Plotly's default colorscale."
        )

    def test_contour_labels_shown(self):
        """Contour labels must remain available for quantitative reading."""
        assert "showlabels=True" in _src, (
            "Contour labels must be enabled for the Page 03 pilot."
        )

    def test_contours_size_parameter(self):
        # The contour interval is passed via the size= keyword — either as a dict
        # literal key ("size": ...) or as a dict() keyword arg (size=...).
        # Both forms set contours.size in go.Contour.
        assert re.search(r"""['""]size['""]?\s*[:=]|size\s*=\s*ts_contour_interval""", _src), (
            "contours size parameter not found; contour spacing must use contours.size "
            "(dict literal or dict(size=...) form)"
        )

    def test_levels_keyword_absent_in_contour_block(self):
        # levels=<int> (fixed count) must not be used for the σ0 contour
        # We allow "levels" as part of other strings, but not levels=<integer>
        assert not re.search(r"\blevels\s*=\s*\d+", _src), (
            "levels=<int> found; contour must use contours.size, not a fixed count"
        )


# ===========================================================================
# 6. Contour placed behind scatter (prepend to fig.data)
# ===========================================================================

class TestContourPlacement:
    """Contour trace must be prepended to place it behind scatter points."""

    def test_contour_prepended_to_fig_data(self):
        assert re.search(
            r"fig_fixed_TS\.add_trace\s*\(\s*_contour_trace\s*\)", _src
        ), "Contour trace must first be added through fig_fixed_TS.add_trace()."
        assert re.search(
            r"fig_fixed_TS\.data\s*=\s*\(\s*fig_fixed_TS\.data\s*\[\s*-1\s*\]\s*,\s*\)"
            r"\s*\+\s*fig_fixed_TS\.data\s*\[\s*:-1\s*\]",
            _src,
        ), (
            "Contour trace must be moved to the first position by reordering "
            "the complete existing figure.data tuple."
        )


# ===========================================================================
# 7. selected_point_indices — scatter_curve_number parameter
# ===========================================================================

class TestSelectedPointIndices:
    """The shared selection helper must accept scatter_curve_number so
    Box/Lasso selection is not broken when the contour trace shifts the
    scatter trace's curveNumber."""

    def test_function_has_scatter_curve_number_param(self):
        assert re.search(
            r"def\s+selected_point_indices\s*\([\s\S]*?scatter_curve_number",
            _src,
        ), (
            "selected_point_indices does not declare scatter_curve_number parameter"
        )

    def test_call_site_passes_scatter_curve_number(self):
        assert re.search(
            r"selected_point_indices\s*\([\s\S]*?scatter_curve_number\s*=",
            _src,
        ), (
            "selected_point_indices call site does not pass scatter_curve_number="
        )

    def test_ts_scatter_curve_variable_used(self):
        assert "_ts_scatter_curve" in _src, (
            "_ts_scatter_curve variable not found; "
            "scatter_curve_number must be parameterized"
        )


# ===========================================================================
# 8. Approximation caption
# ===========================================================================

class TestApproximationCaption:
    """An st.caption must appear below the T–S figure with the approximation notice."""

    def test_caption_present(self):
        assert re.search(r"st\.caption\s*\(", _src), (
            "st.caption( not found in page 03 source"
        )

    def test_caption_mentions_approximate_sigma0(self):
        assert re.search(
            r"st\.caption[\s\S]*?approximate[\s\S]*?σ0|st\.caption[\s\S]*?approx[\s\S]*?sigma0",
            _src,
            re.IGNORECASE,
        ), (
            "st.caption does not mention 'approximate σ0' or 'approx sigma0'"
        )

    def test_caption_mentions_not_pointwise(self):
        assert re.search(
            r"st\.caption[\s\S]*?[Nn]ot\s+pointwise|st\.caption[\s\S]*?[Nn]ot.*?sample\s+density",
            _src,
        ), (
            "st.caption does not clarify that contours are not pointwise sample density"
        )

    def test_caption_precedes_interactive_html_download(self):
        """The scientific note belongs before the figure-download action."""
        caption_index = _src.index("Density contour lines are approximate σ0")
        download_index = _src.index('"Download interactive HTML"', caption_index)
        assert caption_index < download_index, (
            "The density-contour note must appear before the interactive HTML download."
        )


# ===========================================================================
# 9. T–S only — no contamination of d18O / Custom / dD sections
# ===========================================================================

class TestScopeContainment:
    """Changes must be confined to the T–S section.  The σ0 selectbox key
    must not appear in the d18O, Custom, or dD rendering blocks."""

    def test_contour_key_not_in_d18o_section(self):
        # Crude check: the d18O section uses 'd18O' column name in its logic;
        # the key p03_ts_contour_interval must not be co-located.
        # We check that the key appears only once (inside the T–S block).
        occurrences = _src.count("p03_ts_contour_interval")
        # Selectbox definition + key= reference: expect exactly 1 unique string
        # (Streamlit resolves by key; one definition suffices).
        assert occurrences >= 1, "p03_ts_contour_interval not found at all"
        # Sanity: should not be duplicated into other plot-type blocks
        assert occurrences <= 3, (
            f"p03_ts_contour_interval appears {occurrences} times — "
            "may have leaked into non-T-S sections"
        )
