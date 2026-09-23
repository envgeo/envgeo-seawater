"""
pytest tests for the self-contained HTML download feature (sprint #43–#47).

Verifies:
1. envgeo_utils.figure_to_self_contained_html() embeds Plotly.js inline
   (include_plotlyjs=True) and does not reference CDN URLs.
2. The saved HTML contains the expected config flags.
3. pages/05 source uses figure_to_self_contained_html (not CDN).
4. pages/03 and pages/04 source contain st.download_button calls with
   figure_to_self_contained_html.

No real Streamlit server, no network calls.
"""
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
UTILS_PY   = REPO_ROOT / "envgeo_utils.py"
PAGE_03    = REPO_ROOT / "pages" / "03_[Interactive]_2Dplus_Visualizer.py"
PAGE_04    = REPO_ROOT / "pages" / "04_[Interactive]_3D_4D_Visualizer.py"
PAGE_05    = REPO_ROOT / "pages" / "05_User_Data_Check_Quick_Visualizer.py"


# ---------------------------------------------------------------------------
# Helper: read source once
# ---------------------------------------------------------------------------
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ===========================================================================
# 1. envgeo_utils — static source checks
# ===========================================================================

class TestFigureToSelfContainedHtmlSource:
    """Static checks that the helper function exists and is correct in source."""

    def test_function_defined(self):
        src = _read(UTILS_PY)
        assert "def figure_to_self_contained_html(" in src, \
            "figure_to_self_contained_html not found in envgeo_utils.py"

    def test_uses_include_plotlyjs_true(self):
        src = _read(UTILS_PY)
        # Must set include_plotlyjs=True (not "cdn" or False)
        assert 'include_plotlyjs=True' in src, \
            "figure_to_self_contained_html must use include_plotlyjs=True"

    def test_no_cdn_reference_in_helper(self):
        src = _read(UTILS_PY)
        # The helper itself must not reference CDN strings
        # (cdn.plot.ly or cdn.jsdelivr.net/npm/plotly)
        cdn_pattern = re.compile(
            r'include_plotlyjs\s*=\s*["\']cdn["\']', re.IGNORECASE
        )
        assert not cdn_pattern.search(src), \
            "envgeo_utils must not use include_plotlyjs='cdn'"

    def test_config_scroll_zoom(self):
        src = _read(UTILS_PY)
        assert '"scrollZoom": True' in src or "'scrollZoom': True" in src, \
            "_PLOTLY_HTML_CONFIG must include scrollZoom: True"

    def test_config_display_mode_bar(self):
        src = _read(UTILS_PY)
        assert '"displayModeBar": True' in src or "'displayModeBar': True" in src, \
            "_PLOTLY_HTML_CONFIG must include displayModeBar: True"

    def test_config_responsive(self):
        src = _read(UTILS_PY)
        assert '"responsive": True' in src or "'responsive': True" in src, \
            "_PLOTLY_HTML_CONFIG must include responsive: True"

    def test_plotly_html_config_constant_defined(self):
        src = _read(UTILS_PY)
        assert "_PLOTLY_HTML_CONFIG" in src, \
            "_PLOTLY_HTML_CONFIG constant not found in envgeo_utils.py"


# ===========================================================================
# 2. Runtime check: actual HTML output
# ===========================================================================

class TestFigureToSelfContainedHtmlRuntime:
    """Import envgeo_utils and run figure_to_self_contained_html on a trivial fig."""

    @pytest.fixture(scope="class")
    def html_bytes(self):
        import sys, types, importlib

        # Minimal stubs so envgeo_utils can be imported without a full env
        stub_names = [
            "streamlit", "streamlit.components", "streamlit.components.v1",
            "plotly", "plotly.graph_objects", "plotly.express",
            "cartopy", "cartopy.crs", "cartopy.feature", "cartopy.io",
            "cartopy.io.shapereader",
            "numpy", "pandas", "scipy", "scipy.interpolate", "scipy.stats",
            "netCDF4", "xarray", "matplotlib", "matplotlib.pyplot",
            "matplotlib.cm", "matplotlib.colors",
            "streamlit_plotly_events",
        ]
        # Only stub names not already present
        inserted = []
        for name in stub_names:
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
                inserted.append(name)

        # Make sure plotly.graph_objects.Figure exists (we need the real one)
        try:
            import plotly.graph_objects as go
            import importlib as _il
            eu = _il.import_module("envgeo_utils")  # may already be loaded
        except Exception:
            # Fall back: remove stubs and load for real
            for name in inserted:
                del sys.modules[name]
            import importlib.util
            spec = importlib.util.spec_from_file_location("envgeo_utils", str(UTILS_PY))
            eu = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eu)

        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        result = eu.figure_to_self_contained_html(fig)

        # Cleanup stubs
        for name in inserted:
            sys.modules.pop(name, None)

        return result

    def test_returns_bytes(self, html_bytes):
        assert isinstance(html_bytes, bytes), \
            "figure_to_self_contained_html must return bytes"

    def test_html_is_substantial(self, html_bytes):
        # Plotly.js embedded inline is ~4–5 MB; even a tiny figure should be > 1 MB
        assert len(html_bytes) > 1_000_000, \
            f"HTML output is only {len(html_bytes)} bytes — Plotly.js may not be embedded"

    def test_no_cdn_script_src_in_output(self, html_bytes):
        """Verify no <script src=...> CDN tag references plotly.

        Note: 'cdn.plot.ly' appears as a string literal inside the bundled
        Plotly.js source (a default config URL); this is acceptable and does
        not cause a network request.  We check for actual <script src=> tags.
        """
        import re
        html_str = html_bytes.decode("utf-8", errors="replace")
        # Match <script src="https://cdn.plot.ly/..."> or similar CDN loads
        cdn_script_pattern = re.compile(
            r'<script[^>]+src=["\']https?://(cdn\.plot\.ly|cdn\.jsdelivr\.net/npm/plotly)',
            re.IGNORECASE,
        )
        assert not cdn_script_pattern.search(html_str), \
            "CDN <script src=...> tag for Plotly found in self-contained HTML"

    def test_plotlyjs_tag_present(self, html_bytes):
        html_str = html_bytes.decode("utf-8", errors="replace")
        # Inline Plotly.js starts with a <script> tag containing the library
        assert "<script" in html_str, "No <script> tag found in output HTML"

    def test_config_scroll_zoom_in_html(self, html_bytes):
        """scrollZoom must be set to true in the serialised Plotly config block."""
        html_str = html_bytes.decode("utf-8", errors="replace")
        assert '"scrollZoom": true' in html_str or '"scrollZoom":true' in html_str, \
            "scrollZoom not found in HTML config (must be true)"

    def test_config_display_mode_bar_in_html(self, html_bytes):
        """displayModeBar must be set to true in the serialised Plotly config block."""
        html_str = html_bytes.decode("utf-8", errors="replace")
        assert '"displayModeBar": true' in html_str or '"displayModeBar":true' in html_str, \
            "displayModeBar not found in HTML config (must be true)"

    def test_config_responsive_in_html(self, html_bytes):
        """responsive must be set to true in the serialised Plotly config block."""
        html_str = html_bytes.decode("utf-8", errors="replace")
        assert '"responsive": true' in html_str or '"responsive":true' in html_str, \
            "responsive not found in HTML config (must be true)"


# ===========================================================================
# 3. Page 05 — no longer uses CDN
# ===========================================================================

class TestPage05NoCDN:

    def test_no_include_plotlyjs_cdn(self):
        src = _read(PAGE_05)
        cdn_pattern = re.compile(
            r'include_plotlyjs\s*=\s*["\']cdn["\']', re.IGNORECASE
        )
        assert not cdn_pattern.search(src), \
            "page 05 download_figure() still uses include_plotlyjs='cdn'"

    def test_uses_figure_to_self_contained_html(self):
        src = _read(PAGE_05)
        assert "figure_to_self_contained_html" in src, \
            "page 05 download_figure() does not use figure_to_self_contained_html"

    def test_download_button_present(self):
        src = _read(PAGE_05)
        assert "st.download_button" in src or "download_button" in src, \
            "page 05 has no download button"


# ===========================================================================
# 4. Page 03 — has download buttons
# ===========================================================================

class TestPage03DownloadButtons:

    def test_uses_figure_to_self_contained_html(self):
        src = _read(PAGE_03)
        assert "figure_to_self_contained_html" in src, \
            "page 03 does not call figure_to_self_contained_html"

    def test_download_button_label(self):
        src = _read(PAGE_03)
        assert "Download interactive HTML" in src, \
            "page 03 has no 'Download interactive HTML' button label"

    def test_no_cdn_reference(self):
        src = _read(PAGE_03)
        cdn_pattern = re.compile(
            r'include_plotlyjs\s*=\s*["\']cdn["\']', re.IGNORECASE
        )
        assert not cdn_pattern.search(src), \
            "page 03 contains include_plotlyjs='cdn'"

    def test_multiple_download_buttons(self):
        src = _read(PAGE_03)
        count = src.count("figure_to_self_contained_html")
        assert count >= 6, \
            f"page 03 should have ≥6 download buttons (map/TS/TS-map/d18O-map/xy/d18O); found {count}"

    def test_render_linked_xy_plot_has_download(self):
        """render_linked_xy_plot() must contain a download button for fig_xy."""
        src = _read(PAGE_03)
        # Find the function body (heuristic: between def render_linked_xy_plot and the next top-level def)
        start = src.find("def render_linked_xy_plot(")
        assert start != -1, "render_linked_xy_plot not found in page 03"
        # Find next top-level def after this function
        next_def = src.find("\n    def ", start + 1)
        if next_def == -1:
            next_def = len(src)
        fn_body = src[start:next_def]
        assert "figure_to_self_contained_html" in fn_body, \
            "render_linked_xy_plot() does not call figure_to_self_contained_html (fig_xy download missing)"
        assert "xy_html_dl" in fn_body, \
            "render_linked_xy_plot() download button key must contain 'xy_html_dl'"

    def test_fig_d18o_has_download(self):
        """The d18O scatter plot section must have a download button."""
        src = _read(PAGE_03)
        assert "p03_d18O_html_dl" in src, \
            "page 03 fig_d18O download button key 'p03_d18O_html_dl' not found"

    def test_no_duplicate_download_keys(self):
        """Every st.download_button in page 03 must have a unique key."""
        keys = re.findall(r'key\s*=\s*["\']([^"\']+)["\']', _read(PAGE_03))
        dl_keys = [k for k in keys if "html_dl" in k]
        assert len(dl_keys) == len(set(dl_keys)), \
            f"Duplicate download button keys in page 03: {dl_keys}"

    def test_map_style_widget_keys_unique_across_pages(self):
        """map_style widget keys must be unique across pages 03/04/31/32/34."""
        pages = {
            "03": _read(PAGE_03),
            "04": _read(PAGE_04),
            "31": _read(REPO_ROOT / "pages" / "31_Salinity-d18O_Relationship.py"),
            "32": _read(REPO_ROOT / "pages" / "32_Isotope_Hydrographic_Mapping.py"),
            "34": _read(REPO_ROOT / "pages" / "34_T-S_diagram.py"),
        }
        seen = {}
        for page_num, src in pages.items():
            for key in re.findall(r'key\s*=\s*["\']([^"\']+)["\']', src):
                if key.startswith("map_style_"):
                    assert key not in seen, (
                        f"Widget key '{key}' appears in both page {seen[key]} and page {page_num}"
                    )
                    seen[key] = page_num


# ===========================================================================
# 5. Page 04 — has download buttons
# ===========================================================================

class TestPage04DownloadButtons:

    def test_uses_figure_to_self_contained_html(self):
        src = _read(PAGE_04)
        assert "figure_to_self_contained_html" in src, \
            "page 04 does not call figure_to_self_contained_html"

    def test_download_button_label(self):
        src = _read(PAGE_04)
        assert "Download interactive HTML" in src, \
            "page 04 has no 'Download interactive HTML' button label"

    def test_no_cdn_reference(self):
        src = _read(PAGE_04)
        cdn_pattern = re.compile(
            r'include_plotlyjs\s*=\s*["\']cdn["\']', re.IGNORECASE
        )
        assert not cdn_pattern.search(src), \
            "page 04 contains include_plotlyjs='cdn'"

    def test_two_download_buttons(self):
        src = _read(PAGE_04)
        count = src.count("figure_to_self_contained_html")
        assert count >= 2, \
            f"page 04 should have ≥2 download buttons; found {count}"
