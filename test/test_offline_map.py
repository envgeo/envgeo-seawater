"""Tests for the offline map sprint helpers in envgeo_utils.

No real network calls are made: check_online_connectivity() is always
monkeypatched via the ``online`` / ``offline`` fixtures below, and
socket.create_connection is patched wherever the inner cached function is
tested directly.

Run from the app root:
    pytest -q test/test_offline_map.py
"""

import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Minimal Streamlit stub so envgeo_utils can be imported without a running
# Streamlit server.  Only the symbols actually used at module-load time are
# needed; runtime calls are either tested in-process or not called here.
# ---------------------------------------------------------------------------

def _make_streamlit_stub():
    st = types.ModuleType("streamlit")

    # cache_data / cache_resource – return identity decorator
    def _noop_deco(*args, **kwargs):
        def _wrap(fn):
            return fn
        # handle both @st.cache_data and @st.cache_data(ttl=…)
        if args and callable(args[0]):
            return args[0]
        return _wrap

    st.cache_data = _noop_deco
    st.cache_resource = _noop_deco

    # session_state
    st.session_state = {}

    # sidebar / widgets – silent stubs
    for _name in ("sidebar", "radio", "selectbox", "warning", "caption",
                  "expander", "columns", "write", "info", "error"):
        setattr(st, _name, mock.MagicMock())

    return st


if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = _make_streamlit_stub()
else:
    # Patch the decorators if a real Streamlit is somehow present but
    # not running a server.
    _st = sys.modules["streamlit"]
    if not callable(getattr(_st, "cache_data", None)):
        _st.cache_data = lambda *a, **kw: (a[0] if a and callable(a[0]) else lambda f: f)
        _st.cache_resource = _st.cache_data

import envgeo_utils  # noqa: E402  (must come after stub)


# ---------------------------------------------------------------------------
# Connectivity fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def online(monkeypatch):
    """Make check_online_connectivity() return True without any network call.
    Accepts any positional args to match the updated mode-aware signature.
    """
    monkeypatch.setattr(envgeo_utils, "check_online_connectivity", lambda *args: True)


@pytest.fixture()
def offline(monkeypatch):
    """Make check_online_connectivity() return False without any network call."""
    monkeypatch.setattr(envgeo_utils, "check_online_connectivity", lambda *args: False)


# ---------------------------------------------------------------------------
# Helper to build a minimal Plotly figure for apply_map_style tests
# ---------------------------------------------------------------------------

def _blank_mapbox_fig():
    """Return a bare Plotly Figure with a mapbox layout."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(mapbox=dict(style="open-street-map"))
    return fig


# ---------------------------------------------------------------------------
# 1. MAP_MODE_OPTIONS structure
# ---------------------------------------------------------------------------

class TestMapModeOptions:
    def test_offline_first(self):
        assert envgeo_utils.MAP_MODE_OPTIONS[0] == "Coastline (offline)"

    def test_offline_in_options(self):
        assert "Coastline (offline)" in envgeo_utils.MAP_MODE_OPTIONS

    def test_default_is_standard(self):
        assert envgeo_utils.MAP_MODE_DEFAULT == "Standard"

    def test_default_index_points_to_standard(self):
        idx = envgeo_utils.MAP_MODE_DEFAULT_INDEX
        assert envgeo_utils.MAP_MODE_OPTIONS[idx] == "Standard"

    def test_default_index_is_not_zero(self):
        # index=0 would select "Coastline (offline)" by default — wrong.
        assert envgeo_utils.MAP_MODE_DEFAULT_INDEX != 0


# ---------------------------------------------------------------------------
# 2. resolve_map_mode
# ---------------------------------------------------------------------------

class TestResolveMapMode:
    def test_offline_mode_no_check(self, monkeypatch):
        """Offline mode skips the connectivity check entirely."""
        called = []
        monkeypatch.setattr(
            envgeo_utils, "check_online_connectivity", lambda *a: called.append(1) or True
        )
        mode, fell = envgeo_utils.resolve_map_mode("Coastline (offline)")
        assert mode == "Coastline (offline)"
        assert fell is False
        assert called == []  # never called

    def test_online_mode_when_connected(self, online):
        mode, fell = envgeo_utils.resolve_map_mode("Standard")
        assert mode == "Standard"
        assert fell is False

    def test_online_mode_falls_back_when_offline(self, offline):
        mode, fell = envgeo_utils.resolve_map_mode("Standard")
        assert mode == "Coastline (offline)"
        assert fell is True

    def test_satellite_falls_back_when_offline(self, offline):
        mode, fell = envgeo_utils.resolve_map_mode("Satellite")
        assert mode == "Coastline (offline)"
        assert fell is True

    def test_bathymetry_falls_back_when_offline(self, offline):
        mode, fell = envgeo_utils.resolve_map_mode("Bathymetry (Sea)")
        assert mode == "Coastline (offline)"
        assert fell is True

    def test_contour_falls_back_when_offline(self, offline):
        mode, fell = envgeo_utils.resolve_map_mode("Contour (GSI)")
        assert mode == "Coastline (offline)"
        assert fell is True


# ---------------------------------------------------------------------------
# 3. apply_map_style — offline mode
# ---------------------------------------------------------------------------

class TestApplyMapStyleOffline:
    def test_offline_mode_uses_white_bg(self, online):
        """Coastline (offline) must set mapbox_style to white-bg."""
        fig = _blank_mapbox_fig()
        envgeo_utils.apply_map_style(fig, "Coastline (offline)")
        style = fig.layout.mapbox.style
        assert style == "white-bg"

    def test_offline_mode_has_no_external_tile_layers(self, online):
        """Offline mode must produce zero external tile URLs."""
        fig = _blank_mapbox_fig()
        envgeo_utils.apply_map_style(fig, "Coastline (offline)")
        layers = fig.layout.mapbox.layers or []
        for layer in layers:
            src = getattr(layer, "source", None)
            if isinstance(src, (list, tuple)):
                for url in src:
                    assert not str(url).startswith("http"), (
                        f"Offline mode must not include external URL: {url}"
                    )

    def test_online_fallback_also_uses_white_bg(self, offline):
        """When online mode falls back, apply_map_style still gets white-bg."""
        fig = _blank_mapbox_fig()
        envgeo_utils.apply_map_style(fig, "Standard")
        # apply_map_style calls resolve_map_mode internally; offline → white-bg
        style = fig.layout.mapbox.style
        assert style == "white-bg"


# ---------------------------------------------------------------------------
# 4. add_coastline_overlay
# ---------------------------------------------------------------------------

class TestAddCoastlineOverlay:
    def test_adds_scattermapbox_trace(self, monkeypatch):
        """add_coastline_overlay must append exactly one Scattermapbox trace."""
        import plotly.graph_objects as go

        # Stub load_coastline_data to return a tiny synthetic coastline
        monkeypatch.setattr(
            envgeo_utils,
            "load_coastline_data",
            lambda ref_data, resolution="50m": ([0.0, 1.0, None], [0.0, 1.0, None]),
        )

        fig = _blank_mapbox_fig()
        before = len(fig.data)
        result = envgeo_utils.add_coastline_overlay(fig)
        assert result is True
        assert len(fig.data) == before + 1

        trace = fig.data[-1]
        assert isinstance(trace, go.Scattermapbox)

    def test_trace_name_is_coastline_overlay(self, monkeypatch):
        monkeypatch.setattr(
            envgeo_utils,
            "load_coastline_data",
            lambda ref_data, resolution="50m": ([0.0, 1.0], [0.0, 1.0]),
        )
        fig = _blank_mapbox_fig()
        envgeo_utils.add_coastline_overlay(fig)
        assert fig.data[-1].name == "_coastline_overlay"

    def test_trace_mode_is_lines(self, monkeypatch):
        monkeypatch.setattr(
            envgeo_utils,
            "load_coastline_data",
            lambda ref_data, resolution="50m": ([0.0, 1.0], [0.0, 1.0]),
        )
        fig = _blank_mapbox_fig()
        envgeo_utils.add_coastline_overlay(fig)
        assert fig.data[-1].mode == "lines"

    def test_no_trace_when_coastline_empty(self, monkeypatch):
        """If the coastline CSV is empty/unavailable, no trace is added, returns False."""
        monkeypatch.setattr(
            envgeo_utils,
            "load_coastline_data",
            lambda ref_data, resolution="50m": ([], []),
        )
        fig = _blank_mapbox_fig()
        before = len(fig.data)
        result = envgeo_utils.add_coastline_overlay(fig)
        assert len(fig.data) == before  # nothing added
        assert result is False           # signals missing data


# ---------------------------------------------------------------------------
# 5. add_graticule_overlay
# ---------------------------------------------------------------------------

class TestAddGraticuleOverlay:
    def test_adds_scattermapbox_trace(self):
        """add_graticule_overlay must append exactly one Scattermapbox trace."""
        import plotly.graph_objects as go

        fig = _blank_mapbox_fig()
        before = len(fig.data)
        envgeo_utils.add_graticule_overlay(fig)
        assert len(fig.data) == before + 1
        assert isinstance(fig.data[-1], go.Scattermapbox)

    def test_trace_name_is_graticule_overlay(self):
        fig = _blank_mapbox_fig()
        envgeo_utils.add_graticule_overlay(fig)
        assert fig.data[-1].name == "_graticule_overlay"

    def test_trace_mode_is_lines(self):
        fig = _blank_mapbox_fig()
        envgeo_utils.add_graticule_overlay(fig)
        assert fig.data[-1].mode == "lines"

    def test_no_external_urls_in_graticule(self):
        """Graticule trace must contain only numeric/None coords, no URLs."""
        fig = _blank_mapbox_fig()
        envgeo_utils.add_graticule_overlay(fig)
        trace = fig.data[-1]
        for lon in trace.lon:
            assert lon is None or isinstance(lon, float), f"Unexpected lon value: {lon!r}"

    def test_covers_all_longitudes(self):
        """lon values should span -180 to 180."""
        fig = _blank_mapbox_fig()
        envgeo_utils.add_graticule_overlay(fig)
        lons = [v for v in fig.data[-1].lon if v is not None]
        assert min(lons) == -180.0
        assert max(lons) == 180.0

    def test_dateline_safe_no_jump(self):
        """Adjacent non-None lon values must not jump by ≥ 180° (antimeridian artifact)."""
        fig = _blank_mapbox_fig()
        envgeo_utils.add_graticule_overlay(fig)
        lons = list(fig.data[-1].lon)
        prev = None
        for v in lons:
            if v is not None and prev is not None:
                assert abs(v - prev) < 180.0, (
                    f"Antimeridian jump detected: {prev} → {v}"
                )
            prev = v if v is not None else None


# ---------------------------------------------------------------------------
# 6. check_online_connectivity — per-mode hosts, caching / mocking
# ---------------------------------------------------------------------------

class TestCheckOnlineConnectivity:
    def test_returns_bool(self, monkeypatch):
        """check_online_connectivity() must return a plain bool."""
        import socket

        monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: mock.MagicMock())
        result = envgeo_utils.check_online_connectivity()
        assert isinstance(result, bool)

    def test_returns_false_on_os_error(self, monkeypatch):
        import socket

        def _fail(*a, **kw):
            raise OSError("simulated network failure")

        monkeypatch.setattr(socket, "create_connection", _fail)
        # Clear the LRU cache so the monkeypatch takes effect
        envgeo_utils._check_connectivity_cached.cache_clear()
        result = envgeo_utils._check_connectivity_cached(-9999, "example.com", 443)
        assert result is False

    def test_socket_is_closed_on_success(self, monkeypatch):
        """The socket returned by create_connection must be closed."""
        import socket

        closed = []
        mock_conn = mock.MagicMock()
        mock_conn.close.side_effect = lambda: closed.append(True)
        monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: mock_conn)
        envgeo_utils._check_connectivity_cached.cache_clear()
        envgeo_utils._check_connectivity_cached(-8888, "example.com", 443)
        assert closed, "Socket.close() was never called"

    def test_mode_specific_host_standard(self, monkeypatch):
        """Standard mode must check tile.openstreetmap.org."""
        import socket

        checked_hosts = []

        def _capture(address, **kw):
            checked_hosts.append(address[0])
            return mock.MagicMock()

        monkeypatch.setattr(socket, "create_connection", _capture)
        envgeo_utils._check_connectivity_cached.cache_clear()
        envgeo_utils.check_online_connectivity("Standard")
        assert any("openstreetmap" in h for h in checked_hosts), checked_hosts

    def test_mode_specific_host_satellite(self, monkeypatch):
        """Satellite mode must check basemap.nationalmap.gov."""
        import socket

        checked_hosts = []

        def _capture(address, **kw):
            checked_hosts.append(address[0])
            return mock.MagicMock()

        monkeypatch.setattr(socket, "create_connection", _capture)
        envgeo_utils._check_connectivity_cached.cache_clear()
        envgeo_utils.check_online_connectivity("Satellite")
        assert any("nationalmap" in h for h in checked_hosts), checked_hosts

    def test_mode_specific_host_bathymetry(self, monkeypatch):
        """Bathymetry mode must check services.arcgisonline.com."""
        import socket

        checked_hosts = []

        def _capture(address, **kw):
            checked_hosts.append(address[0])
            return mock.MagicMock()

        monkeypatch.setattr(socket, "create_connection", _capture)
        envgeo_utils._check_connectivity_cached.cache_clear()
        envgeo_utils.check_online_connectivity("Bathymetry (Sea)")
        assert any("arcgisonline" in h for h in checked_hosts), checked_hosts

    def test_mode_specific_host_gsi(self, monkeypatch):
        """Contour (GSI) mode must check cyberjapandata.gsi.go.jp."""
        import socket

        checked_hosts = []

        def _capture(address, **kw):
            checked_hosts.append(address[0])
            return mock.MagicMock()

        monkeypatch.setattr(socket, "create_connection", _capture)
        envgeo_utils._check_connectivity_cached.cache_clear()
        envgeo_utils.check_online_connectivity("Contour (GSI)")
        assert any("gsi.go.jp" in h for h in checked_hosts), checked_hosts


# ===========================================================================
# OFFLINE_FALLBACK_WARNING exact text
# ===========================================================================

class TestOfflineFallbackWarningText:
    """Verify OFFLINE_FALLBACK_WARNING is English-only with the required text."""

    def test_warning_text_english_only(self):
        msg = envgeo_utils.OFFLINE_FALLBACK_WARNING
        assert "Online map tiles are unavailable" in msg, \
            f"Expected 'Online map tiles are unavailable' in OFFLINE_FALLBACK_WARNING; got: {msg!r}"
        assert "Showing the local coastline map" in msg, \
            f"Expected 'Showing the local coastline map' in OFFLINE_FALLBACK_WARNING; got: {msg!r}"

    def test_warning_starts_with_emoji(self):
        assert envgeo_utils.OFFLINE_FALLBACK_WARNING.startswith("⚠️"), \
            "OFFLINE_FALLBACK_WARNING should start with ⚠️"

    def test_warning_no_japanese(self):
        msg = envgeo_utils.OFFLINE_FALLBACK_WARNING
        # Japanese characters fall in Unicode ranges 3040-9FFF (hiragana, katakana, CJK)
        has_japanese = any(0x3040 <= ord(c) <= 0x9FFF for c in msg)
        assert not has_japanese, \
            f"OFFLINE_FALLBACK_WARNING must be English-only, got: {msg!r}"

    def test_warning_no_slash_separator(self):
        """Must not contain ' / ' bilingual separator."""
        assert " / " not in envgeo_utils.OFFLINE_FALLBACK_WARNING, \
            "OFFLINE_FALLBACK_WARNING must not contain ' / ' bilingual separator"


# ===========================================================================
# add_coastline_overlay idempotency
# ===========================================================================

class TestAddCoastlineOverlayIdempotency:
    """add_coastline_overlay must not add a duplicate trace on second call."""

    def test_idempotent_second_call(self, offline):
        """Calling add_coastline_overlay twice on the same fig adds only 1 trace."""
        import plotly.graph_objects as go

        fig = go.Figure()
        before = len(fig.data)
        envgeo_utils.add_coastline_overlay(fig)
        after_first = len(fig.data)
        envgeo_utils.add_coastline_overlay(fig)
        after_second = len(fig.data)

        assert after_first == before + 1, "First call should add exactly 1 trace"
        assert after_second == after_first, \
            "Second call must not add another trace (idempotency broken)"

    def test_returns_true_on_duplicate(self, offline):
        """Second call must return True (trace already present)."""
        import plotly.graph_objects as go

        fig = go.Figure()
        envgeo_utils.add_coastline_overlay(fig)
        result = envgeo_utils.add_coastline_overlay(fig)
        assert result is True, \
            "add_coastline_overlay should return True when trace already exists"

    def test_overlay_trace_named_correctly(self, offline):
        import plotly.graph_objects as go

        fig = go.Figure()
        envgeo_utils.add_coastline_overlay(fig)
        names = [getattr(t, "name", None) for t in fig.data]
        assert "_coastline_overlay" in names, \
            f"Trace named '_coastline_overlay' not found; got: {names}"


# ===========================================================================
# apply_map_style adds coastline trace for all modes
# ===========================================================================

class TestApplyMapStyleAddsCoastline:
    """apply_map_style must embed _coastline_overlay for every map mode."""

    @pytest.mark.parametrize("mode", [
        "Coastline (offline)",
        "Standard",
        "Satellite",
        "Bathymetry (Sea)",
        "Contour (GSI)",
    ])
    def test_mode_has_coastline_trace(self, mode, offline):
        import plotly.graph_objects as go

        fig = go.Figure()
        envgeo_utils.apply_map_style(fig, mode)
        names = [getattr(t, "name", None) for t in fig.data]
        assert "_coastline_overlay" in names, \
            f"apply_map_style('{mode}') did not add _coastline_overlay; traces: {names}"

    @pytest.mark.parametrize("mode", [
        "Coastline (offline)",
        "Standard",
        "Satellite",
        "Bathymetry (Sea)",
        "Contour (GSI)",
    ])
    def test_double_apply_no_duplicate(self, mode, offline):
        """Calling apply_map_style twice must not produce two _coastline_overlay traces."""
        import plotly.graph_objects as go

        fig = go.Figure()
        envgeo_utils.apply_map_style(fig, mode)
        envgeo_utils.apply_map_style(fig, mode)
        count = sum(1 for t in fig.data if getattr(t, "name", None) == "_coastline_overlay")
        assert count == 1, \
            f"Expected exactly 1 _coastline_overlay after double apply_map_style('{mode}'); got {count}"


# ===========================================================================
# MAP_MODE_DEFAULT_INDEX — default selection is Standard, not Coastline
# ===========================================================================

class TestMapModeDefaultIndex:
    """MAP_MODE_DEFAULT_INDEX must point to Standard, not Coastline (offline)."""

    def test_default_is_standard(self):
        default = envgeo_utils.MAP_MODE_OPTIONS[envgeo_utils.MAP_MODE_DEFAULT_INDEX]
        assert default == "Standard", (
            f"MAP_MODE_DEFAULT_INDEX should select 'Standard'; got '{default}'"
        )

    def test_coastline_offline_is_at_index_zero(self):
        """Coastline (offline) must remain first in the list."""
        assert envgeo_utils.MAP_MODE_OPTIONS[0] == "Coastline (offline)", (
            "MAP_MODE_OPTIONS[0] must be 'Coastline (offline)'"
        )

    def test_default_index_not_zero(self):
        """Default must not be index 0 (which is Coastline (offline))."""
        assert envgeo_utils.MAP_MODE_DEFAULT_INDEX != 0, (
            "MAP_MODE_DEFAULT_INDEX is 0 — this would default to Coastline (offline) on fresh sessions"
        )

    def test_page03_radio_map_mode_has_default_index(self):
        """page 03 map_mode radio must specify index=MAP_MODE_DEFAULT_INDEX."""
        import re
        src = Path(__file__).parent.parent / "pages" / "03_[Interactive]_2Dplus_Visualizer.py"
        text = src.read_text(encoding="utf-8")
        # Find radio blocks for map_mode, map_mode_ts, map_mode_d18o
        radios = re.findall(
            r'(map_mode(?:_ts|_d18o)?\s*=\s*st\.radio\([^)]+\))',
            text, re.DOTALL
        )
        for block in radios:
            assert "MAP_MODE_DEFAULT_INDEX" in block, (
                f"A map_mode st.radio in page 03 is missing index=MAP_MODE_DEFAULT_INDEX:\n{block[:200]}"
            )

    def test_page04_radio_map_mode_has_default_index(self):
        """page 04 map_mode radio must specify index=MAP_MODE_DEFAULT_INDEX."""
        src = Path(__file__).parent.parent / "pages" / "04_[Interactive]_3D_4D_Visualizer.py"
        text = src.read_text(encoding="utf-8")
        assert "MAP_MODE_DEFAULT_INDEX" in text, \
            "page 04 map_mode st.radio is missing index=MAP_MODE_DEFAULT_INDEX"

    def test_page32_radio_map_mode_has_default_index(self):
        """page 32 map_mode radio must specify index=MAP_MODE_DEFAULT_INDEX."""
        src = Path(__file__).parent.parent / "pages" / "32_Isotope_Hydrographic_Mapping.py"
        text = src.read_text(encoding="utf-8")
        assert "MAP_MODE_DEFAULT_INDEX" in text, \
            "page 32 map_mode st.radio is missing index=MAP_MODE_DEFAULT_INDEX"
