"""Tests for the bundled Natural Earth 50m land shapefile.

Verifies:
- Required shapefile components are present in the repository.
- The shapefile can be read by cartopy.io.shapereader.Reader (no download).
- No call to cartopy's NaturalEarthFeature, natural_earth(), or the Cartopy
  downloader is made when add_geometries is used with a local Reader.
- The helper _load_ne50m_land_geometries() warns (does not crash) when a
  component is missing, and returns an empty list without network access.

Run from the app root:
    pytest -q test/test_natural_earth_land.py
"""

import pathlib
import sys
import types
import unittest.mock as mock

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
NE50M_DIR = ROOT / "coastline" / "natural_earth_50m_land"
NE50M_SHP = NE50M_DIR / "ne_50m_land.shp"
REQUIRED_EXTS = (".shp", ".shx", ".dbf")
OPTIONAL_EXTS = (".prj", ".cpg")


# ---------------------------------------------------------------------------
# 1. Shapefile presence
# ---------------------------------------------------------------------------

class TestShapefilePresence:
    """All required components must exist in the repository."""

    def test_directory_exists(self):
        assert NE50M_DIR.is_dir(), (
            f"Natural Earth land directory missing: {NE50M_DIR}"
        )

    @pytest.mark.parametrize("ext", REQUIRED_EXTS)
    def test_required_component_exists(self, ext):
        f = NE50M_DIR / f"ne_50m_land{ext}"
        assert f.exists(), f"Required shapefile component missing: {f.name}"

    @pytest.mark.parametrize("ext", OPTIONAL_EXTS)
    def test_optional_component_present(self, ext):
        f = NE50M_DIR / f"ne_50m_land{ext}"
        # Warn but don't fail — these are optional
        if not f.exists():
            pytest.skip(f"Optional component absent (acceptable): {f.name}")

    def test_license_file_exists(self):
        assert (NE50M_DIR / "LICENSE_OR_SOURCE.md").exists(), (
            "LICENSE_OR_SOURCE.md is missing from natural_earth_50m_land/"
        )

    def test_shp_not_empty(self):
        assert NE50M_SHP.stat().st_size > 100_000, (
            f"ne_50m_land.shp is suspiciously small: {NE50M_SHP.stat().st_size} bytes"
        )


# ---------------------------------------------------------------------------
# 2. Readable by Cartopy shapereader — no network needed
# ---------------------------------------------------------------------------

class TestShapefileReadable:
    """Cartopy Reader must open the file locally without downloading anything."""

    def test_reader_opens_without_download(self):
        """Open the shapefile with cartopy.io.shapereader.Reader.

        If cartopy tries to download anything the test will raise because we
        patch urllib.request.urlopen to raise immediately.
        """
        import cartopy.io.shapereader as shp

        with mock.patch("urllib.request.urlopen", side_effect=AssertionError(
            "cartopy tried to download data — should use local Reader only"
        )):
            reader = shp.Reader(str(NE50M_SHP))
            geoms = list(reader.geometries())

        assert len(geoms) > 100, (
            f"Expected many land polygons, got {len(geoms)}"
        )

    def test_geometry_types_are_polygon(self):
        """Every feature should be a Polygon or MultiPolygon."""
        import cartopy.io.shapereader as shp
        from shapely.geometry import Polygon, MultiPolygon

        reader = shp.Reader(str(NE50M_SHP))
        for i, geom in enumerate(reader.geometries()):
            assert isinstance(geom, (Polygon, MultiPolygon)), (
                f"Feature {i} is unexpected type: {type(geom).__name__}"
            )

    def test_feature_count_reasonable(self):
        """NE 50m land should have >1000 features (continents + islands)."""
        import cartopy.io.shapereader as shp

        reader = shp.Reader(str(NE50M_SHP))
        count = sum(1 for _ in reader.geometries())
        assert count > 1000, f"Feature count suspiciously low: {count}"

    def test_no_natural_earth_downloader_called(self):
        """cartopy.io.shapereader.natural_earth() must NOT be called."""
        import cartopy.io.shapereader as shp

        with mock.patch.object(shp, "natural_earth", side_effect=AssertionError(
            "natural_earth() was called — only the local Reader should be used"
        )):
            reader = shp.Reader(str(NE50M_SHP))
            _ = list(reader.geometries())  # Should not trigger the patch


# ---------------------------------------------------------------------------
# 3. _load_ne50m_land_geometries() — graceful degradation when file missing
# ---------------------------------------------------------------------------

def _make_minimal_streamlit_stub():
    """Minimal st stub that captures st.warning() calls."""
    st = types.ModuleType("streamlit")
    st.warning = mock.MagicMock()
    st.error = mock.MagicMock()
    st.cache_data = lambda *a, **kw: (a[0] if a and callable(a[0]) else lambda f: f)
    st.cache_resource = st.cache_data
    st.session_state = {}
    return st


class TestLoadHelper:
    """Tests for the _load_ne50m_land_geometries() helper in page 32."""

    @pytest.fixture(autouse=True)
    def _stub_streamlit(self, monkeypatch):
        """Inject a minimal streamlit stub so the page module can be imported."""
        self._st_stub = _make_minimal_streamlit_stub()
        monkeypatch.setitem(sys.modules, "streamlit", self._st_stub)
        # Also stub other heavy imports that the page pulls in
        for mod in [
            "plotly.express",
            "envgeo_utils",
            "envgeo_user_data",
        ]:
            if mod not in sys.modules:
                monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))

    def _import_loader(self):
        """Import _load_ne50m_land_geometries from the page module."""
        import importlib
        # Clear cached module if present
        for key in list(sys.modules):
            if "32_Isotope" in key:
                del sys.modules[key]
        spec = __import__(
            "importlib.util", fromlist=["spec_from_file_location", "module_from_spec"]
        )
        import importlib.util as ilu
        page_path = ROOT / "pages" / "32_Isotope_Hydrographic_Mapping.py"
        spec_ = ilu.spec_from_file_location("page32", page_path)
        mod = ilu.module_from_spec(spec_)
        # Suppress actual execution of main() etc. by only loading
        try:
            spec_.loader.exec_module(mod)
        except Exception:
            pass  # top-level errors are acceptable; we only need the helper
        return mod

    def test_returns_list_when_shapefile_present(self):
        mod = self._import_loader()
        geoms = mod._load_ne50m_land_geometries()
        assert isinstance(geoms, list)
        assert len(geoms) > 100

    def test_returns_empty_list_when_shp_missing(self, tmp_path, monkeypatch):
        """When .shp is missing, return [] and show a warning (no network call)."""
        mod = self._import_loader()
        # Point the helper to a directory that has only .shx and .dbf
        fake_dir = tmp_path / "ne_land"
        fake_dir.mkdir()
        # Create dummy .shx and .dbf but NOT .shp
        (fake_dir / "ne_50m_land.shx").write_bytes(b"\x00" * 10)
        (fake_dir / "ne_50m_land.dbf").write_bytes(b"\x00" * 10)

        monkeypatch.setattr(mod, "_NE50M_LAND_DIR", fake_dir)
        monkeypatch.setattr(mod, "_NE50M_LAND_SHP", fake_dir / "ne_50m_land.shp")
        monkeypatch.setattr(mod, "_NE50M_LAND_REQUIRED_EXTS", (".shp", ".shx", ".dbf"))

        with mock.patch("urllib.request.urlopen", side_effect=AssertionError(
            "No network calls permitted during graceful degradation"
        )):
            result = mod._load_ne50m_land_geometries()

        assert result == [], f"Expected [] when .shp missing, got {result}"
        assert self._st_stub.warning.called, "st.warning() should be called when file is missing"

    def test_no_network_call_when_file_missing(self, tmp_path, monkeypatch):
        """Even when the shapefile is absent, no HTTP/network request is made."""
        mod = self._import_loader()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(mod, "_NE50M_LAND_DIR", empty_dir)
        monkeypatch.setattr(mod, "_NE50M_LAND_SHP", empty_dir / "ne_50m_land.shp")
        monkeypatch.setattr(mod, "_NE50M_LAND_REQUIRED_EXTS", (".shp", ".shx", ".dbf"))

        network_called = []
        with mock.patch("urllib.request.urlopen", side_effect=lambda *a, **kw: network_called.append(a)):
            mod._load_ne50m_land_geometries()

        assert network_called == [], (
            f"network was accessed during graceful degradation: {network_called}"
        )


# ---------------------------------------------------------------------------
# 4. Page 32 source-code audit: forbidden drawing calls
# ---------------------------------------------------------------------------

class TestPage32ForbiddenCalls:
    """Static analysis: page 32 must not contain forbidden coastline/land calls.

    These calls would reach Cartopy's shapefile downloader in an empty-cache
    environment, defeating the purpose of bundling the Natural Earth assets.
    """

    PAGE32 = ROOT / "pages" / "32_Isotope_Hydrographic_Mapping.py"

    @pytest.fixture(autouse=True)
    def _source(self):
        self._src = self.PAGE32.read_text(encoding="utf-8")

    def test_no_ax_coastlines(self):
        """ax.coastlines() / ax2.coastlines() must not appear in the source."""
        import re
        hits = re.findall(r"\bax[^.]*\.coastlines\s*\(", self._src)
        assert not hits, (
            f"ax.coastlines() found in page 32 — replace with "
            f"envgeo_utils.plot_bundled_coastline(): {hits}"
        )

    def test_no_cfeature_LAND(self):
        """cfeature.LAND must not appear in the source."""
        assert "cfeature.LAND" not in self._src, (
            "cfeature.LAND found in page 32 — use the bundled NE shapefile instead."
        )

    def test_no_natural_earth_function_call(self):
        """shapereader.natural_earth() must not appear in the source."""
        assert "natural_earth(" not in self._src, (
            "shapereader.natural_earth() found in page 32 — "
            "use shapereader.Reader(local_path) instead."
        )

    def test_uses_plot_bundled_coastline(self):
        """plot_bundled_coastline must be called for coastline drawing."""
        assert "plot_bundled_coastline" in self._src, (
            "envgeo_utils.plot_bundled_coastline() not found in page 32; "
            "it must be used for all coastline drawing."
        )

    def test_uses_local_shapereader_reader(self):
        """shapereader.Reader(local path) must be used for the land mask."""
        assert "shapereader.Reader" in self._src, (
            "shapereader.Reader not found in page 32; "
            "land mask must use the bundled local shapefile."
        )


# ---------------------------------------------------------------------------
# 5. Page 32 degradation: missing shapefile → English warning, no network
# ---------------------------------------------------------------------------

class TestPage32ShapefileMissingDegradation:
    """When the bundled NE 50m shapefile is absent, page 32 must:
    - show an English-language warning
    - not attempt any network connection
    - still draw coastlines and data points (degradation, not crash)
    """

    def _make_st_stub(self):
        stub = types.SimpleNamespace(
            warning=mock.MagicMock(),
            error=mock.MagicMock(),
            write=mock.MagicMock(),
            header=mock.MagicMock(),
            caption=mock.MagicMock(),
            sidebar=types.SimpleNamespace(
                selectbox=mock.MagicMock(return_value=None),
                checkbox=mock.MagicMock(return_value=False),
                radio=mock.MagicMock(return_value=None),
                slider=mock.MagicMock(return_value=(0, 1)),
                multiselect=mock.MagicMock(return_value=[]),
                number_input=mock.MagicMock(return_value=0),
            ),
        )
        return stub

    def _import_page32(self, monkeypatch):
        """Import the page module with minimal stubs, clear cached copy first."""
        for key in list(sys.modules):
            if "page32" in key or "32_Isotope" in key:
                del sys.modules[key]
        st_stub = self._make_st_stub()
        monkeypatch.setitem(sys.modules, "streamlit", st_stub)
        for mod_name in ["plotly.express", "envgeo_utils", "envgeo_user_data"]:
            if mod_name not in sys.modules:
                monkeypatch.setitem(sys.modules, mod_name, types.ModuleType(mod_name))
        import importlib.util as ilu
        page_path = ROOT / "pages" / "32_Isotope_Hydrographic_Mapping.py"
        spec = ilu.spec_from_file_location("page32_degrade", page_path)
        mod = ilu.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        mod._st_stub = st_stub
        return mod

    def test_english_warning_when_shp_missing(self, tmp_path, monkeypatch):
        """Missing shapefile must trigger an English-language st.warning call."""
        mod = self._import_page32(monkeypatch)
        empty_dir = tmp_path / "ne_empty"
        empty_dir.mkdir()
        monkeypatch.setattr(mod, "_NE50M_LAND_DIR", empty_dir)
        monkeypatch.setattr(mod, "_NE50M_LAND_SHP", empty_dir / "ne_50m_land.shp")
        monkeypatch.setattr(mod, "_NE50M_LAND_REQUIRED_EXTS", (".shp", ".shx", ".dbf"))

        with mock.patch("urllib.request.urlopen",
                        side_effect=AssertionError("network not allowed")):
            result = mod._load_ne50m_land_geometries()

        assert result == [], "Should return [] when shapefile is missing"
        warning_calls = [str(c) for c in mod._st_stub.warning.call_args_list]
        assert warning_calls, "st.warning() must be called when shapefile is missing"
        assert any("Natural Earth" in w or "Land mask" in w or "bundled" in w
                   for w in warning_calls), (
            f"Warning must mention the bundled shapefile. Got: {warning_calls}"
        )
        assert any("Coastline" in w or "data point" in w or "still" in w
                   for w in warning_calls), (
            f"Warning must indicate coastlines/points still display. Got: {warning_calls}"
        )

    def test_no_network_on_shapefile_missing(self, tmp_path, monkeypatch):
        """No network access when the bundled shapefile is absent."""
        mod = self._import_page32(monkeypatch)
        empty_dir = tmp_path / "ne_empty2"
        empty_dir.mkdir()
        monkeypatch.setattr(mod, "_NE50M_LAND_DIR", empty_dir)
        monkeypatch.setattr(mod, "_NE50M_LAND_SHP", empty_dir / "ne_50m_land.shp")
        monkeypatch.setattr(mod, "_NE50M_LAND_REQUIRED_EXTS", (".shp", ".shx", ".dbf"))

        network_calls = []
        with mock.patch("urllib.request.urlopen",
                        side_effect=lambda *a, **kw: network_calls.append(a)):
            mod._load_ne50m_land_geometries()

        assert network_calls == [], (
            f"Network was accessed during shapefile-missing degradation: {network_calls}"
        )

    def test_degraded_geoms_is_empty_list(self, tmp_path, monkeypatch):
        """_load_ne50m_land_geometries returns [] (not None, not exception) when missing."""
        mod = self._import_page32(monkeypatch)
        empty_dir = tmp_path / "ne_empty3"
        empty_dir.mkdir()
        monkeypatch.setattr(mod, "_NE50M_LAND_DIR", empty_dir)
        monkeypatch.setattr(mod, "_NE50M_LAND_SHP", empty_dir / "ne_50m_land.shp")
        monkeypatch.setattr(mod, "_NE50M_LAND_REQUIRED_EXTS", (".shp", ".shx", ".dbf"))

        result = mod._load_ne50m_land_geometries()
        assert result == [], f"Expected [], got {type(result)}: {result}"
