"""Focused tests for static Cartopy projection choices on the Mapping page."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAGE_PATH = ROOT / "pages" / "32_Isotope_Hydrographic_Mapping.py"


def test_mapping_page_exposes_the_static_projection_choices():
    """Keep the public static-map projection menu from silently regressing."""
    page_text = PAGE_PATH.read_text(encoding="utf-8")

    assert "Map projection / 投影法" in page_text
    for label in (
        "Plate Carrée (longitude–latitude)",
        "Mercator",
        "Robinson",
        "Azimuthal Equidistant",
        "Orthographic (map centre)",
        "Orthographic (South Polar)",
        "Orthographic (North Polar)",
    ):
        assert label in page_text

    for japanese_label in (
        "正距円筒図法（経緯度）",
        "メルカトル図法",
        "ロビンソン図法",
        "正距方位図法",
        "正射図法（地図中心）",
        "正射図法（南極中心）",
        "正射図法（北極中心）",
    ):
        assert japanese_label in page_text

    assert "def _read_ne50m_land_geometries" in page_text
    assert "close_reader()" in page_text
    assert "plt.close(fig)" in page_text
    assert "plt.close(fig_contour)" in page_text


@pytest.fixture(scope="module")
def mapping_module():
    """Load Page 32 without running its Streamlit ``main`` function."""
    pytest.importorskip("cartopy")
    spec = importlib.util.spec_from_file_location("test_page32_projections", PAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("label", "class_name"),
    [
        ("Plate Carrée (longitude–latitude)", "PlateCarree"),
        ("Mercator", "Mercator"),
        ("Robinson", "Robinson"),
        ("Azimuthal Equidistant", "AzimuthalEquidistant"),
        ("Orthographic (map centre)", "Orthographic"),
        ("Orthographic (South Polar)", "Orthographic"),
        ("Orthographic (North Polar)", "Orthographic"),
    ],
)
def test_static_map_projection_options_build_cartopy_crs(
    mapping_module, label, class_name
):
    projection = mapping_module.build_static_map_projection(label, 135.0, 35.0)

    assert type(projection).__name__ == class_name


def test_unknown_static_map_projection_is_rejected(mapping_module):
    with pytest.raises(ValueError, match="Unknown static map projection"):
        mapping_module.build_static_map_projection("Unknown", 0.0, 0.0)


def test_mercator_extent_excludes_poles(mapping_module):
    extent, was_clamped = mapping_module.safe_static_projection_extent(
        (-180.0, 180.0, -90.0, 90.0), "Mercator"
    )

    assert extent == (-180.0, 180.0, -84.5, 84.5)
    assert was_clamped is True


def test_non_mercator_extent_is_unchanged(mapping_module):
    extent, was_clamped = mapping_module.safe_static_projection_extent(
        (120.0, 155.0, 20.0, 50.0), "Azimuthal Equidistant"
    )

    assert extent == (120.0, 155.0, 20.0, 50.0)
    assert was_clamped is False


class _AxesStub:
    def __init__(self):
        self.calls = []

    def gridlines(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs

    def set_extent(self, extent, **kwargs):
        self.calls.append(("set_extent", extent, kwargs))

    def set_global(self):
        self.calls.append(("set_global",))


def test_curved_projections_use_unlabelled_gridlines(mapping_module):
    axes = _AxesStub()
    mapping_module.add_static_map_gridlines(axes, "Robinson")

    assert axes.calls == [{"draw_labels": False, "zorder": 4}]


def test_rectangular_projections_keep_grid_labels(mapping_module):
    axes = _AxesStub()
    mapping_module.add_static_map_gridlines(axes, "Mercator")

    assert axes.calls == [{"draw_labels": True, "zorder": 4}]


@pytest.mark.parametrize(
    "projection_label",
    [
        "Azimuthal Equidistant",
        "Orthographic (map centre)",
        "Orthographic (South Polar)",
    ],
)
def test_global_curved_view_uses_cartopy_global_extent(
    mapping_module, projection_label
):
    axes = _AxesStub()
    mapping_module.set_cartopy_extent_safely(
        axes,
        (-179.999, 179.999, -78.0, 90.0),
        projection_label,
    )

    assert axes.calls == [("set_global",)]


def test_southern_ocean_azimuthal_view_keeps_its_regional_extent(mapping_module):
    """A full-longitude polar band must not be mistaken for a global map."""
    axes = _AxesStub()
    mapping_module.set_cartopy_extent_safely(
        axes,
        (-180.0, 180.0, -78.0, -45.0),
        "Azimuthal Equidistant",
    )

    assert axes.calls[0][0] == "set_extent"


@pytest.mark.parametrize(
    ("extent", "expected_latitude"),
    [
        ((-180.0, 180.0, -78.0, -45.0), -90.0),
        ((-180.0, 180.0, 65.0, 90.0), 90.0),
    ],
)
def test_polar_azimuthal_presets_are_centred_on_the_pole(
    mapping_module, extent, expected_latitude
):
    _, central_latitude = mapping_module.static_projection_center(
        extent, "Azimuthal Equidistant"
    )

    assert central_latitude == expected_latitude


@pytest.mark.parametrize(
    ("projection_label", "expected_latitude"),
    [
        ("Orthographic (South Polar)", -90.0),
        ("Orthographic (North Polar)", 90.0),
    ],
)
def test_named_orthographic_views_are_centred_on_the_selected_pole(
    mapping_module, projection_label, expected_latitude
):
    _, central_latitude = mapping_module.static_projection_center(
        (120.0, 155.0, 20.0, 50.0), projection_label
    )

    assert central_latitude == expected_latitude


def test_map_centre_orthographic_uses_the_display_extent_midpoint(mapping_module):
    central_longitude, central_latitude = mapping_module.static_projection_center(
        (-70.0, 20.0, -60.0, 60.0), "Orthographic (map centre)"
    )

    assert (central_longitude, central_latitude) == (-25.0, 0.0)


def test_global_globe_style_contour_uses_observed_sample_scatter(mapping_module):
    effective_map_type = mapping_module.resolve_static_map_type_for_extent(
        (-180.0, 180.0, -78.0, 90.0),
        "Contour Map",
        "Azimuthal Equidistant",
    )

    assert effective_map_type == "Scatter Map"


def test_global_robinson_contour_uses_observed_sample_scatter(mapping_module):
    effective_map_type = mapping_module.resolve_static_map_type_for_extent(
        (-180.0, 180.0, -78.0, 90.0),
        "Contour Map",
        "Robinson",
    )

    assert effective_map_type == "Scatter Map"


def test_regional_contour_retains_exploratory_interpolation(mapping_module):
    effective_map_type = mapping_module.resolve_static_map_type_for_extent(
        (120.0, 155.0, 20.0, 50.0),
        "Contour Map",
        "Azimuthal Equidistant",
    )

    assert effective_map_type == "Contour Map"
