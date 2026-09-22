#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AppTests for shared uploaded-data filtering and foreground overlays.

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

from pathlib import Path
import sys
import importlib.util

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import envgeo_utils


def _vertical_section_module():
    page_path = ROOT / "pages" / "53_Vertical_Section_Visualizer.py"
    spec = importlib.util.spec_from_file_location(
        "test_vertical_section_visualizer",
        page_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _app_test():
    return pytest.importorskip("streamlit.testing.v1").AppTest


def _run_page(page_name, data):
    app = _app_test().from_file(str(ROOT / "pages" / page_name))
    app.session_state[envgeo_utils.UPLOAD_SESSION_DATA_KEY] = (
        envgeo_utils.prepare_uploaded_data(pd.DataFrame(data))
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_FILENAME_KEY] = "apptest.csv"
    app.run(timeout=45)
    return app


def _visible_text(app):
    values = []
    for element_type in (
        "markdown",
        "text",
        "caption",
        "success",
        "warning",
        "info",
        "error",
    ):
        values.extend(str(item.value) for item in getattr(app, element_type))
    return values


@pytest.mark.parametrize(
    "page_name",
    [
        "31_Salinity-d18O_Relationship.py",
        "32_Isotope_Hydrographic_Mapping.py",
        "34_T-S_diagram.py",
        "35_Custom_Parameter_Plot_beta.py",
        "37_Depth_Profile.py",
        "53_Vertical_Section_Visualizer.py",
    ],
)
def test_uploaded_dataset_option_is_visible_in_common_filter(page_name):
    if page_name == "32_Isotope_Hydrographic_Mapping.py":
        pytest.importorskip("cartopy")
    app = _app_test().from_file(str(ROOT / "pages" / page_name))
    app.run(timeout=60)

    assert not app.exception
    dataset_selector = next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    )
    assert "Uploaded data" in dataset_selector.options


@pytest.mark.parametrize(
    ("page_name", "data"),
    [
        (
            "31_Salinity-d18O_Relationship.py",
            {"Salinity": [34.5, 35.0], "d18O": [0.1, 0.2]},
        ),
        (
            "34_T-S_diagram.py",
            {"Salinity": [34.5, 35.0], "Temperature_degC": [20.0, 22.0]},
        ),
        (
            "37_Depth_Profile.py",
            {"d18O": [0.1, 0.2], "Depth_m": [10.0, 20.0], "Month": [1, 2]},
        ),
        (
            "35_Custom_Parameter_Plot_beta.py",
            {"d18O": [0.1, 0.2], "dD": [1.0, 2.0], "NovelElement": [3.0, 4.0]},
        ),
    ],
)
def test_uploaded_overlay_pages_render_shared_controls_and_points(page_name, data):
    app = _run_page(page_name, data)

    assert not app.exception
    expander_labels = [item.label for item in app.expander]
    assert "Uploaded data columns" in expander_labels
    assert "Uploaded marker style" in expander_labels
    assert "Uploaded data quality check" in expander_labels
    dataset_selector = next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    )
    assert "Uploaded data" in dataset_selector.value
    assert any("(Uploaded data: 2)" in value for value in _visible_text(app))
    assert any("Uploaded overlay:" in value for value in _visible_text(app))


def test_mapping_page_uploaded_overlay_with_cartopy():
    pytest.importorskip("cartopy")
    app = _run_page(
        "32_Isotope_Hydrographic_Mapping.py",
        {
            "Longitude_degE": [135.0, 136.0],
            "Latitude_degN": [35.0, 36.0],
            "d18O": [0.1, 0.2],
        },
    )

    assert not app.exception
    expander_labels = [item.label for item in app.expander]
    assert "Uploaded data columns" in expander_labels
    assert "Uploaded marker style" in expander_labels
    dataset_selector = next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    )
    assert "Uploaded data" in dataset_selector.value
    assert any("Uploaded overlay:" in value for value in _visible_text(app))


def test_mapping_contour_uses_uploaded_only_selected_data_without_error():
    pytest.importorskip("cartopy")
    app = _run_page(
        "32_Isotope_Hydrographic_Mapping.py",
        {
            # Two locations intentionally exercise the nearest-neighbour
            # fallback: linear interpolation requires three non-collinear rows.
            "Longitude_degE": [135.0, 136.0],
            "Latitude_degN": [35.0, 36.0],
            "d18O": [0.1, 0.2],
        },
    )
    next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    ).set_value(["Uploaded data"])
    next(item for item in app.radio if item.label == "Map type").set_value(
        "Contour Map"
    )
    app.run(timeout=60)

    assert not app.exception
    assert not any(
        "could not be interpolated" in str(item.value).lower()
        for item in app.warning
    )


def test_quick_visualizer_supports_shared_uploaded_4d_data():
    app = _run_page(
        "05_User_Data_Check_Quick_Visualizer.py",
        {
            "Longitude_degE": [135.0, 136.0, 137.0],
            "Latitude_degN": [35.0, 36.0, 37.0],
            "Depth_m": [10.0, 20.0, 30.0],
            "Salinity": [34.5, 34.7, 34.9],
            "d18O": [0.1, 0.2, 0.3],
            "dD": [1.0, 2.0, 3.0],
            "Temperature_degC": [18.0, 17.0, 16.0],
        },
    )

    assert not app.exception
    assert any("User Data Check & Quick Visualizer" in str(item.value) for item in app.header)
    assert {"✅ Overview & Quality", "📈 2D Explore", "🧊 3D / 4D", "🗺️ 2D Map", "🌍 3D Map", "🗂️ Data & Export"}.issubset(
        {item.label for item in app.tabs}
    )
    assert "Visualization settings" in {item.value for item in app.sidebar.subheader}
    comparison_source = next(
        item for item in app.radio if item.label.startswith("Comparison data source")
    )
    assert comparison_source.value == "None"
    dataset_selector = next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    )
    assert "Uploaded data" in dataset_selector.options
    assert next(item for item in app.sidebar.selectbox if item.label == "X axis").value == "Salinity"
    assert next(item for item in app.sidebar.selectbox if item.label == "Y axis").value == "d18O"
    assert next(item for item in app.sidebar.selectbox if item.label == "Z axis").value == "Depth_m"
    assert next(
        item for item in app.sidebar.selectbox
        if item.label == "Color / fourth dimension (optional)"
    ).value == "Temperature_degC"
    assert len(app.get("plotly_chart")) == 4


def test_quick_visualizer_geographic_scene_has_page04_style_projection(monkeypatch):
    page_path = ROOT / "pages" / "05_User_Data_Check_Quick_Visualizer.py"
    spec = importlib.util.spec_from_file_location("quick_visualizer", page_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.envgeo_utils,
        "load_coastline_data",
        lambda *_args, **_kwargs: ([135.0, 136.0], [35.0, 36.0]),
    )
    data = pd.DataFrame({
        "Longitude_degE": [135.0, 136.0, 137.0],
        "Latitude_degN": [35.0, 36.0, 37.0],
        "Depth_m": [10.0, 50.0, 100.0],
        "Temperature_degC": [18.0, 17.0, 16.0],
    })

    figure, count = module.create_geographic(
        data,
        "Temperature_degC",
        {"color": "#cccccc", "size": 16, "marker": "D", "alpha": 0.9, "outline_width": 1, "outline_color": "#111111"},
        True,
        "None",
    )

    assert count == 3
    assert figure.layout.scene.aspectmode == "manual"
    assert figure.layout.scene.zaxis.autorange is False
    assert figure.layout.scene.aspectratio.z == 0.58
    coastline_traces = {trace.name: trace for trace in figure.data if str(trace.name).startswith("Coastline")}
    assert set(coastline_traces) == {"Coastline (surface)", "Coastline projection"}
    assert coastline_traces["Coastline projection"].line.color == "#8c959b"


def test_quick_visualizer_hover_includes_station_and_oceanographic_fields():
    page_path = ROOT / "pages" / "05_User_Data_Check_Quick_Visualizer.py"
    spec = importlib.util.spec_from_file_location("quick_visualizer_hover", page_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    hover = module.rich_hover_text(pd.DataFrame({
        "Station": ["14_5"], "Cruise": ["KT-01"], "Salinity": [34.56],
        "Temperature_degC": [15.2], "d18O": [0.24], "Custom element": ["present"],
    }))

    assert "Station: 14_5" in hover[0]
    assert "Salinity: 34.56" in hover[0]
    assert "Custom element: present" in hover[0]


@pytest.mark.parametrize(
    ("page_name", "data"),
    [
        (
            "31_Salinity-d18O_Relationship.py",
            {"Salinity": [34.5, 35.0], "d18O": [0.1, 0.2]},
        ),
        (
            "32_Isotope_Hydrographic_Mapping.py",
            {
                "Longitude_degE": [135.0, 136.0],
                "Latitude_degN": [35.0, 36.0],
                "d18O": [0.1, 0.2],
            },
        ),
        (
            "34_T-S_diagram.py",
            {"Salinity": [34.5, 35.0], "Temperature_degC": [20.0, 22.0]},
        ),
        (
            "35_Custom_Parameter_Plot_beta.py",
            {"d18O": [0.1, 0.2], "dD": [1.0, 2.0]},
        ),
    ],
)
def test_uploaded_only_subdataset_does_not_raise(page_name, data):
    if page_name == "32_Isotope_Hydrographic_Mapping.py":
        pytest.importorskip("cartopy")
    app = _run_page(page_name, data)
    next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    ).set_value(["Uploaded data"])
    app.run(timeout=60)

    assert not app.exception


def test_depth_profile_embedded_mode_uses_integrated_upload_owner():
    app = _app_test().from_file(str(ROOT / "pages" / "37_Depth_Profile.py"))
    app.session_state[envgeo_utils.UPLOAD_SESSION_DATA_KEY] = (
        envgeo_utils.prepare_uploaded_data(
            pd.DataFrame({"d18O": [0.1], "Depth_m": [10.0]})
        )
    )
    app.session_state[envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY] = (
        "37_Depth_Profile.py"
    )
    app.run(timeout=45)

    assert not app.exception
    expander_labels = [item.label for item in app.expander]
    assert "Uploaded data overlay" not in expander_labels
    assert "Uploaded data columns" in expander_labels
    assert any(item.label == "Month (optional)" for item in app.selectbox)
    marker_size = next(
        item for item in app.number_input if item.label == "Marker size"
    )
    assert marker_size.value == 10
    assert any(item.label == "Line width" for item in app.number_input)
    assert any(item.label == "Line style" for item in app.selectbox)


def test_depth_profile_draws_when_only_uploaded_subdataset_is_selected():
    app = _run_page(
        "37_Depth_Profile.py",
        {
            "d18O": [0.1, 0.2],
            "Depth_m": [10.0, 20.0],
            "Month": [1, 2],
        },
    )
    next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    ).set_value(["Uploaded data"])
    app.run(timeout=60)

    assert not app.exception
    assert not any(
        "No valid d18O(VSMOW) depth-profile data" in str(item.value)
        for item in app.warning
    )
    assert any("Uploaded overlay: 2 / 2 plotted" in value for value in _visible_text(app))


def test_custom_plot_supports_uploaded_only_numeric_axes():
    app = _run_page(
        "35_Custom_Parameter_Plot_beta.py",
        {
            "Experimental_X": [1.0, 2.0, "bad"],
            "Experimental_Y": [10.0, 20.0, 30.0],
        },
    )

    next(item for item in app.selectbox if item.label == "X axis").set_value(
        "Experimental_X"
    )
    next(item for item in app.selectbox if item.label == "Y axis").set_value(
        "Experimental_Y"
    )
    app.run(timeout=45)

    assert not app.exception
    assert any(
        "Uploaded overlay: 2 / 3 plotted" in value for value in _visible_text(app)
    )


def test_custom_plot_regression_accepts_uploaded_only_selected_rows():
    app = _run_page(
        "35_Custom_Parameter_Plot_beta.py",
        {"d18O": [0.1, 0.2], "dD": [1.0, 2.0]},
    )
    next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    ).set_value(["Uploaded data"])
    next(item for item in app.radio if item.label == "Regression line").set_value(
        "Yes"
    )
    app.run(timeout=60)

    assert not app.exception
    assert not any(
        "Regression line was skipped" in value for value in _visible_text(app)
    )


def test_custom_plot_embedded_mode_uses_integrated_upload_owner():
    app = _app_test().from_file(
        str(ROOT / "pages" / "35_Custom_Parameter_Plot_beta.py")
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_DATA_KEY] = (
        envgeo_utils.prepare_uploaded_data(
            pd.DataFrame({"d18O": [0.1], "dD": [1.0]})
        )
    )
    app.session_state[envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY] = (
        "35_Custom_Parameter_Plot_beta.py"
    )
    app.run(timeout=45)

    assert not app.exception
    expander_labels = [item.label for item in app.expander]
    assert "Uploaded data overlay" not in expander_labels
    assert "Uploaded data columns" in expander_labels


def test_vertical_section_includes_selected_uploaded_subdataset_in_interpolation():
    app = _run_page(
        "53_Vertical_Section_Visualizer.py",
        {
            "Longitude_degE": [124.249483, 144.003],
            "Latitude_degN": [30.500233, 42.573483],
            "Depth_m": [10.0, 20.0],
            "d18O": [0.1, 0.2],
        },
    )
    next(item for item in app.radio if item.label == "A-B input").set_value(
        "Manual"
    )
    endpoint_values = {
        "A lat": 30.500233,
        "A lon": 124.249483,
        "B lat": 42.573483,
        "B lon": 144.003,
    }
    for label, value in endpoint_values.items():
        next(
            item for item in app.number_input if item.label == label
        ).set_value(value)
    app.run(timeout=60)

    assert not app.exception
    expander_labels = [item.label for item in app.expander]
    assert "Uploaded data columns" in expander_labels
    assert "Uploaded marker style" in expander_labels
    assert "Uploaded data quality check" in expander_labels
    dataset_selector = next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    )
    assert "Uploaded data" in dataset_selector.value
    assert any(
        "Uploaded section input: 2 / 2 filtered uploaded rows are included"
        in value for value in _visible_text(app)
    )
    assert any(
        "(Uploaded data: 2)" in value for value in _visible_text(app)
    )
    assert any(
        "Selected d18O:" in value for value in _visible_text(app)
    )
    assert any(
        "Uploaded map overlay: 2 / 2 rows with valid coordinates."
        in value
        for value in _visible_text(app)
    )
    assert any(item.label == "Colormap" for item in app.sidebar.selectbox)
    assert {
        "Colorbar thickness",
        "Colorbar length",
    }.issubset({item.label for item in app.sidebar.slider})
    assert any(
        item.label == "Colorbar font size" for item in app.sidebar.number_input
    )
    assert any(
        item.label == "Colorbar tick count"
        for item in app.sidebar.select_slider
    )
    assert len(app.get("plotly_chart")) == 3


def test_vertical_section_accepts_uploaded_subdataset_without_reference_rows():
    app = _run_page(
        "53_Vertical_Section_Visualizer.py",
        {
            "Longitude_degE": [135.0, 136.0],
            "Latitude_degN": [35.0, 36.0],
            "Depth_m": [10.0, 20.0],
            "d18O": [0.1, 0.2],
        },
    )
    next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    ).set_value(["Uploaded data"])
    next(item for item in app.radio if item.label == "Section mode").set_value(
        "Axis-based"
    )
    app.run(timeout=60)

    assert not app.exception
    assert not any("no data found" in str(item.value).lower() for item in app.warning)
    assert any(
        "Uploaded section input: 2 / 2 filtered uploaded rows are included"
        in value for value in _visible_text(app)
    )


def test_vertical_section_colorbar_uses_neat_horizontal_ticks():
    vertical_section = _vertical_section_module()
    tickvals, ticktext = vertical_section.build_neat_colorbar_ticks(
        -5.0,
        2.0,
        5,
    )
    colorbar = vertical_section.build_section_colorbar(
        "d18O",
        -5.0,
        2.0,
        70,
        20,
        11,
        5,
    )

    assert tickvals == [-4.0, -2.0, 0.0, 2.0]
    assert ticktext == ["-4.0", "-2.0", "0.0", "2.0"]
    assert colorbar["tickangle"] == 0
    assert colorbar["len"] == 0.7
    assert colorbar["thickness"] == 20


def test_vertical_section_draws_uploaded_trace_last():
    vertical_section = _vertical_section_module()
    points = pd.DataFrame(
        {
            "SectionDistance_km": [0.0, 1.0],
            "Depth_m": [10.0, 20.0],
            "d18O": [0.1, 0.2],
            "Longitude_degE": [135.0, 136.0],
            "Latitude_degN": [35.0, 36.0],
            "CrossTrack_km": [0.0, 0.0],
        }
    )
    style = {
        "size": 36,
        "marker": "Diamond",
        "alpha": 1.0,
        "outline_color": "#000000",
        "outline_width": 1.0,
        "color_mode": "Single color",
        "color": "#ff0000",
    }

    figure = vertical_section.create_section_plot(
        z_grid=[[0.1, 0.2], [0.1, 0.2]],
        xi=[0.0, 1.0],
        yi=[0.0, 20.0],
        df_points=points,
        target_col="d18O",
        z_min=-1.0,
        z_max=1.0,
        uploaded_points=points,
        uploaded_style=style,
    )

    assert figure.data[-1].name == "Uploaded data"


def test_vertical_section_sidebar_uses_shared_upload_and_filter_layout():
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.run(timeout=60)

    assert not app.exception
    data_source_radio = next(
        item for item in app.radio if item.label == "Select Data Source:"
    )
    assert data_source_radio.proto.horizontal
    sidebar_expanders = [item.label for item in app.sidebar.expander]
    assert sidebar_expanders[:3] == [
        "Uploaded data overlay",
        "Uploaded data columns",
        "Uploaded marker style",
    ]
    assert "Uploaded data" not in sidebar_expanders
    assert [item.value for item in app.sidebar.header] == ["Data filtering"]
    assert [item.label for item in app.sidebar.multiselect] == [
        "Choose datasets",
        "Cruise / Area / Transect",
    ]
    dataset_selector = next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    )
    assert "Uploaded data" in dataset_selector.options
    assert any(
        item.label == "Area filter preset" for item in app.sidebar.selectbox
    )
    assert any(
        item.label == "Month" for item in app.sidebar.get("button_group")
    )
    assert [item.label for item in app.sidebar.button] == [
        "Apply settings",
        "Apply settings!",
    ]


def test_vertical_section_embedded_mode_uses_integrated_upload_owner():
    app = _app_test().from_file(
        str(ROOT / "pages" / "53_Vertical_Section_Visualizer.py")
    )
    app.session_state[envgeo_utils.UPLOAD_SESSION_DATA_KEY] = (
        envgeo_utils.prepare_uploaded_data(
            pd.DataFrame(
                {
                    "Longitude_degE": [135.0],
                    "Latitude_degN": [35.0],
                    "Depth_m": [10.0],
                    "d18O": [0.1],
                }
            )
        )
    )
    app.session_state[envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY] = (
        "53_Vertical_Section_Visualizer.py"
    )
    app.run(timeout=60)

    assert not app.exception
    expander_labels = [item.label for item in app.expander]
    assert "Uploaded data overlay" not in expander_labels
    assert "Uploaded data columns" in expander_labels
    dataset_selector = next(
        item for item in app.sidebar.multiselect if item.label == "Choose datasets"
    )
    assert "Uploaded data" in dataset_selector.value
