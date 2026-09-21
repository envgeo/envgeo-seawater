from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import envgeo_utils


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
    assert any("Uploaded overlay:" in value for value in _visible_text(app))


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
    assert any(item.label == "Line width" for item in app.number_input)
    assert any(item.label == "Line style" for item in app.selectbox)


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
