#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an EnvGeo-Seawater runtime and dependency diagnostic report.

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

import importlib
import importlib.metadata
import inspect
import io
import platform
import subprocess
import sys
import textwrap
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


st.set_page_config(page_title="Environment Check")


APP_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV_NAMES = [
    "envgeo_st142_py310_plotly5",
    "envgeo_streamlit134",
]
PDF_FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
]


def stretch_width_kwargs(widget):
    """Return full-width arguments for both old and new Streamlit releases."""
    width_parameter = inspect.signature(widget).parameters.get("width")
    supports_stretch = width_parameter is not None and (
        "Width" in str(width_parameter.annotation)
        or width_parameter.default in {"stretch", "content"}
    )
    if supports_stretch:
        return {"width": "stretch"}
    return {"use_container_width": True}


PACKAGES = [
    {
        "name": "Streamlit",
        "import_name": "streamlit",
        "package_name": "streamlit",
        "required": True,
    },
    {
        "name": "Pandas",
        "import_name": "pandas",
        "package_name": "pandas",
        "required": True,
    },
    {
        "name": "NumPy",
        "import_name": "numpy",
        "package_name": "numpy",
        "required": True,
    },
    {
        "name": "Plotly",
        "import_name": "plotly",
        "package_name": "plotly",
        "required": True,
    },
    {
        "name": "Matplotlib",
        "import_name": "matplotlib",
        "package_name": "matplotlib",
        "required": True,
    },
    {
        "name": "SciPy",
        "import_name": "scipy",
        "package_name": "scipy",
        "required": True,
    },
    {
        "name": "scikit-learn",
        "import_name": "sklearn",
        "package_name": "scikit-learn",
        "required": True,
    },
    {
        "name": "openpyxl",
        "import_name": "openpyxl",
        "package_name": "openpyxl",
        "required": True,
    },
    {
        "name": "gsw",
        "import_name": "gsw",
        "package_name": "gsw",
        "required": True,
    },
    {
        "name": "Cartopy",
        "import_name": "cartopy",
        "package_name": "cartopy",
        "required": True,
    },
    {
        "name": "cmocean",
        "import_name": "cmocean",
        "package_name": "cmocean",
        "required": True,
    },
    {
        "name": "Folium",
        "import_name": "folium",
        "package_name": "folium",
        "required": False,
    },
    {
        "name": "streamlit-folium",
        "import_name": "streamlit_folium",
        "package_name": "streamlit-folium",
        "required": False,
    },
    {
        "name": "streamlit-plotly-events",
        "import_name": "streamlit_plotly_events",
        "package_name": "streamlit-plotly-events",
        "required": False,
    },
    {
        "name": "statsmodels",
        "import_name": "statsmodels",
        "package_name": "statsmodels",
        "required": False,
    },
]


def get_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "-"


def check_package(package):
    try:
        importlib.import_module(package["import_name"])
        status = "OK"
    except Exception as exc:
        status = f"Not available: {type(exc).__name__}"

    return {
        "Package": package["name"],
        "Import name": package["import_name"],
        "Installed version": get_package_version(package["package_name"]),
        "Required": "Yes" if package["required"] else "Optional",
        "Status": status,
    }


def expected_environment_status():
    executable = str(sys.executable)
    matched_env = next((name for name in EXPECTED_ENV_NAMES if name in executable), None)

    if matched_env:
        return "OK", f"Running in the expected Anaconda environment: {matched_env}"
    if "homebrew" in executable.lower():
        return (
            "Warning",
            "Homebrew-managed Python is active. Anaconda packages may not be available.",
        )
    return (
        "Warning",
        "The active Python executable does not match the expected EnvGeo Anaconda environment.",
    )


def get_environment_rows():
    return [
        ("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Python executable", sys.executable),
        ("Python version", sys.version.replace("\n", " ")),
        ("Platform", platform.platform()),
        ("Streamlit module", st.__file__),
        ("App root", str(APP_ROOT)),
        ("Current working directory", str(Path.cwd())),
    ]


def get_package_check_df():
    return pd.DataFrame([check_package(package) for package in PACKAGES])


def get_project_file_check_df():
    paths = [
        APP_ROOT / "envgeo_utils.py",
        APP_ROOT / "requirements.txt",
        APP_ROOT / "dataset",
        APP_ROOT / "data_text",
        APP_ROOT / "pages",
        APP_ROOT / "tools",
    ]
    return pd.DataFrame(
        [
            {
                "Path": str(path.relative_to(APP_ROOT)),
                "Exists": path.exists(),
                "Type": "directory" if path.is_dir() else "file",
            }
            for path in paths
        ]
    )


def build_report_csv(environment_df, package_df, project_df):
    runtime_export = environment_df.copy()
    runtime_export.insert(0, "Section", "Runtime")

    package_export = package_df.copy()
    package_export.insert(0, "Section", "Package Check")

    project_export = project_df.copy()
    project_export.insert(0, "Section", "Project File Check")

    return pd.concat(
        [runtime_export, package_export, project_export],
        ignore_index=True,
        sort=False,
    ).to_csv(index=False).encode("utf-8-sig")


def get_pdf_font_properties():
    for font_path in PDF_FONT_CANDIDATES:
        if font_path.exists():
            return font_manager.FontProperties(fname=str(font_path))
    return font_manager.FontProperties()


def normalize_pdf_text(value):
    return unicodedata.normalize("NFC", str(value))


def add_pdf_text_page(pdf, title, lines):
    font_props = get_pdf_font_properties()
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    fig.text(0.08, 0.95, title, fontsize=16, weight="bold", va="top", fontproperties=font_props)

    y = 0.9
    for line in lines:
        wrapped_lines = textwrap.wrap(normalize_pdf_text(line), width=92) or [""]
        for wrapped_line in wrapped_lines:
            fig.text(0.08, y, wrapped_line, fontsize=8.5, va="top", fontproperties=font_props)
            y -= 0.018
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                fig = Figure(figsize=(8.27, 11.69))
                ax = fig.subplots()
                ax.axis("off")
                y = 0.95

    pdf.savefig(fig, bbox_inches="tight")


def add_pdf_table_page(pdf, title, df, max_rows=32):
    font_props = get_pdf_font_properties()
    fig = Figure(figsize=(11.69, 8.27))
    ax = fig.subplots()
    ax.axis("off")
    fig.text(0.04, 0.95, title, fontsize=15, weight="bold", va="top", fontproperties=font_props)

    display_df = df.head(max_rows).copy().map(normalize_pdf_text)
    table = ax.table(
        cellText=display_df.astype(str).values,
        colLabels=display_df.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    for cell in table.get_celld().values():
        cell.get_text().set_fontproperties(font_props)
        cell.get_text().set_fontsize(7)

    if len(df) > max_rows:
        fig.text(
            0.04,
            0.04,
            f"Showing first {max_rows} rows of {len(df)} rows. Use CSV export for full table.",
            fontsize=8,
            fontproperties=font_props,
        )

    pdf.savefig(fig, bbox_inches="tight")


def build_report_pdf(environment_df, package_df, project_df, environment_message):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        runtime_lines = [
            f"{row['Item']}: {normalize_pdf_text(row['Value'])}"
            for _, row in environment_df.iterrows()
        ]
        runtime_lines.append("")
        runtime_lines.append(f"Environment status: {environment_message}")
        add_pdf_text_page(pdf, "EnvGeo-Seawater Environment Report", runtime_lines)
        add_pdf_table_page(pdf, "Package Check", package_df)
        add_pdf_table_page(pdf, "Project File Check", project_df)

    buffer.seek(0)
    return buffer.getvalue()


def render_environment_summary():
    st.header("1. Runtime")

    environment_df = pd.DataFrame(get_environment_rows(), columns=["Item", "Value"])
    st.dataframe(environment_df, **stretch_width_kwargs(st.dataframe))

    status, message = expected_environment_status()
    if status == "OK":
        st.success(message)
    else:
        st.warning(message)

    return environment_df, message


def render_package_table():
    st.header("2. Package Check")

    result_df = get_package_check_df()
    st.dataframe(
        result_df, hide_index=True, **stretch_width_kwargs(st.dataframe)
    )

    missing_required = result_df[
        (result_df["Required"] == "Yes") & (result_df["Status"] != "OK")
    ]
    if missing_required.empty:
        st.success("All required packages can be imported.")
    else:
        st.error("Some required packages could not be imported.")
        st.dataframe(
            missing_required,
            hide_index=True,
            **stretch_width_kwargs(st.dataframe),
        )

    return result_df


def render_project_file_check():
    st.header("3. Project File Check")

    project_df = get_project_file_check_df()
    st.dataframe(
        project_df, hide_index=True, **stretch_width_kwargs(st.dataframe)
    )

    return project_df


def render_report_downloads(environment_df, package_df, project_df, environment_message):
    st.header("4. Report Export")

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    csv_data = build_report_csv(environment_df, package_df, project_df)
    pdf_data = build_report_pdf(
        environment_df,
        package_df,
        project_df,
        environment_message,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV report",
            data=csv_data,
            file_name=f"envgeo_environment_report_{timestamp}.csv",
            mime="text/csv",
            key="env_check_download_csv_report",
        )
    with col2:
        st.download_button(
            "Download PDF report",
            data=pdf_data,
            file_name=f"envgeo_environment_report_{timestamp}.pdf",
            mime="application/pdf",
            key="env_check_download_pdf_report",
        )


def render_pip_list():
    st.header("5. Installed Packages Inventory")

    if st.button("Display detailed package list", key="env_check_display_pip_list"):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list"],
                capture_output=True,
                check=False,
                text=True,
            )
            if result.stdout:
                st.code(result.stdout, language="text")
            if result.stderr:
                st.warning(result.stderr)
        except Exception as exc:
            st.error(f"Failed to retrieve package list: {exc}")


def main():
    st.title("Environment Diagnostic Tool")
    st.caption("Checks the active Python environment and EnvGeo-Seawater dependencies.")

    environment_df, environment_message = render_environment_summary()
    package_df = render_package_table()
    project_df = render_project_file_check()
    render_report_downloads(environment_df, package_df, project_df, environment_message)
    render_pip_list()


if __name__ == "__main__":
    main()
