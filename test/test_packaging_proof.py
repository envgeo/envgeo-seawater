"""Focused checks for the installable EnvGeo-Seawater package configuration.

The tests preserve declared runtime dependencies, package data, public launch
commands, and exclusions for development-only or generated files.
実行時依存関係、package data、公開用起動command、開発専用・生成物の除外設定を
確認する。
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
import envgeo_launcher
import envgeo_diagnostic_launcher


def _pyproject():
    with (ROOT / "pyproject.toml").open("rb") as file_obj:
        return tomllib.load(file_obj)


def test_launcher_targets_home_beside_pages_directory():
    app_script = envgeo_launcher.application_script()

    assert app_script == ROOT / "home.py"
    assert app_script.is_file()
    assert (app_script.parent / "pages").is_dir()


def test_package_derives_wheel_dependencies_from_requirements():
    project = _pyproject()["project"]
    dynamic = _pyproject()["tool"]["setuptools"]["dynamic"]

    assert project["name"] == "envgeo-seawater"
    assert "dependencies" not in project
    assert "dependencies" in project["dynamic"]
    assert dynamic["dependencies"] == {"file": ["requirements.txt"]}
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / "runtime.txt").is_file()


def test_wheel_proof_uses_current_root_without_src_migration():
    setuptools_config = _pyproject()["tool"]["setuptools"]

    assert setuptools_config["packages"] == ["envgeo_seawater"]
    assert setuptools_config["package-dir"] == {"envgeo_seawater": "."}
    assert not (ROOT / "src").exists()


def test_package_declares_required_asset_families():
    package_data = _pyproject()["tool"]["setuptools"]["package-data"]
    patterns = set(package_data["envgeo_seawater"])

    assert {
        "CITATION.cff",
        "pages/*.py",
        "coastline/*.csv",
        "coastline/natural_earth_50m_land/*",
        "data/d18O_all.mp4",
        "data/sites_20230515.gif",
        "data/year_20230517.gif",
        "bathymetry/*.nc",
        "data_text/*.md",
        "dataset/*.xlsx",
        "images/*",
        "local_data/user_data.xlsx",
        "tools/*.py",
    } <= patterns


def test_package_excludes_local_and_generated_files():
    exclude_data = _pyproject()["tool"]["setuptools"]["exclude-package-data"]
    patterns = set(exclude_data["envgeo_seawater"])

    assert "pages/99_Environment_Check.py" in patterns
    assert "bathymetry/*.py" in patterns
    assert "docs/wheel_proof_report*.md" in patterns
    assert "**/.DS_Store" in patterns
    assert "**/__pycache__/*" in patterns


def test_installed_launcher_name_is_explicit():
    scripts = _pyproject()["project"]["scripts"]

    assert scripts == {
        "envgeo-seawater": "envgeo_seawater.envgeo_launcher:main",
        "envgeo-seawater-check": "envgeo_seawater.envgeo_diagnostic_launcher:main",
    }


def test_diagnostic_launcher_targets_tool_outside_page_navigation():
    diagnostic = envgeo_diagnostic_launcher.diagnostic_script()

    assert diagnostic == ROOT / "tools" / "env_check_streamlit.py"
    assert diagnostic.is_file()
    assert diagnostic != ROOT / "pages" / "99_Environment_Check.py"
