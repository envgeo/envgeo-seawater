#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checks for the public-facing landing page, documents, and page inventory.

公開するHome、文書、ページ構成を確認する。開発作業フォルダだけに許可するページと、
安定版cloneで公開するページを区別する。
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_home_keeps_the_public_landing_identity_and_version_display():
    """Protect the title and introduction shown on the public landing page."""
    home_text = (ROOT / "home.py").read_text(encoding="utf-8")

    assert "st.title('EnvGeo Seawater')" in home_text
    assert "An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data" in home_text
    assert "Interactive 3D/4D Seawater Isotope and Hydrographic Database" in home_text
    assert "Seawater d18O, dD, temperature, salinity, d-excess" in home_text
    assert "Version {envgeo_utils.APP_VERSION_LABEL}" in home_text


# These terms should stay out of public-facing README/app update text for now.
# JOSS関連の話は内部メモに置き、公開向けREADMEやアプリ履歴には出さない方針を確認する。
PUBLIC_SUBMISSION_TERMS = [
    "JOSS",
    "joss",
    "submission",
    "resubmission",
    "re-submission",
    "再投稿",
    "リジェクト",
]


def test_public_readmes_do_not_mention_submission_status():
    public_docs = [
        ROOT / "README.md",
        ROOT / "README_Japanese.md",
        ROOT / "data_text" / "update_log.md",
        ROOT / "data_text" / "update_log_Japanese.md",
    ]

    for path in public_docs:
        text = path.read_text(encoding="utf-8")
        for term in PUBLIC_SUBMISSION_TERMS:
            assert term not in text, f"{term!r} remains in {path.name}"


# Streamlit automatically shows Python files directly under pages/.
# Public pages are required in every repository. Local-only pages are allowed
# in the development workspace but are not required in public repositories.
# pages/直下には公開必須ページと、ローカル開発時だけ許可するページを置く。
def test_pages_directory_contains_only_stable_or_explicit_beta_pages():
    page_names = {path.name for path in (ROOT / "pages").glob("*.py")}
    public_pages = {
        "03_[Interactive]_2Dplus_Visualizer.py",
        "04_[Interactive]_3D_4D_Visualizer.py",
        "31_Salinity-d18O_Relationship.py",
        "32_Isotope_Hydrographic_Mapping.py",
        "34_T-S_diagram.py",
        "35_Custom_Parameter_Plot.py",
        "37_Depth_Profile.py",
        "05_[Utils]_User_Data_Check_Quick_Visualizer.py",
        "06_[Utils]_Data_Overlap_Check.py",
        "80_Correlation_Overview.py",
        "53_Vertical_Section_Visualizer.py",
        "90_Integrated_Visualizer_beta.py",
        "91_EnvGeo_Earthquake.py",
    }
    local_only_pages = {"99_Environment_Check.py"}

    assert public_pages <= page_names
    assert page_names <= public_pages | local_only_pages


def test_retired_pages_are_not_kept_in_public_source_tree():
    assert not (ROOT / "archived_pages").exists()


def test_overlap_check_page_is_a_public_audit_tool_not_an_auto_deleter():
    page_text = (ROOT / "pages" / "06_[Utils]_Data_Overlap_Check.py").read_text(encoding="utf-8")

    assert "Run overlap screen" in page_text
    assert "Download Strong candidate audit CSV" in page_text
    assert "Download Review candidate audit CSV" in page_text
    assert "Download provisional one-to-one candidate CSV" in page_text
    assert 'f"Inspect one {key} candidate pair"' in page_text
    assert 'render_candidate_inspector(strong_audit' in page_text
    assert 'render_candidate_inspector(review_audit' in page_text
    assert '"Provisional one-to-one"' in page_text
    assert "Record a provisional-candidate decision" in page_text
    assert "Download provisional review decisions CSV" in page_text
    assert "Display action after confirmation" in page_text
    assert "Strong candidates (" in page_text
    assert "Review candidates (" in page_text
    assert "Unique {left_source} rows" in page_text
    assert "Source rows loaded for this screen" in page_text
    assert "Review difference category" in page_text
    assert "does not remove records" in page_text
    assert "screen_dataset_pair_for_overlaps" in page_text
    assert "Uploaded data ×" in page_text
    assert "two uploaded tables are not compared" in page_text
    assert "render_upload_panel" in page_text
    assert "How to read the audit table" in page_text
    assert "監査表の読み方" in page_text
    assert "Download English audit-column guide CSV" in page_text
    assert "監査列ガイドCSVをダウンロード（日本語）" in page_text
    assert "Strong and Review candidates" in page_text
    assert "The two classes are disjoint" in page_text
    assert 'layout="centered"' in page_text


def test_shared_sidebar_offers_reference_filter_before_transect_filter():
    """All bundled seawater rows carry a source-reference label."""
    utils_text = (ROOT / "envgeo_utils.py").read_text(encoding="utf-8")

    assert 'with st.expander("Reference / Citation", expanded=False)' in utils_text
    assert '"Reference"' in utils_text
    assert "default=reference_list" in utils_text
    assert utils_text.index("Reference filtering") < utils_text.index("Transect filtering")
