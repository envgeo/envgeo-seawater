from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
# pages/直下はアプリのサイドバーに出るため、安定版、明示的な試作ページ、
# ローカル開発用診断ページだけに保つ。
def test_pages_directory_contains_only_stable_or_explicit_beta_pages():
    page_names = sorted(path.name for path in (ROOT / "pages").glob("*.py"))

    assert page_names == [
        "03_3D_Visualizer.py",
        "04_4D_Visualizer.py",
        "05_3D4D_Visualizer_Uploader.py",
        "31_Salinity-d18O_Relationship.py",
        "32_Isotope_Hydrographic_Mapping.py",
        "34_T-S_diagram.py",
        "35_Custom_Parameter_Plot_beta.py",
        "37_Depth_Profile.py",
        "51_Correlation_Overview.py",
        "53_Vertical_Section_Visualizer.py",
        "90_Integrated_Visualizer_beta.py",
        "99_Environment_Check.py",
    ]


def test_retired_pages_are_not_kept_in_public_source_tree():
    assert not (ROOT / "archived_pages").exists()
