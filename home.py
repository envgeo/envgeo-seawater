#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnvGeo-Seawater home page and application overview.

EnvGeo-SeawaterのHome画面とアプリケーション概要を表示する。
"""

import re
from pathlib import Path

import streamlit as st

import envgeo_assets
import envgeo_utils


# =============================================================================
# Page configuration / ページ設定
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent


# =============================================================================
# Online documentation / オンライン利用ガイド
# =============================================================================
# The stable site is the canonical public manual. Development clones may use a
# clone-specific URL when they publish the same reviewed guide for testing.
# 安定版siteを正規の公開マニュアルとする。開発cloneでは、同じ確認済みガイドを
# テスト公開する場合にclone固有のURLへ差し替えられる。
ONLINE_MANUAL_EN_URL = "https://envgeo.github.io/envgeo-seawater/"
ONLINE_MANUAL_JA_URL = "https://envgeo.github.io/envgeo-seawater/index_Japanese.html"

st.set_page_config(
    page_title="EnvGeo Seawater Isotope Database",
    initial_sidebar_state="auto",
    menu_items={
         'Get Help': 'https://envgeo.h.kyoto-u.ac.jp/sw_jpn/',
         'Report a bug': "https://www.h.kyoto-u.ac.jp/en_f/faculty_f/ishimura_toyoho_4dea/#mailform",
         'About': """
         Interactive 3D/4D Seawater Isotope and Hydrographic Database
         – Japan Marginal Seas and Global Ocean
            by T. Ishimura
              https://envgeo.h.kyoto-u.ac.jp/sw_jpn/"""
     })


# =============================================================================
# Markdown and bundled-file rendering / Markdownと同梱ファイルの表示
# =============================================================================
def render_markdown_streamlit(md_text: str, base_dir: Path | None = None) -> None:
    """Render Markdown, resolving standalone local-image lines when available.

    単独行のローカル画像を解決できる場合は画像として、それ以外はMarkdownとして表示する。
    """
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    buffer = []

    for line in md_text.splitlines():
        match = image_pattern.fullmatch(line.strip())

        if match:
            # Render preceding Markdown before the image. / 画像の前のMarkdownを先に表示する。
            if buffer:
                st.markdown("\n".join(buffer), unsafe_allow_html=True)
                buffer = []

            caption, image_path = match.groups()
            if image_path.startswith(("http://", "https://", "data:")):
                st.image(image_path, caption=caption if caption else None)
                continue

            if base_dir:
                resolved_path = (base_dir / image_path).resolve()
                if resolved_path.exists() and resolved_path.is_file():
                    st.image(str(resolved_path), caption=caption if caption else None)
                    continue

            buffer.append(line)
        else:
            buffer.append(line)

    # Render the remaining Markdown. / 残りのMarkdownを表示する。
    if buffer:
        st.markdown("\n".join(buffer), unsafe_allow_html=True)


def resolve_path(*parts: str) -> Path:
    """Resolve a path relative to the application root via ``envgeo_assets``.

    ``envgeo_assets``を通じてアプリケーションrootからのパスを解決する。
    """
    return envgeo_assets.asset_path(*parts, required=False)


def read_text_file(path: Path) -> str:
    """Read a bundled UTF-8 text file. / 同梱UTF-8テキストを読む。"""
    return path.read_text(encoding="utf-8")


def render_markdown_file(file_path: Path, not_found_message: str | None = None) -> None:
    """Render a bundled Markdown file with a safe user-facing failure message.

    同梱Markdownを表示し、読込失敗時は安全な利用者向けメッセージを出す。
    """
    if not file_path.exists():
        st.info(not_found_message or f"Error: {file_path.name} not found.")
        return

    try:
        render_markdown_streamlit(read_text_file(file_path), base_dir=BASE_DIR)
    except Exception:
        st.error(f"Could not load {file_path.name}.")


def render_external_link(label: str, url: str) -> None:
    """Render an external link with a Streamlit-version fallback.

    Streamlitの版によるfallbackを備えて外部リンクを表示する。
    """
    if hasattr(st, "link_button"):
        st.link_button(label, url)
    else:
        st.markdown(f"[{label}]({url})")


# =============================================================================
# Home display styles and update history / Home表示スタイルと更新履歴
# =============================================================================
def render_tab_style() -> None:
    """
    Render compact, readable tabs for the Home page.

    Home画面のタブを、境界と選択状態が分かりやすい表示に整えます。
    """
    st.markdown(
        """
        <style>
        div[data-baseweb="tab-list"] {
            gap: 0.25rem;
            flex-wrap: wrap;
            border-bottom: 1px solid rgba(49, 51, 63, 0.18);
            padding-bottom: 0;
        }
        div[data-baseweb="tab-list"] button[role="tab"] {
            background: rgba(248, 249, 250, 0.95);
            color: #1f2937;
            border: 1px solid rgba(49, 51, 63, 0.20);
            border-bottom-color: rgba(49, 51, 63, 0.12);
            border-radius: 6px 6px 0 0;
            padding: 0.38rem 0.72rem;
            min-height: 2.15rem;
            white-space: nowrap;
            font-weight: 600;
        }
        div[data-baseweb="tab-list"] button[role="tab"] p {
            margin: 0;
            color: inherit;
        }
        div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #e8f2ff 0%, #ddeaff 100%);
            border-color: #4a90e2;
            border-bottom-color: #ddeaff;
            color: #0b3e75;
            box-shadow: inset 0 0 0 1px rgba(74, 144, 226, 0.35);
        }
        html[data-theme="dark"] div[data-baseweb="tab-list"] {
            border-bottom-color: rgba(240, 244, 250, 0.18);
        }
        html[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"],
        body[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"] {
            background: rgba(44, 49, 61, 0.96);
            color: rgba(245, 247, 250, 0.95);
            border-color: rgba(240, 244, 250, 0.26);
        }
        html[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"],
        body[data-theme="dark"] div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #204061 0%, #1a314a 100%);
            color: #e9f2ff;
            border-color: #76adff;
            box-shadow: inset 0 0 0 1px rgba(118, 173, 255, 0.42);
        }
        @media (prefers-color-scheme: dark) {
            div[data-baseweb="tab-list"] {
                border-bottom-color: rgba(240, 244, 250, 0.18);
            }
            div[data-baseweb="tab-list"] button[role="tab"] {
                background: rgba(44, 49, 61, 0.96);
                color: rgba(245, 247, 250, 0.95);
                border-color: rgba(240, 244, 250, 0.26);
            }
            div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
                background: linear-gradient(180deg, #204061 0%, #1a314a 100%);
                color: #e9f2ff;
                border-color: #76adff;
                box-shadow: inset 0 0 0 1px rgba(118, 173, 255, 0.42);
            }
        }
        @media (max-width: 900px) {
            div[data-baseweb="tab-list"] button[role="tab"] {
                font-size: 0.86rem;
                padding: 0.32rem 0.54rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_update_history() -> None:
    """Render the concise public update history. / 簡潔な公開更新履歴を表示する。"""
    st.markdown(
        """
### Version 1.3.4 (2026-10-03)

- Released the installable stable version through PyPI and the v1.3.4 GitHub Release; the archived version DOI is <https://doi.org/10.5281/zenodo.23117784>.
- Verified the test suite, wheel build, and isolated wheel installation on Python 3.10 and 3.12 in CI.
- Retained the complete current cited dataset collection with documented provenance and source-to-workbook transformations.
- Published figure-supported English and Japanese user guides through GitHub Pages; the Manual and Japanese tabs link to the stable guide.

### Post-v1.3.3 maintenance updates (2026-09-24)

- Clarified that T–S density contours are approximate σ0 reference contours, and added selectable contour intervals. A full TEOS-10 (SA–CT) calculation mode remains planned work.
- Added an optional approximate σ0 contour overlay pilot to the interactive T–S view on page 03, without changing its existing point colour, hover, or selection workflow.

### Version 1.3.3 (2026-09-23)

- Added offline-safe Plotly map behaviour: local coastline overlays remain visible when web tiles are unavailable, while `Coastline (offline)` provides a tile-free white-background map.
- Bundled Natural Earth 50m land polygons for page 32 static Cartopy maps, avoiding first-run Natural Earth downloads while retaining correct land masking.
- Added self-contained interactive HTML downloads for the main Plotly figures on pages 03, 04, and 05. Saved figures embed Plotly.js and retain zoom, mode-bar controls, and responsive sizing; online basemap tiles are not bundled.
- Completed the persistent `User Excel data` workflow and clarified data origin in Quick Visualizer hover information.
- Kept Vertical Section as a beta workflow with documented scientific and offline-input limitations.

### Version 1.3.2 (2026-09-22)

- Added shared, in-memory CSV/XLSX user-data upload and filtering for User Data Check and compatible specialist workflows. `Uploaded data` can be selected in Data filtering on supported pages; selected uploads are used in compatible plotting and calculation workflows and are drawn in the foreground.
- Added `User Data Check & Quick Visualizer` as the public upload-first page for quality review, missing-value checks, shared filtering, 2D/3D/4D exploration, 2D maps, geographic 3D, and filtered CSV export.
- Improved Vertical Section Visualizer with shared upload filtering, uploaded-data section inputs, target-parameter availability summaries, and configurable readable colorbars. Its interpolation outputs remain experimental.
- Updated tab styling for Streamlit 1.63, kept the Plotly 5.24 baseline, and expanded targeted regression and AppTest coverage.
- Replaced the fixed legacy workbook with a configurable always-loaded table labeled `User Excel data`; the bundled zero-value public sample is used by default, while researcher-owned data are selected from an external path. Browser uploads remain separate.

### Version 1.3.1 (2026-09-19)

- Added compatibility handling for Python 3.10-3.12 and Streamlit 1.42-1.63 migration testing.
- Preserved the Plotly 5.24 baseline while preparing a staged Mapbox-to-MapLibre migration for Plotly 7.
- Began the shared, memory-only user-data upload rollout for individual visualization pages.

### Version 1.3.0 (2026-09-11)

- Cleaner interface wording, page titles, and figure controls.
- Expanded parameter plotting for d18O, dD, d-excess, salinity, temperature, depth, latitude, and longitude.
- Improved map display with region presets, API-key-free basemaps, and cmocean/EnvGeo colormap options.
- Added user-data upload support in the integrated beta workflow, including overlay styling and quality summaries.
- Added shared quality checks, d-excess calculation, filename handling, and filtered-data summaries.
- Expanded pytest coverage for core utilities, data loading, and public app structure.

### Earlier Versions

- `1.0.1` (2026-03-24): Improved the stable public Streamlit app structure and documentation.
- `1.0.0` (2026-03-18): Updated the main seawater isotope and hydrographic visualization workflows.
- `0.2.0` (2026-02-18): Improved the pre-1.0 integrated app structure.
- `b20` (2024-12-14): Added Excel upload support, custom plotting, and additional datasets.
- `b03` (2023-05-22): Initial pre-release version.
        """
    )



# =============================================================================
# Home-page interface / Home画面
# =============================================================================
def main() -> None:
    """Render the EnvGeo-Seawater Home page. / EnvGeo-SeawaterのHome画面を表示する。"""
    st.title('EnvGeo Seawater')
    st.subheader("An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data")
    st.write("Interactive 3D/4D Seawater Isotope and Hydrographic Database – Japan Marginal Seas and Global Ocean")
    st.write(":blue[Seawater d18O, dD, temperature, salinity, d-excess, and seasonal to interannual variations]")
    st.markdown(
        f"Version {envgeo_utils.APP_VERSION_LABEL} · "
        "[DOI: 10.5281/zenodo.23117784](https://doi.org/10.5281/zenodo.23117784)"
    )
    render_tab_style()
    envgeo_utils.render_card_tab_style()
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🏠 Main", "ℹ️ About", "📚 Data Sources", "📖 Manual", "📝 Updates", "🇯🇵 日本語"]
    )
    
    
    # -------------------------------------------------------------------------
    # Main tab / メインタブ
    # -------------------------------------------------------------------------
    with tab1:
        def display_autoplay_video(video_path_or_url):
            """Render a bundled or remote looping video. / 同梱または外部のループ動画を表示する。"""
            import base64

            # Resolve a bundled path unless the value is a URL. / URL以外は同梱パスとして解決する。
            if not video_path_or_url.startswith(('http://', 'https://')):
                video_path = resolve_path(video_path_or_url)

                if not video_path.exists():
                    st.error(f"動画ファイルが見つかりません: {video_path_or_url}")
                    return
                
                with open(video_path, "rb") as f:
                    data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    video_url = f"data:video/mp4;base64,{bin_str}"
            else:
                video_url = video_path_or_url
        
            # Use HTML to request muted autoplay. / muted自動再生を指定するためHTMLを使う。
            video_html = f'''
                <video width="100%" autoplay loop muted playsinline>
                    <source src="{video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            '''
            st.markdown(video_html, unsafe_allow_html=True)


        target_video = "data/d18O_all.mp4"
        display_autoplay_video(target_video)

        st.markdown("<h6 style='text-align: center; color: grey;'>Animation created with GMT (The Generic Mapping Tools)</h6>", unsafe_allow_html=True)

        st.markdown(
            """
### Start exploring

- **Core dataset:** explore the multi-year Kodama et al. (2024) seawater-isotope dataset centred on the East China Sea and Japan Sea.
- **Reference datasets:** compare it with cited regional and global datasets, including NASA GISS and PAGES CoralHydro2k.
- **Visualisation:** use the sidebar to open maps, 3D/4D views, T--S diagrams, depth profiles, and other specialist tools.
- **Browser uploads:** use **User Data Check Quick Visualizer** to inspect and plot an uploaded CSV/XLSX file for the current browser session. Upload overlays are also available on the salinity--d18O, mapping, T--S, custom-parameter, depth-profile, and vertical-section pages; 2Dplus and 3D 4D do not currently accept browser uploads.
- **Detailed guidance:** open the **Manual** tab for a quick guide and links to the page-by-page English and Japanese manuals.
            """
        )

    


    # -------------------------------------------------------------------------
    # About tab / Aboutタブ
    # -------------------------------------------------------------------------
    with tab2:
        st.header("About")
        about_file = resolve_path('data_text', 'about.md')
        render_markdown_file(about_file)
        st.header("README")

        readme_file = resolve_path("README.md")
        japanese_readme_file = resolve_path("README_Japanese.md")
        
        if readme_file.exists():
            readme_content = read_text_file(readme_file)
            # Streamlit treats relative links as browser URLs; render the bundled Japanese README below.
            # Streamlitは相対linkをbrowser URLとして扱うため、同梱の日本語READMEを下に表示する。
            readme_content = readme_content.replace(
                "[日本語版 README](README_Japanese.md)",
                "日本語版は下の「日本語版 README」を開いてください。",
            )
        
            with st.expander("Show README"):
                render_markdown_streamlit(readme_content, base_dir=BASE_DIR)

        if japanese_readme_file.exists():
            with st.expander("日本語版 README"):
                render_markdown_streamlit(
                    read_text_file(japanese_readme_file), base_dir=BASE_DIR
                )
        st.divider()
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    
    

    # -------------------------------------------------------------------------
    # Data-sources tab / データソースタブ
    # -------------------------------------------------------------------------
    with tab3:
        st.header("Data Sources")
        ref_file_main = resolve_path('data_text', 'main_references.md')
        render_markdown_file(ref_file_main)
        ref_file_others = resolve_path('data_text', 'other_references.md')
        render_markdown_file(ref_file_others)
        st.divider()
        st.markdown("## :red[External Reference Databases for Comparison]")
        st.caption("NASA GISS and PAGES CoralHydro2k are cited third-party datasets used for comparison; see the source records below.")
        st.subheader("PAGES CoralHydro2k Seawater Oxygen Isotope Database")
        with st.expander("View PAGES CoralHydro2k source record", expanded=False):
            render_markdown_file(
                resolve_path('data_text', 'CoralHydro2_references.md'),
                "The reference list file cannot be found. Please visit https://doi.org/10.25921/ap7d-2k16",
            )


        st.divider()
        st.subheader("NASA GISS Global Seawater Oxygen Isotope Database")
        with st.expander("View NASA GISS source references", expanded=False):
            render_markdown_file(
                resolve_path('data_text', 'NASA_references.md'),
                "The reference list file cannot be found. Please visit https://data.giss.nasa.gov/o18data/ref.html",
            )
        st.divider()
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    
    
        
    # -------------------------------------------------------------------------
    # Manual tab / マニュアルタブ
    # -------------------------------------------------------------------------
    with tab4:
        st.header("User Manual")
        st.markdown(
            "**For a detailed, figure-based, page-by-page guide, visit the "
            f"[English user guide]({ONLINE_MANUAL_EN_URL}) or the "
            f"[日本語利用ガイド]({ONLINE_MANUAL_JA_URL}).**"
        )
        st.divider()
        st.subheader("Quick guide / 簡易ガイド")
        manual_file = resolve_path("data_text", "manual.md")
        render_markdown_file(manual_file, f"Information: {manual_file.name} was not found.")
        manual_japanese_file = resolve_path("data_text", "manual_Japanese.md")
        with st.expander("日本語版マニュアル"):
            render_markdown_file(
                manual_japanese_file,
                f"情報: {manual_japanese_file.name} が見つかりません。",
            )
        st.video(
            'https://envgeo.h.kyoto-u.ac.jp/wp-content/uploads/2024/10/envgeo20241016-HD-720p.mp4',
            format="video/mp4",
            start_time=0,
            end_time=None,
            loop=True,
        )
        st.divider()
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    

        
    # -------------------------------------------------------------------------
    # Updates tab / 更新履歴タブ
    # -------------------------------------------------------------------------
    with tab5:
        st.header("Update History")
        render_update_history()

        update_log_file = resolve_path('data_text', 'update_log.md')
        update_log_ja_file = resolve_path('data_text', 'update_log_Japanese.md')
        with st.expander("Detailed update log (English)"):
            render_markdown_file(update_log_file, f"Information: {update_log_file.name} was not found.")
        with st.expander("Detailed update log (Japanese)"):
            render_markdown_file(update_log_ja_file, f"Information: {update_log_ja_file.name} was not found.")

        st.divider()
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    

    



    # -------------------------------------------------------------------------
    # Japanese tab / 日本語タブ
    # -------------------------------------------------------------------------
    with tab6:
        st.header("EnvGeo-Seawaterについて")
        st.markdown(
            "**図付きの詳細なページ別操作手順は、"
            f"[オンライン日本語利用ガイド]({ONLINE_MANUAL_JA_URL})をご覧ください。**"
        )
        st.divider()
        japanese_file = resolve_path("data_text", "japanese.md")
        render_markdown_file(japanese_file, f"情報: {japanese_file.name} が見つかりません。")
        st.divider()
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    


    

    
if __name__ == "__main__":
    main()
    
