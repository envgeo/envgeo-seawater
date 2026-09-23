#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnvGeo-Seawater home page and application overview.

Created: 2023-05-21
Author: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-23
"""

import streamlit as st
import re
from pathlib import Path

import envgeo_utils


BASE_DIR = Path(__file__).resolve().parent


# page info
st.set_page_config(
    page_title="EnvGeo Seawater Isotope Database", 
    # page_icon=image, 
    # layout="wide", 
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


# to show markdown files with local images
def render_markdown_streamlit(md_text: str, base_dir: Path | None = None) -> None:
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    buffer = []

    for line in md_text.splitlines():
        match = image_pattern.fullmatch(line.strip())

        if match:
            # それまでのMarkdownを先に表示
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

    # 残りを表示
    if buffer:
        st.markdown("\n".join(buffer), unsafe_allow_html=True)


def resolve_path(*parts: str) -> Path:
    return BASE_DIR.joinpath(*parts)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_markdown_file(file_path: Path, not_found_message: str | None = None) -> None:
    if not file_path.exists():
        st.info(not_found_message or f"Error: {file_path.name} not found.")
        return

    try:
        render_markdown_streamlit(read_text_file(file_path), base_dir=BASE_DIR)
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {e}")


def render_external_link(label: str, url: str) -> None:
    if hasattr(st, "link_button"):
        st.link_button(label, url)
    else:
        st.markdown(f"[{label}]({url})")


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
    st.markdown(
        """
### Version 1.3.3 (2026-09-23)

- Added offline-safe Plotly map behaviour: local coastline overlays remain visible when web tiles are unavailable, while `Coastline (offline)` provides a tile-free white-background map.
- Bundled Natural Earth 50m land polygons for page 32 static Cartopy maps, avoiding first-run Natural Earth downloads while retaining correct land masking.
- Added self-contained interactive HTML downloads for the main Plotly figures on pages 03, 04, and 05. Saved figures embed Plotly.js and retain zoom, mode-bar controls, and responsive sizing; online basemap tiles are not bundled.
- Completed the persistent `User Excel data` workflow and clarified data origin in Quick Visualizer hover information.
- Kept Vertical Section as a beta workflow with documented scientific and offline-input limitations.

### Version 1.3.2 (2026-09-22)

- Added shared, in-memory CSV/XLSX user-data upload and filtering to the active specialist pages. `Uploaded data` can be selected in Data filtering; selected uploads are used in compatible plotting and calculation workflows and are drawn in the foreground.
- Added `User Data Check & Quick Visualizer` as the public upload-first page for quality review, missing-value checks, shared filtering, 2D/3D/4D exploration, 2D maps, geographic 3D, and filtered CSV export.
- Improved Vertical Section Visualizer with shared upload filtering, uploaded-data section inputs, target-parameter availability summaries, and configurable readable colorbars. Its interpolation outputs remain experimental.
- Updated tab styling for Streamlit 1.63, kept the Plotly 5.24 baseline, and expanded targeted regression and AppTest coverage.
- Replaced the fixed legacy workbook with a configurable, Git-ignored always-loaded local table labeled `User Excel data`; it is appended to each selected reference source while browser uploads remain separate.

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



def main():

    st.title('EnvGeo Seawater')
    
    # st.title(':red[Unpublished version]')
    # st.title(':red[for internal use only]')
    
    st.subheader("An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data")
    st.write('Interactive 3D/4D Seawater Isotope and Hydrographic Database – Japan Marginal Seas and Global Ocean')
    st.write(':blue[Seawater d18O, dD, temperature, salinity, d-excess, and seasonal to interannual variations]')
    st.write(f'Version {envgeo_utils.APP_VERSION_LABEL}')
    # st.write('Current Version: Version 1.0 _(v220-20260316)_')
    # st.write(':red[NEW!! Mar 18, 2026: MAJOR UPDATE]')




    render_tab_style()
    envgeo_utils.render_earthquake_tab_style()
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Main", "About", "Data Sources", "Manual", "Updates", "Japanese"]
    )
    
    
    ##############################################################
    with tab1:
        def display_autoplay_video(video_path_or_url):
            import base64
            
            # 動画ファイルが存在するか、URLであるかを確認
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
        
            # HTMLで自動再生設定を書き込む
            video_html = f'''
                <video width="100%" autoplay loop muted playsinline>
                    <source src="{video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            '''
            st.markdown(video_html, unsafe_allow_html=True)


        target_video = 'data/d18O_all.mp4' 
        display_autoplay_video(target_video)

        st.markdown("<h6 style='text-align: center; color: grey;'>Animation created with GMT (The Generic Mapping Tools)</h6>", unsafe_allow_html=True)

        st.markdown(
            """
### Start Exploring

- **Core dataset:** explore the highly comparable Japan-region seawater isotope dataset.
- **Reference datasets:** expand the view to regional and global comparison datasets.
- **Visualization pages:** use the sidebar to open maps, 3D/4D views, T-S diagrams, depth profiles, and beta tools.
- **User data:** upload your own seawater dataset for temporary, session-only comparison.
            """
        )

    


    ##############################################################
    with tab2:


        st.header("About")

        
        ###############
        # 外部ファイル (about.md) の読み込みと実行 
        about_file = resolve_path('data_text', 'about.md')
        render_markdown_file(about_file)
        ###############
            
        ###############
        # データソース読み込み
        st.header('Read me')
        ###############
        

        ###############

        readme_file = resolve_path("README.md")
        japanese_readme_file = resolve_path("README_Japanese.md")
        
        if readme_file.exists():
            readme_content = read_text_file(readme_file)
            # Relative Markdown links are interpreted as browser URLs in
            # Streamlit. Show the bundled Japanese README below instead.
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
        ###############


        st.write('_____')
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    
    

    ##############################################################
    with tab3:

            
        ###############
        # データソース読み込み
        st.header('Data Sources')
        ###############
        
        ###############
        # --- メイン引用文献　外部ファイル (main_references.md) の読み込みと実行 ---
        ref_file_main = resolve_path('data_text', 'main_references.md')
        render_markdown_file(ref_file_main)
        ###############

        
        ###############
        # --- その他の引用文献　外部ファイル (other_references.md) の読み込みと実行 ---
        ref_file_others = resolve_path('data_text', 'other_references.md')
        render_markdown_file(ref_file_others)
           
           
        ###############
        st.write('  ')

        st.subheader(':red[Global Database Integration]')
        
        ###############

        st.subheader('CoralHydro2 Seawater Oxygen Isotope Database')
        
        # --- CoralHydro2データベースの引用文献　外部ファイル(テキスト/Markdown)からの読み込み ---
        render_markdown_file(
            resolve_path('data_text', 'CoralHydro2_references.md'),
            "The reference list file cannot be found. Please visit https://doi.org/10.25921/ap7d-2k16",
        )


        ###############

        
        st.write('  ')

        st.subheader('NASA GISS Global Seawater Oxygen Isotope Database')
        
        # --- NASAデータベースの引用文献　外部ファイル(テキスト/Markdown)からの読み込み ---
        render_markdown_file(
            resolve_path('data_text', 'NASA_references.md'),
            "The reference list file cannot be found. Please visit https://data.giss.nasa.gov/o18data/ref.html",
        )
        ###############
        
        





        st.write('_____')
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    
    
        
    ##############################################################
    with tab4:

        st.header('User Manual')
        
        ###############
        # --- その他の引用文献　外部ファイル (other_references.md) の読み込みと実行 ---
        manual_file = resolve_path('data_text', 'manual.md')
        render_markdown_file(manual_file, f"情報: {manual_file.name} が見つかりません。")
        st.video(
            'https://envgeo.h.kyoto-u.ac.jp/wp-content/uploads/2024/10/envgeo20241016-HD-720p.mp4',
            format="video/mp4",
            start_time=0,
            end_time=None,
            loop=True,
        )
        ###############
        st.write('_____')
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    

        
    ##############################################################
    with tab5:

        st.header('Update History')
        render_update_history()

        update_log_file = resolve_path('data_text', 'update_log.md')
        update_log_ja_file = resolve_path('data_text', 'update_log_Japanese.md')
        with st.expander("Detailed update log (English)"):
            render_markdown_file(update_log_file, f"情報: {update_log_file.name} が見つかりません。")
        with st.expander("Detailed update log (Japanese)"):
            render_markdown_file(update_log_ja_file, f"情報: {update_log_ja_file.name} が見つかりません。")

        st.write('_____')
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    

    



    ##############################################################
    with tab6:


        st.header('このデータベースとwebアプリについて')
        
        ###############
        # --- 日本語簡易説明　外部ファイル (japanese.md) の読み込みと実行 ---
        japanese_file = resolve_path('data_text', 'japanese.md')
        render_markdown_file(japanese_file, f"情報: {japanese_file.name} が見つかりません。")
        ###############
        st.write('_____')
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    


    

    
if __name__ == '__main__':
    main()
    
    
