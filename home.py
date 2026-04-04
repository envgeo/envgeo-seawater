
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:00:21 2023

@author: Toyoho Ishimura @Kyoto-U

2026/04/03 update
"""

import streamlit as st
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


# page info
st.set_page_config(
    page_title="Ocean Geochemical Database", 
    # page_icon=image, 
    # layout="wide", 
    initial_sidebar_state="auto", 
    menu_items={
         'Get Help': 'https://envgeo.h.kyoto-u.ac.jp/sw_jpn/',
         'Report a bug': "https://www.h.kyoto-u.ac.jp/en_f/faculty_f/ishimura_toyoho_4dea/#mailform",
         'About': """
         Interactive 3D-4D Seawater Isotope & Geochemical Database
         – Japan Marginal Seas & Global Ocean –
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
            if base_dir and not image_path.startswith(("http://", "https://", "data:")):
                image_path = str((base_dir / image_path).resolve())
            st.image(image_path, caption=caption if caption else None)
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



def main():

    st.title('EnvGeo Seawater')
    st.subheader("An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data")
    st.write('Interactive 3D-4D Seawater Isotope & Geochemical Database – Japan Marginal Seas & Global Ocean –')
    st.write(':blue[seawater isotopes (d18O, dD), temperature, salinity, seasonality, and annual variations around JAPAN]')
    st.write('Version 1.0.1 (2026-03-24)')
    # st.write('Current Version: Version 1.0 _(v220-20260316)_')
    # st.write(':red[NEW!! Mar 18, 2026: MAJOR UPDATE]')




    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["MAIN", "ABOUT", "DATA SOURCES", "MANUAL", "UPDATE LOG", "JAPANESE"])
    
    
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

        st.warning('Note: This application may have limited performance under heavy traffic. If the page fails to load or respond, please refresh your browser or try again after a short while.')

    


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
        
        if readme_file.exists():
            readme_content = read_text_file(readme_file)
        
            with st.expander("Show README"):
                render_markdown_streamlit(readme_content, base_dir=BASE_DIR)
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

        st.header('Update Log')
        
        ###############
        # --- アップデートログ　外部ファイル (update_log.md) の読み込みと実行 ---
        update_log_file = resolve_path('data_text', 'update_log.md')
        render_markdown_file(update_log_file, f"情報: {update_log_file.name} が見つかりません。")
        ###############
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
    
    
