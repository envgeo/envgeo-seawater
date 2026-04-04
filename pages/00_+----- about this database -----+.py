#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:15:03 2023
@author: Toyoho Ishimura @Kyoto-U
2026/03/11 update 
"""

import streamlit as st
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def render_markdown_streamlit(md_text: str, base_dir: Path | None = None) -> None:
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    buffer = []

    for line in md_text.splitlines():
        match = image_pattern.fullmatch(line.strip())

        if match:
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

    if buffer:
        st.markdown("\n".join(buffer), unsafe_allow_html=True)


def resolve_path(*parts: str) -> Path:
    return BASE_DIR.joinpath(*parts)


def render_markdown_file(file_path: Path, not_found_message: str | None = None) -> None:
    if not file_path.exists():
        st.info(not_found_message or f"Error: {file_path.name} not found.")
        return

    try:
        render_markdown_streamlit(file_path.read_text(encoding="utf-8"), base_dir=BASE_DIR)
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {e}")


def render_external_link(label: str, url: str) -> None:
    if hasattr(st, "link_button"):
        st.link_button(label, url)
    else:
        st.markdown(f"[{label}]({url})")

def main():
    #タブページ
    st.title('EnvGeo Seawater')
    st.subheader("An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data")
    st.write('Interactive 3D-4D Seawater Isotope & Geochemical Database – Japan Marginal Seas & Global Ocean –')
    st.write(':blue[seawater isotopes (d18O, dD), temperature, salinity, seasonality, and annual variations around JAPAN]')
    st.write('Current Version: v1.0.1')
    # st.write(':red[NEW!! Mar 18, 2026: MAJOR UPDATE]')
    # st.write('<span style="color:red;background:white">NEW!!</span> NEW',unsafe_allow_html=True)
    # st.write("This is :blue[test]")
    # st.warning('This site is not able to support multiple simultaneous accesses, so if the display does not work, please reload the page or try again after awhile.')
    # st.warning('Note: This application may have limited performance under heavy traffic. If the page fails to load or respond, please refresh your browser or try again after a short while.')

        
    tab1, tab2= st.tabs(["English", "日本語"])
    
    
    
    


    ##############################################################
    with tab1:
       st.header("about")

       
       ###############
       # 外部ファイル (about.md) の読み込みと実行 
       about_file = resolve_path('data_text', 'about.md')
       render_markdown_file(about_file)
       ###############
           
       ###############
       # データソース読み込み
       st.write('_____')
       st.header('data sources')
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
    with tab2:
        # st.title('SEAWATER DATA JAPAN (b01)')



        ###############
        # --- 日本語簡易説明　外部ファイル (japanese.md) の読み込みと実行 ---
        japanese_file = resolve_path('data_text', 'japanese.md')
        render_markdown_file(japanese_file, f"情報: {japanese_file.name} が見つかりません。")
        ###############

    ##############################################################

    
    
        st.header("about")
        # st.header('by TOYOHO ISHIMURA')
        
        ###############
        # --- 冒頭のabout　外部ファイル (about.md) の読み込みと実行 ---
        about_file = resolve_path('data_text', 'about.md')
        render_markdown_file(about_file, f"情報: {about_file.name} が見つかりません。")
        ###############

        # 純粋なテキスト
        st.write('_____')
        st.header('data sources')

        
        ###############
        # --- メイン引用文献　外部ファイル (main_references.md) の読み込みと実行 ---
        ref_file_main = resolve_path('data_text', 'main_references.md')
        render_markdown_file(ref_file_main, f"情報: {ref_file_main.name} が見つかりません。")
        ###############

        
        ###############
        # --- その他の引用文献　外部ファイル (other_references.md) の読み込みと実行 ---
        ref_file_others = resolve_path('data_text', 'other_references.md')
        render_markdown_file(ref_file_others, f"情報: {ref_file_others.name} が見つかりません。")
        ###############

        
        st.write('  ')

        st.subheader(':red[Global Database Integration]')
        st.subheader('NASA GISS Global Seawater Oxygen Isotope Database')
        st.write('https://data.giss.nasa.gov/o18data/')
        
        
        ###############
        # --- NASAデータベースの引用文献　外部ファイル(テキスト/Markdown)からの読み込み ---
        render_markdown_file(
            resolve_path('data_text', 'NASA_references.md'),
            "The reference list file cannot be found. Please visit https://data.giss.nasa.gov/o18data/ref.html",
        )
        ###############

        st.write('_____')
        render_external_link("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    
    
    
    
   


if __name__ == '__main__':
    main()
