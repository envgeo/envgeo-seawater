#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:15:03 2023
@author: Toyoho Ishimura @Kyoto-U
2026/03/11 update 
"""

import streamlit as st
import os

def main():
    #タブページ
    st.title('EnvGeo Seawater')
    st.subheader("An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data")
    st.write('Interactive 3D-4D Seawater Isotope & Geochemical Database – Japan Marginal Seas & Global Ocean –')
    st.write(':blue[seawater isotopes (d18O, dD), temperature, salinity, seasonality, and annual variations around JAPAN]')
    st.write('Current Version: v1.0.0')
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
       about_file = 'data_text/about.md'

       if os.path.exists(about_file):
           try:
               # ファイルを UTF-8 で読み込み
               with open(about_file, 'r', encoding='utf-8') as f:
                   ref_content = f.read()
               
               # 読み込んだテキスト（st.writeなど）を Python コードとして実行して表示
               exec(ref_content)
               
           except Exception as e:
               st.error(f"Error loading {about_file}: {e}")
       else:
           st.info(f"Error: {about_file} not found.")
       ###############
           
       ###############
       # データソース読み込み
       st.write('_____')
       st.header('data sources')
       ###############
       
       ###############
       # --- メイン引用文献　外部ファイル (main_references.md) の読み込みと実行 ---
       ref_file_main = 'data_text/main_references.md'

       if os.path.exists(ref_file_main):
           try:
               # ファイルを UTF-8 で読み込み
               with open(ref_file_main, 'r', encoding='utf-8') as f:
                   ref_content = f.read()
               
               # 読み込んだテキスト（st.writeなど）を Python コードとして実行して表示
               exec(ref_content)
               
           except Exception as e:
               st.error(f"Error loading {ref_file_main}: {e}")
       else:
          st.info(f"Error: {about_file} not found.")
       ###############

       
       ###############
       # --- その他の引用文献　外部ファイル (other_references.md) の読み込みと実行 ---
       ref_file_others = 'data_text/other_references.md'

       if os.path.exists(ref_file_others):
           try:
               # ファイルを UTF-8 で読み込みます
               with open(ref_file_others, 'r', encoding='utf-8') as f:
                   ref_content = f.read()
               
               # 読み込んだテキスト（st.writeなど）を Python コードとして実行して表示します
               exec(ref_content)
               
           except Exception as e:
               st.error(f"Error loading {ref_file_others}: {e}")
       else:
          st.info(f"Error: {about_file} not found.")
          
          
       ###############
       st.write('  ')

       st.subheader(':red[Global Database Integration]')
       
       ###############

       st.subheader('CoralHydro2 Seawater Oxygen Isotope Database')
       
       # --- CoralHydro2データベースの引用文献　外部ファイル(テキスト/Markdown)からの読み込み ---
       try:
           with open('data_text/CoralHydro2_references.md', 'r', encoding='utf-8') as f:
               nasa_refs = f.read()
           st.markdown(nasa_refs, unsafe_allow_html=True)
       except FileNotFoundError:
           st.error("The reference list file cannot be found. Please visit https://doi.org/10.25921/ap7d-2k16")


       ###############

       
       st.write('  ')

       st.subheader('NASA GISS Global Seawater Oxygen Isotope Database')
       
       # --- NASAデータベースの引用文献　外部ファイル(テキスト/Markdown)からの読み込み ---
       try:
           with open('data_text/NASA_references.md', 'r', encoding='utf-8') as f:
               nasa_refs = f.read()
           st.markdown(nasa_refs, unsafe_allow_html=True)
       except FileNotFoundError:
           st.error("The reference list file cannot be found. Please visit https://data.giss.nasa.gov/o18data/ref.html")
       ###############
       
       





       st.write('_____')
       st.link_button("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    
   
   
   

    ##############################################################
    with tab2:
        # st.title('SEAWATER DATA JAPAN (b01)')



        ###############
        # --- 日本語簡易説明　外部ファイル (japanese.md) の読み込みと実行 ---
        japanese_file = 'data_text/japanese.md'

        if os.path.exists(japanese_file):
            try:
                # ファイルを UTF-8 で読み込みます
                with open(japanese_file, 'r', encoding='utf-8') as f:
                    ref_content = f.read()
                
                # 読み込んだテキスト（st.writeなど）を Python コードとして実行して表示します
                exec(ref_content)
                
            except Exception as e:
                st.error(f"Error loading {japanese_file}: {e}")
        else:
            # ファイルがない場合の予備表示（必要なければ消して構いません）
            # st.info(f"情報: {ref_file} が見つかりません。")
            st.info(f"情報: {japanese_file} が見つかりません。")
        ###############

    ##############################################################

    
    
        st.header("about")
        # st.header('by TOYOHO ISHIMURA')
        
        ###############
        # --- 冒頭のabout　外部ファイル (about.md) の読み込みと実行 ---
        about_file = 'data_text/about.md'

        if os.path.exists(about_file):
            try:
                # ファイルを UTF-8 で読み込みます
                with open(about_file, 'r', encoding='utf-8') as f:
                    ref_content = f.read()
                
                # 読み込んだテキスト（st.writeなど）を Python コードとして実行して表示します
                exec(ref_content)
                
            except Exception as e:
                st.error(f"Error loading {about_file}: {e}")
        else:
            # ファイルがない場合の予備表示（必要なければ消して構いません）
            # st.info(f"情報: {about_file} が見つかりません。")
            st.info(f"情報: {about_file} が見つかりません。")
        ###############

        # 純粋なテキスト
        st.write('_____')
        st.header('data sources')

        
        ###############
        # --- メイン引用文献　外部ファイル (main_references.md) の読み込みと実行 ---
        ref_file_main = 'data_text/main_references.md'

        if os.path.exists(ref_file_main):
            try:
                # ファイルを UTF-8 で読み込みます
                with open(ref_file_main, 'r', encoding='utf-8') as f:
                    ref_content = f.read()
                
                # 読み込んだテキスト（st.writeなど）を Python コードとして実行して表示します
                exec(ref_content)
                
            except Exception as e:
                st.error(f"Error loading {ref_file_main}: {e}")
        else:
            # ファイルがない場合の予備表示（必要なければ消して構いません）
            # st.info(f"情報: {ref_file_main} が見つかりません。")
            st.info(f"情報: {ref_file_main} が見つかりません。")
        ###############

        
        ###############
        # --- その他の引用文献　外部ファイル (other_references.md) の読み込みと実行 ---
        ref_file_others = 'data_text/other_references.md'

        if os.path.exists(ref_file_others):
            try:
                # ファイルを UTF-8 で読み込みます
                with open(ref_file_others, 'r', encoding='utf-8') as f:
                    ref_content = f.read()
                
                # 読み込んだテキスト（st.writeなど）を Python コードとして実行して表示します
                exec(ref_content)
                
            except Exception as e:
                st.error(f"Error loading {ref_file_others}: {e}")
        else:
            # ファイルがない場合の予備表示（必要なければ消して構いません）
            # st.info(f"情報: {ref_file} が見つかりません。")
            st.info(f"情報: {ref_file_others} が見つかりません。")
        ###############

        
        st.write('  ')

        st.subheader(':red[Global Database Integration]')
        st.subheader('NASA GISS Global Seawater Oxygen Isotope Database')
        st.write('https://data.giss.nasa.gov/o18data/')
        
        
        ###############
        # --- NASAデータベースの引用文献　外部ファイル(テキスト/Markdown)からの読み込み ---
        try:
            with open('data_text/NASA_references.md', 'r', encoding='utf-8') as f:
                nasa_refs = f.read()
            st.markdown(nasa_refs, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("The reference list file cannot be found. Please visit https://data.giss.nasa.gov/o18data/ref.html")
        ###############

        st.write('_____')
        st.link_button("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    
    
    
    
   


if __name__ == '__main__':
    main()
