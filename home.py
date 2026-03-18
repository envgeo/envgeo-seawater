
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:00:21 2023

@author: Toyoho Ishimura @Kyoto-U

2026/03/18 update
"""

import streamlit as st
import os



# page info
st.set_page_config(
    page_title="Ocean Geochemical Database", 
    # page_icon=image, 
    # layout="wide", 
    initial_sidebar_state="auto", 
    menu_items={
         'Get Help': 'https://envgeo.h.kyoto-u.ac.jp/sw_jpn/',
         'Report a bug': "https://www.h.kyoto-u.ac.jp/academic_f/faculty_f/ishimura_toyoho_4dea/#mailform",
         'About': """
         Interactive 3D-4D Seawater Isotope & Geochemical Database
         – Japan Marginal Seas & Global Ocean –
            by T. Ishimura
              https://envgeo.h.kyoto-u.ac.jp/sw_jpn/"""
     })




def main():

    st.write('Interactive 3D-4D Seawater Isotope & Geochemical Database – Japan Marginal Seas & Global Ocean –')
    st.title('SEAWATER GEOCHEM. DATABASE')
    st.subheader("Around Japan & Global Oceans / 2D-3D-4D visualizer")
    st.write(':blue[seawater isotopes (d18O, dD), temperature, salinity, seasonality, and annual variations around JAPAN]')
    
    st.write('Current Version: Version 1.0 _(v220-20260316)_')
    st.write(':red[NEW!! Mar 18, 2026: MAJOR UPDATE]')




    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["MAIN", "ABOUT", "DATA SOURCES", "MANUAL", "UPDATE LOG", "JAPANESE"])
    
    
    ##############################################################
    with tab1:
        def display_autoplay_video(video_path_or_url):
            import base64
            import os
            
            # 動画ファイルが存在するか、URLであるかを確認
            if not video_path_or_url.startswith(('http://', 'https://')):
                if not os.path.exists(video_path_or_url):
                    st.error(f"動画ファイルが見つかりません: {video_path_or_url}")
                    return
                
                with open(video_path_or_url, "rb") as f:
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
        st.header('Read me')
        ###############
        
        ###############
        readme_file = 'README.md'   # ← ファイル名も統一推奨

        if os.path.exists(readme_file):
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                with st.expander("Show README"):
                    st.markdown(readme_content)

            except Exception as e:
                st.error(f"Error loading {readme_file}: {e}")
        else:
            st.info(f"Error: {readme_file} not found.")
        ###############

    



        st.write('_____')
        st.link_button("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    
    

    ##############################################################
    with tab3:

            
        ###############
        # データソース読み込み
        st.header('Data Sources')
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
    with tab4:

        st.header('User Manual')
        
        ###############
        # --- その他の引用文献　外部ファイル (other_references.md) の読み込みと実行 ---
        manual_file = 'data_text/manual.md'

        if os.path.exists(manual_file):
            try:
                # ファイルを UTF-8 で読み込みます
                with open(manual_file, 'r', encoding='utf-8') as f:
                    ref_content = f.read()
                
                # 読み込んだテキスト（st.writeなど）を Python コードとして実行して表示します
                exec(ref_content)
                
            except Exception as e:
                st.error(f"Error loading {manual_file}: {e}")
        else:
            # ファイルがない場合の予備表示（必要なければ消して構いません）
            # st.info(f"情報: {ref_file} が見つかりません。")
            st.info(f"情報: {manual_file} が見つかりません。")
        ###############
        st.write('_____')
        st.link_button("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    

        
    ##############################################################
    with tab5:

        st.header('Update Log')
        
        ###############
        # --- アップデートログ　外部ファイル (update_log.md) の読み込みと実行 ---
        update_log_file = 'data_text/update_log.md'

        if os.path.exists(update_log_file):
            try:
                # ファイルを UTF-8 で読み込みます
                with open(update_log_file, 'r', encoding='utf-8') as f:
                    ref_content = f.read()
                
                # 読み込んだテキスト（st.writeなど）を Python コードとして実行して表示します
                exec(ref_content)
                
            except Exception as e:
                st.error(f"Error loading {update_log_file}: {e}")
        else:
            # ファイルがない場合の予備表示（必要なければ消して構いません）
            # st.info(f"情報: {ref_file} が見つかりません。")
            st.info(f"情報: {update_log_file} が見つかりません。")
        ###############
        st.write('_____')
        st.link_button("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    

    



    ##############################################################
    with tab6:


        st.header('このデータベースとwebアプリについて')
        
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
        st.write('_____')
        st.link_button("Go to Lab.", "https://envgeo.h.kyoto-u.ac.jp/sw_jpn/")
    


    

    
if __name__ == '__main__':
    main()
    
    

