#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redirect users to the dedicated EnvGeo-Earthquake application.

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

import streamlit as st


EARTHQUAKE_APP_URL = "https://envgeo-earthquake.streamlit.app"


st.set_page_config(
    page_title="EnvGeo-Earthquake",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://envgeo.h.kyoto-u.ac.jp/simple-earthquake-hypocenter-visualization/",
        "Report a bug": "https://www.h.kyoto-u.ac.jp/en_f/faculty_f/ishimura_toyoho_4dea/#mailform",
        "About": (
            "EnvGeo-Earthquake is now maintained as a separate application. "
            "Toyoho Ishimura, Kyoto University (2026)."
        ),
    },
)

st.title("EnvGeo-Earthquake")
st.info(
    "This page has moved to the dedicated EnvGeo-Earthquake site. "
    "/ このページはEnvGeo-Earthquake専用サイトへ移動しました。"
)
st.link_button("Open EnvGeo-Earthquake", EARTHQUAKE_APP_URL)
st.markdown(f"[{EARTHQUAKE_APP_URL}]({EARTHQUAKE_APP_URL})")
