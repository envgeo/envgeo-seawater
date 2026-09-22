#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for shared uploaded-data controls and map overlays.

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import envgeo_user_data
import envgeo_utils


def test_add_uploaded_map_overlay_adds_outline_shared_color_and_fallback_traces():
    fig = go.Figure()
    uploaded = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 136.0, None],
            "Latitude_degN": [35.0, 36.0, 37.0],
            "d18O": [0.1, None, 0.3],
            "Salinity": [34.5, 35.0, 35.2],
        }
    )
    style = {
        "color_mode": "Use current colorbar when possible",
        "color": "#F2C14E",
        "size": 144,
        "alpha": 0.9,
        "outline_color": "#111111",
        "outline_width": 1.0,
    }

    result, plotted_count = envgeo_user_data.add_uploaded_map_overlay(
        fig,
        uploaded,
        style,
        color_column="d18O",
        colorscale=[[0, "blue"], [1, "red"]],
        color_range=(-1.0, 1.0),
    )

    assert result is fig
    assert plotted_count == 2
    assert [trace.name for trace in fig.data] == [
        "Uploaded data outline",
        "Uploaded data",
        "Uploaded data (no d18O)",
    ]
    assert len(fig.data[1].lon) == 1
    assert len(fig.data[2].lon) == 1


def test_add_uploaded_map_overlay_skips_data_without_coordinates():
    fig = go.Figure()
    style = {
        "color_mode": "Fixed marker color",
        "color": "#F2C14E",
        "size": 100,
        "alpha": 1.0,
        "outline_color": "#111111",
        "outline_width": 1.0,
    }

    result, plotted_count = envgeo_user_data.add_uploaded_map_overlay(
        fig,
        pd.DataFrame({"Salinity": [34.5]}),
        style,
    )

    assert result is fig
    assert plotted_count == 0
    assert not fig.data


def test_add_uploaded_map_overlay_reports_only_rows_actually_drawn():
    fig = go.Figure()
    uploaded = pd.DataFrame(
        {
            "Longitude_degE": [135.0, 136.0],
            "Latitude_degN": [35.0, 36.0],
            "d18O": [0.1, None],
        }
    )
    style = {
        "color_mode": "Use current colorbar when possible",
        "color": "#F2C14E",
        "size": 100,
        "alpha": 1.0,
        "outline_color": "#111111",
        "outline_width": 1.0,
    }

    _, plotted_count = envgeo_user_data.add_uploaded_map_overlay(
        fig,
        uploaded,
        style,
        color_column="d18O",
        colorscale=[[0, "blue"], [1, "red"]],
        color_range=(-1.0, 1.0),
        show_nodata=False,
    )

    assert plotted_count == 1
    assert [trace.name for trace in fig.data] == [
        "Uploaded data outline",
        "Uploaded data",
    ]


def test_uploaded_map_hover_includes_metadata_and_arbitrary_columns():
    uploaded = pd.DataFrame(
        {
            "Longitude_degE": [135.0],
            "Latitude_degN": [35.0],
            "Year": [2026],
            "Month": [4],
            "Cruise": ["TEST-01"],
            "NovelElement": [1.25],
            envgeo_utils.QUALITY_FLAG_COLUMN: ["internal flag"],
        }
    )

    hover_text = envgeo_user_data.uploaded_map_hover_text(uploaded)[0]

    assert "Year: 2026" in hover_text
    assert "Month: 4" in hover_text
    assert "Cruise: TEST-01" in hover_text
    assert "NovelElement: 1.25" in hover_text
    assert envgeo_utils.QUALITY_FLAG_COLUMN not in hover_text
