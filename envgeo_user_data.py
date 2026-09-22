#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Streamlit controls for memory-only uploaded user data.

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import envgeo_utils


COLUMN_NOT_ASSIGNED = "Not assigned"
MARKER_OPTIONS = {
    "Diamond": "D",
    "Circle": "o",
    "Square": "s",
    "Triangle up": "^",
    "Star": "*",
    "X": "X",
}
LINE_STYLE_OPTIONS = {
    "Dotted": ":",
    "Dashed": "--",
    "Solid": "-",
    "Dash-dot": "-.",
}
MAX_MAP_HOVER_COLUMNS = 30
MAX_MAP_HOVER_VALUE_LENGTH = 120


def render_upload_panel(page_key, requirement_text):
    """Load or reuse shared session data without changing reference filters."""
    uploaded_df = envgeo_utils.get_uploaded_data()

    with st.sidebar.expander("Uploaded data overlay", expanded=False):
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        st.caption(
            "Uploaded files remain in memory for this Streamlit session only. "
            "They are not saved to local or server storage."
        )
        st.download_button(
            "Download CSV template",
            data=envgeo_utils.build_upload_template_csv(),
            file_name="envgeo_seawater_upload_template.csv",
            mime="text/csv",
            key=f"{page_key}_upload_template",
        )

        if not uploaded_df.empty:
            filename = envgeo_utils.get_uploaded_filename() or "current session"
            st.success(f"Using {len(uploaded_df):,} shared rows from {filename}.")
            if st.button(
                "Clear shared uploaded data",
                key=f"{page_key}_clear_uploaded_data",
            ):
                envgeo_utils.clear_uploaded_data()
                for key in list(st.session_state):
                    if key.startswith(f"{page_key}_uploaded_column_"):
                        st.session_state.pop(key, None)
                generation_key = f"{page_key}_upload_generation"
                st.session_state[generation_key] = (
                    st.session_state.get(generation_key, 0) + 1
                )
                st.rerun()

        generation_key = f"{page_key}_upload_generation"
        upload_generation = st.session_state.get(generation_key, 0)
        uploaded_file = st.file_uploader(
            "Upload user data",
            type=["xlsx", "xls", "csv"],
            key=f"{page_key}_uploaded_file_{upload_generation}",
            help=(
                requirement_text
                + " Common English and unambiguous Japanese column names are recognized."
            ),
        )
        if uploaded_file is not None:
            try:
                uploaded_df = envgeo_utils.prepare_uploaded_data(
                    envgeo_utils.read_uploaded_table(uploaded_file)
                )
                envgeo_utils.store_uploaded_data(uploaded_df, uploaded_file.name)
            except Exception as exc:
                st.error(f"Could not read uploaded file: {exc}")
                return envgeo_utils.get_uploaded_data()

            renamed_columns = uploaded_df.attrs.get("standardized_column_renames", {})
            if renamed_columns:
                rename_text = ", ".join(
                    f"{source} -> {target}"
                    for source, target in renamed_columns.items()
                )
                st.caption(f"Standardized column names: {rename_text}")
            st.success(f"Loaded {len(uploaded_df):,} uploaded rows.")

    return uploaded_df


def render_column_controls(
    uploaded_df,
    required_roles,
    page_key,
    optional_roles=None,
):
    """Render editable assignments for required and optional plot roles."""
    optional_roles = optional_roles or {}
    missing_roles = [
        target for target in required_roles.values() if target not in uploaded_df.columns
    ]
    with st.sidebar.expander(
        "Uploaded data columns",
        expanded=bool(not uploaded_df.empty and missing_roles),
    ):
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        st.caption(
            "Recognized labels are selected automatically. Choose the source "
            "column manually when a label is unknown."
        )
        if uploaded_df.empty:
            st.caption("Upload or reuse session data to enable column selection.")
            return uploaded_df

        excluded_columns = {
            envgeo_utils.QUALITY_FLAG_COLUMN,
            envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN,
        }
        source_columns = [
            column for column in uploaded_df.columns if column not in excluded_columns
        ]
        options = [COLUMN_NOT_ASSIGNED] + source_columns
        selected_mapping = {}

        role_items = [
            (label, target_column, True)
            for label, target_column in required_roles.items()
        ] + [
            (label, target_column, False)
            for label, target_column in optional_roles.items()
        ]
        for label, target_column, is_required in role_items:
            state_key = f"{page_key}_uploaded_column_{target_column}"
            default_value = (
                target_column
                if target_column in source_columns
                else COLUMN_NOT_ASSIGNED
            )
            if st.session_state.get(state_key) not in options:
                st.session_state[state_key] = default_value
            selected_source = st.selectbox(
                label,
                options,
                key=state_key,
                help=(
                    f"Select the uploaded column that represents {target_column}. "
                    "Automatic recognition remains editable."
                ),
            )
            if selected_source == COLUMN_NOT_ASSIGNED:
                requirement = "required" if is_required else "optional"
                st.caption(f"{target_column}: not assigned ({requirement})")
            else:
                selected_mapping[target_column] = selected_source
                status = (
                    "Automatically detected"
                    if selected_source == target_column
                    else "Manually selected"
                )
                st.caption(f"{target_column}: {status}")

        selected_sources = list(selected_mapping.values())
        if len(selected_sources) != len(set(selected_sources)):
            st.error("Each assigned role must use a different source column.")
            return _without_required_roles(uploaded_df, required_roles.values())
        assigned_required = {
            target for target in required_roles.values() if target in selected_mapping
        }
        if len(assigned_required) != len(required_roles):
            st.warning("Assign all required columns to enable the uploaded overlay.")
            return _without_required_roles(uploaded_df, required_roles.values())

        return envgeo_utils.apply_uploaded_column_mapping(
            uploaded_df,
            selected_mapping,
        )


def _without_required_roles(uploaded_df, required_columns):
    return uploaded_df.drop(
        columns=[column for column in required_columns if column in uploaded_df],
        errors="ignore",
    )


def uploaded_map_hover_text(uploaded_df):
    """Build bounded hover text while retaining arbitrary uploaded fields."""
    excluded_columns = {
        envgeo_utils.QUALITY_FLAG_COLUMN,
        envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN,
    }
    coordinate_columns = {"Longitude_degE", "Latitude_degN"}
    hover_columns = [
        column
        for column in uploaded_df.columns
        if column not in excluded_columns and column not in coordinate_columns
    ]
    shown_columns = hover_columns[:MAX_MAP_HOVER_COLUMNS]
    omitted_count = len(hover_columns) - len(shown_columns)
    hover_text = []

    for _, row in uploaded_df.iterrows():
        lines = [
            "Uploaded data",
            f"Longitude: {row['Longitude_degE']:g}",
            f"Latitude: {row['Latitude_degN']:g}",
        ]
        for column in shown_columns:
            value = row[column]
            if pd.isna(value):
                continue
            value_text = str(value)
            if len(value_text) > MAX_MAP_HOVER_VALUE_LENGTH:
                value_text = value_text[: MAX_MAP_HOVER_VALUE_LENGTH - 3] + "..."
            lines.append(f"{column}: {value_text}")
        if omitted_count:
            lines.append(f"({omitted_count} additional columns not shown)")
        hover_text.append("<br>".join(lines))
    return hover_text


def render_marker_style_controls(
    uploaded_df,
    page_key,
    include_line=False,
    marker_size_default=140,
    marker_size_min=10,
    marker_size_step=10,
):
    """Render common uploaded-marker controls and optional line controls."""
    controls_disabled = uploaded_df.empty
    with st.sidebar.expander("Uploaded marker style", expanded=False):
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)
        if controls_disabled:
            st.caption("Upload or reuse session data to enable these controls.")

        color_mode = st.selectbox(
            "Marker color mode",
            ["Use current colorbar when possible", "Fixed marker color"],
            key=f"{page_key}_uploaded_color_mode",
            disabled=controls_disabled,
            help=(
                "The current colorbar is used only when the selected color "
                "parameter also exists in the uploaded data."
            ),
        )
        marker_color = st.color_picker(
            "Fixed marker color",
            value="#D4D4D4",
            key=f"{page_key}_uploaded_marker_color",
            disabled=controls_disabled,
            help="Also used when a shared color value is missing.",
        )
        marker_label = st.selectbox(
            "Marker shape",
            list(MARKER_OPTIONS),
            index=0,
            key=f"{page_key}_uploaded_marker_shape",
            disabled=controls_disabled,
        )
        marker_col1, marker_col2 = st.columns(2)
        with marker_col1:
            marker_size = st.number_input(
                "Marker size",
                min_value=marker_size_min,
                max_value=600,
                value=marker_size_default,
                step=marker_size_step,
                key=f"{page_key}_uploaded_marker_size",
                disabled=controls_disabled,
            )
            outline_width = st.number_input(
                "Outline width",
                min_value=0.0,
                max_value=5.0,
                value=1.2,
                step=0.1,
                key=f"{page_key}_uploaded_outline_width",
                disabled=controls_disabled,
            )
        with marker_col2:
            marker_alpha = st.slider(
                "Opacity",
                min_value=0.1,
                max_value=1.0,
                value=0.95,
                step=0.05,
                key=f"{page_key}_uploaded_marker_opacity",
                disabled=controls_disabled,
            )
            outline_color = st.color_picker(
                "Outline color",
                value="#111111",
                key=f"{page_key}_uploaded_outline_color",
                disabled=controls_disabled,
            )

        line_width = 2.0
        line_style = ":"
        if include_line:
            line_col1, line_col2 = st.columns(2)
            with line_col1:
                line_width = st.number_input(
                    "Line width",
                    min_value=0.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    key=f"{page_key}_uploaded_line_width",
                    disabled=controls_disabled,
                    help="Line width used to connect uploaded profile samples.",
                )
            with line_col2:
                line_style_label = st.selectbox(
                    "Line style",
                    list(LINE_STYLE_OPTIONS),
                    key=f"{page_key}_uploaded_line_style",
                    disabled=controls_disabled,
                    help="Line style used to connect uploaded profile samples.",
                )
                line_style = LINE_STYLE_OPTIONS[line_style_label]

    return {
        "color_mode": color_mode,
        "color": marker_color,
        "marker": MARKER_OPTIONS[marker_label],
        "size": marker_size,
        "alpha": marker_alpha,
        "outline_color": outline_color,
        "outline_width": outline_width,
        "line_width": line_width,
        "line_style": line_style,
    }


def add_uploaded_map_overlay(
    fig,
    uploaded_df,
    style,
    color_column=None,
    colorscale=None,
    color_range=None,
    show_nodata=True,
):
    """Add uploaded locations as frontmost outlined Scattermapbox traces."""
    required = {"Longitude_degE", "Latitude_degN"}
    if uploaded_df.empty or not required.issubset(uploaded_df.columns):
        return fig, 0

    map_df = uploaded_df.copy()
    map_df["Longitude_degE"] = pd.to_numeric(
        map_df["Longitude_degE"], errors="coerce"
    )
    map_df["Latitude_degN"] = pd.to_numeric(
        map_df["Latitude_degN"], errors="coerce"
    )
    map_df = map_df.dropna(subset=["Longitude_degE", "Latitude_degN"])
    map_df = map_df.loc[map_df["Latitude_degN"].between(-90, 90)].copy()
    if map_df.empty:
        return fig, 0

    marker_size = max(6.0, math.sqrt(float(style["size"])))
    outline_size = marker_size + 2.0 * float(style["outline_width"])
    hover_text = uploaded_map_hover_text(map_df)

    # --- カラーバー共有の判定（アウトライン前に確定させる）---
    use_shared_color = (
        style["color_mode"] == "Use current colorbar when possible"
        and color_column
        and color_column in map_df.columns
        and colorscale is not None
        and color_range is not None
    )
    fixed_color_rows = pd.Series(True, index=map_df.index)
    valid_color = pd.Series(False, index=map_df.index)
    color_values = None
    if use_shared_color:
        color_values = pd.to_numeric(map_df[color_column], errors="coerce")
        valid_color = color_values.notna()
        if valid_color.any():
            fixed_color_rows = ~valid_color

    show_fixed = fixed_color_rows.any() and (not use_shared_color or show_nodata)

    # アウトライン：実際に描画される行のみ対象にする
    if use_shared_color:
        outline_rows = valid_color | (fixed_color_rows & show_fixed)
    else:
        outline_rows = pd.Series(True, index=map_df.index)

    if outline_rows.any():
        fig.add_trace(
            go.Scattermapbox(
                lon=map_df.loc[outline_rows, "Longitude_degE"],
                lat=map_df.loc[outline_rows, "Latitude_degN"],
                mode="markers",
                marker={
                    "size": outline_size,
                    "color": style["outline_color"],
                    "opacity": style["alpha"],
                },
                hoverinfo="skip",
                showlegend=False,
                name="Uploaded data outline",
            )
        )

    # カラーバー共有トレース
    if use_shared_color and valid_color.any():
        fig.add_trace(
            go.Scattermapbox(
                lon=map_df.loc[valid_color, "Longitude_degE"],
                lat=map_df.loc[valid_color, "Latitude_degN"],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": color_values[valid_color],
                    "colorscale": colorscale,
                    "cmin": color_range[0],
                    "cmax": color_range[1],
                    "showscale": False,
                    "opacity": style["alpha"],
                },
                text=[
                    text for text, is_valid in zip(hover_text, valid_color) if is_valid
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Uploaded data",
            )
        )

    # 固定カラートレース（show_nodataがFalseのとき非表示）
    if show_fixed:
        fill_color = style["color"]
        fig.add_trace(
            go.Scattermapbox(
                lon=map_df.loc[fixed_color_rows, "Longitude_degE"],
                lat=map_df.loc[fixed_color_rows, "Latitude_degN"],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": fill_color,
                    "opacity": style["alpha"],
                },
                text=[
                    text
                    for text, is_fixed in zip(hover_text, fixed_color_rows)
                    if is_fixed
                ],
                hovertemplate="%{text}<extra></extra>",
                name=(
                    f"Uploaded data (no {color_column})"
                    if use_shared_color
                    else "Uploaded data"
                ),
            )
        )
    return fig, int(outline_rows.sum())
