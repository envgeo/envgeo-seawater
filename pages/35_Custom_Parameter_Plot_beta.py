#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Parameter Plot beta for EnvGeo-Seawater.

任意の数値パラメーターをX軸、Y軸、色、サイズとして選び、
海水同位体・水文データの関係を試験的に確認するページです。

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

import io

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import streamlit as st

import envgeo_user_data
import envgeo_utils


version = "1.3.3"
fig_title = "envgeo-seawater-database"


PARAMETER_LABELS = {
    "d18O": "δ18O (‰)",
    "dD": "δD (‰)",
    "d-excess": "d-excess (‰)",
    "Salinity": "Salinity",
    "Temperature_degC": "Temperature (degC)",
    "Depth_m": "Depth (m)",
    "Latitude_degN": "Latitude (degN)",
    "Longitude_degE": "Longitude (degE)",
    "Year": "Year",
    "Month": "Month",
}


DEFAULT_PARAMETERS = [
    "d18O",
    "dD",
    "d-excess",
    "Salinity",
    "Temperature_degC",
    "Depth_m",
    "Latitude_degN",
    "Longitude_degE",
    "Year",
    "Month",
]


def numeric_parameter_options(*dataframes):
    """
    Return numeric columns suitable for plotting.

    描画に使いやすい数値列だけを候補として返します。
    """
    excluded = {
        envgeo_utils.QUALITY_FLAG_COLUMN,
        envgeo_utils.QUALITY_ORIGINAL_VALUE_COLUMN,
    }
    available = {}
    for df in dataframes:
        if df is None or df.empty:
            continue
        for column in df.columns:
            if column in excluded:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().any():
                available.setdefault(column, True)

    known = [column for column in DEFAULT_PARAMETERS if column in available]
    experimental = sorted(column for column in available if column not in known)
    return known + experimental


def combined_numeric_values(column, *dataframes):
    """Return valid numeric values for one column across available datasets."""
    values = []
    for df in dataframes:
        if df is not None and not df.empty and column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce").dropna()
            if not numeric.empty:
                values.append(numeric)
    if not values:
        return pd.Series(dtype="float64")
    return pd.concat(values, ignore_index=True)


def prepare_plot_rows(df, required_columns):
    """Return numeric plot rows only when all selected columns are available."""
    if df is None or df.empty or not set(required_columns).issubset(df.columns):
        return pd.DataFrame(columns=required_columns)
    result = df.copy()
    for column in required_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=required_columns).copy()


def scaled_marker_sizes(df, size_by, base_size, size_contrast):
    """Calculate marker areas for fixed or parameter-scaled plotting."""
    if size_by == "Fixed size":
        return base_size
    values = pd.to_numeric(df[size_by], errors="coerce")
    value_min = float(values.min())
    value_max = float(values.max())
    if value_min == value_max:
        return np.full(len(df), base_size)
    scaled = (values - value_min) / (value_max - value_min)
    return base_size * (0.15 + scaled * (size_contrast - 0.15))


def default_axis_range(series):
    """
    Return a padded numeric axis range.

    データ範囲に少し余白を足した軸範囲を返します。
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0, 1.0

    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        margin = max(abs(vmin) * 0.1, 1.0)
    else:
        margin = (vmax - vmin) * 0.08
    return vmin - margin, vmax + margin


def numeric_input_pair(label, default_min, default_max, key_prefix, step=0.1):
    """
    Render paired min/max numeric inputs.

    2値スライダーではなく、最小値と最大値を個別に入力します。
    """
    col_min, col_max = st.columns(2)
    with col_min:
        selected_min = st.number_input(
            f"{label} min",
            value=float(default_min),
            step=step,
            key=f"{key_prefix}_min",
        )
    with col_max:
        selected_max = st.number_input(
            f"{label} max",
            value=float(default_max),
            step=step,
            key=f"{key_prefix}_max",
        )

    if selected_min >= selected_max:
        st.warning(f"{label} min must be smaller than max. Default range is used.")
        return float(default_min), float(default_max)
    return float(selected_min), float(selected_max)


def main():
    st.header(f"Custom Parameter Plot beta ({version})")
    st.caption("Experimental page for flexible 2D plots such as dD x d18O colored by depth.")
    st.button("Reload")

    data_source_japan_sea = envgeo_utils.data_source_JAPAN_SEA
    data_source_around_japan = envgeo_utils.data_source_AROUND_JAPAN
    data_source_global = envgeo_utils.data_source_GLOBAL

    ref_data = st.radio(
        "Data source (see Home > About):",
        (data_source_japan_sea, data_source_around_japan, data_source_global),
        horizontal=True,
    )

    if ref_data == data_source_japan_sea:
        st.write(envgeo_utils.refs_JAPAN_SEA)
    elif ref_data == data_source_around_japan:
        st.write(envgeo_utils.refs_AROUND_JAPAN)
    elif ref_data == data_source_global:
        st.write(envgeo_utils.refs_GLOBAL)
    else:
        st.warning("Invalid data source selection.")

    df_original = envgeo_utils.load_isotope_data(ref_data)
    if df_original.empty:
        st.warning("No data available for the selected conditions.")
        return

    embedded_in_integrated = (
        st.session_state.get(envgeo_utils.INTEGRATED_EMBEDDED_PAGE_KEY)
        == "35_Custom_Parameter_Plot_beta.py"
    )
    if embedded_in_integrated:
        uploaded_df = envgeo_utils.get_uploaded_data()
    else:
        uploaded_df = envgeo_user_data.render_upload_panel(
            "custom_plot",
            "Choose any two numeric columns for the X and Y axes.",
        )
    uploaded_df = envgeo_user_data.render_column_controls(
        uploaded_df,
        {},
        "custom_plot",
        optional_roles={
            "Longitude column (optional)": "Longitude_degE",
            "Latitude column (optional)": "Latitude_degN",
            "Depth column (optional)": "Depth_m",
            "Temperature column (optional)": "Temperature_degC",
            "Salinity column (optional)": "Salinity",
            "d18O column (optional)": "d18O",
            "dD column (optional)": "dD",
        },
    )
    uploaded_style = envgeo_user_data.render_marker_style_controls(
        uploaded_df,
        "custom_plot",
    )

    (
        df_filtered,
        sld_year_min, sld_year_max,
        selected_months,
        sld_lon_min, sld_lon_max,
        sld_lat_min, sld_lat_max,
        sld_depth_min, sld_depth_max,
        sld_sal_min, sld_sal_max,
        sld_d18O_min, sld_d18O_max,
        sld_temp_min, sld_temp_max,
        selected_cruise,
        submitted,
    ) = envgeo_utils.sidebar_filter_and_display(
        envgeo_utils.combine_reference_and_uploaded_for_filtering(
            df_original, uploaded_df
        ),
        ref_data,
        data_source_japan_sea,
        data_source_around_japan,
        uploaded_df=uploaded_df,
        uploaded_filter_key="custom_plot",
        uploaded_dataset_label=envgeo_utils.UPLOADED_DATA_LABEL,
    )
    # Plot styling keeps uploaded rows separate so they can be redrawn in the
    # foreground.  Statistical calculations use the full sidebar-selected
    # reference-plus-upload table.
    filtered_integrated_df = df_filtered.copy()
    df_filtered, uploaded_df = envgeo_utils.split_uploaded_rows(
        filtered_integrated_df, envgeo_utils.UPLOADED_DATA_LABEL
    )

    options = numeric_parameter_options(df_filtered, uploaded_df)
    if len(options) < 2:
        st.warning("At least two numeric parameters are required for this plot.")
        return

    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "CUSTOM_PLOT_SETTINGS_LABEL", "Custom plot settings"))
        st.caption(envgeo_utils.AUTO_APPLY_NOTE)

        x_axis = st.selectbox(
            "X axis",
            options,
            index=options.index("d18O") if "d18O" in options else 0,
            help="Choose the numeric parameter for the horizontal axis.",
        )
        y_default = "dD" if "dD" in options else options[min(1, len(options) - 1)]
        y_axis = st.selectbox(
            "Y axis",
            options,
            index=options.index(y_default),
            help="Choose the numeric parameter for the vertical axis.",
        )

        color_options = ["Single color"] + options
        color_by = st.selectbox(
            "Color by",
            color_options,
            index=color_options.index("Depth_m") if "Depth_m" in color_options else 0,
            help="Use a fixed blue color, or color points by another numeric parameter.",
        )

        size_options = ["Fixed size"] + options
        size_by = st.selectbox(
            "Size by",
            size_options,
            index=0,
            help="Use fixed marker size, or scale marker size by another numeric parameter.",
        )

        marker_size = st.number_input(
            "Base marker size",
            min_value=5,
            max_value=300,
            value=80,
            step=5,
            help="Base marker size for the selected data points.",
        )
        if size_by != "Fixed size":
            size_contrast = st.number_input(
                "Size contrast",
                min_value=1.5,
                max_value=8.0,
                value=4.0,
                step=0.5,
                help=(
                    "Controls the maximum marker-size multiplier for the "
                    "selected Size by parameter. Larger values emphasize "
                    "differences more strongly."
                ),
            )
        else:
            size_contrast = 1.0
        marker_alpha = st.number_input(
            "Marker transparency",
            min_value=0.05,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Marker opacity for the selected data points.",
        )

        display_col1, display_col2 = st.columns([1, 1])
        with display_col1:
            show_background = st.radio(
                "Show background data",
                ("Yes", "No"),
                index=1,
                horizontal=True,
                help=getattr(envgeo_utils, "BACKGROUND_DATA_HELP_TEXT", "Show the unfiltered dataset behind the currently filtered data for context."),
            )
        with display_col2:
            show_legend = st.radio(
                "Show legend",
                ("Yes", "No"),
                index=0,
                horizontal=True,
                help="Show or hide the legend for plotted data groups.",
            )

        add_regression_line = st.radio(
            "Regression line",
            ("Yes", "No"),
            index=1,
            horizontal=True,
            help=getattr(envgeo_utils, "REGRESSION_HELP_TEXT", "Add a simple least-squares regression line for quick visual reference."),
        )

        x_values_all = combined_numeric_values(x_axis, df_filtered, uploaded_df)
        y_values_all = combined_numeric_values(y_axis, df_filtered, uploaded_df)
        x_default_min, x_default_max = default_axis_range(x_values_all)
        y_default_min, y_default_max = default_axis_range(y_values_all)
        x_min, x_max = numeric_input_pair(
            "X axis range",
            x_default_min,
            x_default_max,
            f"custom_plot_x_range::{ref_data}::{x_axis}",
        )
        y_min, y_max = numeric_input_pair(
            "Y axis range",
            y_default_min,
            y_default_max,
            f"custom_plot_y_range::{ref_data}::{y_axis}",
        )

        color_range = None
        colormap_label = None
        if color_by != "Single color":
            colormap_options = envgeo_utils.get_plotly_colormap_options(color_by)
            colormap_label = st.selectbox(
                "Colormap",
                list(colormap_options.keys()),
                index=list(colormap_options.keys()).index(
                    envgeo_utils.recommended_plotly_colormap_label(color_by)
                ),
                key=f"custom_plot_colormap::{color_by}",
                help="Choose the colormap used for the Custom Parameter Plot colorbar.",
            )
            color_values_all = combined_numeric_values(
                color_by,
                df_filtered,
                uploaded_df,
            )
            color_default_min, color_default_max = default_axis_range(color_values_all)
            color_range = numeric_input_pair(
                "Color range",
                color_default_min,
                color_default_max,
                f"custom_plot_color_range::{ref_data}::{color_by}",
            )

        size_col1, size_col2 = st.columns(2)
        with size_col1:
            fig_width = st.number_input(
                "Fig width (x)", min_value=4, max_value=24, value=12, step=1
            )
        with size_col2:
            fig_height = st.number_input(
                "Fig height (y)", min_value=4, max_value=24, value=9, step=1
            )

        font_col1, font_col2 = st.columns(2)
        with font_col1:
            tick_font_size = st.number_input(
                "Tick font size", min_value=6, max_value=32, value=15, step=1
            )
        with font_col2:
            label_font_size = st.number_input(
                "Label font size", min_value=6, max_value=32, value=16, step=1
            )

        tick_col1, tick_col2 = st.columns(2)
        with tick_col1:
            x_tick_count = st.number_input(
                "X tick count", min_value=3, max_value=30, value=9, step=1
            )
        with tick_col2:
            y_tick_count = st.number_input(
                "Y tick count", min_value=3, max_value=30, value=9, step=1
            )

    required_columns = [x_axis, y_axis]
    if size_by != "Fixed size":
        required_columns.append(size_by)

    df_plot = prepare_plot_rows(df_filtered, required_columns)
    uploaded_required_columns = [x_axis, y_axis]
    if size_by != "Fixed size":
        uploaded_required_columns.append(size_by)
    uploaded_plot = prepare_plot_rows(uploaded_df, uploaded_required_columns)
    excluded_count = len(df_filtered) - len(df_plot) if set(required_columns).issubset(df_filtered.columns) else len(df_filtered)
    uploaded_excluded_count = len(uploaded_df) - len(uploaded_plot)
    if excluded_count > 0 and set(required_columns).issubset(df_filtered.columns):
        st.caption(
            f":blue[Custom plot: {len(df_plot):,} samples plotted and "
            f"{excluded_count:,} excluded due to missing selected parameter values.]"
        )

    if not uploaded_df.empty:
        st.caption(
            f":blue[Uploaded overlay: {len(uploaded_plot):,} / {len(uploaded_df):,} "
            f"plotted ({uploaded_excluded_count:,} excluded due to missing or invalid "
            "selected parameter values).]"
        )
        uploaded_quality_df = envgeo_utils.get_quality_rows(uploaded_df)
        with st.expander("Uploaded data quality check", expanded=False):
            envgeo_utils.render_quality_flag_criteria_note()
            st.write(
                f"Quality-flagged rows: {len(uploaded_quality_df):,} / "
                f"{len(uploaded_df):,}"
            )
            if uploaded_quality_df.empty:
                st.success("No uploaded rows triggered the current quality rules.")
            else:
                st.dataframe(
                    uploaded_quality_df,
                    **envgeo_utils.stretch_width_kwargs(st.dataframe),
                )

    if df_plot.empty and uploaded_plot.empty:
        st.warning("No valid data are available for the selected plot settings.")
        return

    df_background = prepare_plot_rows(df_original, [x_axis, y_axis])

    x_label = PARAMETER_LABELS.get(x_axis, x_axis)
    y_label = PARAMETER_LABELS.get(y_axis, y_axis)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    if show_background == "Yes" and not df_background.empty:
        ax.scatter(
            df_background[x_axis],
            df_background[y_axis],
            s=max(marker_size * 0.4, 5),
            c="lightgray",
            alpha=0.35,
            linewidths=0,
            label="All data",
        )

    point_sizes = scaled_marker_sizes(df_plot, size_by, marker_size, size_contrast)

    scatter = None
    if color_by == "Single color" and not df_plot.empty:
        ax.scatter(
            df_plot[x_axis],
            df_plot[y_axis],
            s=point_sizes,
            c="blue",
            alpha=marker_alpha,
            edgecolors="black",
            linewidths=0.5,
            label="Filtered data",
        )
    elif color_by != "Single color" and not df_plot.empty:
        color_values = pd.to_numeric(df_plot[color_by], errors="coerce")
        color_valid = color_values.notna()
        if color_valid.any():
            valid_sizes = (
                point_sizes[color_valid]
                if not np.isscalar(point_sizes)
                else point_sizes
            )
            scatter = ax.scatter(
                df_plot.loc[color_valid, x_axis],
                df_plot.loc[color_valid, y_axis],
                s=valid_sizes,
                c=color_values[color_valid],
                cmap=envgeo_utils.get_matplotlib_colormap(color_by, colormap_label),
                vmin=color_range[0] if color_range is not None else None,
                vmax=color_range[1] if color_range is not None else None,
                alpha=marker_alpha,
                edgecolors="black",
                linewidths=0.5,
                label="Filtered data",
            )
            cbar = fig.colorbar(
                scatter,
                ax=ax,
                orientation="vertical",
                pad=0.02,
                fraction=0.04,
            )
            cbar.set_label(
                PARAMETER_LABELS.get(color_by, color_by),
                fontsize=label_font_size,
            )
            cbar.ax.tick_params(labelsize=tick_font_size)

        missing_color = ~color_valid
        if missing_color.any():
            missing_sizes = (
                point_sizes[missing_color]
                if not np.isscalar(point_sizes)
                else point_sizes
            )
            ax.scatter(
                df_plot.loc[missing_color, x_axis],
                df_plot.loc[missing_color, y_axis],
                s=missing_sizes,
                c="#8C8C8C",
                alpha=marker_alpha,
                edgecolors="black",
                linewidths=0.5,
                label=f"Filtered data (no {color_by})",
            )

    if not uploaded_plot.empty:
        uploaded_sizes = scaled_marker_sizes(
            uploaded_plot,
            size_by,
            uploaded_style["size"],
            size_contrast,
        )
        use_shared_colorbar = (
            uploaded_style["color_mode"] == "Use current colorbar when possible"
            and color_by != "Single color"
            and color_by in uploaded_plot.columns
        )
        if use_shared_colorbar:
            uploaded_color_values = pd.to_numeric(
                uploaded_plot[color_by],
                errors="coerce",
            )
            uploaded_color_valid = uploaded_color_values.notna()
            uploaded_scatter = None
            if uploaded_color_valid.any():
                valid_sizes = (
                    uploaded_sizes[uploaded_color_valid]
                    if not np.isscalar(uploaded_sizes)
                    else uploaded_sizes
                )
                uploaded_scatter = ax.scatter(
                    uploaded_plot.loc[uploaded_color_valid, x_axis],
                    uploaded_plot.loc[uploaded_color_valid, y_axis],
                    s=valid_sizes,
                    c=uploaded_color_values[uploaded_color_valid],
                    cmap=envgeo_utils.get_matplotlib_colormap(color_by, colormap_label),
                    vmin=color_range[0] if color_range is not None else None,
                    vmax=color_range[1] if color_range is not None else None,
                    marker=uploaded_style["marker"],
                    alpha=uploaded_style["alpha"],
                    edgecolors=uploaded_style["outline_color"],
                    linewidths=uploaded_style["outline_width"],
                    label="Uploaded data",
                    zorder=10,
                )
            missing_color = ~uploaded_color_valid
            if missing_color.any():
                missing_sizes = (
                    uploaded_sizes[missing_color]
                    if not np.isscalar(uploaded_sizes)
                    else uploaded_sizes
                )
                ax.scatter(
                    uploaded_plot.loc[missing_color, x_axis],
                    uploaded_plot.loc[missing_color, y_axis],
                    s=missing_sizes,
                    c=uploaded_style["color"],
                    marker=uploaded_style["marker"],
                    alpha=uploaded_style["alpha"],
                    edgecolors=uploaded_style["outline_color"],
                    linewidths=uploaded_style["outline_width"],
                    label=f"Uploaded data (no {color_by})",
                    zorder=10,
                )
            if scatter is None and uploaded_scatter is not None:
                cbar = fig.colorbar(
                    uploaded_scatter,
                    ax=ax,
                    orientation="vertical",
                    pad=0.02,
                    fraction=0.04,
                )
                cbar.set_label(
                    PARAMETER_LABELS.get(color_by, color_by),
                    fontsize=label_font_size,
                )
                cbar.ax.tick_params(labelsize=tick_font_size)
        else:
            ax.scatter(
                uploaded_plot[x_axis],
                uploaded_plot[y_axis],
                s=uploaded_sizes,
                c=uploaded_style["color"],
                marker=uploaded_style["marker"],
                alpha=uploaded_style["alpha"],
                edgecolors=uploaded_style["outline_color"],
                linewidths=uploaded_style["outline_width"],
                label="Uploaded data",
                zorder=10,
            )

    if add_regression_line == "Yes":
        # Regression deliberately uses only the two axis columns.  It should
        # not discard otherwise valid points merely because a styling column
        # (colour or marker size) is missing.
        regression_source = prepare_plot_rows(
            filtered_integrated_df, [x_axis, y_axis]
        )
        x_values = pd.to_numeric(regression_source[x_axis], errors="coerce")
        y_values = pd.to_numeric(regression_source[y_axis], errors="coerce")
        regression_df = pd.DataFrame({"x": x_values, "y": y_values}).dropna()

        if len(regression_df) >= 2 and regression_df["x"].nunique() > 1:
            coef = np.polyfit(regression_df["x"], regression_df["y"], 1)
            x_line = np.linspace(x_min, x_max, 100)
            y_line = np.poly1d(coef)(x_line)
            ax.plot(
                x_line,
                y_line,
                color="black",
                linewidth=1.6,
                linestyle="-",
                label="Regression line",
            )

            r_value = np.corrcoef(regression_df["x"], regression_df["y"])[0, 1]
            ax.text(
                0.99,
                0.02,
                f"y = {coef[0]:.3g}x + {coef[1]:.3g}; R = {r_value:.2f}; N = {len(regression_df)}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=max(8, tick_font_size - 3),
                color="black",
            )
        else:
            st.caption(":gray[Regression line was skipped because fewer than two valid x-y points are available.]")

    ax.set_xlabel(x_label, fontsize=label_font_size)
    ax.set_ylabel(y_label, fontsize=label_font_size)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.linspace(x_min, x_max, int(x_tick_count)))
    ax.set_yticks(np.linspace(y_min, y_max, int(y_tick_count)))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(labelsize=tick_font_size, length=6)
    ax.grid(True, color="0.88", linewidth=0.6)
    if show_legend == "Yes":
        ax.legend(fontsize=tick_font_size)

    month_display = "All" if len(selected_months) == 12 else ", ".join(map(str, sorted(selected_months)))
    subtitle = (
        f"Lon:{sld_lon_min}-{sld_lon_max}, Lat:{sld_lat_min}-{sld_lat_max}, "
        f"Y:{sld_year_min}-{sld_year_max}, M:{month_display}, "
        f"S:{sld_sal_min}-{sld_sal_max}, D:{sld_depth_min}-{sld_depth_max}m"
    )
    title = f"{fig_title}\n{x_axis} x {y_axis}"
    if color_by != "Single color":
        title += f" colored by {color_by}"
    if size_by != "Fixed size":
        title += f", sized by {size_by}"
    title += f"\n{subtitle}"
    fig.suptitle(title.replace("_", " "), fontsize=label_font_size + 4)
    fig.tight_layout()

    fn = envgeo_utils.build_figure_filename(f"Fig_custom_{x_axis}_x_{y_axis}", subtitle)
    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=300, bbox_inches="tight")
    img.seek(0)

    st.caption("Adjust Custom plot settings in the sidebar.")
    st.pyplot(fig)

    st.download_button(
        "Download image",
        img,
        fn,
        "image/png",
    )

    with st.expander("Filtered dataset (CSV)", expanded=False):
        table_columns = [
            "reference",
            "Cruise",
            "Station",
            "Date",
            "Year",
            "Month",
            "Longitude_degE",
            "Latitude_degN",
            "Depth_m",
            "Temperature_degC",
            "Salinity",
            "d18O",
            "dD",
            "d-excess",
        ]
        selected_table_columns = list(dict.fromkeys(table_columns + required_columns))
        df_table = df_plot[
            [column for column in selected_table_columns if column in df_plot.columns]
        ].copy()
        st.dataframe(df_table.astype(str))

    if not uploaded_plot.empty:
        with st.expander("Uploaded plotted dataset (CSV)", expanded=False):
            st.dataframe(
                uploaded_plot.astype(str),
                **envgeo_utils.stretch_width_kwargs(st.dataframe),
            )


if __name__ == "__main__":
    main()
