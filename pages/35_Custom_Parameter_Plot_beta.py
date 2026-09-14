#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Parameter Plot beta for EnvGeo-Seawater.

任意の数値パラメーターをX軸、Y軸、色、サイズとして選び、
海水同位体・水文データの関係を試験的に確認するページです。
"""

import io

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import streamlit as st

import envgeo_utils


version = "1.3.0"
fig_title = "envgeo-seawater-database"


PARAMETER_LABELS = {
    "d18O": r"$\delta^{18}$O",
    "dD": r"$\delta$D",
    "d-excess": "d-excess",
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


def numeric_parameter_options(df):
    """
    Return numeric columns suitable for plotting.

    描画に使いやすい数値列だけを候補として返します。
    """
    options = []
    for column in DEFAULT_PARAMETERS:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().any():
                options.append(column)
    return options


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
        args=[1, 0],
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
        df_original.copy(),
        ref_data,
        data_source_japan_sea,
        data_source_around_japan,
    )

    options = numeric_parameter_options(df_filtered)
    if len(options) < 2:
        st.warning("At least two numeric parameters are required for this plot.")
        return

    with st.sidebar.container(border=True):
        st.subheader(getattr(envgeo_utils, "CUSTOM_PLOT_SETTINGS_LABEL", "Custom plot settings"))

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

        x_default_min, x_default_max = default_axis_range(df_filtered[x_axis])
        y_default_min, y_default_max = default_axis_range(df_filtered[y_axis])
        x_min, x_max = numeric_input_pair(
            "X axis range",
            x_default_min,
            x_default_max,
            f"custom_plot_x_range::{x_axis}",
        )
        y_min, y_max = numeric_input_pair(
            "Y axis range",
            y_default_min,
            y_default_max,
            f"custom_plot_y_range::{y_axis}",
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
            color_default_min, color_default_max = default_axis_range(df_filtered[color_by])
            color_range = numeric_input_pair(
                "Color range",
                color_default_min,
                color_default_max,
                f"custom_plot_color_range::{color_by}",
            )

        fig_width = st.number_input("Fig width (x)", min_value=4, max_value=24, value=12, step=1)
        fig_height = st.number_input("Fig height (y)", min_value=4, max_value=24, value=9, step=1)
        tick_font_size = st.number_input("Tick font size", min_value=6, max_value=32, value=15, step=1)
        label_font_size = st.number_input("Label font size", min_value=6, max_value=32, value=16, step=1)
        x_tick_count = st.number_input("X tick count", min_value=3, max_value=30, value=9, step=1)
        y_tick_count = st.number_input("Y tick count", min_value=3, max_value=30, value=9, step=1)

    required_columns = [x_axis, y_axis]
    if color_by != "Single color":
        required_columns.append(color_by)
    if size_by != "Fixed size":
        required_columns.append(size_by)

    df_plot = df_filtered.dropna(subset=required_columns).copy()
    excluded_count = len(df_filtered.dropna(how="all")) - len(df_plot)
    if excluded_count > 0:
        st.caption(
            f":blue[Custom plot: {len(df_plot):,} samples plotted and "
            f"{excluded_count:,} excluded due to missing selected parameter values.]"
        )

    if df_plot.empty:
        st.warning("No valid data are available for the selected plot settings.")
        return

    df_background = df_original.dropna(subset=[x_axis, y_axis]).copy()

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

    if size_by == "Fixed size":
        point_sizes = marker_size
    else:
        size_values = pd.to_numeric(df_plot[size_by], errors="coerce")
        size_min = float(size_values.min())
        size_max = float(size_values.max())
        if size_min == size_max:
            point_sizes = np.full(len(df_plot), marker_size)
        else:
            scaled = (size_values - size_min) / (size_max - size_min)
            point_sizes = marker_size * (0.15 + scaled * (size_contrast - 0.15))

    if color_by == "Single color":
        ax.scatter(
            df_plot[x_axis],
            df_plot[y_axis],
            s=point_sizes,
            c="blue",
            alpha=marker_alpha,
            edgecolors="black",
            linewidths=0.5,
            label="Selected data",
        )
    else:
        color_values = pd.to_numeric(df_plot[color_by], errors="coerce")
        scatter = ax.scatter(
            df_plot[x_axis],
            df_plot[y_axis],
            s=point_sizes,
            c=color_values,
            cmap=envgeo_utils.get_matplotlib_colormap(color_by, colormap_label),
            vmin=color_range[0] if color_range is not None else None,
            vmax=color_range[1] if color_range is not None else None,
            alpha=marker_alpha,
            edgecolors="black",
            linewidths=0.5,
            label="Selected data",
        )
        cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", pad=0.02, fraction=0.04)
        cbar.set_label(PARAMETER_LABELS.get(color_by, color_by), fontsize=label_font_size)
        cbar.ax.tick_params(labelsize=tick_font_size)

    if add_regression_line == "Yes":
        x_values = pd.to_numeric(df_plot[x_axis], errors="coerce")
        y_values = pd.to_numeric(df_plot[y_axis], errors="coerce")
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

    st.pyplot(fig)

    st.download_button(
        "Download image",
        img,
        fn,
        "image/png",
    )

    with st.expander("Selected dataset (CSV)", expanded=False):
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
        df_table = df_plot[[column for column in table_columns if column in df_plot.columns]].copy()
        st.dataframe(df_table.astype(str))


if __name__ == "__main__":
    main()
