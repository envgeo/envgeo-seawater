---
layout: default
title: EnvGeo-Seawater User Guide
---

# EnvGeo-Seawater User Guide

**Version 1.3.4** · Interactive seawater isotope and hydrographic data exploration

[日本語で読む](index_Japanese.html) · [Source repository](https://github.com/envgeo/seawater_map)

EnvGeo-Seawater helps users explore curated seawater isotope and hydrographic data, including regional and global reference datasets, alongside their own uploaded observations. This guide documents the supported stable workflows.

> The application is intended for interactive exploration and quality checking. For analysis, figures, publications, or presentations, cite both EnvGeo-Seawater and the original data providers used.

## A first workflow

1. Open a visualization page from the application sidebar.
2. Choose a reference dataset and adjust the shared data filters.
3. Select **Apply settings** to update tables and figures.
4. Explore maps, T–S space, depth profiles, and other views.
5. Optionally upload a CSV or XLSX file in pages that provide the user-data workflow; uploaded data remain in the current session.

## Choose a guide

### Start here

- [Overview](manual/00_overview.html)
- [Data filtering](manual/01_data_filtering.html)
- [User Data Check & Quick Visualizer](manual/05_user_data_check_quick_visualizer.html)
- [Data Overlap Check](manual/06_data_overlap_check.html)

### Interactive exploration

- [Interactive 2Dplus Visualizer](manual/03_2dplus_visualizer.html)
- [Interactive 3D/4D Visualizer](manual/04_3d_4d_visualizer.html)

### Analysis and figure generation

- [Isotope & Hydrographic Mapping](manual/32_isotope_hydrographic_mapping.html)
- [Temperature–Salinity Diagram](manual/34_ts_diagram.html)
- [Salinity–δ18O Relationship](manual/31_salinity_d18o.html)
- [Custom Parameter Plot](manual/35_custom_parameter_plot.html)
- [Depth Profile](manual/37_depth_profile.html)
- [Vertical Section Visualizer](manual/53_vertical_section.html)

## Data, citation, and support

- [Data-source and citation information](https://github.com/envgeo/seawater_map/tree/main/data_text)
- [Current update history](https://github.com/envgeo/seawater_map/blob/main/data_text/update_log.md)
- [Offline and degraded-network operation](offline_operation_log.html)
- [Testing and supported environment](testing.html)
- [Stable-release scope](stable_release_publication_notes.html)

## Examples

The following figures illustrate the range of supported exploratory views.

### Global isotope distribution

This contour map shows global seawater δ18O using the integrated reference
datasets (approximately 50,000 records). Contour interpolation is intended to
help explore broad oceanographic patterns and basin-scale variability.

![Global isotope distribution contour map](assets/images/contour_map.png)

### Temperature–salinity diagram

The T–S view overlays approximate σ0 reference contours. Practical Salinity is
used as an approximation of Absolute Salinity, and in-situ temperature as an
approximation of Conservative Temperature; the contours are reference guides,
not a substitute for a full TEOS-10 calculation.

![Temperature–salinity diagram](assets/images/ts_diagram.png)

### 4D visualization

Longitude, latitude, depth, and isotope information can be explored together
to inspect spatial gradients and vertical structure.

![4D visualization](assets/images/4d_d18O.png)

### Linked interactive selection

Selection in T–S space can be linked to sampling locations on the map, helping
users examine the geographic context of a selected water-mass subset.

![Geographic selection map](assets/images/selection_map.png)
![Selected locations linked from T–S space](assets/images/selection_ts.png)

## Scope of this website

This website covers the ten supported stable application pages. Historical development records and excluded experimental pages are not user workflows and are not presented as part of the stable interface.
