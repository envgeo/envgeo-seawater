---
title: 'EnvGeo-Seawater: An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data'
tags:
  - Python
  - Oceanography
  - Stable Isotopes
  - Data Visualization
  - Streamlit
authors:
  - name: Toyoho Ishimura
    orcid: 0000-0001-9708-3743
    affiliation: 1
affiliations:
  - name: Graduate School of Human and Environmental Studies, Kyoto University, Japan
    index: 1
date: 21 March 2026
bibliography: paper.bib
---

# Summary

EnvGeo-Seawater is a web-based interactive visualization platform for exploring marine
geochemical and hydrographic datasets, including stable water isotopes ($\delta^{18}$O and
$\delta$D), salinity, temperature, and depth.

The platform integrates approximately 50,000 seawater isotope records from major global
datasets, including the NASA GISS database [@schmidt1999] and the CoralHydro2k seawater
isotope database [@atwood2026], together with internally consistent regional datasets
analyzed under unified analytical protocols (e.g., around Japan; @kodama2024).

EnvGeo-Seawater enables simultaneous exploration of spatial distributions, cross-variable
relationships, and vertical structures through an integrated interface. By combining
multiple visualization modes—mapping, depth profiles, temperature–salinity diagrams,
regression analysis, and multi-dimensional (3D/4D) plots—the platform supports rapid
exploratory analysis and reproducible comparison of heterogeneous seawater datasets.

# Statement of Need

Seawater isotope measurements ($\delta^{18}$O, $\delta$D, and d-excess) are widely used in
oceanography and paleoclimate research to investigate ocean circulation, freshwater fluxes,
and climate processes. However, despite the availability of large public datasets,
integrated and interactive analysis across multiple variables and datasets remains limited.

Existing platforms primarily focus on data archiving and access, providing limited support
for exploratory visualization and cross-dataset comparison. As a result, researchers
typically rely on custom scripts and fragmented workflows to analyze relationships among
isotopic and hydrographic variables.

EnvGeo-Seawater addresses this gap by providing a unified, interactive environment that
integrates heterogeneous global datasets with internally consistent regional datasets. A
key contribution is the integration of regionally curated datasets for the area around
Japan, which provide consistent analytical quality and enhance reliability of comparative
analyses across spatial scales.

## State of the field

Oceanographic and geochemical datasets, particularly those including seawater stable
isotopes, are increasingly available through global databases such as the NASA GISS
seawater isotope database [@schmidt1999] and CoralHydro2k [@atwood2026]. However, these
datasets are often distributed across heterogeneous formats and lack integrated tools for
interactive exploration.

Existing oceanographic visualization tools such as Ocean Data View (ODV) [@schlitzer2018]
provide powerful capabilities for analyzing hydrographic data but are not specifically
designed to handle isotope datasets or to integrate multiple heterogeneous sources in a
unified, web-accessible environment. Furthermore, many tools require local installation and
are not optimized for rapid exploratory analysis or direct comparison with user-supplied
datasets.

EnvGeo-Seawater is specifically designed to integrate isotope datasets with hydrographic
variables in a unified, interactive framework, filling this gap with a lightweight,
browser-accessible tool that requires no local installation by end users of the deployed
application.

## Software design

EnvGeo-Seawater is implemented as a modular Python application using Streamlit
[@streamlit] for the web interface, Plotly [@plotly] for interactive visualization, and
Matplotlib for publication-quality figure generation. Scientific colormaps follow the
guidelines of @thyng2016 using the cmocean package, ensuring perceptually uniform and
colorblind-accessible color scales (e.g., *thermal* for temperature, *haline* for
salinity).

The application is organized around a shared utility module (`envgeo_utils`) that
centralizes data loading, column normalization, filtering, quality flagging, and
visualization helpers. Individual Streamlit pages import from this module, keeping
page-level code focused on layout and user interaction.

Core data-processing functions are exposed as a Python API independent of the web
interface, allowing programmatic access for custom analyses and reproducible workflows:

```python
import envgeo_utils

df = envgeo_utils.load_isotope_data("with [Global data sets]")
df_filtered = envgeo_utils.sidebar_filter_and_display(df, ...)
```

All core scientific datasets are included in the repository (total size < 30 MB).
After the Python environment has been installed locally, bundled data and most non-map
analysis workflows can be used without network access. User-supplied CSV and XLSX files
can be uploaded directly through the interface for in-session comparison with reference
datasets; uploaded data are handled in memory only and are not persisted to disk. Full
offline operation of web basemaps and browser assets remains under development and
verification.

This local-first design is particularly valuable aboard research vessels, where
satellite connectivity may be limited, unstable, or unavailable. Researchers can inspect
newly collected data for outliers, coordinate errors, missing values, unexpected depth
profiles, temperature–salinity structure, and isotope–hydrographic relationships while
still at sea. Early detection can inform remeasurement, additional sampling, and changes
to the remaining observation plan before the cruise ends.

The test suite covers core non-UI functionality including data loading, required-column
checking, numeric type conversion, gap-row insertion for depth profiles, and colorscale
selection. Tests are run with `pytest` from the repository root.

# Capabilities

The platform provides the following capabilities:

- Interactive spatial mapping with adaptive zoom
- Depth profile visualization with gap-aware plotting for discrete sampling data
- Temperature–salinity (T–S) diagrams with density contours ($\sigma_\theta$)
- Cross-variable analysis (e.g., salinity–$\delta^{18}$O relationships with regression)
- Multi-dimensional visualization (3D/4D exploration of spatial–temporal structures)
- Integration of global datasets (~50,000 records) and internally consistent regional datasets
- User data upload (CSV/XLSX) for direct comparison with reference datasets
- Export of publication-quality figures

# Research Impact

EnvGeo-Seawater has been used directly in multiple peer-reviewed research workflows in
marine and fisheries science. Because the platform provides rapid, interactive access to
the seawater isotope dataset of @kodama2024, it has supported isotope-based data
selection, visualization, and interpretation in studies that analyze fish migration and
habitat use using otolith or water isotope records.

Documented research applications include:

- @aono2024: investigation of Japanese sardine (*Sardinops melanostictus*) migration routes
  in the Sea of Japan, utilizing seawater isotope data for environmental reference.
- @sakamoto2024: analysis of vertical habitat selection by sardine juveniles, combining
  biological and isotopic records.
- @kuroki2025: reconstruction of experienced-temperature histories during anguillid eel
  (*Anguilla japonica*) larval migration using seawater isotope data.
- @sakamoto2026: tracing cross-shelf movements of early-life Japanese jack mackerel
  (*Trachurus japonicus*) using isotopic signatures of the surrounding seawater.

The platform has additionally been presented at three scientific conferences in Japan and
is used by collaborators at multiple institutions for exploratory isotope data analysis.
One manuscript currently under review independently cites EnvGeo-Seawater as a primary
data exploration tool.

# Implementation

The software is implemented in Python. The web interface uses Streamlit [@streamlit],
interactive figures use Plotly [@plotly], and publication-quality figures use Matplotlib.
Scientific colormaps use cmocean [@thyng2016]. The codebase is modular and designed to
support extension to additional datasets and visualization methods.

The application is designed for reproducibility and lightweight deployment, with all
required datasets included in the repository (< 30 MB) and minimal setup required for
local execution.

## AI Usage Disclosure

Portions of code structuring, documentation refinement, and language editing were assisted
by AI tools (ChatGPT, OpenAI; Claude, Anthropic).
All scientific design, data interpretation, and validation were performed solely by the author.

## Conflict of Interest

The author declares no conflicts of interest.

# References
