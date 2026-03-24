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

EnvGeo-Seawater is a web-based interactive visualization platform for exploring marine geochemical and hydrographic datasets, including stable water isotopes ($\delta^{18}$O and $\delta$D), salinity, temperature, and depth.

The platform integrates approximately 50,000 seawater isotope records from major global datasets, including the NASA GISS database [@schmidt1999] and the CoralHydro2k seawater isotope database [@atwood2026], together with internally consistent regional datasets (e.g., around Japan) [@kodama2024] analyzed under unified analytical protocols.

EnvGeo-Seawater enables simultaneous exploration of spatial distributions, cross-variable relationships, and vertical structures through an integrated interface. By combining multiple visualization modes—such as mapping, depth profiles, temperature–salinity diagrams, regression analysis, and multi-dimensional (3D/4D) plots—the platform supports rapid exploratory analysis and reproducible comparison of heterogeneous seawater datasets.

# Statement of Need

Seawater isotope measurements (e.g., $\delta^{18}$O, $\delta$D, and d-excess) are widely used in oceanography and paleoclimate research to investigate ocean circulation, freshwater fluxes, and climate processes. However, despite the availability of large public datasets, integrated and interactive analysis across multiple variables and datasets remains limited.

Existing platforms primarily focus on data archiving and access, providing limited support for exploratory visualization and cross-dataset comparison. As a result, researchers typically rely on custom scripts and fragmented workflows to analyze relationships among isotopic and hydrographic variables.

EnvGeo-Seawater addresses this gap by providing a unified, interactive environment that integrates heterogeneous global datasets with internally consistent regional datasets analyzed under unified analytical protocols. This design enables rigorous cross-comparison across datasets while minimizing methodological inconsistencies.

A key contribution of this platform is the integration of regionally curated datasets for around Japan, which provide consistent analytical quality and enhance the reliability of comparative analyses across spatial scales.

## State of the field

Oceanographic and geochemical datasets, particularly those including seawater stable isotopes (e.g., $\delta^{18}$O, $\delta$D), are increasingly available through global and regional databases such as the NASA GISS seawater isotope database and CoralHydro2k. However, these datasets are often distributed across heterogeneous formats and lack integrated tools for interactive exploration.

Existing oceanographic visualization tools, such as Ocean Data View (ODV)  [@schlitzer2018], provide powerful capabilities for analyzing hydrographic data but are not specifically designed to handle isotope datasets or to integrate multiple sources in a unified, web-accessible environment. Furthermore, many existing tools require local installation and are not optimized for rapid exploratory analysis or comparison with user-supplied datasets.

As a result, there is a gap in the availability of lightweight, accessible tools that enable integrated visualization and analysis of seawater isotope and hydrographic data across multiple datasets.

In contrast, EnvGeo-Seawater is specifically designed to integrate isotope datasets with hydrographic variables in a unified, interactive framework.

## Software design

EnvGeo-Seawater is designed as a modular Python-based application that separates data processing, visualization, and user interface components.

The modular structure allows individual components to be reused independently of the web interface.

The design emphasizes reusability and extensibility, enabling integration with additional datasets and facilitating future development.

The core functionality is implemented as reusable Python modules that handle data loading, filtering, and visualization. These modules are exposed through a simple API, allowing users to access key functions programmatically for custom analyses.

In addition, command-line execution can be used to support reproducible workflows without relying on the graphical interface.

The interactive web interface is implemented using Streamlit, which provides an accessible platform for exploratory analysis. The application supports multiple visualization types, including map-based exploration, temperature–salinity diagrams, depth profiles, and regression analyses.

## Research impact

EnvGeo-Seawater provides a unified platform for exploring seawater isotope and hydrographic datasets, enabling researchers to more efficiently investigate relationships between physical and geochemical parameters.

By integrating multiple datasets into a single interface, the software reduces barriers to data access and comparison, supporting both regional and global analyses. The ability to upload and compare user datasets further enhances its utility for research and education.

This tool is particularly relevant for studies of ocean circulation, water mass mixing, and paleoclimate reconstruction, where isotope data play a critical role. By improving accessibility and usability of these datasets, EnvGeo-Seawater has the potential to accelerate data-driven research in oceanography and geochemistry.

The software has been used in the author's research workflows and has supported analyses presented at multiple scientific conferences in Japan (three presentations to date). It has also been adopted by collaborators for their own research and has been used in their presentations.

# Capabilities

The platform provides the following capabilities:

- Interactive spatial mapping with adaptive zoom  
- Depth profile visualization with gap-aware plotting for discrete sampling data  
- Temperature–salinity (T–S) diagrams with density contours ($\sigma_\theta$)  
- Cross-variable analysis (e.g., salinity–$\delta^{18}$O relationships with regression)  
- Multi-dimensional visualization (3D/4D exploration of spatial–temporal structures)  
- Integration of global datasets (~50,000 records) and internally consistent regional datasets  
- User data upload for direct comparison with reference datasets  
- Export of publication-quality figures  

# Implementation

The software is implemented in Python using Streamlit [@streamlit] for the web interface, Plotly [@plotly] for interactive visualization, and Matplotlib for high-quality figure generation. The codebase is modular and designed to support extension to additional datasets and visualization methods.

The application is designed for reproducibility and lightweight deployment, with all required datasets included in the repository (<30 MB) and minimal setup required for local execution.

# Example Use Case

EnvGeo-Seawater supports exploratory analysis across spatial and temporal scales, including visualization of global $\delta^{18}$O distributions, salinity–isotope relationships, and vertical structures.

The platform also enables direct comparison between user-provided datasets and curated reference datasets within a unified analytical framework.

## AI Usage Disclosure

Portions of code structuring, documentation refinement, and language editing were assisted by AI tools (ChatGPT, OpenAI).  
All scientific design, data interpretation, and validation were performed solely by the author.

## Conflict of Interest

The author declares no conflicts of interest.

# References
