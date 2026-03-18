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
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Kyoto University, Japan
    index: 1
date: 19 March 2026
bibliography: paper.bib
---

# Summary

EnvGeo-Seawater is a web-based interactive visualization platform for exploring marine geochemical and hydrographic datasets, including stable water isotopes ($\delta^{18}$O and $\delta$D), salinity, temperature, and depth. The platform integrates nearly 50,000 seawater isotope records from global datasets, including the NASA GISS database and the CoralHydro2k seawater isotope database, together with regionally curated datasets around Japan.

Users can investigate spatial distributions, cross-variable relationships, and vertical structures through an intuitive interface. By combining multiple visualization modes—such as mapping, depth profiles, temperature–salinity diagrams, regression analysis, and multi-dimensional (3D/4D) plots—the platform enables rapid exploratory analysis and comparison of seawater datasets across regions and time periods.

# Statement of Need

Seawater isotope measurements (e.g., $\delta^{18}$O, $\delta$D, and d-excess) are widely used in oceanography and paleoclimate research to investigate ocean circulation, freshwater fluxes, and climate processes. Although several public repositories provide access to such datasets, interactive exploration and consistent cross-dataset comparison remain challenging.

Existing platforms such as NOAA Paleoclimatology and PANGAEA primarily focus on data access and archiving, offering limited support for integrated visualization and exploratory analysis. As a result, researchers often rely on custom scripts and local workflows to analyze relationships among isotopic and hydrographic variables.

EnvGeo-Seawater addresses this gap by providing a unified, web-based environment for interactive exploration of seawater isotope and hydrographic data. The platform integrates approximately 50,000 observations from global databases with internally consistent regional datasets (e.g., around Japan) analyzed under unified criteria. This enables more rigorous cross-comparison across regions and time periods than is typically possible with heterogeneous datasets.

# Capabilities

The platform provides the following capabilities:

- Interactive spatial mapping of seawater isotope observations with adaptive zoom  
- Depth profile visualization with gap-aware plotting for discrete sampling data  
- Temperature–salinity (T–S) diagrams with density contours ($\sigma_\theta$)  
- Cross-variable analysis (e.g., salinity–$\delta^{18}$O relationships and regression)  
- Multi-dimensional visualization (3D and 4D exploration of spatial–temporal structures)  
- Integration of large-scale global datasets (~50,000 records) and curated regional datasets  
- User-upload functionality for direct comparison with reference datasets  
- Export of high-resolution figures for publication use  

# Implementation

The software is implemented in Python using Streamlit for the web interface, Plotly for interactive visualization, and Matplotlib for high-quality figure generation.

# Example Use Case

EnvGeo-Seawater can be used for rapid exploratory analysis of seawater isotope datasets across multiple spatial and temporal scales. Users can visualize global distributions of $\delta^{18}$O, examine relationships between salinity and isotopic composition, and analyze vertical structures using depth profiles.

The platform also enables comparison of user-provided datasets with curated reference datasets, allowing researchers to assess consistency and identify anomalies within a unified analytical framework.

# References
