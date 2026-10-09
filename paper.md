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
date: 26 September 2026
bibliography: paper.bib
---

# Summary

EnvGeo-Seawater is a web-based interactive visualization platform for exploring marine geochemical and hydrographic datasets, including stable water isotopes ($\delta^{18}$O and $\delta$D), salinity, temperature, and depth.

The platform integrates approximately 50,000 seawater isotope records from
major global datasets, including the NASA GISS database [@schmidt1999] and the
CoralHydro2k seawater isotope database [@atwood2026], together with cited
regional reference data and the internally consistent **EnvGeo Dataset
[ECS–Japan Sea]** core collection (primary reference: Kodama et al. 2024)
[@kodama2024].

EnvGeo-Seawater enables simultaneous exploration of spatial distributions,
cross-variable relationships, and vertical structures through an integrated
interface. By combining mapping, depth profiles, temperature–salinity diagrams,
regression analysis, and multi-dimensional (3D/4D) plots, it supports rapid
exploratory analysis and documented comparison of heterogeneous seawater
datasets.

# Statement of Need

Seawater isotope measurements (e.g., $\delta^{18}$O, $\delta$D, and d-excess) are widely used in oceanography and paleoclimate research to investigate ocean circulation, freshwater fluxes, and climate processes. However, despite the availability of large public datasets, integrated and interactive analysis across multiple variables and datasets remains limited.

Existing platforms primarily focus on data archiving and access, providing limited support for exploratory visualization and cross-dataset comparison. As a result, researchers typically rely on custom scripts and fragmented workflows to analyze relationships among isotopic and hydrographic variables.

EnvGeo-Seawater addresses this gap by providing a unified, interactive
environment that integrates heterogeneous global datasets, cited regional
reference data, and an internally consistent Japan-region core collection.
Source-specific citations and provenance remain visible so that comparisons can
be interpreted in light of their respective sampling and analytical contexts.

Integrating independently curated reference datasets also creates a practical
quality-assurance challenge: records inherited from a common original source,
or records affected by rounding and revised metadata, can resemble duplicates.
EnvGeo-Seawater therefore provides a provenance-aware, read-only
cross-dataset overlap screen. It uses explicit user-visible criteria for
matching sampling time and for comparing coordinates, depth, salinity, and
$\delta^{18}$O; it separates strong candidates from cases requiring review and
exports the matching evidence for inspection. Source records are preserved and
candidate status is not treated as confirmation of duplication. The shared
filter sidebar retains all records by default and offers reversible,
opt-in sensitivity screens: a deterministic one-to-one subset whose recorded
differences are compatible with inferred rounding precision, and a broader
Strong-candidate screen that is explicitly not a confirmed de-duplication
result. Current provisional Strong defaults are ≤0.1° for latitude and
longitude, ≤5 m for depth, ≤0.1 for salinity, and ≤0.1‰ for $\delta^{18}$O;
the broader Review defaults are ≤0.2°, ≤10 m, ≤0.2, and ≤0.2‰, respectively.
These visible, adjustable values are screening criteria rather than universal
measurement-error thresholds. This makes cross-dataset integration more
transparent while retaining the source-level context needed for scientific
interpretation.

A key contribution is the integration of the regionally curated EnvGeo Dataset
[ECS–Japan Sea] collection, currently represented by Kodama et al. (2024),
which provides a consistent analytical baseline for
exploratory comparison across its sampled locations and periods.

## State of the field

Oceanographic and geochemical datasets, particularly those including seawater stable isotopes (e.g., $\delta^{18}$O, $\delta$D), are increasingly available through global and regional databases such as the NASA GISS seawater isotope database and CoralHydro2k. However, these datasets are often distributed across heterogeneous formats and lack integrated tools for interactive exploration.

Existing oceanographic visualization tools, such as Ocean Data View (ODV)
[@schlitzer2002], provide powerful capabilities for analyzing hydrographic
data. EnvGeo-Seawater instead provides a preconfigured web interface for
seawater-isotope and hydrographic exploration across its bundled sources and
for session-only comparison with user-supplied data.

As a result, there is a gap in the availability of lightweight, accessible tools that enable integrated visualization and analysis of seawater isotope and hydrographic data across multiple datasets.

In contrast, EnvGeo-Seawater is specifically designed to integrate isotope datasets with hydrographic variables in a unified, interactive framework.

## Software design

EnvGeo-Seawater is organized as a modular Python application with shared
modules for asset resolution, data loading, filtering, and visualization, plus
Streamlit page scripts for the interactive interface. These modules are
reused within the application; the project does not currently claim a separate
stable public API for programmatic analysis.

The design supports the addition of new, source-documented datasets and future
maintenance without changing the existing page-oriented workflow. Installed
console commands start the application or its local diagnostic tool; the
interactive interface remains the primary analysis route.

The interactive web interface is implemented using Streamlit, which provides an accessible platform for exploratory analysis. The application supports multiple visualization types, including map-based exploration, temperature–salinity diagrams, depth profiles, and regression analyses.

A figure-supported, page-by-page user guide is available in English and
Japanese at <https://envgeo.github.io/seawater_map/>.

Oceanographic colour scales follow the cmocean design guidance [@thyng2016].
T–S diagrams include approximate $\sigma_0$ reference contours calculated
with the Gibbs SeaWater (GSW) implementation of TEOS-10 [@mcdougall2011],
using Practical Salinity and in-situ temperature as display-oriented proxies.
They are visual reference contours rather than fully converted TEOS-10 density
values for individual observations. The Vertical Section Visualizer can use a
project-derived, downsampled GEBCO 2025 Grid for bathymetric context
[@gebco2025].

## Research impact

EnvGeo-Seawater provides a unified platform for exploring seawater isotope and hydrographic datasets, enabling researchers to more efficiently investigate relationships between physical and geochemical parameters.

By integrating multiple datasets into a single interface, the software reduces barriers to data access and comparison, supporting both regional and global analyses. The ability to upload and compare user datasets further enhances its utility for research and education.

This tool is particularly relevant for studies of ocean circulation, water mass mixing, and paleoclimate reconstruction, where isotope data play a critical role. By improving accessibility and usability of these datasets, EnvGeo-Seawater has the potential to accelerate data-driven research in oceanography and geochemistry.

Before EnvGeo-Seawater had an archival software DOI, it was used in the
author's and collaborators' research workflows to select, explore, and
visualize subsets of the regional seawater isotope dataset reported by Kodama
et al. (2024). These workflows supported isotope-based interpretation in
studies of Japanese sardine migration [@aono2024], sardine juvenile habitat
selection [@sakamoto2024], anguillid eel larval experienced-temperature
reconstruction [@kuroki2025], and early-life Japanese jack mackerel movement
[@sakamoto2026]. The resulting publications cite the underlying Kodama et al.
(2024) dataset paper rather than EnvGeo-Seawater; they are evidence of
workflow use, not direct software citations. The platform has also been used
in scientific conference presentations in Japan.

## Future directions

The data model is designed to accommodate additional datasets after their
sources, provenance, and redistribution status have been recorded. Reusable
visualization, asset-resolution, and distribution components may also support
future related EnvGeo applications. These are future directions and are not
part of the v1.3.4 public release scope.

# Capabilities

The platform provides the following capabilities:

- Interactive spatial mapping with adaptive zoom  
- Depth profile visualization with gap-aware plotting for discrete sampling data  
- Temperature–salinity (T–S) diagrams with approximate $\sigma_0$ reference contours
- Cross-variable analysis (e.g., salinity–$\delta^{18}$O relationships with regression)  
- Multi-dimensional visualization (3D/4D exploration of spatial–temporal structures)  
- Integration of global datasets (~50,000 records), cited regional reference data, and the EnvGeo Dataset [ECS–Japan Sea] core collection (primary reference: Kodama et al. 2024)
- Provenance-aware overlap screening with explicit criteria, strong/review candidate classes, downloadable audit evidence, and reversible display sensitivity screens
- User data upload for direct comparison with reference datasets  
- Export of publication-quality figures  

# Implementation

The software is implemented in Python using Streamlit [@streamlit] for the web interface, Plotly [@plotly] for interactive visualization, and Matplotlib for high-quality figure generation. The codebase is modular and designed to support extension to additional datasets and visualization methods.

The bundled application data and assets occupy less than 30 MB. Local execution
requires installation of the declared Python dependencies. The package declares
Python >=3.10; wheel builds and isolated installs are verified in continuous
integration on Python 3.10 and 3.12, with Python 3.12 recommended for local
installation. The v1.3.4 release
artifact is prepared for package-index distribution: once published, users can
install it with `pip install envgeo-seawater` and start the local application
with `envgeo-seawater`.

# Availability

The stable public source repository is
https://github.com/envgeo/seawater_map. Version 1.3.4 was released on 3 October
2026 and archived in Zenodo at https://doi.org/10.5281/zenodo.23117784. The
all-versions concept DOI is https://doi.org/10.5281/zenodo.23117783.

# Example Use Case

EnvGeo-Seawater supports exploratory analysis across spatial and temporal scales, including visualization of global $\delta^{18}$O distributions, salinity–isotope relationships, and vertical structures.

The platform also enables direct comparison between user-provided datasets and curated reference datasets within a unified analytical framework.

A particularly important use case is rapid data checking aboard research vessels,
where satellite connectivity may be limited or unavailable. After local installation,
researchers can use bundled reference data and local measurements to inspect data
quality, sampling positions, depth profiles, temperature–salinity structure, and
isotope–hydrographic relationships. Detecting outliers, coordinate errors, missing
values, or unexpected profiles while still at sea can inform remeasurement,
additional sampling, and adjustments to the remaining observation plan.

## AI Usage Disclosure

Development of this software, from version 1.3 onward, has made substantial use of AI
coding assistants for code review, implementation drafting, refactoring, test design and
authoring, bug investigation, and documentation. The tools used were OpenAI Codex and
Anthropic Claude Code. Portions of language editing in earlier drafts of this manuscript
were also assisted by a general-purpose AI language tool (ChatGPT, OpenAI).

All AI-assisted output was reviewed, edited, and verified by the corresponding author
before being adopted into the codebase, tests, documentation, or this manuscript. The
author performed all scientific design, data interpretation, and validation decisions,
and takes full responsibility for the accuracy, originality, licensing, and ethical and
legal compliance of the software and this manuscript. AI tools are not authors or
co-developers of this work.

## Conflict of Interest

The author declares no conflicts of interest.

# References
