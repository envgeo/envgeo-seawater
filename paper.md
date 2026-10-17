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
date: 17 October 2026
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

Although public datasets provide essential archival access, researchers often
still assemble custom and fragmented workflows to explore relationships among
isotopic and hydrographic variables across separately curated sources.

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
result. The default thresholds are user-visible and adjustable; they are
screening criteria rather than universal measurement-error thresholds. This
makes cross-dataset integration more
transparent while retaining the source-level context needed for scientific
interpretation.

The locally installable application also supports a reproducible workflow that
does not depend on the continued availability of a hosted web service:
bundled-data loading, filtering, analysis, and figure generation can be run
locally after installation. Online basemap tiles and external source links
remain network-dependent features and are not represented as offline data.

A key contribution is the integration of the regionally curated EnvGeo Dataset
[ECS–Japan Sea] collection, currently represented by Kodama et al. (2024),
which provides a consistent analytical baseline for
exploratory comparison across its sampled locations and periods.

# State of the field

Oceanographic and geochemical datasets, particularly those including seawater stable isotopes (e.g., $\delta^{18}$O, $\delta$D), are increasingly available through global and regional databases such as the NASA GISS seawater isotope database and CoralHydro2k. However, these datasets are often distributed across heterogeneous formats and lack integrated tools for interactive exploration.

Existing oceanographic visualization tools, such as Ocean Data View (ODV)
[@schlitzer2002], provide powerful capabilities for analyzing hydrographic
data. EnvGeo-Seawater complements rather than replaces such general-purpose
tools: it provides a lightweight, local, seawater-isotope-specific workflow
across bundled cited sources and session-only user-supplied data, while
retaining source-specific provenance and citations, supporting an
overlap-candidate audit, and providing figure-oriented interactive
exploration. This specialization supports comparison of isotope and
hydrographic variables across heterogeneous, cited seawater datasets without
first assembling a workflow from general-purpose components.

# Software design

EnvGeo-Seawater is organized as a modular Python application with shared
modules for asset resolution, data loading, filtering, and visualization, plus
Streamlit page scripts for the interactive interface. These modules are
reused within the application, while the interactive interface is the primary
analysis route.

The design supports the addition of new, source-documented datasets and future
maintenance without changing the existing page-oriented workflow. Installed
console commands start the application or its local diagnostic tool.

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

# Research impact

Before the software received an archival DOI, EnvGeo-Seawater was used in the
author's and collaborators' research workflows to select, explore, and
visualize subsets of the regional seawater isotope dataset reported by Kodama
et al. (2024). These workflows supported isotope-based interpretation in
studies of Japanese sardine migration [@aono2024], sardine juvenile habitat
selection [@sakamoto2024], anguillid eel larval experienced-temperature
reconstruction [@kuroki2025], and early-life Japanese jack mackerel movement
[@sakamoto2026]. The resulting publications cite the underlying Kodama et al.
(2024) dataset paper rather than EnvGeo-Seawater; they are evidence of
workflow use, not direct software citations. The platform has also been used
in four scientific conference presentations in Japan. For example, a 2024
presentation introducing visualisation of seawater stable-isotope data around
Japan and the online interactive database is publicly archived [@ishimura2024conference].

# Future directions

The data model is designed to accommodate additional datasets after their
sources, provenance, and redistribution status have been recorded. Reusable
visualization, asset-resolution, and distribution components may also support
future related EnvGeo applications. These are future directions and are not
part of the v1.3.5 stable release scope.

# Capabilities

The platform provides the following capabilities:

- Interactive spatial mapping with adaptive zoom  
- Shared interactive filtering by geographic extent, sampling year and month, depth, dataset, and citation
- Depth profile visualization with gap-aware plotting for discrete sampling data  
- Temperature–salinity (T–S) diagrams with approximate $\sigma_0$ reference contours
- Cross-variable analysis (e.g., salinity–$\delta^{18}$O relationships with regression)  
- Consistent calculation and visualization of d-excess from paired $\delta^{18}$O and $\delta$D measurements, where available, across spatial, profile, and multi-dimensional views
- Multi-dimensional visualization (3D/4D exploration of spatial–temporal structures)  
- Integration of global datasets (~50,000 records), cited regional reference data, and the EnvGeo Dataset [ECS–Japan Sea] core collection (primary reference: Kodama et al. 2024)
- Provenance-aware dataset and citation selection, transparent quality/exclusion reporting, and overlap screening with explicit criteria, strong/review candidate classes, downloadable audit evidence, and reversible display sensitivity screens
- Session-only user-data upload, quality checking, and direct comparison with reference datasets
- Optional local-coastline map fallback for offline or degraded-network use
- Export of publication-quality figures  

# Implementation

The software is implemented in Python using Streamlit [@streamlit] for the web interface, Plotly [@plotly] for interactive visualization, and Matplotlib [@hunter2007] for high-quality figure generation. The codebase is modular and designed to support extension to additional datasets and visualization methods.

The bundled data and visual assets occupy approximately 30 MB when unpacked.
Local execution requires installation of the declared Python dependencies. The
package declares Python >=3.10; wheel builds and isolated installs are verified
in continuous integration on Python 3.10 and 3.12, with Python 3.12
recommended for local installation. The v1.3.5 release is published on PyPI:
users can install it with `python -m pip install envgeo-seawater` and start the local
application with `envgeo-seawater`.

# Availability

The stable public source repository is
https://github.com/envgeo/seawater_map. Version 1.3.5 was released on 17 October
2026 (GitHub Release: https://github.com/envgeo/seawater_map/releases/tag/v1.3.5)
and is distributed through PyPI at https://pypi.org/project/envgeo-seawater/1.3.5/.
The version-specific archival record is https://doi.org/10.5281/zenodo.23428743;
the all-versions concept DOI is https://doi.org/10.5281/zenodo.23117783.

# Example Use Case

EnvGeo-Seawater supports exploratory analysis in research and education settings across spatial and temporal scales, including visualization of global $\delta^{18}$O distributions, salinity–isotope relationships, and vertical structures.

The platform also enables direct comparison between user-provided datasets and curated reference datasets within a unified analytical framework.

A particularly important use case is rapid data checking aboard research vessels,
where satellite connectivity may be limited or unavailable. After local installation,
researchers can use bundled reference data and local measurements to inspect data
quality, sampling positions, depth profiles, temperature–salinity structure, and
isotope–hydrographic relationships. Detecting outliers, coordinate errors, missing
values, or unexpected profiles while still at sea can inform remeasurement,
additional sampling, and adjustments to the remaining observation plan.

# AI Usage Disclosure

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

# Conflict of Interest

The author declares no conflicts of interest.

# Acknowledgements

This work was supported in part by JSPS KAKENHI Grant Number JP24H00523;
the funder had no role in the software design, implementation, validation,
or decision to submit this manuscript.

# References
