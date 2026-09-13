# EnvGeo-Seawater 🌊

EnvGeo-Seawater is an interactive platform for exploring seawater isotope and hydrographic data.

[日本語版 README](README_Japanese.md)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://envgeo.h.kyoto-u.ac.jp/sw_jpn/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Current version:** 1.3.0

**An interactive platform for exploring seawater isotope and hydrographic data.**

---

## Overview

EnvGeo-Seawater has been used for exploratory analysis of seawater isotope datasets in marine geochemistry research.

EnvGeo-Seawater is a web-based interactive visualization platform for marine geochemical and hydrographic datasets, including stable water isotopes (δ¹⁸O, δD), salinity, temperature, and depth.

It integrates curated regional datasets (e.g., around Japan) and major global datasets (~50,000 records), enabling consistent cross-comparison under unified analytical conditions.

The platform is designed to support both **exploratory data analysis** and **reproducible research workflows** in marine geochemistry and oceanography.

---

## Key Features

- 🌍 Interactive map visualization with adaptive zoom  
- 📊 Depth profiles with gap-aware plotting  
- 📈 Temperature–Salinity (T–S) diagrams with density contours (σθ)  
- 📉 Regression analysis (e.g., salinity–δ¹⁸O relationships)  
- 🧭 3D / 4D visualization of spatial–temporal structures  
- 📂 User data upload for comparison with reference datasets (currently limited functionality)
- 🧾 Transparent data handling (filtered / excluded samples clearly reported)  
- 🖼️ Export of publication-quality figures  

---

## Main Pages

The stable Streamlit app uses `home.py` for the about, data-source, manual,
update-log, and Japanese information tabs. The `pages/` directory is reserved
for the main visualization tools:

- `pages/03_3D_Visualizer.py`  
  Interactive 3D plots for isotope-hydrographic relationships and selected sample locations.

- `pages/04_4D_Visualizer.py`  
  Main 3D/4D seawater visualizer for longitude, latitude, depth, and selected variables.

- `pages/05_3D4D_Visualizer_Uploader.py`  
  User-upload visualizer for Excel-based custom datasets and comparison with reference data.

- `pages/31_Salinity-d18O_Relationship.py`  
  Salinity-δ18O relationship plots with optional regression lines.

- `pages/32_Isotope_Hydrographic_Mapping.py`  
  Isotope and hydrographic maps for d18O, dD, d-excess, salinity, and temperature distributions.

- `pages/34_T-S_diagram.py`  
  Temperature-salinity diagrams with density contours.

- `pages/37_Depth_Profile.py`  
  Depth profiles for δ18O, δD, d-excess, temperature, and salinity.

- `pages/35_Custom_Parameter_Plot_beta.py`  
  Experimental custom 2D parameter plots with selectable X axis, Y axis, color, and marker size.

- `pages/53_Vertical_Section_Visualizer.py`  
  Vertical Section Visualizer beta. This experimental page is used to refine section-line selection, interpolation, bathymetry handling, and vertical-section plotting.

- `pages/90_Integrated_Visualizer_beta.py`  
  Experimental integrated visualizer. It can run the original visualization workflows inside one beta page, temporarily merge uploaded user data into those workflows, and test a shared-filter tab workflow with ocean-region map presets for future integration. The standalone uploader page remains separate because it uses its own upload-first workflow.

The former standalone about page was merged into `home.py`.

---

## Local Diagnostic Tool

The package includes a local Streamlit-based environment checker. It is intended
for local use when confirming Python paths and installed dependency versions.
The diagnostic results can be exported as CSV or PDF reports:

```bash
streamlit run tools/env_check_streamlit.py
```

---

## User Data Integration

`91_USER_UPLOAD_UNPUB.xlsx` is provided as a template for user-defined comparison data.

- Replace the sample rows with your own measurements  
- Keep the same column structure  
- Run the app locally to integrate your dataset  

This allows direct comparison between user datasets and curated reference datasets across all supported visualizations.

Uploaded files are intended to be handled in memory during the current
Streamlit session only. The integrated beta workflow does not save uploaded
files or merged user/reference datasets to local or server storage.

---

## Why this tool?

EnvGeo-Seawater enables integrated exploration of isotope and hydrographic data, which are typically analyzed separately.

Unlike conventional tools that treat datasets and visualization separately, EnvGeo-Seawater provides an integrated, interactive framework for exploring isotope–hydrographic relationships and enables direct comparison with user-provided data.

- Integrated multi-parameter visualization  
- Consistent filtering across datasets  
- Direct comparison between user data and curated datasets  
- A unified analytical framework across regional and global datasets  

---

## Data Characteristics

The platform includes internally consistent datasets (e.g., around Japan) analyzed under unified criteria, enabling rigorous cross-comparison.

A key strength is that a substantial portion of the regional datasets (around Japan) has been analyzed by the author using consistent analytical protocols, ensuring high comparability across sampling campaigns.

In particular, regional datasets curated by the author provide:

- Consistent analytical methods  
- Harmonized data structure  
- High comparability across sampling campaigns  

This ensures that observed patterns reflect environmental signals rather than methodological differences.

---

## Data Availability

All datasets included in this repository are either publicly available or redistributed in accordance with their respective licenses.

- Provided in a standardized format for immediate use  

Unpublished or restricted datasets are **not included**.

---

## Installation & Requirements

The currently verified environment is **Python 3.10.15** in the Anaconda
`envgeo_streamlit142` environment. Python 3.12 support should be re-verified
with the dependency set and Streamlit pages before it is recommended as the
default environment.

### 💡 Special Note for macOS (Apple Silicon) Users:
To avoid build errors with geospatial libraries, it is highly recommended to use **Conda** to install core dependencies before running pip:

```bash
# 1. Create and activate environment
conda create -n envgeo python=3.10
conda activate envgeo

# 2. Install pre-built geospatial binaries
conda install -c conda-forge proj pyproj cartopy -y

# 3. Install remaining requirements
pip install -r requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/envgeo/seawater_map.git
cd seawater_map
pip install -r requirements.txt
streamlit run home.py
```
Then open the local URL shown in the terminal (typically http://localhost:8501).

---

## API Usage

Core functionality can be accessed programmatically:

```python
import envgeo_utils

df = envgeo_utils.load_isotope_data("with [Global data sets]")

df_filtered = envgeo_utils.sidebar_filter_and_display(
    df,
    ref_data="with [Global data sets]",
    data_source_JAPAN_SEA="Kodama et al. (2024) [ECS - Japan Sea]",
    data_source_AROUND_JAPAN="with [Around Japan]"
)
```

---

## Testing

Basic functionality can be verified using pytest:

```bash
pytest
```

---

## Directory Structure

- `home.py`  
  Main Streamlit entry point for EnvGeo-Seawater.

- `envgeo_utils.py`  
  Shared utilities for dataset loading, data cleaning, filtering, common Plotly
  layout, map styles, coastline loading, and table display.

- `pages/`  
  Stable visualization page files shown in the app sidebar.

- `tools/`  
  Local support tools that are included in the package but not shown in the
  public Streamlit app sidebar.

- `dataset/`  
  Curated seawater isotope and hydrographic datasets used by the application.

- `data/`  
  Sample files and media used by the app, including template data, GIF/MP4
  examples, and temporary coastline sample files.

- `data_text/`  
  Markdown resources for about text, references, manuals, Japanese notes, and
  English/Japanese update logs.

- `images/`  
  Figures and example outputs used in README and documentation.

- `coastline/`  
  Local coastline coordinate files for map and 3D reference overlays.

- `test/`  
  Basic pytest tests for imports, dataset loading, numeric conversion, gap-row
  insertion, colorscales, and coastline loading.

Core functionality is implemented as reusable Python functions in `envgeo_utils.py`, allowing programmatic access outside the Streamlit interface.

---

## Usage

1. Select a dataset (Japan Sea / Around Japan / Global)  
2. Apply filters (location, depth, time, parameters)  
3. Explore:  
   - Maps  
   - T–S diagrams  
   - Depth profiles  
   - Regression plots  
4. Upload your own data for comparison (optional)  
5. Export figures  

---

## Reproducibility

This repository contains:

- The full source code of the visualization platform  
- All required public datasets (lightweight, <30 MB total)  
- A template for user-defined data integration  

Therefore, all figures and analyses generated by this platform are **fully reproducible** in a local environment.

All figures can be reproduced by running the application locally using the provided datasets and following the Quick Start instructions above.

Due to the interactive nature of the application, functionality is validated through manual testing and visual inspection.

Experimental features are included for development purposes and may change in future versions. These features are clearly separated from the core functionalities.

---

## Limitations

- Interactive 3D/4D visualizations can be slow with large global selections.
- Plotly 3D interaction is best on a PC; 2D plots are more suitable for
  smartphones and tablets.
- User data upload is currently Excel-centered and depends on the expected
  column structure.
- Dataset interpretation depends on the original data sources, analytical
  methods, and metadata. Users should cite and evaluate the underlying data
  providers when using outputs in publications.

---

## Data Sources and Attribution

The platform integrates major seawater isotope datasets:

- CoralHydro2k: a global seawater oxygen isotope database (Atwood et al., 2026, ESSD)  
- NASA GISS Global Seawater Oxygen-18 Database (Schmidt et al., 1999)  
- Kodama et al. (2024), *Geochemical Journal*  
- Additional regional datasets  

For publication, teaching material, or redistributed outputs, cite both
EnvGeo-Seawater and the original dataset providers used in the selected
visualization. Source details are shown in the app and in the Markdown files
under `data_text/`.

## Live Demo

Primary stable demo:
https://envgeo-seawater-map.streamlit.app

Stable demo with experimental updates:
https://envgeo-seawater-pre.streamlit.app

---

## Citation

Ishimura, T. (2026).  
EnvGeo-Seawater: An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data.  
Archival citation details will be added after release.

---

## Examples

EnvGeo-Seawater enables multi-scale exploration of seawater isotope and hydrographic data, from global distributions to detailed interactive analysis.

---

### Global Isotope Distribution (Contour Map)

Spatial distribution of seawater δ18O at the global scale, based on integrated datasets (approximately 50,000 records).  
Contour interpolation highlights large-scale oceanographic patterns and basin-scale variability.

![Global map](images/contour_map.png)

---

### Temperature–Salinity Diagram

Temperature–salinity (T–S) relationships with overlaid density contours (σθ).  
This visualization supports identification of water masses and examination of isotope–hydrography relationships.

![TS diagram](images/ts_diagram.png)

---

### 4D Visualization (Longitude–Latitude–Depth–δ18O)

Multi-dimensional visualization of seawater isotope data, incorporating spatial coordinates and depth.  
This allows exploration of vertical structure and spatial gradients simultaneously.

![4D](images/4d_d18O.png)

---

### Interactive Selection (Map–T–S Linkage)

Linked visualization between T–S space and geographic location.  
Selected subsets in the T–S diagram are dynamically highlighted on the map, enabling intuitive interpretation of water mass origins.


![](images/selection_map.png)
![Highlight the corresponding sampling locations on the map.](images/selection_ts.png)


---

## License

MIT License
