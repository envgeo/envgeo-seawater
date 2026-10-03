# Geospatial Assets — Usage, Placement, Licence, and Update Policy

This document describes the geographic and scientific assets bundled in the
EnvGeo-Seawater repository: the coastline CSV files, the Natural Earth 50m
land polygon shapefile, and the GEBCO bathymetry grid.

**Status:** current bundled files and package-data scope rechecked 2026-09-30.
The corresponding Japanese record is
[`geospatial_assets_Japanese.md`](geospatial_assets_Japanese.md).

---

## 1. Coastline CSV files

| Item | Details |
|---|---|
| Location | `coastline/` (repository root) |
| Files | `world_coastline_coordinates_50m.csv`, `world_coastline_coordinates_110m.csv` |
| Purpose | Lightweight offline coastline overlay for all interactive Plotly/Mapbox maps and the Cartopy static maps in page 32 |
| Format | CSV with `Longitude`, `Latitude` columns; segments are separated by missing-coordinate rows where present |
| Loaded by | `envgeo_utils.load_coastline_data()` |
| Drawn by | `envgeo_utils.add_coastline_overlay()` (Plotly Scattermapbox) and `envgeo_utils.plot_bundled_coastline()` (Matplotlib/Cartopy) |

### Licence

The coastline data is derived from Natural Earth public-domain sources.
The detailed provenance/checksum record for the land shapefile is stored with
that asset. The retained coastline source workspace identifies both the 50m and
110m coastline shapefiles as Natural Earth v4.1.0; their source archives and
the corresponding coordinate workbooks were saved on 2025-01-24. The current
CSV files were created in that workspace from those source files.
Natural Earth data is public domain — no licence required.

### Update policy

These CSV files are generated from Natural Earth vector data and are stored
directly in the repository.  They should be regenerated when Natural Earth
issues a new 50m/110m release with meaningful geometry changes.  After
regeneration, update the SHA-256 checksums in the associated
`LICENSE_OR_SOURCE.md` and record the source version and date.

Do **not** replace these files with data from other providers without updating
the licence record.

---

## 2. Natural Earth 50m land polygon shapefile

| Item | Details |
|---|---|
| Location | `coastline/natural_earth_50m_land/` |
| Files | `ne_50m_land.shp`, `.shx`, `.dbf`, `.prj`, `.cpg` |
| Purpose | Land mask for static Cartopy maps in `pages/32_Isotope_Hydrographic_Mapping.py` |
| Loaded by | `cartopy.io.shapereader.Reader(local_path)` inside `_load_ne50m_land_geometries()` in page 32 |
| Used for | `ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor="white", ...)` to paint land white and clip contours at coastlines |
| Bundled | 2026-09-23; see `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md` for SHA-256 checksums and provenance |

### Design rationale

Using a local `shapereader.Reader` instead of `cfeature.LAND` or
`shapereader.natural_earth()` prevents Cartopy from contacting the Natural
Earth download server, which is essential for offline and shipboard use.
The coastline **line** overlay is drawn separately via
`envgeo_utils.plot_bundled_coastline()` (which reads the CSV in `coastline/`),
ensuring both the land mask and the coastline outline are fully offline.

### Degradation behaviour

If any required shapefile component (`.shp`, `.shx`, `.dbf`) is missing, the
land mask is skipped and an English-language `st.warning()` is shown.
Coastline outlines and observation points continue to be drawn normally.
No network access is attempted.

### Licence

Natural Earth data is in the public domain.
See `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md` for full details.
Recommended attribution: *Made with Natural Earth (https://www.naturalearthdata.com/)*

### Update policy

Update when a Natural Earth 50m land release contains meaningful geometry
changes.  After replacing the files:

1. Recompute and record SHA-256 checksums in
   `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md` and
   `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE_Japanese.md`.
2. Record the new commit SHA, raw GitHub URLs, and retrieval date.
3. Run `pytest test/test_natural_earth_land.py` to confirm the shapefile is
   still readable and the feature count is reasonable.

---

## 3. GEBCO bathymetry grid

| Item | Details |
|---|---|
| Location | `bathymetry/` |
| File | `GEBCO_2025_6min.nc` (derived NetCDF3 grid, ~13 MB) |
| Purpose | Depth interpolation and seafloor estimation for the Vertical Section Visualizer (`pages/53_Vertical_Section_Visualizer.py`) |
| Loaded by | `scipy.io.netcdf_file()` inside the Vertical Section page |
| Scope | Used for cross-section analysis only; unrelated to map land masking |

### Licence

The GEBCO Grid is in the public domain and may be copied, adapted,
distributed, and commercially used, subject to source acknowledgement, no
implied GEBCO/IHO/IOC endorsement, and the stated disclaimer. It must not be
used for navigation or any purpose involving safety at sea. Cite GEBCO when
using outputs in publications:

> GEBCO Compilation Group (2025) GEBCO 2025 Grid,
> doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29.

See <https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use>
for the current terms.

`GEBCO_2025_6min.nc` is a project-derived 6 arc-minute subsample of the
official GEBCO_2025 NetCDF grid. `bathymetry/make_lightweight_gebco.py` records
the input, conversion, and stride procedure. Preserve this script and update
the provenance record if regenerating the derived file.

### Update policy

Replace `GEBCO_2025_6min.nc` when a newer GEBCO annual release is available
and the improved bathymetry is needed for the Vertical Section workflow.
Update the citation and licence note in documentation accordingly.
Do **not** alter the GEBCO loading code in page 53 without re-testing the
Vertical Section cross-section output.

---

## 4. Current packaged directory layout

```
coastline/
    world_coastline_coordinates_50m.csv   # offline Plotly/Mapbox coastline overlay
    world_coastline_coordinates_110m.csv  # lower-resolution variant
    natural_earth_50m_land/     # land mask for static Cartopy maps (page 32)
        ne_50m_land.shp
        ne_50m_land.shx
        ne_50m_land.dbf
        ne_50m_land.prj
        ne_50m_land.cpg
        LICENSE_OR_SOURCE.md
        LICENSE_OR_SOURCE_Japanese.md

bathymetry/
    GEBCO_2025_6min.nc          # GEBCO bathymetry for Vertical Section (page 53)
```

All items shown above are included in the current wheel through the explicit
`[tool.setuptools.package-data]` allowlist in `pyproject.toml`; the GEBCO
generation script is intentionally excluded. **Do not reorganize these
directories in the current version.**

---

## 5. Future asset-loader refinement (do not implement yet)

The project is already an installable Python package: `pyproject.toml` lists
the bundled assets as package data, and `envgeo_assets.asset_path()` resolves
them relative to the installed package module. A cleaner asset layout remains
a future candidate only. Paths must not be changed until a backward-compatible
loader and installed-wheel tests have been designed.

```
assets/
    geospatial/    ← coastline CSVs + Natural Earth land shapefile
    bathymetry/    ← GEBCO grid
    metadata/      ← LICENSE_OR_SOURCE files, checksums, source records
```

Migration requirements before this reorganization:

- Preserve the current installed-package resolution through
  `envgeo_assets.asset_path()` (or an equivalently tested resource API); do
  not reintroduce paths relative to individual page scripts or the CWD.
- A backward-compatible fallback must allow existing local checkouts without
  a package install to still work.
- Move `LICENSE_OR_SOURCE.md` / `LICENSE_OR_SOURCE_Japanese.md` files into
  `assets/metadata/` at the same time.
- Verify that Streamlit Cloud and the local Conda environment both find the
  bundled files after migration.
- Keep Natural Earth land (static map land mask) and GEBCO (bathymetry /
  section analysis) in separate subdirectories even after reorganization.

## 6. Current asset recheck (2026-09-30)

The canonical working folder and stable `seawater_map` clone contain
byte-identical copies of the two coastline CSVs, the five Natural Earth
shapefile components, `GEBCO_2025_6min.nc`, and
`make_lightweight_gebco.py`. The current snapshot identifiers are:

| Asset | Current check |
|---|---|
| 50m coastline CSV | 61,844 data rows; SHA-256 `c3d7bee4fb696b011fa34bb13bed0c335c5250eeaf37d8739d77d29a27fe385c` |
| 110m coastline CSV | 5,261 data rows; SHA-256 `a31df3aeee9dc4195af35a31b0605fdb572c7c7dd7cde17f773c9438f5ec7f3f` |
| Natural Earth land | Component checksums match `LICENSE_OR_SOURCE.md`; 1,420 polygon features are recorded there. |
| GEBCO derived grid | NetCDF variables `lon`, `lat`, and `Height`; dimensions 3,600 × 1,800; SHA-256 `0afdf1d0e023b0529c56b69a2684e505c7e2ea28d78a3af8814817af59b09030` |

The retained 2025-01-24 coordinate workbooks for both resolutions match the
current CSV coordinates within floating-point representation precision (maximum
absolute difference about 1.4 × 10⁻¹⁴); the files have matching row counts and
the same missing-coordinate segment separators. The original source archives
identify Natural Earth v4.1.0. The project maintainer confirms that the
current CSV files were created from the source files in that retained workspace.
Before any future CSV update, record the new source version, retrieval date,
conversion procedure, and checksum alongside the replacement files.

---

*Last updated: 2026-09-30*
