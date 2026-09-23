# Geospatial Assets — Usage, Placement, Licence, and Update Policy

This document describes the geographic and scientific assets bundled in the
EnvGeo-Seawater repository: the coastline CSV files, the Natural Earth 50m
land polygon shapefile, and the GEBCO bathymetry grid.

---

## 1. Coastline CSV files

| Item | Details |
|---|---|
| Location | `coastline/` (repository root) |
| Files | `coastline_50m.csv`, `coastline_110m.csv` (and associated files) |
| Purpose | Lightweight offline coastline overlay for all interactive Plotly/Mapbox maps and the Cartopy static maps in page 32 |
| Format | CSV with `lon`, `lat` columns; polygon segments separated by `None` rows |
| Loaded by | `envgeo_utils.load_coastline_data()` |
| Drawn by | `envgeo_utils.add_coastline_overlay()` (Plotly Scattermapbox) and `envgeo_utils.plot_bundled_coastline()` (Matplotlib/Cartopy) |

### Licence

The coastline data is derived from Natural Earth public-domain sources.
See the `LICENSE_OR_SOURCE.md` or equivalent file in `coastline/` for details.
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
| Location | `data_beta/` |
| File | `GEBCO_2025_6min.nc` (NetCDF, ~90 MB) |
| Purpose | Depth interpolation and seafloor estimation for the Vertical Section Visualizer (`pages/53_Vertical_Section_Visualizer.py`) |
| Loaded by | `xarray.open_dataset()` inside the Vertical Section page |
| Scope | Used for cross-section analysis only; unrelated to map land masking |

### Licence

GEBCO (General Bathymetric Chart of the Oceans) data is provided under a
non-commercial attribution licence.  Cite GEBCO when using outputs in
publications:

> GEBCO Compilation Group (2025) GEBCO 2025 Grid.
> https://doi.org/10.5285/...

See the GEBCO website (https://www.gebco.net/) for the current DOI and
full licence terms.

### Update policy

Replace `GEBCO_2025_6min.nc` when a newer GEBCO annual release is available
and the improved bathymetry is needed for the Vertical Section workflow.
Update the citation and licence note in documentation accordingly.
Do **not** alter the GEBCO loading code in page 53 without re-testing the
Vertical Section cross-section output.

---

## 4. Current directory layout

```
coastline/
    coastline_50m.csv           # offline Plotly/Mapbox coastline overlay
    coastline_110m.csv          # lower-resolution variant
    natural_earth_50m_land/     # land mask for static Cartopy maps (page 32)
        ne_50m_land.shp
        ne_50m_land.shx
        ne_50m_land.dbf
        ne_50m_land.prj
        ne_50m_land.cpg
        LICENSE_OR_SOURCE.md
        LICENSE_OR_SOURCE_Japanese.md

data_beta/
    GEBCO_2025_6min.nc          # GEBCO bathymetry for Vertical Section (page 53)
```

**Do not reorganize these directories in the current version.**
See section 5 for the planned future layout.

---

## 5. Future packaging plan (do not implement yet)

When the project moves to a proper Python package (`pyproject.toml` with
`package_data`), the geographic and scientific assets should be reorganized
into a cleaner structure.  This is a future candidate layout only — paths
must not be changed until a backward-compatible asset loader has been designed
and tested.

```
assets/
    geospatial/    ← coastline CSVs + Natural Earth land shapefile
    bathymetry/    ← GEBCO grid
    metadata/      ← LICENSE_OR_SOURCE files, checksums, source records
```

Migration requirements before this reorganization:

- The asset loader must resolve paths relative to the installed package
  (e.g. via `importlib.resources`), not relative to `__file__` of individual
  page scripts.
- A backward-compatible fallback must allow existing local checkouts without
  a package install to still work.
- Move `LICENSE_OR_SOURCE.md` / `LICENSE_OR_SOURCE_Japanese.md` files into
  `assets/metadata/` at the same time.
- Verify that Streamlit Cloud and the local Conda environment both find the
  bundled files after migration.
- Keep Natural Earth land (static map land mask) and GEBCO (bathymetry /
  section analysis) in separate subdirectories even after reorganization.

---

*Last updated: 2026-09-23*
