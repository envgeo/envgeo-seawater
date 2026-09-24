# Citation and License Plan

This is a release-readiness checklist and provenance record for
EnvGeo-Seawater. It does not replace verification of source licences or legal
advice. Complete the applicable items before a public GitHub release, Zenodo
archive, Streamlit deployment, or JOSS submission.

## Principles

- Cite the project, the data actually used, and method-critical software or
  standards in scientific outputs.
- Record a source URL, version or retrieval date, licence/terms, recommended
  citation, redistribution status, and any transformation for every bundled
  dataset or map asset.
- Keep concise acknowledgements in `README.md`; maintain the full record in
  this document or a future machine-readable provenance table.
- A dependency's open-source licence does not by itself grant redistribution
  rights for its data products.

## Release checklist

### Project and release

- [ ] Add `CITATION.cff` with author, title, repository URL, version, release
      date, and DOI after Zenodo archival.
- [ ] Replace the provisional README citation with the versioned DOI citation.
- [ ] Tag the release and preserve its dependency lock/requirements record.
- [ ] Add a `LICENSE` notice for the project and a third-party notices document
      covering redistributed assets and any required licence texts.

### Scientific data

- [ ] For every file in `dataset/`, record its source publication/data DOI,
      dataset version, retrieval date, licence or reuse permission, and any
      filtering, column normalization, or aggregation applied by this project.
- [ ] Keep CoralHydro2k, NASA GISS, Kodama et al., and each regional source
      individually traceable; do not rely only on a collective README list.
- [ ] Verify that sample and user-facing exports preserve or link to source
      attribution where appropriate.
- [ ] Do not add unpublished, restricted, or researcher-owned measurements to
      the public repository. `local_data/user_data.xlsx` is the tracked
      zero-value public sample: it may be edited locally, but must be restored
      before a commit or public synchronization. An external researcher-owned
      file may instead be selected with `ENVGEO_LOCAL_USER_DATA_PATH`.

### Map and geospatial assets

- [ ] Retain the Natural Earth credit, “Made with Natural Earth
      (https://www.naturalearthdata.com/)”. Natural Earth data are public
      domain; the bundled land shapefile provenance and checksums are recorded
      in `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md`.
- [ ] Cite the GEBCO 2025 Grid in Vertical Section outputs and documentation:
      `GEBCO Compilation Group (2025) GEBCO 2025 Grid,
      doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29`.
- [ ] Describe `GEBCO_2025_6min.nc` as a derived, downsampled product and
      preserve the script and parameters used to create it.
- [ ] Before release, correct and verify the GEBCO terms note: the official
      terms place the grid in the public domain, permit commercial use, require
      attribution, and state that it is not for navigation. See
      https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use

### Methods and software

- [ ] Cite the TEOS-10/GSW standard whenever density or related thermodynamic
      calculations are used in a scientific result; state whether inputs have
      been converted to Absolute Salinity and Conservative Temperature or are
      an explicitly labelled approximation.
- [ ] Cite the cmocean design reference when its palettes are presented as a
      scientific visualization choice: Thyng et al. (2016), *Oceanography*,
      29(3), 9–13, doi:10.5670/oceanog.2016.66.
- [ ] Acknowledge Streamlit and Plotly as core framework and interactive
      visualization software, with versions used for the release. Their
      existing entries in `paper_revised.bib` can be reused for papers.
- [ ] Record Cartopy, Matplotlib, Folium, NumPy, pandas, SciPy,
      scikit-learn, openpyxl, and other runtime dependencies with versions and
      licences in third-party notices. Cite a package or method in scientific
      writing when it is material to a reported algorithm or result.

### Publication and review

- [ ] Ensure each figure/table caption identifies data sources and indicates
      when GEBCO, Natural Earth, or cmocean materially contributes.
- [ ] Verify that README, in-app Data Sources, `data_text/`, and the
      bibliography contain matching citations and links.
- [ ] Perform a final manual licence/reuse review at the tagged release,
      because upstream terms and recommended citations can change.

## Current known follow-ups

- The README already lists major isotope datasets and Natural Earth, but should
  add concise GEBCO, TEOS-10/GSW, cmocean, Streamlit, and Plotly acknowledgements.
- `docs/geospatial_assets.md` currently describes GEBCO as non-commercial;
  update it to match the official terms above before public release.
- `paper_revised.bib` already contains Streamlit, Plotly, and cmocean entries;
  add and verify TEOS-10/GSW and GEBCO entries when preparing the paper.
