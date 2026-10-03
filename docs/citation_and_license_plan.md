# Citation and License Plan

[日本語版](citation_and_license_plan_Japanese.md)

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

## Current data-distribution scope

All current `dataset/*.xlsx` workbooks remain included under the project's
documented scholarly-use decision. Preserve the source citation, DOI or
canonical source where available, access date, and project-side transformation
record for each workbook; do not describe third-party records as
project-owned. This is the project’s current distribution decision, not a
general claim that citation alone grants reuse rights. See
[`dataset_redistribution_audit.md`](dataset_redistribution_audit.md) and
[`provenance_inventory.md`](provenance_inventory.md) for the file-level
records and conditions for reassessment.

## Release checklist

### Project and release

- [x] Added `CITATION.cff` with author, title, repository URL, version, and
      the documented 1.3.4 release-candidate date. It intentionally has no DOI
      before archival publication; see `release_metadata_draft.md`.
- [ ] Replace the candidate date with the final tag date and add the versioned
      DOI after the Zenodo archive has been created.
- [ ] Replace the provisional README citation with the versioned DOI citation.
- [ ] Tag the release and preserve its dependency lock/requirements record.
- [x] Added the project `LICENSE` and `THIRD_PARTY_NOTICES.md` / Japanese
      counterpart. The notices distinguish the MIT code licence from bundled
      third-party data and geospatial asset records; source-specific records
      remain authoritative.

### Package-index publication (before JOSS resubmission)

- [ ] Build the intended final artifact from the reviewed stable source and run
      `twine check` on the distribution files.
- [ ] Upload that artifact to TestPyPI, then install it in a new macOS test
      environment. Install compiled geospatial prerequisites from conda-forge
      where needed, then install EnvGeo-Seawater through `pip`.
- [ ] After the TestPyPI check succeeds, publish the reviewed final artifact as
      `envgeo-seawater` on PyPI. Use PyPI Trusted Publishing or a secure manual
      upload process; never store an upload token in the repository.
- [ ] Reproduce a `pip install envgeo-seawater` installation and launch in a
      new macOS environment; record the result in the release record.
- [ ] Treat a conda-forge recipe/feedstock as a separate post-v1.3.4
      improvement, not a blocker once the PyPI installation path works.

### Scientific data

- [ ] For every file in `dataset/`, record its source publication/data DOI,
      dataset version, retrieval date, stated licence or terms where
      available, required citation, documented scholarly-use decision, and any
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

- [ ] Retain the project’s chosen Natural Earth credit, “Made with Natural
      Earth (https://www.naturalearthdata.com/)”, in the documented
      provenance and relevant outputs. Natural Earth data are public domain;
      the bundled land shapefile provenance and checksums are recorded in
      `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md`.
- [ ] Cite the GEBCO 2025 Grid in Vertical Section outputs and documentation:
      `GEBCO Compilation Group (2025) GEBCO 2025 Grid,
      doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29`.
- [ ] Describe `GEBCO_2025_6min.nc` as a derived, downsampled product and
      preserve the script and parameters used to create it.
- [ ] Recheck the GEBCO terms at the tagged release. The official terms place
      the grid in the public domain, permit commercial use, require
      attribution, prohibit misleading endorsement claims, and state that it
      is not for navigation. See
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
      entries in `paper.bib` can be reused for papers.
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

- README files now name the roles of GEBCO, TEOS-10/GSW, cmocean, Streamlit,
  Plotly, and Natural Earth. This concise acknowledgement does not replace the
  source-specific citation, provenance record, or future third-party notices.
- `docs/geospatial_assets.md` already records the current GEBCO terms. Recheck
  that wording and the source citation at the tagged release.
- `paper.bib` contains verified entries for Streamlit, Plotly, cmocean,
  TEOS-10/GSW, and the GEBCO 2025 Grid. Recheck each citation at the tagged
  release if the referenced software, data version, or manuscript scope changes.
