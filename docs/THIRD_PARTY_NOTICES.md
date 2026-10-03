# Third-Party Data and Asset Notices

[日本語版](THIRD_PARTY_NOTICES_Japanese.md)

This notice distinguishes the project's MIT-licensed source code from bundled
third-party data and geospatial assets. It is a concise release notice, not a
replacement for the source-specific records or their current terms of use.

## Project source code

The EnvGeo-Seawater source code is licensed under the repository's
[`LICENSE`](../LICENSE) (MIT). That licence does not transfer ownership of, or
create redistribution rights for, third-party datasets, map assets, or their
derived records.

## Bundled datasets

The five `dataset/*.xlsx` workbooks are included under the project's recorded
scholarly-use distribution decision. Preserve source citations, versions or
access dates, and project-side transformations; do not describe third-party
records as project-owned. The file-level source records and reassessment rules
are in [`dataset_redistribution_audit.md`](dataset_redistribution_audit.md),
[`provenance_inventory.md`](provenance_inventory.md), and
[`external_dataset_workbook_notes.md`](external_dataset_workbook_notes.md).

The tracked `local_data/user_data.xlsx` workbook is a project-created,
zero-row public template. It is not a measurement dataset.

## Geospatial assets

- The coastline CSVs and Natural Earth 50m land polygons are derived from
  Natural Earth public-domain data. Retain the credit: “Made with Natural
  Earth (https://www.naturalearthdata.com/)”. See
  [`geospatial_assets.md`](geospatial_assets.md) and
  [`../coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md`](../coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md).
- `bathymetry/GEBCO_2025_6min.nc` is a project-derived, downsampled GEBCO 2025
  Grid. It is context for vertical-section analysis only, is not for
  navigation, and must retain the required GEBCO acknowledgement and
  disclaimer. See [`geospatial_assets.md`](geospatial_assets.md).

## Software dependencies

Dependencies are installed separately from the Python package under their own
licences. Method-critical software and data sources should be cited as
described in the README, `paper.bib`, and the source-specific records.
