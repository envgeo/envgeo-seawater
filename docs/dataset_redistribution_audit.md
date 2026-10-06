# Dataset Redistribution Evidence Audit

**Status:** release-evidence audit, initially reviewed 2026-09-25,
workbook/package scope rechecked 2026-09-30, and derived-asset records
verified 2026-10-03. This is a release-planning record, not legal advice. It supplements the concise
[`provenance_inventory.md`](provenance_inventory.md).

## Purpose and release rule

EnvGeo-Seawater currently reads several consolidated workbooks from
`dataset/`.  A source being downloadable, cited, or visible in a public
repository does **not** by itself establish that the derived workbook may be
redistributed in a Git repository, Python wheel, Zenodo archive, or another
release artifact.

The project policy is to retain the current scholarly-use approach: identify
the dataset DOI or canonical source, cite it as requested, record access date
and transformations, and do not present third-party data as project-owned.
This audit does not call for removing the currently cited public workbooks. No
additional provider enquiry is planned before public release or manuscript
submission. A future installed-package or Zenodo design must preserve this
metadata and revisit an item only if an explicit restriction, reviewer concern,
or rights-holder request arises.

### Recorded current-distribution decision for NASA GISS and CoralHydro2k

The project has decided to continue the current scholarly-use distribution of
the NASA GISS and PAGES CoralHydro2k-derived workbooks through public use and
manuscript submission. This decision is based on their traceable public source
records, the documented source-to-workbook checks and transformations, and the
project's prior communication about data-quality findings. Retain the source
URL, version, DOI where applicable, access date, required citation, and
transformation record. Do not describe either dataset as project-owned.

Private correspondence, contact details, replies, and attachments are not
release metadata and must not be copied into the repository, wheel, release
notes, Zenodo archive, or public documentation. Reconsider the decision only
if an explicit restriction, a rights-holder request, or a concrete reviewer
concern arises.

### Current package-data decision (2026-09-28)

All current `dataset/*.xlsx` workbooks are included in the EnvGeo-Seawater
package. This is the project's documented scholarly-use distribution decision
for the current cited, public data collection. Each workbook must retain its
source citation, DOI or canonical source where available, access date, and
known project-side schema or display transformations. Inclusion does not make
the underlying records project-owned, and does not automatically apply to a
future dataset added to the collection.

### Verified package snapshot (2026-09-30)

A read-only comparison confirmed that the five `dataset/*.xlsx` workbooks
selected by `pyproject.toml` are byte-identical in the canonical working
folder and the stable `seawater_map` clone. Their filenames, worksheet names,
row/column counts, and SHA-256 checksums are recorded in
[`provenance_inventory.md`](provenance_inventory.md). No workbook content was
modified by this audit. The tracked `local_data/user_data.xlsx` sample is a
zero-row, 22-column public template in both locations. On 2026-10-03, its
workbook metadata was regenerated without a local absolute-path field; its
headers, zero-row state, and visible template formatting were retained.

The unused legacy files `data/reference.xlsx` and
`data/seawater_data_sample.xlsx` were removed from both application trees on
2026-10-03. Byte-identical recovery copies are retained only in a local
historical archive outside the application and public-release trees; they are
not wheel, GitHub Release, or Zenodo inputs.

## Findings by resource

| Resource | Evidence confirmed | Ongoing release safeguard | Packaging decision now |
|---|---|---|---|
| `local_data/user_data.xlsx` | Project-created, zero-value template; headers, zero-row state, visible formatting, and absence of a local absolute-path metadata field were verified 2026-10-03. | Maintain its zero-value state and metadata-safe workbook form in public synchronization. | May remain tracked and bundled as a sample. |
| Natural Earth land and derived coastline CSVs | Natural Earth states that its data are public domain. Natural Earth 50m land is documented separately; 50m/110m CSV intermediates, output checksums, and numerical correspondence were verified 2026-10-03. | Retain credit, intermediate-file record, and output checksums. Original raw-download checksums were not retained. | Included under the documented public-domain source record. |
| Derived `bathymetry/GEBCO_2025_6min.nc` | GEBCO states that the Grid is public domain and allows redistribution subject to attribution, disclaimer, and non-navigation terms. NetCDF history, retained generator, grid dimensions, and output checksum were verified 2026-10-03. | Retain the GEBCO 2025 citation, disclaimer, non-navigation notice, generator, and output checksum. Original input-grid checksum was not retained. | Included under the documented GEBCO source record. |
| `01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx` | Publication DOI/citation, analytical record, current row count, and output checksum are recorded. | Preserve the source location, version/access date when available, citation, and transformation record. | Included in the current package under the documented scholarly-use policy. |
| `11_AROUND_JAPAN_PUB_20260305.xlsx` | Four regional source labels, citations, current row count, and output checksum are recorded. | Preserve the row/source mapping, citations, and transformations for contributing records. | Included in the current package under the documented scholarly-use policy. |
| `71_GLOBA_NASA_20260226.xlsx` | `data_text/NASA_references.md` records the GISS reference URL, database v1.22 citation, source URL, and access date **2026-03-01**. `external_dataset_workbook_notes.md` records the 25,514-row comparison, schema labels, and `Transect = Nasa_database`. | Retain the source/version/access record, the `Transect` convention, and the change record. Review only if an explicit restriction, reviewer concern, or rights-holder request arises. | Retain current scholarly-use distribution with citation; do not represent it as project-owned. |
| `71_GLOBAL_Atwood_et_al_2026_v02.xlsx` | `data_text/CoralHydro2_references.md` records the NCEI study URL, project DOI, cited Atwood et al. reference, and access date **2026-03-16**. `external_dataset_workbook_notes.md` records the 18,598-row comparison, source-column mapping, the 2026-10-06 short-reference completion, `Transect` field, and date-format representation. | Retain the DOI, source/access record, requested citation, and change record. Review any study-specific restriction if it is identified. | Retain current scholarly-use distribution with citation; do not represent it as project-owned. |
| `72_GLOBAL_RECENT_REPORTS_20260302.xlsx` | It is a 35-row project compilation currently labelled Sakamoto et al. (2022); its current output checksum is recorded. | Preserve its row/source mapping, citations, and transformations for report-derived records. | Included in the current package under the documented scholarly-use policy. |

## Scientific-provenance consequence for T–S Stage 2

The NASA GISS database documentation cautions that temperature can be either
in-situ or potential temperature and that original references may be needed to
resolve missing information.  Therefore the current global workbook must not
be treated as a uniform, TEOS-10-ready source for automatic SA/CT conversion.
Any Stage 2 calculation needs row-level declarations for salinity type,
temperature type, pressure/depth, and latitude/longitude, with an explicit
skip/fallback for rows without verified metadata.

## Terms review result (2026-09-25)

**NASA GISS.** The database page provides a specified v1.22 citation and a
research download, while describing a collection assembled from independent
sources. The project will use the database under ordinary scholarly practice:
retain its requested citation, source URL, access date, and transformation
record. No additional contact is planned before release or manuscript
submission. If a rights-holder, reviewer, or explicit restriction raises a
concern, reassess the affected records promptly.

**Known NASA selection convention.** The source database has no transect
metadata usable by the application. For the application's sub-dataset selector,
every NASA row is assigned `Transect = Nasa_database`. This is an application
grouping label, not a source-supplied oceanographic transect and must be
described as such in future data documentation.

**PAGES / NCEI.** The collection has a public study page, a dataset DOI, and a
specified scholarly citation. NCEI's general policy distinguishes federally
produced data from external deposits, and the CC BY 4.0 on the ESSD article is
the article licence rather than a substitute dataset licence. The project will
retain the existing cited-workbook distribution with complete provenance. No
additional contact is planned before release or manuscript submission. Any
explicit study-level restriction or request will be recorded and acted on.

## Evidence package for future additions or changes

For a new source, or when replacing a currently bundled workbook, add a small
machine- and human-readable record for each source:

1. Dataset title, owner/publisher, canonical landing URL, DOI or identifier.
2. Dataset version, retrieval date, original filename/checksum, and local
   output checksum.
3. Licence or terms where stated, plus required attribution and any known
   restrictions.
4. Exact input rows/fields used and every transformation, filter, unit change,
   or manually curated correction.
5. Required scientific citation and acknowledgement text.
6. Recorded scholarly-use decision and any restriction or escalation path that
   becomes relevant.

## Recommended release sequence

1. Maintain this evidence package for the project-owned and regional workbooks;
   add source-level detail when a workbook is changed or a new source is added.
2. Retain DOI/source, access-date, citation, and transformation records for
   each third-party source. Do not make additional enquiries before release or
   manuscript submission unless a concrete issue requires it.
3. Keep the selected full-data scholarly-use package scope: all current
   `dataset/*.xlsx` workbooks remain included, with source citations and
   provenance records retained per source.
4. Record this mode in `pyproject.toml` package-data declarations,
   README installation instructions, release notes, Zenodo metadata, and the
   future JOSS archive.

## Primary sources checked

- NASA GISS, *Global Seawater Oxygen-18 Database*: <https://data.giss.nasa.gov/o18data/>.
- NASA, *Science Data Licenses*: <https://science.data.nasa.gov/about/license>.
- NOAA NCEI, *CoralHydro2k directory*: <https://www.ncei.noaa.gov/pub/data/paleo/coral/coralhydro2k/>.
- NOAA NCEI, *Open Data Policy*: <https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf>.
- Atwood et al. (2026), *PAGES CoralHydro2k Seawater δ18O Database*: <https://doi.org/10.5194/essd-18-1921-2026>.
- Natural Earth, *Terms of Use*: <https://www.naturalearthdata.com/about/terms-of-use/>.
- GEBCO, *Grid terms of use*: <https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use>.
