# Data Overlap Check

## Purpose

This utility page screens two datasets for possible repeated observations. It
supports source-level audit before combining datasets for statistics, figures,
or interpretation. It never changes a source workbook, confirms a duplicate,
or removes an observation.

## At a glance

**This utility identifies possible duplicate records only.** It does not alter
source workbooks, confirm duplicates, or remove records from analyses. Use it
to create an auditable candidate list before reviewing source provenance,
cruises, stations, and measurement metadata.

A session-only uploaded dataset can also be screened against a selected bundled
EnvGeo dataset. Use **Uploaded data overlay** in the sidebar and provide the
required columns. On analytical pages, **Data filtering → Duplicate-candidate
display** provides a reversible choice to show or hide candidate records; the
source records remain unchanged.

## Basic workflow

1. Select a dataset pair.
2. Set the Strong and broader Review criteria in the sidebar.
3. Select **Run overlap screen**.
4. Inspect the exported tables and the side-by-side source-row viewer.
5. Check the cited source, cruise, station, date, and analytical metadata
   before recording a final decision outside the source workbook.

Bundled comparisons include EnvGeo Dataset [ECS–Japan Sea], Around Japan, NASA GISS
global, and CoralHydro2k global. The page compares one selected pair at a
time; it does not automatically treat several pairwise candidates as a
confirmed multi-dataset duplicate.

Only rows with the same valid Year and Month are compared. A sampling day can
be checked in the side-by-side source-row review when available, but is not a
matching requirement in the initial screen.

## Candidate classes

- **Strong candidate**: all strict coordinate, depth, salinity, and δ18O
  thresholds are met. The current provisional defaults are latitude ≤0.1°,
  longitude ≤0.1°, depth ≤5 m, salinity ≤0.1, and δ18O ≤0.1‰.
- **Review candidate**: the broader Review thresholds are met, but at least
  one strict threshold is exceeded. The current provisional Review defaults
  are latitude ≤0.2°, longitude ≤0.2°, depth ≤10 m, salinity ≤0.2, and
  δ18O ≤0.2‰.
- **Provisional one-to-one rounding-compatible match**: a stricter subset of
  Strong candidates. Every comparison is compatible with decimal precision
  inferred from imported numeric values, and each source row is linked at most
  once within a Year--Month group.

The provisional list remains a prioritised source-review list, not proof that
two observations are duplicates. Different source files can contain rounded
values, incomplete metadata, or nearby but genuinely distinct samples.
These defaults are transparent initial screening criteria, not universal
measurement-error thresholds; users can adjust them in the audit sidebar and
should report the criteria used for a scientific result.

## Current bundled-data screening snapshot

The following reproducible snapshot was calculated on **2026-10-07** with the
pre-v1.3.5 development code (currently labelled v1.3.4), the bundled `11_AROUND_JAPAN_PUB_20260305.xlsx`,
`71_GLOBA_NASA_20260226.xlsx`, and
`71_GLOBAL_Atwood_et_al_2026_v02.xlsx` workbooks, and the provisional default
criteria listed above.

| Dataset pair | Candidate pairs | Strong | Review | Provisional one-to-one rounding-compatible |
| --- | ---: | ---: | ---: | ---: |
| Around Japan × NASA GISS global | 65 | 55 | 10 | 8 |
| Around Japan × CoralHydro2k global | 0 | 0 | 0 | 0 |
| NASA GISS global × CoralHydro2k global | 2,185 | 1,830 | 355 | 1,347 |

At these criteria, no candidate component joined all three bundled datasets.
The counts are **candidate pairs**, not confirmed duplicate observations, and
will change if source-workbook versions or audit criteria change. Future
screens should record their date, source versions, and criteria alongside any
reported count.

The separately bundled **EnvGeo Dataset [ECS–Japan Sea]** collection
(`01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx`) was also checked on the same date
and criteria. Its primary reference is Kodama et al. (2024). It had **0 candidate pairs** with each of Around Japan, NASA
GISS global, and CoralHydro2k global (therefore 0 Strong and 0 Review
candidates in all three checks). This is a screening result, not evidence that
all possible relationships between the collections have been exhaustively
resolved.

## Optional display screen in analysis pages

The shared data-filter sidebar leaves all records visible by default. If an
analysis needs a sensitivity check, choose one of the reversible bundled-data
display modes and apply the settings:

- **Hide one-to-one rounding-compatible candidates (recommended)** hides only
  the provisional one-to-one subset. For each pair, the row with more filled
  provenance/analytical metadata remains visible; ties use a documented,
  deterministic source priority.
- **Hide Strong candidates broadly (screening use)** uses every Strong pair.
  Strong pairs can be one-to-many, so this is a deliberately broad sensitivity
  screen and must not be reported as confirmed de-duplication.

The first use calculates and caches the bundled-source audit for the current
app session. Neither mode changes a workbook, an uploaded table, or a source
measurement. Selecting **Show all records (default)** restores every row
immediately.

## Outputs

- Separate Strong and Review audit CSV files
- A provisional one-to-one rounding-compatible CSV
- The full candidate audit CSV
- English and Japanese audit-column guide CSV files

Use **Inspect one Provisional one-to-one candidate pair** to view the two
original rows side by side before making a decision.

Use **Record a provisional-candidate decision** to record **Pending**,
**Confirmed duplicate**, or **Keep both**, together with the source evidence
or next check. For a confirmed duplicate, the optional **Display action after
confirmation** records which source row would remain visible. Decisions remain
only in the current browser session until you download the separate
review-decision CSV. This file is intentionally separate from all source
workbooks and is the starting point for a later reviewed confirmed-duplicate
manifest.

The audit includes source names, source-row IDs, citation fallbacks, absolute
and signed differences, strict-threshold flags, and rounding allowances. The
`Right_minus_Left` columns always mean the right selected dataset minus the
left selected dataset.

**Candidate pairs** count combinations of left and right source rows. They can
therefore exceed the number of unique records when one row has several nearby
matches. Before the candidate metrics, the page reports the total rows loaded
from each selected source; it also reports the number of unique source rows
represented in candidate pairs on each side.

## Uploaded data

A session-only uploaded table can be compared with one bundled reference
dataset after required columns are assigned in the sidebar. Two uploaded
tables are not compared on this page. Uploads remain in the current browser
session and are never written by the app.

## Limits and planned use

The decimal-precision check is an audit estimate based on imported numeric
values; it is not a measurement-uncertainty model. Confirmed duplicate
decisions are stored separately from the original workbooks. The optional
display screens are reversible sensitivity tools; the default retains all
observations.
