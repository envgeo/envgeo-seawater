# EnvGeo-Seawater Documentation

This directory collects project documents that are too detailed for the top-level README.
Historical and development records are retained when useful, but they are not
release approval or a statement of the stable public scope.

[日本語版](README_Japanese.md)

For the public, task-oriented documentation entry point, start with the
[User Guide](index.md). It links only to the supported stable workflows and is
the source directory for the published [GitHub Pages website](https://envgeo.github.io/envgeo-seawater/).

## Current Documents

- `release_checklist.md` / `release_checklist_Japanese.md`
  Checklist for local testing, Streamlit deployment, GitHub release preparation, and Zenodo/DOI archiving.

- `testing.md` / `testing_Japanese.md`  
  Public-facing explanation of the current pytest suite, its scope, limitations, and planned expansion.

- `streamlit_migration.md` / `streamlit_migration_Japanese.md`
  Local migration log for Python 3.12, Streamlit 1.63, Plotly compatibility, environment comparisons, and visual checks.

- `integrated_visualizer_strategy.md` / `integrated_visualizer_strategy_Japanese.md`
  Accepted architecture and migration policy for individual pages, shared user-data upload, and Integrated Visualizer.
  The Page 90 workflow is development history and is outside the stable-release scope.

- `development_notes.md` / `development_notes_Japanese.md`
  Historical 1.3.3 development notes: offline-map design, user-data
  architecture, bundled geospatial assets, deferred work, and safe cleanup
  conditions. They are not the current release checklist.

- `citation_and_license_plan.md` / `citation_and_license_plan_Japanese.md`
  Release-readiness checklist for project citation, third-party software,
  data and map assets, attribution, redistribution terms, Zenodo, and JOSS.

- `code_guide.md` / `code_guide_Japanese.md`
  Concise bilingual map of the source modules, page responsibilities, resource
  directories, and code-documentation conventions.

- `publication_content_review_checklist.md` /
  `publication_content_review_checklist_Japanese.md`
  File-by-file public-content review checklist for the canonical working copy,
  stable-release scope, code, documents, data, assets, and exclusion checks.

- `sprint3_distribution_design_and_acceptance.md` /
  `sprint3_distribution_design_and_acceptance_Japanese.md`
  Current distribution design and verification record for the 1.3.4
  development/pre-release repository; “Sprint 3” is explained in the document.

- `distribution_foundation_audit_and_plan.md` /
  `distribution_foundation_audit_and_plan_Japanese.md`
  Historical Sprint 2 foundation-audit record for asset resolution, packaging,
  installation, offline modes, and deployment validation. It is not the
  current release configuration.

- `wheel_proof_report.md` / `wheel_proof_report_Japanese.md`
  Historical Sprint 2B–2C local wheel build/install evidence, acceptance
  results, limitations, and deferred production-package decisions. The current
  1.3.4 verification record is `sprint3_distribution_design_and_acceptance*.md`.

- `provenance_inventory.md` / `provenance_inventory_Japanese.md`
  Working bilingual inventory of bundled assets and data, their known source
  records, redistribution status, and release-blocking follow-ups.

- `dataset_redistribution_audit.md` / `dataset_redistribution_audit_Japanese.md`
  Evidence-focused bilingual audit separating public availability from a
  documented right to redistribute each bundled dataset; also records the
  scientific-provenance gate for T–S Stage 2.

- `external_dataset_workbook_notes.md` /
  `external_dataset_workbook_notes_Japanese.md`
  Current-state record for the cited NASA GISS and PAGES workbooks, including
  source references and the application-only NASA `Transect` convention.

- `offline_operation_log.md` / `offline_operation_log_Japanese.md`
  Current offline/degraded-network map behavior, self-contained HTML exports,
  known limitations, and verification scope.

- `manual/` / `manual_Japanese/`  
  Page-by-page bilingual user manuals for the stable public workflows. They are
  the content source for the published static documentation website and may also
  be reused for laboratory-web guidance.

## Planned Documents

- `joss_checklist.md`  
  Checklist for future JOSS resubmission, including tests, documentation, examples, citation, license, and archival DOI.
