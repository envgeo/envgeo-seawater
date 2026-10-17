# Version Update Checklist

Use this reusable checklist whenever preparing a new EnvGeo-Seawater version.
It complements `release_checklist.md`: this file prevents version strings,
documentation, and release state from drifting; the release checklist records
the evidence for a particular release.

Do not describe a version as a published stable release until its reviewed
stable commit, GitHub Release, and intended archive state exist. Never reuse a
Zenodo version DOI from an older release.

## 1. Define the release state

- [ ] Record the target version, candidate date, repository, and intended
  public-page scope.
- [ ] Decide whether the repository is a development candidate or a published
  release. Use *release candidate* wording until GitHub Release publication.
- [ ] List repository-specific exceptions before synchronizing files (for
  example, private `test2` data and Home differences, or stable-only pages).

## 2. Update version metadata and regression tests

- [ ] Update `pyproject.toml`.
- [ ] Update `envgeo_utils.APP_VERSION` and `APP_VERSION_DATE`.
- [ ] Update each public page's displayed `version` value where it is defined.
- [ ] Update version assertions in `test/test_basic.py`,
  `test/test_envgeo_utils.py`, and any other fixed-version tests.
- [ ] Search the code and tests for stale current-version strings. Preserve
  older version strings only when they are explicitly historical records.

## 3. Review data and reproducibility state

- [ ] Confirm every workbook selected by the loader exists in each intended
  repository and has the intended filename.
- [ ] When a bundled workbook changes, update its loader path, overlap/source
  catalogue, provenance inventory, redistribution audit, workbook notes, and
  recorded checksum together.
- [ ] Keep private or unpublished data outside public repositories. Do not
  overwrite `test2`-specific data without an explicit comparison and decision.

## 4. Update user-facing documentation

- [ ] Update English and Japanese `README` files, including supported Python
  versions, public-page scope, and release state.
- [ ] Update `CITATION.cff` and `data_text/citation*.md`. Before Zenodo archives
  the release, use the concept DOI and state that the version DOI is pending.
- [ ] Update Home's **Update History** and `data_text/update_log*.md`.
- [ ] Update the embedded manual (`data_text/manual*.md`), the page-by-page
  manuals, and the GitHub Pages entries `docs/index*.md`.
- [ ] Update `release_metadata_draft*.md`, `release_checklist*.md`, and any
  current-status development note. Leave historical release evidence intact.
- [ ] Update `paper.md` only once the version-specific Zenodo DOI is known.

## 5. Synchronize repositories deliberately

- [ ] Synchronize common code and documentation from the canonical working
  copy to `envgeo-seawater`, `test2`, and `seawater_map`.
- [ ] Verify `git status` in every clone. Check that a copied file is tracked;
  do not accidentally introduce GitHub Pages files into repositories that do
  not publish them.
- [ ] Preserve documented clone-specific differences, especially test-only
  assets and the stable repository's exclusion of development pages.

## 6. Verify the candidate

- [ ] Run focused tests for changed code and version assertions locally.
- [ ] Run the repository health tests and construct a wheel from the stable
  repository.
- [ ] Confirm Python 3.10 and 3.12 CI are green in each affected repository.
- [ ] Perform a short Cloud smoke check of Home, filtering, mapping, uploads,
  and any changed page. Check GitHub Pages after a documentation-site update.

## 7. Publish and finalize archival metadata

- [ ] Commit and push the reviewed stable `seawater_map` state; record the
  exact commit SHA.
- [ ] Create the GitHub Release and tag from that exact commit (for example,
  `v1.3.5`).
- [ ] Publish to PyPI only when that distribution is part of the release plan.
- [ ] Wait for Zenodo to archive the tagged release and record the new
  version-specific DOI separately from the all-versions concept DOI.
- [ ] Make a final DOI-only documentation update to `paper.md`, citations,
  README, update history, and release metadata; rerun CI and confirm the
  deployed app and GitHub Pages.

## Completion record

Record the following in the release notes or metadata draft:

- target version and publication date;
- stable commit SHA and Git tag;
- GitHub Release URL;
- Zenodo version DOI and concept DOI;
- CI run URLs or results, wheel filename/checksum where relevant; and
- any repository-specific exclusions or deferred items.
