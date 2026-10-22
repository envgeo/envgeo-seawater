# Post-release maintenance log

This development-only log records maintenance work completed after the v1.3.5
release. It excludes JOSS manuscript work. Entries here do not alter the
tagged v1.3.5 GitHub Release, Zenodo archive, or PyPI distribution; confirm
them again and incorporate the applicable items into release notes for a
future version.

[日本語版](post_release_maintenance_log_Japanese.md)

## After v1.3.5 (October 2026)

- Updated the Home animation asset: removed its progress indicator and retained
  a visually equivalent, smaller encoded video.
- Clarified the browser-upload sidebar guidance. Browser uploads remain
  session-scoped and are not written by the application to local or server
  storage.
- Corrected Custom Parameter Plot isotope labels so \(\delta^{18}\mathrm{O}\)
  uses superscript isotope notation in axes, titles, and captions.
- Aligned the root package `__init__.__version__` with v1.3.5 source metadata
  in development copies and added an explicit future-release checklist check
  for this field and installed-package metadata.
- Made matplotlib legend creation conditional in Custom Parameter Plot, T–S
  Diagram, Depth Profile, and Correlation Overview. This avoids empty-legend
  warnings and avoids the slow automatic legend-placement search for large
  Custom Plot datasets.

## Recording rule

For each non-JOSS change after a tagged release, add a concise entry here with
its user-facing effect, affected pages or assets, validation status, and
whether it should be considered for the next release. Do not copy detailed
development planning into the stable public repository.
