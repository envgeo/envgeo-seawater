# EnvGeo-Seawater ToDo

This file records development notes that should not be forgotten during local
cleanup and refactoring. Items here can be moved into README, documentation, or
GitHub issues when they become stable enough.

## High Priority

### User-data upload support for individual pages

Date added: 2026-09-13

Current status:
- Keep the existing implementation in `90_Integrated_Visualizer_beta.py` as the
  main test bed for user-data upload workflows.
- Do not add upload widgets to every individual page yet.
- A possible final publication style is to expose mainly the Integrated
  Visualizer, while it internally calls selected individual-page workflows.
  Individual pages can remain available for development, checking, and advanced
  maintenance.

Planned direction:
- Move the common upload workflow into `envgeo_utils.py` before expanding it to
  individual pages.
- Treat user-data upload support as a core EnvGeo utility, not only as a
  seawater-specific feature. The same foundation should eventually support other
  EnvGeo applications, such as earthquake and other geoscience visualizers.
- The shared workflow should handle CSV/XLSX reading, column-name
  standardization, required-column checks, d-excess calculation, quality flags,
  and session-only memory handling.
- After the shared workflow is stable, test it first in a small number of
  representative pages:
  - Temperature-Salinity Diagram
  - Salinity-d18O Relationship
  - Mapping / isotope map pages
  - Depth Profile
- Treat `3D/4D Visualizer Uploader` as a likely private/development-only page
  unless its purpose is redesigned. Its original role as a generic uploaded-data
  3D visualizer is becoming less central as uploaded-data support moves toward
  the Integrated Visualizer and shared `envgeo_utils.py` core functions.

Notes:
- Uploaded user data should remain in memory only and should not be saved to the
  local machine or server.
- Uploaded user data should be visually distinguishable from reference data.
- Uploaded-data quality flags should be visible and downloadable where relevant.

## Streamlit Upgrade Follow-ups

### Form submit button keys

Date added: 2026-09-13

Current status:
- The current local/test environment uses Streamlit 1.42.0.
- In this version, `st.form_submit_button()` does not support the `key`
  argument.
- `51_Correlation_Overview.py` therefore uses two different labels,
  `Apply settings` and `Apply settings!`, for the top and bottom submit buttons
  inside the same form.

Planned direction:
- After updating Streamlit to a version where `st.form_submit_button()` supports
  `key`, revisit the two-button form layout.
- Use matching button labels, such as `Apply settings`, with separate keys for
  the top and bottom buttons.
- Apply the same pattern to other long sidebar forms if needed.
