# Integrated Visualizer development record (beta)

> Historical development record. Page 90 is retained here to explain earlier
> integration experiments. It is excluded from the stable-release application
> and the user-facing static documentation website.

## What This Page Does

This beta page is a permanent development workspace for testing existing
visualization pages with the same session-only uploaded data. It is not a
stable-release workflow.

## Basic Workflow

1. Upload data if needed; it remains available for the current Streamlit session.
2. Choose an existing page workflow.
3. Test that page with its own filters, figure controls, and uploaded-data overlay.
4. Use **User Data Check & Quick Visualizer** for combined filtering, quality
   review, and quick 2D--4D exploration.

## Main Controls

- Full existing page workflows
- Uploaded data overlay controls
- Legacy Shared-filter beta enablement (compatibility checks only)

## Outputs

- Existing-page workflows using the same uploaded session data
- Page-specific uploaded-data overlays and scientific figure controls
- Legacy diagnostic plots only when the compatibility workflow is explicitly enabled

## Notes And Limitations

- This page is retained as development history, not a supported stable-release workflow.
- Uploaded data are intended to remain in memory only during the Streamlit session.
- Some workflows call existing page files for compatibility.
- Individual visualization pages remain first-class workflows. The public
  **User Data Check & Quick Visualizer** now provides the upload-first
  validation, reference comparison, and simple diagnostic role formerly
  planned as an independent User Data Validator.
- Integrated Visualizer remains a permanent development workspace, rather than a
  normal public workflow or a stable-release feature.
- The duplicate Shared-filter beta is hidden behind an explicit development
  compatibility toggle and will be removed after it is no longer needed for
  regression checks. It must not receive new user-facing features.
- See the [Integrated Visualizer strategy](../integrated_visualizer_strategy.md) for the accepted architecture and migration plan.
