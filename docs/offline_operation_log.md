# Offline Operation Log

## Overview

This document records the design decisions and implementation details for the
offline / degraded-network operation of the Plotly interactive maps in
**EnvGeo Seawater** (v1.3.3).

---

## Background

EnvGeo Seawater uses Plotly Mapbox for interactive 2-D and 3-D maps.  By
default, Plotly Mapbox fetches background tile images from third-party CDN
servers at render time.  When those servers are unreachable — e.g. in
air-gapped research environments or aboard ships — the map background goes
blank and the user loses geographic context.

The local coastline overlay (Natural Earth 50 m resolution, stored as a
bundled CSV) was introduced to guarantee that at least the coastlines are
always visible, regardless of network state.

---

## Tile availability by map mode

| Map mode | Tile source | Requires network? |
|---|---|---|
| Coastline (offline) | None (white background) | No |
| Standard | OpenStreetMap tiles | Yes |
| Satellite | USGS National Map | Yes |
| Bathymetry (Sea) | Esri World Ocean Base | Yes |
| Contour (GSI) | Geospatial Information Authority of Japan | Yes |

**Note:** All online tile sources are used under their respective public
web-use licences.  None requires an API key as of v1.3.3.  CARTO basemaps,
which previously required an API key, have been removed.

---

## Coastline overlay — always shown

Since sprint #48 (2026-09), `apply_map_style()` calls
`add_coastline_overlay()` for **every** map mode before returning.  This
means:

* **Coastline (offline):** white background + local coastline + data points.
* **Online modes (Standard, Satellite, …):** tile background + local coastline
  overlay + data points.  If the browser fails to fetch tiles, the local
  coastline still gives spatial reference.

The overlay is idempotent: if a `_coastline_overlay` trace already exists on
the figure, a second call to `add_coastline_overlay()` is a no-op.

---

## Offline detection and user feedback

`resolve_map_mode(map_mode)` performs a lightweight socket probe and returns
`(effective_mode, fell_back)`.  If `fell_back` is True the calling page
displays:

```
⚠️ Online map tiles are unavailable. Showing the local coastline map.
```

Pages should call `resolve_map_mode()` before building the figure and pass the
`effective_mode` result to `apply_map_style()`.

---

## Self-contained HTML downloads

`figure_to_self_contained_html(fig) → bytes` exports a Plotly figure as a
fully offline-capable HTML file with Plotly.js (~4.5 MB) embedded inline
(`include_plotlyjs=True`).  No CDN reference is emitted.  The file opens in
any modern browser without a network connection.

Config flags in the exported HTML:

| Flag | Value |
|---|---|
| `scrollZoom` | `true` |
| `displayModeBar` | `true` |
| `responsive` | `true` |

Download buttons labelled **"Download interactive HTML"** appear on pages 03
and 04.  Page 05 also provides a download button via `download_figure()`.

---

## Constraints not changed by this sprint

* Cartopy static contour maps (page 32), Vertical Section, and GEBCO logic are
  unchanged.
* Persistent user-data storage locations and upload/share specifications are
  unchanged.
* Folium / Leaflet / Draw CDN assets are not bundled locally.
* No earthquake catalogue, plate boundary, or Bering Sea date-line logic was
  modified.
* Version is 1.3.3.

---

## Test coverage

| Test file | Suite | What is tested |
|---|---|---|
| `test/test_offline_map.py` | `TestAddCoastlineOverlayIdempotency` | Second call is a no-op; returns True; trace named correctly |
| `test/test_offline_map.py` | `TestApplyMapStyleAddsCoastline` | All 5 map modes produce `_coastline_overlay`; double-apply produces exactly 1 trace |
| `test/test_offline_map.py` | `TestOfflineFallbackWarningText` | English-only, starts with ⚠️, no Japanese, no `/` separator |
| `test/test_self_contained_html.py` | `TestFigureToSelfContainedHtmlSource` | Helper exists, `include_plotlyjs=True`, config constants present |
| `test/test_self_contained_html.py` | `TestFigureToSelfContainedHtmlRuntime` | Returns bytes >1 MB, no CDN script src, config flags serialised in HTML |
| `test/test_self_contained_html.py` | `TestPage03/04/05*` | Pages call helper, have download buttons, no CDN reference |
