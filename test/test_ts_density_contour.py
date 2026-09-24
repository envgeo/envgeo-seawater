"""
test_ts_density_contour.py
──────────────────────────────────────────────────────────────────────────────
Focused tests for T-S Density Contours Stage 1 acceptance conditions.

Acceptance conditions (from docs/ts_density_contour_review_and_plan.md):
  1. Contour grid is bounded by the selected displayed axis limits, not by
     data extrema plus a fixed extension.
  2. Contour grid is clipped to the valid GSW input domain so invalid values
     (e.g. negative salinity from the Global axis) are never passed to gsw.sigma0.
  3. Density contour interval is user-selectable (0.2 / 0.5 / 1.0 kg m⁻³;
     default 0.5 kg m⁻³). levels=50 is replaced by an explicit level array.
  4. English and Japanese README / manual text states that the layer is an
     approximate σ0 reference grid, not pointwise sample density.

These tests do NOT run the Streamlit app (no AppTest dependency) and do NOT
require a network connection or external data files.
"""

import re
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Locate the project root (two levels up from this test file)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# 1. Contour grid bounds — static analysis of 34_T-S_diagram.py
# ===========================================================================

class TestContourGridBounds:
    """The contour linspace calls must reference axis-limit variables,
    not data-extrema variables."""

    _src = (ROOT / "pages" / "34_T-S_diagram.py").read_text(encoding="utf-8")

    def test_axis_limit_variables_used_for_tempL(self):
        """tempL linspace must use lim_min_Y / lim_max_Y (possibly clipped)."""
        # Accept either the direct form or the clipped _T_lo/_T_hi form
        assert re.search(
            r"tempL\s*=\s*np\.linspace\s*\(\s*(_T_lo|lim_min_Y)",
            self._src,
        ), (
            "tempL linspace does not use axis-limit variables (lim_min_Y or _T_lo). "
            "Data-extrema-based grid (e.g. mint-5/maxt+5) must not be used."
        )

    def test_axis_limit_variables_used_for_salL(self):
        """salL linspace must use lim_min_X / lim_max_X (possibly clipped)."""
        assert re.search(
            r"salL\s*=\s*np\.linspace\s*\(\s*(_S_lo|lim_min_X)",
            self._src,
        ), (
            "salL linspace does not use axis-limit variables (lim_min_X or _S_lo). "
            "Data-extrema-based grid (e.g. mins-5/maxs+5) must not be used."
        )

    def test_no_data_extrema_extension(self):
        """Old pattern mint-5 / maxt+5 / mins-5 / maxs+5 must be absent."""
        for forbidden in ("mint-5", "maxt+5", "mins-5", "maxs+5"):
            assert forbidden not in self._src, (
                f"Forbidden data-extrema extension pattern '{forbidden}' "
                "still present in 34_T-S_diagram.py."
            )

    def test_sigma0_approx_variable_name(self):
        """Variable must be named sigma0_approx (not sigma_theta)."""
        assert "sigma0_approx" in self._src, (
            "sigma0_approx not found in 34_T-S_diagram.py."
        )
        assert "sigma_theta" not in self._src, (
            "Deprecated variable name sigma_theta still present in 34_T-S_diagram.py."
        )

    def test_sigma0_approx_uses_gsw_sigma0(self):
        """sigma0_approx must be computed via gsw.sigma0."""
        assert re.search(
            r"sigma0_approx\s*=\s*gsw\.sigma0\s*\(",
            self._src,
        ), "sigma0_approx = gsw.sigma0(...) not found in 34_T-S_diagram.py."

    def test_contour_call_uses_sigma0_approx(self):
        """ax.contour must reference sigma0_approx, not sigma_theta."""
        assert re.search(
            r"ax\.contour\s*\([\s\S]*?sigma0_approx",
            self._src,
        ), "ax.contour call does not reference sigma0_approx."


# ===========================================================================
# 2. Domain clipping — GSW valid input range
# ===========================================================================

class TestDomainClipping:
    """The contour grid must be clipped to the quality-checked valid domain
    before being passed to gsw.sigma0."""

    _src = (ROOT / "pages" / "34_T-S_diagram.py").read_text(encoding="utf-8")

    def test_temperature_gsw_bounds_defined(self):
        """Source must define the GSW temperature clipping bounds."""
        assert re.search(
            r"_T_GSW_MIN\s*,\s*_T_GSW_MAX\s*=\s*-5",
            self._src,
        ), "_T_GSW_MIN / _T_GSW_MAX = -5 ... not found in 34_T-S_diagram.py."

    def test_salinity_gsw_bounds_defined(self):
        """Source must define the GSW salinity clipping bounds."""
        assert re.search(
            r"_S_GSW_MIN\s*,\s*_S_GSW_MAX\s*=\s*0",
            self._src,
        ), "_S_GSW_MIN / _S_GSW_MAX = 0 ... not found in 34_T-S_diagram.py."

    def test_temperature_lower_clamp_applied(self):
        """Temperature grid lower bound must use max(axis_min, GSW_MIN)."""
        assert re.search(
            r"_T_lo\s*=\s*max\s*\(\s*float\s*\(\s*lim_min_Y\s*\)\s*,\s*_T_GSW_MIN\s*\)",
            self._src,
        ), "_T_lo = max(float(lim_min_Y), _T_GSW_MIN) not found."

    def test_salinity_lower_clamp_applied(self):
        """Salinity grid lower bound must use max(axis_min, GSW_MIN)."""
        assert re.search(
            r"_S_lo\s*=\s*max\s*\(\s*float\s*\(\s*lim_min_X\s*\)\s*,\s*_S_GSW_MIN\s*\)",
            self._src,
        ), "_S_lo = max(float(lim_min_X), _S_GSW_MIN) not found."

    def test_guard_against_degenerate_grid(self):
        """Source must guard that T_lo < T_hi and S_lo < S_hi before gridding."""
        assert re.search(
            r"if\s+_T_lo\s*<\s*_T_hi\s+and\s+_S_lo\s*<\s*_S_hi",
            self._src,
        ), "Degenerate-range guard 'if _T_lo < _T_hi and _S_lo < _S_hi' not found."


# ===========================================================================
# 3. Contour interval UI and explicit levels
# ===========================================================================

class TestContourIntervalControl:
    """Density contour interval must be user-selectable; levels=50 must be absent."""

    _src = (ROOT / "pages" / "34_T-S_diagram.py").read_text(encoding="utf-8")

    def test_levels_50_absent(self):
        """The legacy levels=50 fixed count must not appear in the contour call."""
        assert "levels=50" not in self._src, (
            "levels=50 still present in 34_T-S_diagram.py. "
            "It must be replaced by an explicit contour level array."
        )

    def test_contour_interval_selectbox_present(self):
        """A selectbox for contour_interval must exist in the sidebar."""
        assert "contour_interval" in self._src, (
            "contour_interval variable not found in 34_T-S_diagram.py."
        )
        assert re.search(r"st\.selectbox\s*\(", self._src), (
            "st.selectbox not found in 34_T-S_diagram.py."
        )

    def test_default_interval_is_0_5(self):
        """Default selection must be 0.5 kg m⁻³ (index=1 among [0.2, 0.5, 1.0])."""
        assert re.search(
            r"options\s*=\s*\[\s*0\.2\s*,\s*0\.5\s*,\s*1\.0\s*\]",
            self._src,
        ), "options=[0.2, 0.5, 1.0] not found for the contour interval selectbox."
        assert re.search(
            r"index\s*=\s*1",
            self._src,
        ), "index=1 (default 0.5) not found for the contour interval selectbox."

    def test_explicit_contour_levels_used(self):
        """ax.contour must use contour_levels (not a plain integer)."""
        assert re.search(
            r"levels\s*=\s*contour_levels",
            self._src,
        ), "levels=contour_levels not found in the ax.contour call."

    def test_contour_levels_arange_from_interval(self):
        """Explicit levels must be built from the interval using np.arange or similar."""
        assert re.search(
            r"np\.arange\s*\([\s\S]*?contour_interval",
            self._src,
        ), "np.arange(..., contour_interval) level generation not found."


# ===========================================================================
# 4. Numerical contour level generation — direct unit test
# ===========================================================================

class TestContourLevelGeneration:
    """Test the explicit level generation logic directly (no Streamlit needed)."""

    @staticmethod
    def _make_levels(s0_min: float, s0_max: float, interval: float):
        """Mirror of the level generation logic in 34_T-S_diagram.py."""
        if not (np.isfinite(s0_min) and np.isfinite(s0_max) and s0_max > s0_min):
            return np.array([])
        first = np.ceil(s0_min / interval) * interval
        levels = np.arange(first, s0_max + interval * 0.5, interval)
        levels = levels[levels <= s0_max]
        return levels

    def test_typical_range_0_5_interval(self):
        """For a typical 20–28 kg/m³ range at 0.5 interval, levels should land on 0.5 multiples."""
        levels = self._make_levels(20.1, 27.9, 0.5)
        assert len(levels) >= 1, "Expected levels in range [20.1, 27.9] at 0.5 interval."
        assert all(abs(v % 0.5) < 1e-9 or abs(v % 0.5 - 0.5) < 1e-9 for v in levels), (
            "Levels are not multiples of 0.5."
        )
        assert levels[0] >= 20.1, "First level below s0_min."
        assert levels[-1] <= 27.9, "Last level above s0_max."

    def test_0_2_interval_finer_grid(self):
        """0.2 interval should produce more levels than 0.5 for the same range."""
        lv_02 = self._make_levels(20.0, 28.0, 0.2)
        lv_05 = self._make_levels(20.0, 28.0, 0.5)
        assert len(lv_02) > len(lv_05), "0.2 interval should yield more contours than 0.5."

    def test_1_0_interval_coarser_grid(self):
        """1.0 interval should produce fewer levels than 0.5 for the same range."""
        lv_10 = self._make_levels(20.0, 28.0, 1.0)
        lv_05 = self._make_levels(20.0, 28.0, 0.5)
        assert len(lv_10) < len(lv_05), "1.0 interval should yield fewer contours than 0.5."

    def test_degenerate_range_returns_empty(self):
        """Equal min/max should return an empty array (no levels)."""
        levels = self._make_levels(25.0, 25.0, 0.5)
        assert len(levels) == 0, "Degenerate range should return zero levels."

    def test_nan_returns_empty(self):
        """NaN bounds should return an empty array safely."""
        levels = self._make_levels(float("nan"), float("nan"), 0.5)
        assert len(levels) == 0, "NaN bounds should return zero levels without error."

    def test_levels_do_not_exceed_bounds(self):
        """No generated level should lie outside [s0_min, s0_max]."""
        s0_min, s0_max = 22.3, 29.7
        for interval in [0.2, 0.5, 1.0]:
            levels = self._make_levels(s0_min, s0_max, interval)
            if len(levels) > 0:
                assert levels[0] >= s0_min - 1e-9, (
                    f"First level {levels[0]} < s0_min {s0_min} at interval {interval}."
                )
                assert levels[-1] <= s0_max + 1e-9, (
                    f"Last level {levels[-1]} > s0_max {s0_max} at interval {interval}."
                )


# ===========================================================================
# 5. Approximation wording — English sources
# ===========================================================================

class TestApproximationWordingEnglish:
    """Key approximation phrases must appear in English README and manual."""

    _readme = (ROOT / "README.md").read_text(encoding="utf-8")
    _manual = (ROOT / "docs" / "manual" / "34_ts_diagram.md").read_text(encoding="utf-8")
    _page   = (ROOT / "pages" / "34_T-S_diagram.py").read_text(encoding="utf-8")

    def test_readme_no_sigma_theta(self):
        """README.md must not annotate contours as (σθ)."""
        assert "density contours (σθ)" not in self._readme, (
            "README.md still labels contours as (σθ). Stage 1 requires removal."
        )

    def test_readme_mentions_approximate_sigma0(self):
        """README.md must mention approximate σ0 in the context of contours."""
        assert re.search(r"approximate\s+σ0", self._readme), (
            "README.md does not contain 'approximate σ0' near the contour description."
        )

    def test_manual_mentions_approximate_sigma0(self):
        """English manual must mention approximate σ0 reference contours."""
        assert re.search(r"approximate\s+σ0", self._manual), (
            "docs/manual/34_ts_diagram.md does not contain 'approximate σ0'."
        )

    def test_manual_not_pointwise_sample_density(self):
        """English manual must state contours are not pointwise sample density."""
        assert "pointwise sample density" in self._manual, (
            "English manual does not state 'not pointwise sample density'."
        )

    def test_manual_no_0_4_kg_bound(self):
        """English manual must not state a specific 0.4 kg/m³ audit bound."""
        assert "0.4 kg" not in self._manual, (
            "English manual still contains audit-specific 0.4 kg figure. "
            "Replace with source/location/depth-dependent wording."
        )

    def test_manual_varies_by_source_or_location(self):
        """English manual must say differences vary by source, location, or depth."""
        assert re.search(
            r"vary\s+by\s+.*(source|location|depth)",
            self._manual,
        ), (
            "English manual does not state differences vary by source/location/depth."
        )

    def test_manual_lists_contour_interval_control(self):
        """English manual Main Controls must list the density contour interval setting."""
        assert "Density contour interval" in self._manual, (
            "English manual does not list 'Density contour interval' in Main Controls."
        )

    def test_page_caption_approximate_sigma0(self):
        """34_T-S_diagram.py must contain the σ0 approximation caption."""
        assert "approximate" in self._page and "σ0" in self._page, (
            "34_T-S_diagram.py does not contain the approximation caption."
        )

    def test_page_caption_not_pointwise(self):
        """34_T-S_diagram.py caption must say 'Not pointwise sample density'."""
        assert "Not pointwise sample density" in self._page, (
            "Approximation caption missing 'Not pointwise sample density'."
        )


# ===========================================================================
# 6. Approximation wording — Japanese sources
# ===========================================================================

class TestApproximationWordingJapanese:
    """Key approximation phrases must appear in Japanese README and manual."""

    _readme = (ROOT / "README_Japanese.md").read_text(encoding="utf-8")
    _manual = (ROOT / "docs" / "manual_Japanese" / "34_ts_diagram.md").read_text(encoding="utf-8")

    def test_readme_no_sigma_theta(self):
        """README_Japanese.md must not annotate contours as (σθ)."""
        assert "密度等値線（σθ）" not in self._readme, (
            "README_Japanese.md still labels contours as (σθ)."
        )

    def test_readme_mentions_approximate_sigma0(self):
        """README_Japanese.md must mention approximate σ0 (近似的な σ0)."""
        assert "近似的な σ0" in self._readme, (
            "README_Japanese.md does not contain '近似的な σ0'."
        )

    def test_manual_mentions_approximate(self):
        """Japanese manual must contain the approximation notice."""
        assert "近似的な σ0" in self._manual, (
            "Japanese manual does not contain '近似的な σ0'."
        )

    def test_manual_not_pointwise(self):
        """Japanese manual must state contours are not individual observation density."""
        assert (
            "個々の観測値の密度ではありません" in self._manual
            or "観測点ごとの密度ではありません" in self._manual
        ), (
            "Japanese manual does not state the contours are not pointwise sample density."
        )

    def test_manual_no_0_4_kg_bound(self):
        """Japanese manual must not state the specific 0.4 kg/m³ audit bound."""
        assert "0.4 kg" not in self._manual, (
            "Japanese manual still contains audit-specific 0.4 kg figure."
        )

    def test_manual_lists_contour_interval_control(self):
        """Japanese manual Main Controls must list the density contour interval setting."""
        assert "Density contour interval" in self._manual, (
            "Japanese manual does not list 'Density contour interval' in 主な設定."
        )
