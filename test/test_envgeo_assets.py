"""Tests for the :mod:`envgeo_assets` resolver. / :mod:`envgeo_assets` の資産解決テスト。

Run from the application root with ``python -m pytest -q test/test_envgeo_assets.py``.
アプリのrootから上記コマンドで実行する。

The tests cover CWD-independent bundled assets, absolute-path and traversal
rejection, and the behaviour for required and optional missing assets.
作業ディレクトリに依存しない同梱資産、絶対パス・親ディレクトリ遡行の拒否、必須／任意の
欠落資産の挙動を確認する。
"""
from __future__ import annotations

from pathlib import Path

import pytest

import envgeo_assets


# ---------------------------------------------------------------------------
# application_root() / アプリケーションroot
# ---------------------------------------------------------------------------

class TestApplicationRoot:
    def test_returns_path(self):
        assert isinstance(envgeo_assets.application_root(), Path)

    def test_is_absolute(self):
        assert envgeo_assets.application_root().is_absolute()

    def test_is_directory(self):
        assert envgeo_assets.application_root().is_dir()

    def test_contains_home_py(self):
        """The root must contain the Streamlit entry point ``home.py``. / rootにStreamlitの入口``home.py``がある。"""
        assert (envgeo_assets.application_root() / "home.py").exists()

    def test_contains_envgeo_assets_py(self):
        assert (envgeo_assets.application_root() / "envgeo_assets.py").exists()

    def test_cwd_independent(self, tmp_path, monkeypatch):
        """The application root must not change with CWD. / CWDを変えてもアプリrootは変わらない。"""
        expected = envgeo_assets.application_root()
        monkeypatch.chdir(tmp_path)
        assert envgeo_assets.application_root() == expected


# ---------------------------------------------------------------------------
# asset_path() — existing bundled assets / 同梱済み資産
# ---------------------------------------------------------------------------

class TestAssetPathExistingAssets:
    """Representative assets must resolve from any CWD. / 代表的な資産を任意のCWDから解決できる。"""

    def _check(self, *parts, is_dir=False):
        p = envgeo_assets.asset_path(*parts)
        assert isinstance(p, Path)
        assert p.is_absolute()
        if is_dir:
            assert p.is_dir(), f"Expected directory: {p}"
        else:
            assert p.exists(), f"Expected file: {p}"
        return p

    def test_coastline_50m_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = self._check("coastline/world_coastline_coordinates_50m.csv")
        assert p.suffix == ".csv"

    def test_coastline_110m_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._check("coastline", "world_coastline_coordinates_110m.csv")

    def test_natural_earth_50m_land_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._check("coastline/natural_earth_50m_land", is_dir=True)

    def test_natural_earth_50m_shp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._check("coastline/natural_earth_50m_land/ne_50m_land.shp")

    def test_local_user_data_xlsx(self, tmp_path, monkeypatch):
        """The zero-value public user-data sample is present and resolvable. / 値を含まない公開サンプルを解決できる。"""
        monkeypatch.chdir(tmp_path)
        p = envgeo_assets.asset_path("local_data/user_data.xlsx", required=False)
        assert isinstance(p, Path)
        assert p.is_absolute()
        # The file is tracked in git — it must exist in a source checkout.
        assert p.exists(), f"Expected tracked sample: {p}"

    def test_sites_gif(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = self._check("data/sites_20230515.gif")
        assert p.suffix == ".gif"

    def test_gebco_required_false(self, tmp_path, monkeypatch):
        """The large optional GEBCO grid must not raise with ``required=False``. / 任意の大容量GEBCO格子は例外にしない。"""
        monkeypatch.chdir(tmp_path)
        p = envgeo_assets.asset_path("bathymetry/GEBCO_2025_6min.nc", required=False)
        assert isinstance(p, Path)
        assert p.is_absolute()
        # We don't assert existence — it may legitimately be absent in CI.


# ---------------------------------------------------------------------------
# asset_path() — security and validation / 安全性と入力検証
# ---------------------------------------------------------------------------

class TestAssetPathValidation:
    def test_rejects_absolute_posix(self):
        with pytest.raises(ValueError, match="absolute"):
            envgeo_assets.asset_path("/etc/passwd")

    def test_rejects_windows_drive_forward_slash(self):
        """Reject a Windows drive path with forward slashes on any OS. / Windows形式の絶対パスをOSにかかわらず拒否する。"""
        with pytest.raises(ValueError, match="absolute"):
            envgeo_assets.asset_path("C:/Windows/System32")

    def test_rejects_windows_drive_backslash(self):
        """Reject a Windows drive path with backslashes on any OS. / Windows形式の絶対パスをOSにかかわらず拒否する。"""
        with pytest.raises(ValueError, match="absolute"):
            envgeo_assets.asset_path("C:\\Windows\\System32")

    def test_rejects_unc_backslash(self):
        """Reject a Windows UNC path on any OS. / Windows UNCパスをOSにかかわらず拒否する。"""
        with pytest.raises(ValueError, match="absolute"):
            envgeo_assets.asset_path("\\\\server\\share\\file.csv")

    def test_allows_drive_relative_path(self):
        """Drive-relative paths like ``C:relative`` are not treated as absolute.

        Per the spec, only fully absolute Windows paths (drive + root, or UNC)
        are rejected. A drive-relative path has no root component and is not a
        path-injection risk in the same way.
        この形式にはroot成分がなく、絶対WindowsパスやUNCパスと同じ注入リスクとしては扱わない。
        """
        # C:relative_xyz does not exist, so FileNotFoundError is correct here;
        # the important thing is that it is NOT a ValueError('absolute').
        with pytest.raises(FileNotFoundError):
            envgeo_assets.asset_path("C:relative_does_not_exist")

    def test_rejects_parent_traversal_escape(self):
        """Parent traversal must raise ``ValueError``, not ``FileNotFoundError``. / 親ディレクトリ遡行は``ValueError``とする。"""
        with pytest.raises(ValueError, match=r"\.\.|outside"):
            envgeo_assets.asset_path("../../etc/passwd")

    def test_rejects_parent_traversal_multi(self):
        with pytest.raises(ValueError, match=r"\.\.|outside"):
            envgeo_assets.asset_path("..", "..", "etc", "passwd")

    def test_missing_required_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="EnvGeo asset not found"):
            envgeo_assets.asset_path("this_file_does_not_exist_xyz.nc", required=True)

    def test_missing_not_required_returns_path(self):
        p = envgeo_assets.asset_path("this_file_does_not_exist_xyz.nc", required=False)
        assert isinstance(p, Path)
        assert not p.exists()

    def test_no_parts_raises(self):
        with pytest.raises((ValueError, TypeError)):
            envgeo_assets.asset_path()

    def test_returns_path_type(self):
        p = envgeo_assets.asset_path("coastline/world_coastline_coordinates_50m.csv")
        assert isinstance(p, Path)

    def test_result_is_under_app_root(self):
        """The resolved path remains inside the application root. / 解決したパスがアプリrootの内側にある。"""
        p = envgeo_assets.asset_path("coastline/world_coastline_coordinates_50m.csv")
        # Use relative_to() rather than startswith() — relative_to() is
        # semantically correct and immune to prefix collisions such as
        # /app vs /app-extra.
        p.relative_to(envgeo_assets.application_root())  # raises ValueError if not inside
