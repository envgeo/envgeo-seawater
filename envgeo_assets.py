"""Resolve read-only EnvGeo-Seawater assets independently of the CWD.

The application root is the directory containing this module.  Pages should use
``asset_path()`` rather than direct relative paths.

起動時のカレントディレクトリに依存せず、読み取り専用の同梱資産を解決します。
ページ側では直接の相対パスではなく``asset_path()``を使います。

This module does not manage private data, user output, or writable caches.
個人データ、利用者出力、書込み可能なcacheは扱いません。
"""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath

# =============================================================================
# Application root / アプリケーションroot
# =============================================================================
# The application root is stable for checkout and installed-package launches.
# checkoutとインストール済みpackageのどちらで起動しても同じrootを使う。
_APPLICATION_ROOT: Path = Path(__file__).resolve().parent


# -------------------------------------------------------------------
# Application-root lookup / アプリケーションrootの取得
# -------------------------------------------------------------------
def application_root() -> Path:
    """Return the absolute path of the EnvGeo-Seawater application root.

    Always the directory containing ``envgeo_assets.py``, regardless of
    the current working directory.

    カレントディレクトリにかかわらず、``envgeo_assets.py``を含む
    アプリケーションrootの絶対パスを返します。
    """
    return _APPLICATION_ROOT


# =============================================================================
# Bundled-asset path resolution / 同梱資産パスの解決
# =============================================================================
def asset_path(*parts: str, required: bool = True) -> Path:
    """Resolve a bundled application asset to an absolute :class:`~pathlib.Path`.

    同梱アプリケーション資産を絶対``Path``として解決し、root外参照を拒否します。

    Parameters
    ----------
    *parts:
        Path components relative to the application root, e.g.::

            asset_path("coastline", "world_coastline_coordinates_50m.csv")
            asset_path("coastline/world_coastline_coordinates_50m.csv")

    required:
        When ``True`` (default), raise :exc:`FileNotFoundError` with a
        descriptive message if the resolved path does not exist.
        Pass ``False`` for optional assets (e.g. large data files that
        degrade gracefully when absent).

    Returns
    -------
    Path
        Absolute, resolved path to the requested asset.

    Raises
    ------
    ValueError
        If the requested path is absolute (POSIX or Windows drive/UNC
        format) or resolves outside the application root (``..`` traversal).
    FileNotFoundError
        If *required* is ``True`` and the asset does not exist.
    """
    if not parts:
        raise ValueError("asset_path() requires at least one path component.")

    # Accept a single slash-joined string or multiple components.
    joined = parts[0] if len(parts) == 1 else os.path.join(*parts)

    # Reject absolute paths: POSIX-style (/etc/…) and Windows drive/UNC
    # forms (C:\\…, C:/…, \\\\server\\share\\…, //server/share/…).
    # PureWindowsPath is used cross-platform so that Windows-format paths
    # are caught on macOS and Linux as well.
    if os.path.isabs(joined) or PureWindowsPath(joined).is_absolute():
        raise ValueError(
            f"asset_path() does not accept absolute paths: {joined!r}. "
            "Supply a path relative to the application root."
        )

    resolved = (_APPLICATION_ROOT / joined).resolve()

    # Reject paths that escape the application root via '..'.
    try:
        resolved.relative_to(_APPLICATION_ROOT)
    except ValueError:
        raise ValueError(
            f"asset_path() path {joined!r} resolves outside the application "
            f"root ({_APPLICATION_ROOT}). '..' references are not permitted."
        )

    if required and not resolved.exists():
        raise FileNotFoundError(
            f"EnvGeo asset not found: {resolved}\n"
            f"  Requested : {joined!r}\n"
            f"  App root  : {_APPLICATION_ROOT}\n"
            "Ensure the bundled asset is present in the repository."
        )

    return resolved
