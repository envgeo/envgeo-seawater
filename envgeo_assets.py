"""
envgeo_assets.py — EnvGeo-Seawater application asset resolver (Sprint 1).

Provides a CWD-independent path to bundled application assets.
The application root is determined from this module's own location,
not from the working directory.

同梱資産の場所を、起動時のカレントディレクトリ（CWD）に依存せずに
解決する小さな共通モジュールです。ページ側は直接の相対パスではなく
``asset_path()`` を使います。個人データ、書込み可能なキャッシュ、
パッケージデータの実体化はこのSprint 1の対象外です。

API
---
    application_root() -> Path
    asset_path(*parts, required=True) -> Path

Not implemented in this sprint
------------------------------
- importlib.resources / package data lookup (Sprint 2+)
- User or OS cache directory (Sprint 2+)
- pyproject.toml / pip install packaging (Sprint 3+)
- Physical asset relocation (deferred)
"""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath

# The application root is the directory that contains this file.
# Stable whether Streamlit is launched from the repo root, a parent
# directory, or (in a future sprint) from an installed package entry point.
_APPLICATION_ROOT: Path = Path(__file__).resolve().parent


def application_root() -> Path:
    """Return the absolute path of the EnvGeo-Seawater application root.

    Always the directory containing ``envgeo_assets.py``, regardless of
    the current working directory.
    """
    return _APPLICATION_ROOT


def asset_path(*parts: str, required: bool = True) -> Path:
    """Resolve a bundled application asset to an absolute :class:`~pathlib.Path`.

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
