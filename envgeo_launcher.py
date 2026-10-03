"""Console launcher for the installed EnvGeo-Seawater Streamlit app.

It locates ``home.py`` from the installed module location and does not depend
on the process working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# Streamlit entry-script resolution / Streamlitエントリースクリプトの解決
# =============================================================================

def application_script() -> Path:
    """Return the installed-or-checkout Streamlit entry script.

    インストール環境またはcheckout内のStreamlit起点ファイルを返す。
    """
    return Path(__file__).resolve().with_name("home.py")


# =============================================================================
# Streamlit CLI launch / Streamlit CLIの起動
# =============================================================================

def main() -> None:
    """Launch Streamlit with the EnvGeo-Seawater entry script.

    EnvGeo-Seawaterの起点ファイルを指定してStreamlitを起動する。
    """
    app_script = application_script()
    app_dir = app_script.parent

    # Existing application modules use checkout-compatible absolute imports
    # such as ``import envgeo_utils``.  Streamlit normally adds the entry-script
    # directory; add it explicitly so the installed launcher has the same
    # import boundary before Streamlit starts its script thread.
    # 既存モジュールの絶対importが、インストール環境でも同じ境界で解決されるようにする。
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(app_script), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())
