"""Console launcher for the installed EnvGeo-Seawater diagnostic tool.

The diagnostic tool is available to local wheel users without placing its
development-only Streamlit wrapper in the normal application page navigation.

診断ツールは通常のアプリナビゲーションに開発用wrapperを置かず、ローカルの
wheel利用者が明示的なcommandから起動できるようにする。
"""

from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# Diagnostic-script resolution / 診断スクリプトの解決
# =============================================================================

def diagnostic_script() -> Path:
    """Return the installed-or-checkout Streamlit diagnostic script.

    インストール環境またはcheckout内の診断用Streamlitスクリプトを返す。
    """
    return Path(__file__).resolve().with_name("tools") / "env_check_streamlit.py"


# =============================================================================
# Diagnostic CLI launch / 診断CLIの起動
# =============================================================================

def main() -> None:
    """Launch the local diagnostic tool through Streamlit.

    ローカル診断ツールをStreamlit経由で起動する。
    """
    script = diagnostic_script()
    app_dir = script.parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(script), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())
