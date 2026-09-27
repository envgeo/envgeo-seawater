"""Console launcher for the installed EnvGeo-Seawater diagnostic tool.

The diagnostic tool is available to local wheel users without placing its
development-only Streamlit wrapper in the normal application page navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path


def diagnostic_script() -> Path:
    """Return the installed-or-checkout Streamlit diagnostic script."""
    return Path(__file__).resolve().with_name("tools") / "env_check_streamlit.py"


def main() -> None:
    """Launch the local diagnostic tool through Streamlit."""
    script = diagnostic_script()
    app_dir = script.parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(script), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())
