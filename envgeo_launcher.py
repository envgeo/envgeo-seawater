"""Console launcher for the installed EnvGeo-Seawater Streamlit app.

This launcher is part of the Sprint 2B wheel proof.  It locates ``home.py``
from the installed module location and does not depend on the process working
directory.
"""

from __future__ import annotations

import sys
from pathlib import Path


def application_script() -> Path:
    """Return the installed-or-checkout Streamlit entry script."""
    return Path(__file__).resolve().with_name("home.py")


def main() -> None:
    """Launch Streamlit with the EnvGeo-Seawater entry script."""
    app_script = application_script()
    app_dir = app_script.parent

    # Existing application modules use checkout-compatible absolute imports
    # such as ``import envgeo_utils``.  Streamlit normally adds the entry-script
    # directory; add it explicitly so the installed launcher has the same
    # import boundary before Streamlit starts its script thread.
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(app_script), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())
