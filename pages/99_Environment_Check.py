#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expose the EnvGeo-Seawater environment-check tool as a Streamlit page.

Maintainer: Toyoho Ishimura, Kyoto University
Last updated: 2026-09-22
"""

from pathlib import Path
import runpy


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "env_check_streamlit.py"


runpy.run_path(str(TOOL_PATH), run_name="__main__")
