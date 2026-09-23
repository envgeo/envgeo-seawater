"""conftest.py — make envgeo_utils importable from the test directory.

Running ``pytest -q test/test_offline_map.py`` from the app root (or any
working directory) should work without setting PYTHONPATH.  This conftest
adds the app root to sys.path so the bare ``import envgeo_utils`` inside
the test files resolves correctly.
"""
import pathlib
import sys

# Insert the app root (one level above this conftest) at the front of sys.path.
_APP_ROOT = str(pathlib.Path(__file__).parent.parent)
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)
