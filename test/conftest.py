"""Make envgeo_utils importable when pytest starts from the test directory.

テストディレクトリからpytestを起動した場合にも、envgeo_utilsをimportできるようにする。

Running ``pytest -q test/test_offline_map.py`` from the app root (or any
working directory) should work without setting PYTHONPATH.  This conftest
adds the app root to sys.path so the bare ``import envgeo_utils`` inside
the test files resolves correctly.
テストファイル内の ``import envgeo_utils`` が正しく解決される。
"""
import pathlib
import sys

# Insert the app root at the front of sys.path. / アプリのrootをsys.pathの先頭へ追加する。
_APP_ROOT = str(pathlib.Path(__file__).parent.parent)
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)
