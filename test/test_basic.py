import sys
import os

# Add the parent directory of the current file to sys.path
# 現在のファイル（test_basic.py）の1つ上の階層をシステムパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the target module after setting the path
# パスを通した後に、テスト対象のモジュールをインポートします
import envgeo_utils

def test_import():
    """
    Test if the target module can be imported successfully.
    対象のモジュールが正しくインポートできるかをテスト
    """
    assert envgeo_utils is not None

def test_version():
    """
    Test if the version information exists in the module.
    モジュール内にバージョン情報が存在するかをテスト
    """
    assert hasattr(envgeo_utils, "version")
