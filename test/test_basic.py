#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic import and release-version tests for EnvGeo-Seawater.

pytest loads ``test/conftest.py`` before this module, so the application root
is already importable here.
pytestはこのモジュールより先に``test/conftest.py``を読み込むため、アプリのrootは
ここで既にimport可能である。
"""

import envgeo_utils

def test_envgeo_utils_imports():
    """Confirm that the shared utility module imports. / 共通utility moduleをimportできることを確認する。"""
    assert envgeo_utils is not None

def test_release_version_metadata_is_current():
    """Keep the public module version aligned with the v1.3.4 release candidate.

    公開moduleの版情報がv1.3.4 release candidateと一致することを確認する。
    """
    assert envgeo_utils.APP_VERSION == "1.3.4"
    assert envgeo_utils.version == envgeo_utils.APP_VERSION
