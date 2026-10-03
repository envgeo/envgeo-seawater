# 履歴上のwheel実証レポート（Sprint 2B〜2C）

「Sprint 2B〜2C」は、最初のローカルwheel実証を指すプロジェクト内名称である。本レポートは当時の証跡を保存するものであり、現在のRelease検証記録ではない。実装済み1.3.4の配布・CI確認は[`sprint3_distribution_design_and_acceptance_Japanese.md`](sprint3_distribution_design_and_acceptance_Japanese.md)を参照する。

**日付:** 2026-09-25  
**状態:** Python 3.12基準環境での歴史的ローカル技術実証に合格。  
**公開状態:** 公開用成果物でも、サポート済みインストール手順でもない。

## 範囲

`home.py`、`pages/`、同梱資産を`src/`へ移動せず、現行のリポジトリルート配置をインストールできるか検証した。Streamlit Community Cloud設定、依存関係の正本、`asset_path()`、cache／offline動作、`dataset/`内のファイル、GEBCOは変更していない。

## 最小実証ファイル

- `pyproject.toml`: setuptoolsによる現行ルートからpackageへの対応付け、package dataの明示的な収録／除外規則、実証用`envgeo-seawater`起動コマンド。
- `__init__.py`: インストール実証用のpackage識別。
- `envgeo_launcher.py`: プロセスCWDに依存せず、インストール先で同階層の`home.py`を見つけてStreamlitを起動する。
- `test/test_packaging_proof.py`: 非移動型配置、依存関係境界、資産系統、除外対象、起動コマンド定義を保護するテスト。

実証中のruntime依存関係は引き続き`requirements.txt`が管理する。`pyproject.toml`には重複記載していない。

## 環境と成果物

- Python 3.12.14
- Streamlit 1.63／Plotly 5.24環境
- setuptools 83.0.0／wheel 0.47.0
- buildコマンド:
  `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist`
- 成果物: `envgeo_seawater-1.3.3-py3-none-any.whl`
- サイズ: 25,970,909 bytes
- SHA-256: `225699be477cc35028d1edfce7bc32149ba72203116bd3d5f5439be8d12bdab2`

`dist/`内の成果物は生成物としてignoreされる。この実証wheelを公開・commitしてはならない。
実証レポート自体はwheelから除外し、成果物checksumの記録が自己参照buildにならないようにした。

## 確認結果

- アプリのディレクトリを移動せずwheelをbuildできた。
- checkout外の新規仮想環境へインストールした。仮想環境は`--system-site-packages`で検証済みruntime依存関係を継承したため、依存packageの新規解決自体は未検証である。
- CWDをcheckout外にして、`site-packages`からインストール済みpackageをimportできた。
- 予定する公開ページ12件をすべて収録した。
- `pages/99_Environment_Check.py`、`.DS_Store`、cacheファイルを除外した。
- `dataset/`、海岸線CSV、Natural Earth shapefile一式、ゼロ値ユーザーデータサンプル、Homeメディア、GEBCOの代表ファイルをインストール先から確認した。
- インストール済みHomeをStreamlit AppTestで実行し、タイトル表示とアプリ例外なしを確認した。
- 読取り専用にしたインストールpackageからPage 34を実行し、同梱データの読込みとアプリ例外なしを確認した。
- checkout外から`envgeo-seawater`コマンドを起動し、Streamlit health endpointの`ok`応答を確認した。
- インストールpackageの書込み権限を外した後もHomeとPage 34が動作した。通常のユーザーcacheが使えない場合、Matplotlibは独自の一時cacheへ縮退したが、EnvGeo packageへの書込みは不要だった。

## 自動テスト

package／資産／公開面の焦点テスト:

```text
35 passed
```

対応Python 3.12環境の全テスト:

```text
403 passed, 1 warning
```

warningは`test/test_self_contained_html.py`の既存class-scoped fixture将来非推奨警告であり、package失敗ではない。

最初のsandbox内全体テストでは、sandboxが接続確認を遮断したためVertical Section AppTestが正しいoffline分岐へ入り、online時だけ表示される`A-B input` radioを期待したテストが1件失敗した。同じテストと全体テストは通常のローカル接続条件で通過した。このテストのネットワーク依存性は別の保守項目であり、今回テストを弱めたり変更したりしていない。

## この実証では未確認のもの

- runtime packageを事前導入していない環境での依存関係導入。
- Windows、Linux、Intel Mac、Streamlit Community Cloudでの互換性。
- 依存関係の最終的な正本。
- `importlib.resources`、cache、明示的offline modeの最終API。
- フルデータwheel／archiveの公開可否。
- 将来のpackageディレクトリ／`src/`移動が不要であるという最終判断。

これらはSprint 2D、Sprint 3、その後のOS・公開レビューの判定項目として残す。
