# Sprint 3 配布設計と受入条件

**状態:** 2026-09-27に合意したSprint 3B実装用の設計基準。  
**対象:** EnvGeo-Seawaterのみ。公開Releaseの承認ではない。

## 固定する境界

- `home.py`と同階層の`pages/`、`envgeo_assets.asset_path()`、wheelに展開される
  実ファイルを維持する。
- `dataset/`、Cloud設定、明示的offline起動、現行配置、EnvGeo-Earthquakeのpackage化、
  共通core分離は変更しない。
- 2026-09-27作成のSprint 3開始前アーカイブを復元基準とする。

## 依存関係とPythonの方針

- `requirements.txt`を実行時の直接依存関係の正本とする。Community Cloud用にも維持し、
  wheelの依存metadataもここから導出する。
- `pyproject.toml`はbuild metadata、package data、console commandを管理し、独立した
  依存関係リストを重複管理しない。
- 将来のconstraints／lock相当の記録には、各検証環境で実際に解決された版を残す。
  これは検証根拠であり、第2の依存関係正本ではない。
- 完全新規導入とCIの最初の基準はPython 3.12とする。Python 3.10は対応候補として維持するが、
  同等の試験に通るまでは正式な導入確認済みとは表記しない。

## production package-dataの方針

アプリモジュール、現在の公開12ページ、実行時の媒体・テキスト、海岸線、Natural Earth、
GEBCO、ゼロ値User Excelテンプレート、診断ツール本体だけを収録する。

- `dataset/*.xlsx`の技術的な収録確認と再配布／公開判断を分離し、このpackage作業を理由に
  データを無断で除外・公開しない。
- `pages/99_Environment_Check.py`は除外する。`pages/`内にあると、インストール版と
  Cloud版のどちらでも自動的にナビゲーションへ表示されるためである。
- 99ページを含めず、明示的なcommandから診断ツールを使えるようにする。通常の起動では
  公開ページだけを表示する。
- `data_beta/make_lightweight_gebco.py`は実行時参照がないためwheelから除外する。GEBCO
  生成手順の記録としてsource treeには残す。
- `Claude outputs/`、cache、build成果物、`.DS_Store`、内部レビュー記録、ローカル専用
  ラッパーは除外する。

## 完全新規導入の受入条件

Sprint 3Cでは、system-site-packagesなし、user site-packagesなし、checkoutが
`PYTHONPATH`にない、checkout外のCWDという環境で、source treeではなく最終wheelを導入する。

1. wheel metadataから宣言済みruntime依存関係を導入できる。
2. import先がcheckoutではなく新規環境内である。
3. 公開12ページが存在し、99ページと内部資料は存在しない。
4. Homeと代表的なPage 32、34、53が例外なく起動し、必要な同梱資産を読める。
5. dataset workbook、海岸線CSV、Natural Earth sidecar、GEBCO、実行時媒体・テキスト、
   ゼロ値User Excelテンプレートを利用できる。
6. `ENVGEO_LOCAL_USER_DATA_PATH`が動作し、個人データをpackage、cache、logへコピーしない。
7. EnvGeo本体がインストール済みpackageの隣へ書き込まない。MatplotlibやCartopy等の
   dependency cacheは別項目として観測・記録する。
8. 診断用commandからツールを起動でき、通常アプリのナビゲーションには99ページが出ない。

全ページの視覚確認、外部タイル、オンライン連携は別の手動QAとする。

## Sprint 3C ローカル検証記録（2026-09-27）

Apple Silicon上のPython 3.12で、生成物を除いたステージングコピーからwheelを作り、
system-site-packagesとuser-site-packagesのどちらも使わない新規`venv`へ導入した。作業CWDは
source treeとwheelステージング領域の外側とした。検証したwheelのSHA-256は
`c5a34a2a7356b81e0fa42aaa2b61b0855099c77dedb44089d1b4c70c4de728c5`である。

- `pip check`は依存関係破損なしだった。`pyproject.toml`のlicenseは移植性のある明示table形式を
  用い、Python 3.12 Apple Silicon向けwheelを選べる`pyproj==3.6.1`を宣言した。これにより、
  互換しないsource-onlyの最新版へ解決される状態を回避した。
- 新規環境のインストール先からpackageを読み込み、公開12ページと診断ツールが存在し、99ページと
  GEBCO生成scriptがないことを確認した。checkout外CWDから両console commandがStreamlit起動引数を
  受け付け、診断ツール本体もアプリ例外なしで実行した。
- インストール済みファイルを使い、Home、Page 32、34、53をアプリ例外なしで実行した。個別ページには
  checkout互換のtop-level importが残るため、従来どおりlauncherが設定する互換import pathが必要である。
- `ENVGEO_LOCAL_USER_DATA_PATH`で外部CSVを指定し、`User Excel data`として1行を読めた。この確認中に
  インストール先package配下への書込みはなかった。Matplotlibのcacheは一時ディレクトリへ向け、
  アプリ本体の出力ではなく依存ライブラリのcacheとして区別した。

これはローカル技術検証であり、Release成果物や、Windows、Intel Mac、Linux、Python 3.10の対応確認を
意味しない。

## CI、Cloud、OSの順序

1. まずPython 3.12基準でローカル完全新規導入試験を行う。
2. Linux CIへ、build、wheel内容確認、隔離導入、ネットワーク非依存テストを追加する。
3. ローカルとLinuxが安定した後に現行Community Cloud設定を確認する。Sprint 3Bでは
   Cloud設定を変更しない。
4. WindowsとIntel Macのsmoke testを行い、対応範囲を判断する。

CIは環境構築時に宣言済み依存関係を取得してよいが、テストとアプリ確認は外部タイル、
外部download、実ネットワークに依存させない。

## Sprint 3D 実装記録（2026-09-27）

最初のLinux／Python 3.12ワークフローを`.github/workflows/ci.yml`に定義した。開発用依存関係を導入し、
local user dataを無効にしてテストを実行し、wheelを作成して、checkout外の隔離環境へのwheel導入を検証する。
wheelは確認用にworkflow artifactとして保存するだけで、公開・Releaseには使用しない。Cloud設定は変更して
いない。WindowsとIntel Macは、対応を表明する前の手動smoke test対象として残す。

## Releaseの境界

実証版`1.3.3`は正式Release版ではない。将来のReleaseでは、clean checkout、source revision、
Python版、解決済み依存関係記録、wheel SHA-256、テスト結果、Git tag、GitHub Release、
Zenodo DOIを対応付ける。JOSSでは安定したSeawaterワークフローを中心にし、Page 90と91を
公開ナビゲーションへ残すかはSprint 3C後に別途判断する。
