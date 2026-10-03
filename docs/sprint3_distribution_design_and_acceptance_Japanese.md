# 配布設計・検証記録（Sprint 3）

「Sprint 3」は、インストール可能な配布物、CI、公開前確認を整備した開発フェーズを指すプロジェクト内の名称である。この表題では、当時の記録との履歴上のつながりを保つためだけに用いる。

**状態:** Sprint 3の設計・実装検証記録。2026-09-27にSprint 3Bの設計基準として合意し、2026-09-30に再確認した。
**対象:** EnvGeo-Seawaterの開発／公開前repositoryのみ。公開Releaseの承認ではない。英語版は
[`sprint3_distribution_design_and_acceptance.md`](sprint3_distribution_design_and_acceptance.md)。

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
- Python 3.10と3.12はLinux CIで検証済みの対象とする。最初のローカル完全新規導入基準は
  Apple Silicon上のPython 3.12であり、Python 3.10の完全新規導入実証はCIで維持する。

## production package-dataの方針

アプリモジュール、サポート対象の安定版10ページ、実行時の媒体・テキスト、海岸線、Natural Earth、
GEBCO、ゼロ値User Excelテンプレート、診断ツール本体だけを収録する。開発用source cloneには
事前検証のためPage 90・91を残すが、wheelには収録しない。

- `dataset/*.xlsx`の技術的な収録確認と再配布／公開判断を分離し、このpackage作業を理由に
  データを無断で除外・公開しない。
- 現行の判断（2026-09-28）：現行の全`dataset/*.xlsx` workbookを学術利用packageへ含める。
  各workbookの出典引用と来歴／変換記録を維持し、同梱は所有関係を移転せず、将来のデータの
  取扱いを自動的に決めるものでもない。
- `pages/99_Environment_Check.py`は除外する。`pages/`内にあると、インストール版と
  Cloud版のどちらでも自動的にナビゲーションへ表示されるためである。
- 99ページを含めず、明示的なcommandから診断ツールを使えるようにする。通常の起動では
  公開ページだけを表示する。
- `bathymetry/make_lightweight_gebco.py`は実行時参照がないためwheelから除外する。GEBCO
  生成手順の記録としてsource treeには残す。
- `Claude outputs/`、cache、build成果物、`.DS_Store`、内部レビュー記録、ローカル専用
  ラッパーは除外する。

## 完全新規導入の受入条件

Sprint 3Cでは、system-site-packagesなし、user site-packagesなし、checkoutが
`PYTHONPATH`にない、checkout外のCWDという環境で、source treeではなく最終wheelを導入する。

1. wheel metadataから宣言済みruntime依存関係を導入できる。
2. import先がcheckoutではなく新規環境内である。
3. サポート対象の10ページが存在し、Page 90・91・99と内部資料は存在しない。
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
- 新規環境のインストール先からpackageを読み込み、サポート対象の10ページと診断ツールが存在し、
  Page 90・91・99とGEBCO生成scriptがないことを確認した。checkout外CWDから両console commandがStreamlit起動引数を
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
最初の成功workflow runは2026-09-27の#3（4分25秒）であり、test suite、wheel作成、隔離導入検証、
wheel artifact保存を完了した。
文書・配布方針更新についても、2026-09-28の#4（5分58秒）が成功し、同じtest、wheel、隔離導入、
artifact確認を完了した。
Python 3.10と3.12は、2026-09-28の#6（5分57秒）でこの確認にともに成功した。workflowでは、
Python版ごとに別のwheel artifactを保持する。最初のPython 3.10収集errorは、Python 3.11以降の
`tomllib`をtestが使っていたことだけによるものであり、条件付きのtest専用`tomli` fallbackで修正した。
runtime依存関係およびwheel内容は変更していない。

### Community Cloud 手動QA

これはwheelとCIの試験とは別の、ブラウザ上で行う受入確認である。現在の公開案内では、安定版は
`envgeo/seawater_map`からの`envgeo-seawater-map.streamlit.app`、開発／公開前版は
`envgeo/envgeo-seawater`からの`envgeo-seawater-pre.streamlit.app`として案内されている。
結果を記録する前に、Streamlitのdeployment設定で実際のrepository、branch、revisionを確認する。
wheel設定からCloudの構成を推測してはならない。

開発／公開前deploymentについて、確認日、URL、deploymentのrepository・branch・revision、表示された
errorを記録し、次を確認する。

- Homeが開き、主要ナビゲーションを利用できる。
- サイドバーには意図した公開ページだけが表示され、`99_Environment_Check.py`が表示されない。
- Mapping、T-S Diagram、Depth Profile、および現在公開している場合はIntegrated Visualizer betaが、
  アプリケーションerrorなく開いて描画される。
- データセット選択と`Apply settings`で代表ビューが更新される。
- オンライン地図タイルや外部サービスは、CIの要件ではなく、オンライン可用性として別に観察・記録する。
- local filesystem path、private user data、token、診断専用情報がdeployment上に表示されない。

#### 手動QA記録（2026-09-28）

開発／公開前deploymentを手動確認した。Homeが開き、sidebarに`99_Environment_Check.py`は表示されず、
Mapping、T-S Diagram、Depth Profileはアプリケーションerrorなく開いた。データセット選択と
`Apply settings`により代表ビューが更新され、オンライン地図タイルも表示された。通常利用の範囲では
application errorは観察されなかった。error詳細を確認するために意図的にerrorを起こすことは手動QAの
対象にしない。private dataと診断情報の除外は、Release監査とsource/package確認で扱う。

## Releaseの境界

実証版`1.3.3`は正式Release版ではない。予定する正式Releaseは`1.3.4`とし、clean checkout、
source revision、Python版、解決済み依存関係記録、wheel SHA-256、テスト結果、Git tag、
GitHub Release、Zenodo DOIを対応付ける。JOSSでは安定したSeawaterワークフローを中心にする。
Page 90と91は当面開発／公開前repositoryに残すが、予定する安定版`seawater_map`の対象からは外す。
