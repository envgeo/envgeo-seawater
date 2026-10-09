# EnvGeo-Seawater

EnvGeo-Seawater は、海水の安定同位体・水文データを探索するためのインタラクティブ可視化プラットフォームです。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://envgeo.h.kyoto-u.ac.jp/sw_jpn/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20CI-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/envgeo/envgeo-seawater/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/626690773.svg)](https://doi.org/10.5281/zenodo.23117783)

**現在の安定版:** 1.3.4（2026-10-03）

**海水同位体・水文データを、地図・断面・T-S図・3D/4D表示で探索する研究用Webアプリです。**

---

## 概要

EnvGeo-Seawater は、海洋地球化学研究のために海水同位体・水文データを探索・可視化・比較する、Streamlit ベースの対話型Webアプリケーションです。

対象とするデータは、安定水同位体（δ18O、δD）、塩分、水温、水深などを含む海洋地球化学・水文データです。

日本周辺で統一した分析手順により得られた EnvGeo Dataset と、NASA GISS、PAGES CoralHydro2k を含む地域・全球の参照データセットを統合しています。約5万件の引用付きレコードを、共通のフィルタリング条件で比較・可視化できます。

このアプリは、海洋地球化学・海洋学における **探索的データ解析** と **再現可能な研究ワークフロー** の両方を支援することを目的としています。

---

## 主な機能

- ズーム範囲を調整できるインタラクティブ地図表示
- 欠損や観測グループの切れ目を考慮した深度プロファイル
- 近似的な σ0 参照格子付きの Temperature-Salinity（T-S）図
- 塩分-δ18O 関係などの回帰解析
- 空間・深度・時間構造を探索する 3D / 4D 可視化
- EnvGeo Dataset、地域・全球参照データセットを横断した比較・可視化
- データセット名と Reference / Citation によるデータの絞込み
- 参照データセットと比較するためのユーザーデータアップロード機能
- 同梱参照データセットまたはアップロードデータとの重複データ候補の検証とフィルタリング
- フィルタ後データや除外サンプル数の表示
- 論文・発表用図の出力

---

## 主要ページ

以下は development source のページ一覧です。Streamlit アプリでは、`home.py` が About、データソース、マニュアル、更新履歴、日本語説明のタブを担当します。`pages/` ディレクトリには、主要な可視化ツールに加えて、一部の beta ページやローカル開発用ページも含まれます。

- `pages/03_[Interactive]_2Dplus_Visualizer.py`
  同位体・水文データの関係と観測地点を確認する 2D/2.5D 可視化ページ。

- `pages/04_[Interactive]_3D_4D_Visualizer.py`
  経度、緯度、水深、選択変数を扱う 3D/4D 可視化ページ。

- `pages/05_[Utils]_User_Data_Check_Quick_Visualizer.py`
  参照・CSV/XLSXアップロードデータを扱う User Data Check & Quick Visualizer。共通フィルタ、欠損・品質確認、2D Map、Salinity-d18O、Temperature-Salinity、任意2D/3D/4D、地理3D、フィルタ済みCSV出力を一つの入口へまとめます。

- `pages/06_[Utils]_Data_Overlap_Check.py`
  同梱する参照データセットと、セッション限定の一つのアップロード表について、データ重複候補を読取り専用で確認するページです。Strong／Review候補と照合根拠を出力し、元の観測値を保持したまま、任意の可逆的な表示スクリーンにも対応します。

- `pages/31_Salinity-d18O_Relationship.py`  
  塩分-δ18O 関係を表示し、必要に応じて回帰線を加えるページ。

- `pages/32_Isotope_Hydrographic_Mapping.py`  
  d18O、dD、d-excess、塩分、水温などの同位体・海洋環境パラメーターを2D地図やコンター風の図として表示するページ。

- `pages/34_T-S_diagram.py`  
  密度等値線付きの T-S 図を表示するページ。

- `pages/37_Depth_Profile.py`  
  δ18O、δD、d-excess、水温、塩分の深度プロファイルを表示するページ。

- `pages/35_Custom_Parameter_Plot.py`
  X軸、Y軸、色、マーカーサイズを任意の数値パラメーターから選ぶ柔軟な2Dプロットページ。

- `pages/53_Vertical_Section_Visualizer.py`
  Vertical Section Visualizer。測線選択、補間、海底地形、鉛直断面図を表示するページです。

- `pages/80_Correlation_Overview.py`
  手書きで開発してきた元の探索ワークフローを保存するアーカイブ表示ページです。開発記録として残し、新機能は追加しません。

- `pages/90_Integrated_Visualizer_beta.py`
  試作統合ページ。既存の可視化ワークフローを1ページ内から選択実行できる互換モード、既存ページへのユーザーデータ一時結合、海域プリセット付きの共通フィルタ・タブ切り替えモードを含みます。独立アップロードページはアップロード起点の別ワークフローのため、統合ページ内の選択肢からは外しています。

- `pages/99_Environment_Check.py`
  ローカル開発用の環境診断ラッパーページ。ローカル環境確認には有用ですが、公開 Streamlit サイドバーに表示するページではありません。

### 安定版の公開ページ

安定版 v1.3.4 には、Page 03、04、05、31、32、34、35、37、53、80 を含めます。Page 90、91、99 は開発・ローカル診断用であり、安定版の配布パッケージおよび公開 Streamlit サイドバーには含めません。

以前の独立した about ページは `home.py` に統合しました。

---

## ローカル環境診断ツール

インストール済み配布パッケージには、ローカル環境確認用の Streamlit 診断ツールを含めています。Python パスや依存パッケージを確認するためのローカル用ツールです。診断結果は CSV または PDF レポートとして保存できます。

```bash
envgeo-seawater-check
```

source checkoutでは、同じ診断を`streamlit run tools/env_check_streamlit.py`で起動できます。開発用の`pages/99_Environment_Check.py`ラッパーは、インストール済みpackageや公開deploymentには含めないため、通常のアプリナビゲーションには公開可視化ページだけが表示されます。

---

## ユーザーデータの利用

可視化ページのブラウザアップロードから、CSV/XLSX の測定データを現在の
Streamlit セッションへ読み込みます。`User Data Check & Quick Visualizer` は、
品質確認と簡易2D--4D可視化のためのアップロード起点ページです。対応する個別
ページでも、共通の Data filtering 内に `Uploaded data` が表示されます。

リポジトリには、常時読み込みの `User Excel data` 運用を確認するためのゼロ値の
公開サンプル `local_data/user_data.xlsx` を同梱します。各行には
`User Excel data` というデータセット名を付け、選択した各参照データに結合するため、
ブラウザで毎回アップロードしなくても、アプリ起動時からData filteringで選択できます。
研究者自身のデータは、ローカル利用のために `local_data/user_data.xlsx` を編集するか、
`ENVGEO_LOCAL_USER_DATA_PATH` でCSV、XLSX、XLSを指定して利用できます。commitや
公開用コピーの同期前にはゼロ値の公開サンプルへ戻し、研究データをcommitしないでください。
どちらの表を変更した場合も、Streamlitを再起動するかデータキャッシュをクリアして
再読み込みしてください。

常時読み込みの `User Excel data` と、ブラウザの `Uploaded data` は別の
データセットです。両方を選択した場合は、参照データとともに結合して利用します。

ブラウザアップロードは、現在の Streamlit セッション中のメモリ上でのみ扱います。アプリは、アップロードや参照データと一時結合したデータを保存しません。

Webページ、PDF、別のExcel表などから数値をコピー＆ペーストすると、見た目では
判別できないUnicode空白が混入することがあります。EnvGeoは数値変換前に通常空白、
ノーブレークスペース、狭いノーブレークスペース、全角空白を除去し、Unicodeマイナスを
通常のマイナスへ正規化します。それでも数値化できない値は推測せず欠損として扱います。
この処理は、常時読み込み表とブラウザアップロードの両方に適用されます。

---

## このツールの意義

EnvGeo-Seawater は、同位体データと水文データを統合的に探索できる点に特徴があります。

従来は、データセット管理、地図表示、T-S図、深度プロファイル、同位体-塩分関係などが別々に扱われることが多くありました。EnvGeo-Seawater は、それらをひとつのインタラクティブな環境で扱い、ユーザーデータとの直接比較も可能にします。

- 複数パラメータの統合可視化
- データセット間で共通したフィルタリング
- ユーザーデータと公開・整理済みデータの直接比較
- EnvGeo Dataset と全球データをつなぐ統一的な解析環境

---

## データの特徴

このアプリには、日本周辺を対象とする EnvGeo Dataset と、全球規模のデータが含まれています。EnvGeo Dataset の一部は、著者が統一した手法・基準で分析したものであり、航海や観測期間をまたいだ比較に適しています。

EnvGeo Dataset の主な特徴は次の通りです。

- 統一された分析手法
- 整理されたデータ構造
- 観測航海・調査間での高い比較可能性

これにより、観測された分布や関係が、分析手法の違いではなく環境シグナルを反映しているかを検討しやすくなります。

---

## データ公開方針

同梱データは公開出典の記録、またはプロジェクトが記録した学術利用方針に基づく派生workbookです。
これは第三者データをプロジェクト所有とするものでも、出典ごとの記録を超える一般的な再配布ライセンスを
主張するものでもありません。詳細は
[`docs/dataset_redistribution_audit_Japanese.md`](https://github.com/envgeo/envgeo-seawater/blob/main/docs/dataset_redistribution_audit_Japanese.md)、
[`docs/provenance_inventory_Japanese.md`](https://github.com/envgeo/envgeo-seawater/blob/main/docs/provenance_inventory_Japanese.md)、
[`docs/THIRD_PARTY_NOTICES_Japanese.md`](https://github.com/envgeo/envgeo-seawater/blob/main/docs/THIRD_PARTY_NOTICES_Japanese.md)を参照してください。

- アプリで直接利用できる標準化済み形式で提供
- 未公表データや制限付きデータは含めない方針

正式な公開・リリース前には、各データセットのライセンス、再配布条件、推奨引用を再確認する必要があります。

### データセット間の重複候補の確認

NASA GISSとPAGES CoralHydro2kを含む統合データセットでは、同じ原観測に由来する記録が含まれる可能性があります。**Data Overlap Check** では、同梱参照データセット間、またはセッション内のアップロードデータと選択した同梱参照データセットとの重複候補を確認・フィルタリングできます。候補は **Strong** と **Review** に分けて表示されますが、重複を確定するものではなく、元のworkbookは変更されません。

判定基準、候補区分、監査表、フィルタリングの詳しい使い方は、[Data Overlap Check マニュアル](docs/manual_Japanese/06_data_overlap_check.md)を参照してください。

---

## インストールと必要環境

最終wheelは、継続的インテグレーションで **Python 3.10 と 3.12** により検証しています。リリース基準は **Python 3.10.15 / Streamlit 1.42** と **Python 3.12.14 / Streamlit 1.63**、Plotly 5.24です。ローカル導入には **Python 3.12を推奨**します。Python 3.13 は、どのプラットフォームでも依存関係のネイティブビルドが必要になることがあるため、条件付きサポートです。試験記録と環境構成は `docs/streamlit_migration_Japanese.md` を参照してください。

### 公開パッケージの導入

公開パッケージは、ソースリポジトリをcloneせずに次のように導入・起動できます。

```bash
python -m pip install envgeo-seawater
envgeo-seawater
```

### Python 3.13とmacOSの導入

まず上記の標準pipコマンドを試してください。**Python 3.12を推奨**します。通常、pipはコンパイラを必要としない完成済みのパッケージファイル（wheel）を導入します。しかしPython 3.13では、固定しているNumPyまたはPyProjに対応するwheelが常に利用できるとは限りません。その場合、pipはソースからのビルドを試みます。これは数分かかることがあり、ネイティブのビルド環境を必要とします。そのため、Python 3.13はすべてのPCで標準の`pip install`だけによる導入を保証するものではありません。この条件はM1 Mac固有でも、Cartopy固有でもありません。

macOSでPython 3.13を意図して使用する場合は、先にAppleのCommand Line Toolsを導入してから標準コマンドを再試行してください。

```bash
xcode-select --install
python -m pip install --upgrade pip
python -m pip install envgeo-seawater
```

`pyproj`または`cartopy`の導入時に失敗する場合だけ、次のConda fallbackを使ってください。この経路ではPython 3.12を推奨します。

```bash
# 1. fallback用の環境を作成して有効化
conda create -n envgeo python=3.12
conda activate envgeo

# 2. geospatial 系ライブラリを conda-forge から入れる
conda install -c conda-forge proj pyproj=3.6.1 cartopy=0.25.0 -y

# 3. EnvGeo-Seawaterを導入して起動する
python -m pip install envgeo-seawater
envgeo-seawater
```

Python 3.14は、v1.3.4で固定している依存関係では未対応です。

---

## ソースチェックアウト（開発用）

```bash
git clone https://github.com/envgeo/seawater_map.git
cd seawater_map
python -m pip install -r requirements.txt
streamlit run home.py
```

この経路は、ソースの開発、確認、試験用です。ターミナルに表示されるローカルURLをブラウザで開きます。通常は次のURLです。

```text
http://localhost:8501
```

---

## オフライン利用と調査船上でのデータ確認

オフライン運用はEnvGeo-Seawaterの重要な設計目標です。代表的な利用例は、衛星回線が不安定または利用できない調査船上で、採水・測定直後の海水データを確認することです。異常値、座標の誤り、欠測、想定外の深度プロファイルを航海中に発見できれば、再測定、追加採水、残りの観測計画の調整を現場で判断できます。

Python環境を事前に導入すれば、同梱データと地図以外の主要な解析の多くはローカルで実行できます。Plotly地図ページにはローカル海岸線レイヤーがあり、**Coastline (offline)** を選ぶとタイルを使わない白地図になります。オンライン地図タイルが利用できない場合も、明示的な警告とともにこのローカル地図へ自動縮退します。オンラインマニュアル動画は補助情報のため、船上で再生できなくても中核ワークフローには影響しません。Foliumの地図上A–B線描画には引き続きオンラインのブラウザ資産が必要ですが、Vertical SectionはA/B座標の手入力によりオフラインでも利用できます。

page 32（Isotope Hydrographic Mapping）の静的 Cartopy 地図は、リポジトリ内の `coastline/natural_earth_50m_land/` に同梱した Natural Earth 50m 陸地ポリゴン（シェープファイル）を使用して陸地をマスクします。これらの地図を描画するために外部から Natural Earth データをダウンロードする必要はありません。

page 03（2D+ Visualizer）・page 04（3D/4D Visualizer）・page 05（Quick Visualizer）では、各メイン図の下に **「Download interactive HTML」** ボタンが表示されます。ダウンロードしたファイルには Plotly.js がインライン埋め込みされており、ネットワーク接続なしでPlotly図を開いて操作できます。ただし、Standardなどオンライン背景を選んだ地図のタイル画像までは埋め込まれません。地理背景も含めてオフライン利用する地図は、保存前に **Coastline (offline)** を選んでください。

---

## Python API 利用例

一部の機能は `envgeo_utils.py` から Python コードとして利用できます。

```python
import envgeo_utils

df = envgeo_utils.load_isotope_data(envgeo_utils.data_source_GLOBAL)

df_filtered = envgeo_utils.sidebar_filter_and_display(
    df,
    ref_data=envgeo_utils.data_source_GLOBAL,
    data_source_ENVGEO=envgeo_utils.data_source_ENVGEO,
    data_source_AROUND_JAPAN=envgeo_utils.data_source_AROUND_JAPAN
)
```

今後は、UIに依存しないデータ読み込み・検証・変換処理を、より明確なAPIとして整理していく予定です。

---

## テスト

テストはプロジェクトのルートディレクトリで実行します。開発用依存関係を導入した後、全テストを実行してください。

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q test
```

開発中の簡易確認には、共通ユーティリティとリポジトリ健全性テストを実行できます。

```bash
python -m pytest -q test/test_envgeo_utils.py test/test_repository_health.py
```

テストでは、データ読み込み、数値変換と品質情報、共通フィルタ、ユーザーデータ処理、深度プロファイル用の gap row、地図・作図補助、公開ページ構成、文書リンクなどを確認します。詳細な対象と限界は [`docs/testing_Japanese.md`](docs/testing_Japanese.md) を参照してください。

GitHub Actions の CI は push、pull request、手動起動時に Linux 上の Python 3.10 と 3.12 で全テストを実行します。さらに wheel を作成し、checkout 外の新しい仮想環境へ導入して、配布物の import、同梱ページ、診断ツール、開発専用ページが wheel に含まれないことを確認します。

ブラウザ操作、図の見た目、オンライン地図、全ページの科学的な解釈、性能は完全には自動テスト化していません。リリース前には Streamlit Cloud とローカル環境での手動 smoke test も行います。

現在のテスト群の内容と限界については、`docs/testing_Japanese.md` にまとめています。

---

## 追加ドキュメント

公開前チェックリストや開発メモなど、READMEより詳しい補助ドキュメントは `docs/` にまとめます。

- [図付きオンライン利用ガイド](https://envgeo.github.io/seawater_map/): 安定版向けの英日ページ別操作手順を公開しています。

- `docs/THIRD_PARTY_NOTICES_Japanese.md`: 同梱する第三者データ・地理空間資産の出典、帰属表示、適用範囲をまとめた簡潔な通知。

- `docs/release_checklist.md`  
  ローカル確認、Streamlit公開、GitHubリリース、Zenodoアーカイブ前の確認リスト。

- `docs/testing_Japanese.md`  
  pytest 群の内容、現在のテスト範囲、限界、今後の拡充予定の説明。

- `docs/manual_Japanese/`  
  オンライン利用ガイドで公開する、日本語ページ別マニュアルのMarkdown原稿。

---

## ディレクトリ構成

- `home.py`
  EnvGeo-Seawater の Streamlit メインページ。

- `envgeo_utils.py`
  データ読み込み、データクリーニング、フィルタリング、Plotly共通レイアウト、地図スタイル、海岸線読み込み、表表示などを含む共通ユーティリティ。

- `envgeo_assets.py`
  起動ディレクトリに依存せず、アプリに同梱する読み取り専用資産へのパスを解決する。

- `envgeo_user_data.py`
  ブラウザ内だけで扱うアップロード、列の標準化、ユーザー提供データの表示補助を担う。ブラウザからのアップロードをこのモジュールがディスクへ保存することはない。

- `envgeo_launcher.py`
  インストール済みアプリを `envgeo-seawater` コマンドで起動する。

- `envgeo_diagnostic_launcher.py`
  `envgeo-seawater-check` コマンドでローカル診断ツールを起動する。公開アプリのナビゲーションとは分離されている。

- `pages/`  
  アプリのサイドバーに表示される安定版の可視化ページ。

- `tools/`  
  配布パッケージには含めるが、公開 Streamlit アプリのサイドバーには表示しないローカル補助ツール。

- `dataset/`  
  アプリで使用する海水同位体・水文データセット。

- `data/`  
  サンプルファイル、テンプレート、GIF/MP4、海岸線サンプルなど、アプリ表示に使う補助データ。

- `data_text/`  
  About、文献、マニュアル、日本語説明、英語・日本語の更新履歴などの Markdown テキスト。

- `images/`  
  READMEやドキュメントで使う図・出力例。

- `coastline/`  
  地図や3D表示で使う50m・110m海岸線座標CSVファイル。

- `test/`  
  基本的な pytest テスト。

---

## 使い方

1. データセットを選択する（EnvGeo / EnvGeo + Around Japan / EnvGeo + Global）
2. 位置、水深、時期、パラメータなどでフィルタする
3. 図を確認する
   - 地図
   - T-S図
   - 深度プロファイル
   - 回帰図
   - 3D/4D表示
4. 必要に応じてユーザーデータを比較する
5. 図やデータを出力する

---

## 再現性

このリポジトリには、次のものが含まれています。

- 可視化プラットフォームのソースコード
- アプリで使用する公開・整理済みデータセット
- ユーザーデータ比較用テンプレート

そのため、ローカル環境でアプリを実行すれば、提供されたコード、データセット、設定に基づいて主要な可視化を再現できます。

ただし、アプリはインタラクティブな可視化を中心としており、beta ワークフローは開発版ごとに変わる可能性があります。そのため、現時点では自動テストに加えて、主要ページの手動確認と視覚確認も重要です。

---

## 制限事項

- 全球データを大きく選択した場合、3D/4D表示は重くなることがあります。
- Plotly の3D操作はPCでの利用に向いています。スマートフォンやタブレットでは2D表示が適しています。
- ユーザーデータアップロードは現在 Excel 中心で、想定された列構造に依存しています。
- 出力結果の解釈には、元データの出典、分析手法、メタデータの確認が必要です。出版物で利用する場合は、EnvGeo-Seawater と元データ提供者の両方を適切に引用してください。

---

## データソースと引用

このアプリは、主に次の海水同位体データセットを統合しています。

- CoralHydro2k: 全球海水酸素同位体データベース（Atwood et al., 2026, ESSD）
- NASA GISS Global Seawater Oxygen-18 Database（Schmidt et al., 1999）
- Kodama et al. (2024), *Geochemical Journal*
- その他の地域データセット

論文、教材、発表資料、再配布される図表などで利用する場合は、EnvGeo-Seawater だけでなく、選択した可視化に含まれる元データ提供者も引用してください。

詳細な出典情報は、アプリ内および `data_text/` 以下の Markdown ファイルに記載しています。

## ソフトウェアと地理空間データの謝辞

EnvGeo-Seawater は、アプリケーション画面に Streamlit、対話的な図に Plotly を使用しています。
選択可能な科学カラーパレットには cmocean を用いています。T-S 図の σ0 参照等値線は、
Gibbs SeaWater（GSW）による TEOS-10 実装を用いた明示的な近似です。観測ごとの絶対塩分・
保存温度への変換は行っていません。Vertical Section Visualizer では、海底地形の文脈表示に限り、
プロジェクトで軽量化した GEBCO 2025 Grid を使用できます。これは航海・安全目的の製品では
ありません。Natural Earth は同梱する陸地資産に使用しており、帰属表示は上記に示しています。
完全な文献情報と来歴記録は `paper.bib` および `docs/` に記載しています。

## AI支援開発と人間による監督

バージョン1.3以降、EnvGeo-Seawaterの開発では、コードレビュー、実装草案の
作成、リファクタリング、テストの設計・作成、バグ調査、文書整備において、
AIコーディング支援ツール（OpenAI Codex、Anthropic Claude Code）を本格的に
活用しています。

AIツールはあくまで支援ツールであり、著者・共同開発者としては扱っていません。
採用されたすべての変更は、マージ前に人間の著者がレビュー・編集・検証を行い
ます。科学的・技術的判断、ソフトウェアおよび文書の正確性、ライセンス、
本プロジェクトで公開する内容についての責任は、すべて著者が負います。この
方針の詳細は `docs/development_notes_Japanese.md` を参照してください。

## Live Demo

Primary stable demo:
https://envgeo-seawater-map.streamlit.app

Stable demo with experimental updates:
https://envgeo-seawater-pre.streamlit.app

---

## これまでの研究ワークフローでの利用

EnvGeo-SeawaterがソフトウェアとしてのアーカイブDOIを取得する前から、著者および共同研究者の
ワークフローにおいて、EnvGeo Dataset [ECS–Japan Sea]（主要出典：Kodama et al. (2024)）の一部を選択・探索・
可視化するために利用されてきました。これらの研究成果では本ソフトウェアではなく元データセットの
論文が引用されています。したがって、これらは直接のソフトウェア引用ではなく、研究ワークフローでの
利用例です。

---

## 今後の発展

データモデルは、出典、来歴、再配布上の位置づけが記録された後に、追加データセットを統合できるように
設計しています。再利用可能な可視化、資産パス解決、配布の構成要素は、将来の関連EnvGeoアプリケーションを
支えることも想定しています。これらは将来の方向性であり、安定版v1.3.4に含まれる機能や
データセットではありません。

---

## 引用

Ishimura, T. (2026).

*EnvGeo-Seawater: An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data* (Version 1.3.4). Zenodo.
https://doi.org/10.5281/zenodo.23117784

上部の DOI バッジは全版共通の concept DOI（`10.5281/zenodo.23117783`）を示します。v1.3.4を
利用した成果では、上記の version DOI を引用してください。解析で利用した各元データ提供者も併せて引用してください。

---

## 図の例

EnvGeo-Seawater では、全球スケールの分布から詳細な対話的解析まで、海水同位体・水文データを多面的に探索できます。

### 全球 δ18O 分布

統合データセットに基づく全球スケールの海水 δ18O 分布です。コンター補間により、海盆規模の大きな分布パターンを確認できます。

![Global map](https://raw.githubusercontent.com/envgeo/envgeo-seawater/main/images/contour_map.png)

---

### Temperature-Salinity 図

近似的な σ0 参照等値線（実用塩分 ≈ 絶対塩分；現場水温 ≈ 保存温度）を重ねた T-S 図です。水塊の識別や、同位体と水文構造の関係を調べるために利用できます。

![TS diagram](https://raw.githubusercontent.com/envgeo/envgeo-seawater/main/images/ts_diagram.png)

---

### 4D 可視化

経度、緯度、水深、δ18O などの変数を組み合わせた多次元可視化です。空間勾配と鉛直構造を同時に探索できます。

![4D](https://raw.githubusercontent.com/envgeo/envgeo-seawater/main/images/4d_d18O.png)

---

### インタラクティブ選択

T-S 空間と地理的位置を連動させた可視化です。T-S図で選択したデータ群に対応する採水地点を地図上で確認できます。

![](https://raw.githubusercontent.com/envgeo/envgeo-seawater/main/images/selection_map.png)
![Highlight the corresponding sampling locations on the map.](https://raw.githubusercontent.com/envgeo/envgeo-seawater/main/images/selection_ts.png)

---

## ライセンス

MIT License
