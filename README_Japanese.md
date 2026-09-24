# EnvGeo-Seawater

EnvGeo-Seawater は、海水の安定同位体・水文データを探索するためのインタラクティブ可視化プラットフォームです。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://envgeo.h.kyoto-u.ac.jp/sw_jpn/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**現在の開発バージョン:** 1.3.3（2026-09-23）

**海水同位体・水文データを、地図・断面・T-S図・3D/4D表示で探索する研究用Webアプリです。**

---

## 概要

EnvGeo-Seawater は、海洋地球化学研究における海水同位体データの探索的解析に使われてきた Streamlit ベースのWebアプリです。

対象とするデータは、安定水同位体（δ18O、δD）、塩分、水温、水深などを含む海洋地球化学・水文データです。

日本周辺の地域データセットと、約5万件規模の全球データセットを統合し、共通した条件で比較・可視化できるようにしています。

このアプリは、海洋地球化学・海洋学における **探索的データ解析** と **再現可能な研究ワークフロー** の両方を支援することを目的としています。

---

## 主な機能

- ズーム範囲を調整できるインタラクティブ地図表示
- 欠損や観測グループの切れ目を考慮した深度プロファイル
- 近似的な σ0 参照格子付きの Temperature-Salinity（T-S）図
- 塩分-δ18O 関係などの回帰解析
- 空間・深度・時間構造を探索する 3D / 4D 可視化
- 参照データセットと比較するためのユーザーデータアップロード機能
- フィルタ後データや除外サンプル数の表示
- 論文・発表用図の出力

---

## 主要ページ

Streamlit アプリでは、`home.py` が About、データソース、マニュアル、更新履歴、日本語説明のタブを担当します。`pages/` ディレクトリには、主要な可視化ツールに加えて、一部の beta ページやローカル開発用ページも含まれます。

- `pages/03_[Interactive]_2Dplus_Visualizer.py`
  同位体・水文データの関係と観測地点を確認する 2D/2.5D 可視化ページ。

- `pages/04_[Interactive]_3D_4D_Visualizer.py`
  経度、緯度、水深、選択変数を扱う 3D/4D 可視化ページ。

- `pages/05_User_Data_Check_Quick_Visualizer.py`
  参照・CSV/XLSXアップロードデータを扱う User Data Check & Quick Visualizer。共通フィルタ、欠損・品質確認、2D Map、Salinity-d18O、Temperature-Salinity、任意2D/3D/4D、地理3D、フィルタ済みCSV出力を一つの入口へまとめます。

- `pages/31_Salinity-d18O_Relationship.py`  
  塩分-δ18O 関係を表示し、必要に応じて回帰線を加えるページ。

- `pages/32_Isotope_Hydrographic_Mapping.py`  
  d18O、dD、d-excess、塩分、水温などの同位体・海洋環境パラメーターを2D地図やコンター風の図として表示するページ。

- `pages/34_T-S_diagram.py`  
  密度等値線付きの T-S 図を表示するページ。

- `pages/37_Depth_Profile.py`  
  δ18O、δD、d-excess、水温、塩分の深度プロファイルを表示するページ。

- `pages/35_Custom_Parameter_Plot_beta.py`  
  X軸、Y軸、色、マーカーサイズを任意の数値パラメーターから選ぶ試験的な2Dプロットページ。

- `pages/80_Correlation_Overview.py`  
  手書きで開発してきた元の探索ワークフローを保存するアーカイブ表示ページです。開発記録として残し、新機能は追加しません。

- `pages/53_Vertical_Section_Visualizer.py`  
  Vertical Section Visualizer beta。測線選択、補間、海底地形、鉛直断面図の表示方法を調整するための試験版ページです。

- `pages/90_Integrated_Visualizer_beta.py`  
  試作統合ページ。既存の可視化ワークフローを1ページ内から選択実行できる互換モード、既存ページへのユーザーデータ一時結合、海域プリセット付きの共通フィルタ・タブ切り替えモードを含みます。独立アップロードページはアップロード起点の別ワークフローのため、統合ページ内の選択肢からは外しています。

- `pages/99_Environment_Check.py`  
  ローカル開発用の環境診断ラッパーページ。ローカル環境確認には有用ですが、公開 Streamlit サイドバーに表示するページではありません。

以前の独立した about ページは `home.py` に統合しました。

---

## ローカル環境診断ツール

配布パッケージには、ローカル環境確認用の Streamlit 診断ツールを含めています。Python パスや依存パッケージを確認するためのローカル用ツールです。診断結果は CSV または PDF レポートとして保存できます。

```bash
streamlit run tools/env_check_streamlit.py
```

ローカル開発中は、`pages/99_Environment_Check.py` から同じ診断機能をサイドバーに表示できます。公開アプリで可視化ページだけを表示したい場合は、このページを除外または非表示にする方針です。

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
- 地域データと全球データをつなぐ統一的な解析環境

---

## データの特徴

このアプリには、日本周辺などの地域データと、全球規模のデータが含まれています。特に日本周辺の一部データは、著者が統一した手法・基準で分析したものであり、航海や観測期間をまたいだ比較に適しています。

地域データセットの主な特徴は次の通りです。

- 統一された分析手法
- 整理されたデータ構造
- 観測航海・調査間での高い比較可能性

これにより、観測された分布や関係が、分析手法の違いではなく環境シグナルを反映しているかを検討しやすくなります。

---

## データ公開方針

このリポジトリに含まれるデータは、公開データ、または各データ提供元の条件に沿って再配布可能なデータです。

- アプリで直接利用できる標準化済み形式で提供
- 未公表データや制限付きデータは含めない方針

正式な公開・リリース前には、各データセットのライセンス、再配布条件、推奨引用を再確認する必要があります。

---

## インストールと必要環境

現在は **Python 3.10.15 / Streamlit 1.42** と **Python 3.12.14 / Streamlit 1.63** の両環境で互換性を確認し、Plotly 5.24をリリース基準として維持しています。検証環境の組合せと残りの対話操作確認は `docs/streamlit_migration_Japanese.md` を参照してください。

### macOS Apple Silicon ユーザー向けメモ

`pyproj` や `cartopy` などのビルドエラーを避けるため、macOS Apple Silicon 環境では、先に Conda で地理空間系ライブラリを入れる方法を推奨します。

```bash
# 1. 環境を作成して有効化
conda create -n envgeo python=3.10
conda activate envgeo

# 2. geospatial 系ライブラリを conda-forge から入れる
conda install -c conda-forge proj pyproj cartopy -y

# 3. 残りの依存関係を入れる
pip install -r requirements.txt
```

---

## クイックスタート

```bash
git clone https://github.com/envgeo/seawater_map.git
cd seawater_map
pip install -r requirements.txt
streamlit run home.py
```

ターミナルに表示されるローカルURLをブラウザで開きます。通常は次のURLです。

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

df = envgeo_utils.load_isotope_data("with [Global data sets]")

df_filtered = envgeo_utils.sidebar_filter_and_display(
    df,
    ref_data="with [Global data sets]",
    data_source_JAPAN_SEA="Kodama et al. (2024) [ECS - Japan Sea]",
    data_source_AROUND_JAPAN="with [Around Japan]"
)
```

今後は、UIに依存しないデータ読み込み・検証・変換処理を、より明確なAPIとして整理していく予定です。

---

## テスト

基本的な動作確認は pytest で実行できます。

```bash
pytest
```

現在のテストでは、import、データ読み込み、数値変換、gap row挿入、カラースケール、海岸線読み込みなどを確認しています。

現在のテスト群の内容と限界については、`docs/testing_Japanese.md` にまとめています。

---

## 追加ドキュメント

公開前チェックリストや開発メモなど、READMEより詳しい補助ドキュメントは `docs/` にまとめます。

- `docs/release_checklist.md`  
  ローカル確認、Streamlit公開、GitHubリリース、Zenodoアーカイブ前の確認リスト。

- `docs/testing_Japanese.md`  
  pytest 群の内容、現在のテスト範囲、限界、今後の拡充予定の説明。

- `docs/manual_Japanese/`  
  各ページの詳細マニュアルを今後整備するための日本語マニュアル骨組み。

---

## ディレクトリ構成

- `home.py`  
  EnvGeo-Seawater の Streamlit メインページ。

- `envgeo_utils.py`  
  データ読み込み、データクリーニング、フィルタリング、Plotly共通レイアウト、地図スタイル、海岸線読み込み、表表示などを含む共通ユーティリティ。

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

1. データセットを選択する（Japan Sea / Around Japan / Global）
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

## 引用

Ishimura, T. (2026).  
EnvGeo-Seawater: An Interactive Platform for Exploring Seawater Isotope and Hydrographic Data.  
正式なアーカイブ公開後に引用情報を更新します。

---

## 図の例

EnvGeo-Seawater では、全球スケールの分布から詳細な対話的解析まで、海水同位体・水文データを多面的に探索できます。

### 全球 δ18O 分布

統合データセットに基づく全球スケールの海水 δ18O 分布です。コンター補間により、海盆規模の大きな分布パターンを確認できます。

![Global map](images/contour_map.png)

---

### Temperature-Salinity 図

近似的な σ0 参照等値線（実用塩分 ≈ 絶対塩分；現場水温 ≈ 保存温度）を重ねた T-S 図です。水塊の識別や、同位体と水文構造の関係を調べるために利用できます。

![TS diagram](images/ts_diagram.png)

---

### 4D 可視化

経度、緯度、水深、δ18O などの変数を組み合わせた多次元可視化です。空間勾配と鉛直構造を同時に探索できます。

![4D](images/4d_d18O.png)

---

### インタラクティブ選択

T-S 空間と地理的位置を連動させた可視化です。T-S図で選択したデータ群に対応する採水地点を地図上で確認できます。

![](images/selection_map.png)
![Highlight the corresponding sampling locations on the map.](images/selection_ts.png)

---

## ライセンス

MIT License
