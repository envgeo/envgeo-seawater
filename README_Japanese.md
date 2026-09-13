# EnvGeo-Seawater

EnvGeo-Seawater は、海水の安定同位体・水文データを探索するためのインタラクティブ可視化プラットフォームです。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://envgeo.h.kyoto-u.ac.jp/sw_jpn/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**現在のバージョン:** 1.3.0

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
- 密度等値線（σθ）付きの Temperature-Salinity（T-S）図
- 塩分-δ18O 関係などの回帰解析
- 空間・深度・時間構造を探索する 3D / 4D 可視化
- 参照データセットと比較するためのユーザーデータアップロード機能
- フィルタ後データや除外サンプル数の表示
- 論文・発表用図の出力

---

## 主要ページ

安定版アプリでは、`home.py` が About、データソース、マニュアル、更新履歴、日本語説明のタブを担当します。`pages/` ディレクトリは主要な可視化ツール用です。

- `pages/03_3D_Visualizer.py`  
  同位体・水文データの関係と観測地点を確認する 3D 可視化ページ。

- `pages/04_4D_Visualizer.py`  
  経度、緯度、水深、選択変数を扱う主要な 3D/4D 可視化ページ。

- `pages/05_3D4D_Visualizer_Uploader.py`  
  Excel形式のユーザーデータをアップロードし、参照データと比較するページ。

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

- `pages/53_Vertical_Section_Visualizer.py`  
  Vertical Section Visualizer beta。測線選択、補間、海底地形、鉛直断面図の表示方法を調整するための試験版ページです。

- `pages/90_Integrated_Visualizer_beta.py`  
  試作統合ページ。既存の可視化ワークフローを1ページ内から選択実行できる互換モード、既存ページへのユーザーデータ一時結合、海域プリセット付きの共通フィルタ・タブ切り替えモードを含みます。独立アップロードページはアップロード起点の別ワークフローのため、統合ページ内の選択肢からは外しています。

以前の独立した about ページは `home.py` に統合しました。

---

## ローカル環境診断ツール

配布パッケージには、ローカル環境確認用の Streamlit 診断ツールを含めています。Python パスや依存パッケージを確認するためのローカル用ツールです。診断結果は CSV または PDF レポートとして保存できます。

```bash
streamlit run tools/env_check_streamlit.py
```

---

## ユーザーデータの利用

`91_USER_UPLOAD_UNPUB.xlsx` は、ユーザー独自データを比較表示するためのテンプレートです。

- サンプル行を自分の測定値に置き換える
- 列構造はそのまま維持する
- ローカル環境でアプリを実行する

これにより、ユーザーデータを既存の参照データセットと同じ可視化ワークフローの中で比較できます。

アップロードファイルは、現在の Streamlit セッション中のメモリ上でのみ扱う方針です。統合 beta ワークフローでは、アップロードファイルや、参照データと一時結合したデータをローカル/サーバーへ保存しません。

現在のユーザーデータ機能は Excel テンプレート中心です。今後、CSV/XLSXを直接読み込み、必須列を検証する `load_user_data()` のような仕組みに整理していく予定です。

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

検証済み環境として、現在は **Python 3.10.15**（Anaconda `envgeo_streamlit142`）を想定しています。

一部の依存ライブラリ、特に geospatial 系ライブラリや scikit-learn のバージョンにより、Python バージョンとの相性に注意が必要です。Python 3.12系への更新は、依存関係と各ページの動作を再確認してから行う予定です。

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
  地図や3D表示で使う海岸線座標ファイル。

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

そのため、ローカル環境でアプリを実行すれば、READMEやアプリで示されている可視化を再現できます。

ただし、アプリはインタラクティブな可視化を中心としているため、現時点では自動テストに加えて、主要ページの手動確認と視覚確認も重要です。

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

密度等値線（σθ）を重ねた T-S 図です。水塊の識別や、同位体と水文構造の関係を調べるために利用できます。

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
