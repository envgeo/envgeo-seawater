# 地理空間アセット — 用途・配置・ライセンス・更新方針

このドキュメントでは EnvGeo-Seawater リポジトリに同梱されている地理空間・科学アセットを説明します：
海岸線 CSV ファイル、Natural Earth 50m 陸地ポリゴン シェープファイル、GEBCO 測深グリッドです。

---

## 1. 海岸線 CSV ファイル

| 項目 | 内容 |
|---|---|
| 配置場所 | `coastline/`（リポジトリルート） |
| ファイル | `coastline_50m.csv`、`coastline_110m.csv`（および関連ファイル） |
| 用途 | すべてのインタラクティブ Plotly/Mapbox マップおよびページ 32 の Cartopy 静的マップにおけるオフライン海岸線オーバーレイ |
| 形式 | `lon`・`lat` 列を持つ CSV；ポリゴンセグメントを `None` 行で区切る |
| 読込み関数 | `envgeo_utils.load_coastline_data()` |
| 描画関数 | `envgeo_utils.add_coastline_overlay()`（Plotly Scattermapbox）および `envgeo_utils.plot_bundled_coastline()`（Matplotlib/Cartopy） |

### ライセンス

海岸線データは Natural Earth のパブリックドメインソースから生成されています。
詳細は `coastline/` 内の `LICENSE_OR_SOURCE.md` 等を参照してください。
Natural Earth データはパブリックドメインであり、使用にライセンスは不要です。

### 更新方針

これらの CSV ファイルは Natural Earth ベクタデータから生成されており、リポジトリに直接保存されています。
Natural Earth が 50m/110m の新リリースを発行し、ジオメトリに意味のある変更があった場合に再生成してください。
再生成後は、関連する `LICENSE_OR_SOURCE.md` の SHA-256 チェックサム、ソースバージョン、取得日を更新してください。

他のプロバイダのデータに置き換える場合は、ライセンス記録も必ず更新してください。

---

## 2. Natural Earth 50m 陸地ポリゴン シェープファイル

| 項目 | 内容 |
|---|---|
| 配置場所 | `coastline/natural_earth_50m_land/` |
| ファイル | `ne_50m_land.shp`、`.shx`、`.dbf`、`.prj`、`.cpg` |
| 用途 | `pages/32_Isotope_Hydrographic_Mapping.py` の静的 Cartopy マップにおける陸地マスク |
| 読込み | ページ 32 内の `_load_ne50m_land_geometries()` が `cartopy.io.shapereader.Reader(local_path)` で読み込む |
| 使用方法 | `ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor="white", ...)` で陸地を白く塗り、コンターを海岸線でクリップ |
| バンドル日 | 2026-09-23；SHA-256 チェックサムと出典は `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md` を参照 |

### 設計方針

`cfeature.LAND` や `shapereader.natural_earth()` の代わりにローカルの `shapereader.Reader` を使用することで、
Cartopy が Natural Earth ダウンロードサーバーに接続するのを防ぎます。
これはオフライン・船上での使用に不可欠です。
海岸線の**ライン**オーバーレイは `envgeo_utils.plot_bundled_coastline()` で別途描画します（`coastline/` 内の CSV を参照）。
これにより、陸地マスクと海岸線アウトラインの両方が完全にオフラインで動作します。

### 縮退動作

必要なシェープファイルコンポーネント（`.shp`、`.shx`、`.dbf`）のいずれかが欠損している場合、
陸地マスクの描画はスキップされ、英語の `st.warning()` が表示されます。
海岸線アウトラインと観測点は引き続き描画されます。
ネットワークアクセスは行いません。

### ライセンス

Natural Earth データはパブリックドメインです。
詳細は `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md` および
`LICENSE_OR_SOURCE_Japanese.md` を参照してください。
推奨帰属表示：*Made with Natural Earth (https://www.naturalearthdata.com/)*

### 更新方針

Natural Earth 50m 陸地リリースにジオメトリの意味ある変更がある場合に更新してください。
ファイルを置き換えた後：

1. `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md` および
   `LICENSE_OR_SOURCE_Japanese.md` の SHA-256 チェックサムを再計算・記録する。
2. 新しい commit SHA、raw GitHub URL、取得日を記録する。
3. `pytest test/test_natural_earth_land.py` を実行し、シェープファイルが正常に読み込め、フィーチャー数が妥当であることを確認する。

---

## 3. GEBCO 測深グリッド

| 項目 | 内容 |
|---|---|
| 配置場所 | `data_beta/` |
| ファイル | `GEBCO_2025_6min.nc`（NetCDF、約 90 MB） |
| 用途 | 鉛直断面ビジュアライザー（`pages/53_Vertical_Section_Visualizer.py`）における深度補間と海底推定 |
| 読込み | 鉛直断面ページ内の `xarray.open_dataset()` |
| スコープ | 断面解析専用；マップの陸地マスクとは無関係 |

### ライセンス

GEBCO（General Bathymetric Chart of the Oceans）データは非商用帰属ライセンスで提供されます。
出版物で成果を使用する場合は GEBCO を引用してください：

> GEBCO Compilation Group (2025) GEBCO 2025 Grid.
> https://doi.org/10.5285/...

現在の DOI と完全なライセンス条件については GEBCO ウェブサイト（https://www.gebco.net/）を参照してください。

### 更新方針

新しい GEBCO 年次リリースが公開され、鉛直断面ワークフローに改善された測深データが必要な場合に
`GEBCO_2025_6min.nc` を置き換えてください。
ドキュメントの引用とライセンス記述を相応に更新してください。
ページ 53 の GEBCO 読込みコードは、鉛直断面出力を再テストせずに変更しないでください。

---

## 4. 現在のディレクトリ構成

```
coastline/
    coastline_50m.csv                # Plotly/Mapbox オフライン海岸線オーバーレイ
    coastline_110m.csv               # 低解像度版
    natural_earth_50m_land/          # 静的 Cartopy マップ用陸地マスク（ページ 32）
        ne_50m_land.shp
        ne_50m_land.shx
        ne_50m_land.dbf
        ne_50m_land.prj
        ne_50m_land.cpg
        LICENSE_OR_SOURCE.md
        LICENSE_OR_SOURCE_Japanese.md

data_beta/
    GEBCO_2025_6min.nc               # 鉛直断面用 GEBCO 測深データ（ページ 53）
```

**現バージョンではこれらのディレクトリを再編成しないでください。**
将来の予定レイアウトについてはセクション 5 を参照してください。

---

## 5. 将来のパッケージ化計画（現時点では実施しない）

プロジェクトが適切な Python パッケージ（`pyproject.toml` + `package_data`）に移行する際、
地理空間・科学アセットをより整理された構成に再編することが候補として挙げられています。
これはあくまで将来の候補レイアウトであり、後方互換性のあるアセットローダーを設計・テストするまで
パスを変更してはなりません。

```
assets/
    geospatial/    ← 海岸線 CSV + Natural Earth 陸地シェープファイル
    bathymetry/    ← GEBCO グリッド
    metadata/      ← LICENSE_OR_SOURCE ファイル、チェックサム、出典記録
```

この再編成の前提条件：

- アセットローダーは個々のページスクリプトの `__file__` ではなく、インストール済みパッケージからの相対パスで解決すること（例：`importlib.resources` を使用）。
- パッケージインストールなしの既存ローカルチェックアウトでも引き続き動作する後方互換フォールバックを設けること。
- 再編成と同時に `LICENSE_OR_SOURCE.md` / `LICENSE_OR_SOURCE_Japanese.md` を `assets/metadata/` に移動すること。
- 移行後、Streamlit Cloud とローカル Conda 環境の両方でバンドルファイルが正しく参照されることを確認すること。
- Natural Earth 陸地（静的マップ陸地マスク）と GEBCO（測深・断面解析）は再編成後も別サブディレクトリに分離して保持すること。

---

*最終更新：2026-09-23*
