# 地理空間アセット — 用途・配置・ライセンス・更新方針

このドキュメントでは EnvGeo-Seawater リポジトリに同梱されている地理空間・科学アセットを説明します：
海岸線 CSV ファイル、Natural Earth 50m 陸地ポリゴン シェープファイル、GEBCO 測深グリッドです。

**状態:** 現行同梱ファイルとpackage-data範囲を2026-09-30に再確認した。英語版は
[`geospatial_assets.md`](geospatial_assets.md)。

---

## 1. 海岸線 CSV ファイル

| 項目 | 内容 |
|---|---|
| 配置場所 | `coastline/`（リポジトリルート） |
| ファイル | `world_coastline_coordinates_50m.csv`、`world_coastline_coordinates_110m.csv` |
| 用途 | すべてのインタラクティブ Plotly/Mapbox マップおよびページ 32 の Cartopy 静的マップにおけるオフライン海岸線オーバーレイ |
| 形式 | `Longitude`・`Latitude` 列を持つCSV。存在する場合、欠損座標行でセグメントを区切る。 |
| 読込み関数 | `envgeo_utils.load_coastline_data()` |
| 描画関数 | `envgeo_utils.add_coastline_overlay()`（Plotly Scattermapbox）および `envgeo_utils.plot_bundled_coastline()`（Matplotlib/Cartopy） |

### ライセンス

海岸線データは Natural Earth のパブリックドメインソースから生成されています。
陸域シェープファイルの詳細な来歴・チェックサム記録は、当該資産とともに保存している。
保持している海岸線source workspaceでは、50m・110mの両coastline shapefileをNatural Earth v4.1.0と
確認でき、source archiveと対応する座標workbookは2025-01-24に保存されている。現行CSVは、その
source fileを用いて同workspaceで作成したものである。
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
| 配置場所 | `bathymetry/` |
| ファイル | `GEBCO_2025_6min.nc`（派生NetCDF3格子、約13 MB） |
| 用途 | 鉛直断面ビジュアライザー（`pages/53_Vertical_Section_Visualizer.py`）における深度補間と海底推定 |
| 読込み | 鉛直断面ページ内の `scipy.io.netcdf_file()` |
| スコープ | 断面解析専用；マップの陸地マスクとは無関係 |

### ライセンス

GEBCO Gridはパブリックドメインであり、出典の帰属、GEBCO/IHO/IOCの公式承認を示唆しないこと、
および免責条件の下で、複製・改変・配布・商用利用ができる。航海または海上安全に関わる目的には
使用してはならない。出版物で成果を使用する場合はGEBCOを引用してください：

> GEBCO Compilation Group (2025) GEBCO 2025 Grid,
> doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29.

現在の利用条件は<https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use>を参照してください。

`GEBCO_2025_6min.nc`は公式GEBCO_2025 NetCDF格子を6 arc-minuteへ間引いたプロジェクト派生物である。
`bathymetry/make_lightweight_gebco.py`に入力、変換、stride処理を記録している。派生ファイルを再生成する場合は、
このスクリプトと来歴記録を維持・更新する。

### 更新方針

新しい GEBCO 年次リリースが公開され、鉛直断面ワークフローに改善された測深データが必要な場合に
`GEBCO_2025_6min.nc` を置き換えてください。
ドキュメントの引用とライセンス記述を相応に更新してください。
ページ 53 の GEBCO 読込みコードは、鉛直断面出力を再テストせずに変更しないでください。

---

## 4. 現在のpackage内ディレクトリ構成

```
coastline/
    world_coastline_coordinates_50m.csv   # Plotly/Mapbox オフライン海岸線オーバーレイ
    world_coastline_coordinates_110m.csv  # 低解像度版
    natural_earth_50m_land/          # 静的 Cartopy マップ用陸地マスク（ページ 32）
        ne_50m_land.shp
        ne_50m_land.shx
        ne_50m_land.dbf
        ne_50m_land.prj
        ne_50m_land.cpg
        LICENSE_OR_SOURCE.md
        LICENSE_OR_SOURCE_Japanese.md

bathymetry/
    GEBCO_2025_6min.nc               # 鉛直断面用 GEBCO 測深データ（ページ 53）
```

上記の項目はすべて`pyproject.toml`の明示的な`[tool.setuptools.package-data]` allowlistにより、
現在のwheelへ収録される。GEBCO生成スクリプトは意図的に除外する。**現バージョンではこれらの
ディレクトリを再編成しないでください。**

---

## 5. 将来のアセットローダー改善（現時点では実施しない）

本プロジェクトはすでにインストール可能なPython packageである。`pyproject.toml`は同梱アセットを
package dataとして列挙し、`envgeo_assets.asset_path()`はインストール済みpackage moduleを基準に
パスを解決する。より整理したアセット配置は将来の候補にとどめ、後方互換性のあるローダーと
インストール済みwheelのテストを設計するまでパスを変更してはならない。

```
assets/
    geospatial/    ← 海岸線 CSV + Natural Earth 陸地シェープファイル
    bathymetry/    ← GEBCO グリッド
    metadata/      ← LICENSE_OR_SOURCE ファイル、チェックサム、出典記録
```

この再編成の前提条件：

- `envgeo_assets.asset_path()`による現在のインストール済みpackage基準の解決（または同等にテストしたresource API）を維持し、個々のページスクリプトやCWDからの相対パスへ戻さないこと。
- パッケージインストールなしの既存ローカルチェックアウトでも引き続き動作する後方互換フォールバックを設けること。
- 再編成と同時に `LICENSE_OR_SOURCE.md` / `LICENSE_OR_SOURCE_Japanese.md` を `assets/metadata/` に移動すること。
- 移行後、Streamlit Cloud とローカル Conda 環境の両方でバンドルファイルが正しく参照されることを確認すること。
- Natural Earth 陸地（静的マップ陸地マスク）と GEBCO（測深・断面解析）は再編成後も別サブディレクトリに分離して保持すること。

## 6. 現行アセットの再確認（2026-09-30）

正規作業フォルダとstable `seawater_map` cloneについて、2つの海岸線CSV、Natural Earthの5つの
シェープファイル構成要素、`GEBCO_2025_6min.nc`、`make_lightweight_gebco.py`がbyte単位で一致することを
確認した。現行snapshotの識別情報は以下のとおりである。

| アセット | 現行確認 |
|---|---|
| 50m海岸線CSV | データ行数61,844行；SHA-256 `c3d7bee4fb696b011fa34bb13bed0c335c5250eeaf37d8739d77d29a27fe385c` |
| 110m海岸線CSV | データ行数5,261行；SHA-256 `a31df3aeee9dc4195af35a31b0605fdb572c7c7dd7cde17f773c9438f5ec7f3f` |
| Natural Earth陸地 | 構成要素のchecksumは`LICENSE_OR_SOURCE.md`と一致する。同文書には1,420ポリゴンfeatureを記録している。 |
| GEBCO派生格子 | NetCDF変数は`lon`、`lat`、`Height`；次元は3,600 × 1,800；SHA-256 `0afdf1d0e023b0529c56b69a2684e505c7e2ea28d78a3af8814817af59b09030` |

保持している2025-01-24作成の両解像度の座標workbookは、浮動小数表現による丸め誤差
（最大絶対差約1.4 × 10⁻¹⁴）を除いて現行CSVの全座標と一致する。行数と欠損座標による
セグメント区切りも同じである。元source archiveはNatural Earth v4.1.0と示す。将来CSVを更新する前には、
プロジェクト管理者は、現行CSVが同source workspace内のsource fileから作成されたことを確認している。
将来CSVを更新する前には、置換ファイルとともに新しいsource version、取得日、変換手順、checksumを記録する。

---

*最終更新：2026-09-30*
