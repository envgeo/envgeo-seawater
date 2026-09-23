# Natural Earth 50m 陸地ポリゴン — 出典とライセンス

## データ出典

| 項目 | 内容 |
|---|---|
| データセット | Natural Earth – 50m 物理ベクタ：陸地 (Land) |
| 解像度 | 1:5,000万 |
| ソースリポジトリ | https://github.com/nvkelso/natural-earth-vector |
| 取得時のブランチ | `master` |
| 取得時の Git commit SHA | *取得時に記録されていません — 下記注記を参照* |
| 公式配布ページ | https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-land/ |
| 取得日 | 2026-09-23 |

### Git commit SHA に関する注記

2026-09-23 にファイルを取得した時点の `nvkelso/natural-earth-vector` の
正確な commit SHA は取得時に記録されませんでした。
バンドルファイルの同一性の確認は以下の SHA-256 チェックサムを用いてください。

対応するコミットを特定するには、GitHub のコミット履歴と照合してください：

```
https://github.com/nvkelso/natural-earth-vector/commits/master/50m_physical/ne_50m_land.shp
```

または、Natural Earth が定期発行するバージョンタグのファイルとチェックサムを比較することでも確認できます。

## 取得に使用した raw GitHub URL

以下の URL は master ブランチの最新 HEAD を参照します。
**取得時の特定コミットを再現するには、URL 中のブランチ名を commit SHA に置き換えてください。**

| ファイル | raw URL（master HEAD — 完全再現には commit SHA を使用） |
|---|---|
| `ne_50m_land.shp` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.shp |
| `ne_50m_land.shx` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.shx |
| `ne_50m_land.dbf` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.dbf |
| `ne_50m_land.prj` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.prj |
| `ne_50m_land.cpg` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/50m_physical/ne_50m_land.cpg |

## バンドルファイルと SHA-256 チェックサム

以下のチェックサムがリポジトリにコミットされたファイルの唯一の同一性証明です。

| ファイル | サイズ（概算） | SHA-256 |
|---|---|---|
| `ne_50m_land.shp` | 1.1 MB | `140d1419f7ace66bf0ea8ceb068cab544d413170f4e8c48e0a11bde6c978b319` |
| `ne_50m_land.shx` | 12 KB | `c1b93ee40fe83d6322197251517c141c2c91970915ac61d7bf7a4821fc2a68d7` |
| `ne_50m_land.dbf` | 72 KB | `d082add78c82e01b9bf422df3143ad6fbe43ca1bcb798d12a334f9e001b67bc0` |
| `ne_50m_land.prj` | 4 KB | `98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb` |
| `ne_50m_land.cpg` | <1 KB | `3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824` |

フィーチャー総数：1,420 ポリゴン（Polygon 型、WGS 84 / PlateCarree）

バンドルファイルの整合性を確認するには：

```bash
sha256sum coastline/natural_earth_50m_land/ne_50m_land.*
```

## ライセンス

Natural Earth データは**パブリックドメイン**です。

> Natural Earth は 1:1,000万・1:5,000万・1:1億1,000万スケールで
> 提供される公共ドメインの地図データセットです。
> あらゆる種類のプロジェクトで自由に使用できます。
>
> — https://www.naturalearthdata.com/about/terms-of-use/

利用規約の要点：
- Natural Earth データの使用にライセンスは不要です。
- クレジット表示は推奨されますが、必須ではありません。

## 推奨帰属表示（クレジットを入れる場合）

> Made with Natural Earth (https://www.naturalearthdata.com/)

## このプロジェクトでの用途

これらのファイルは `pages/32_Isotope_Hydrographic_Mapping.py` における
**静的 Cartopy マップの陸地マスク描画**にのみ使用します。
実行時に `cartopy.io.shapereader.Reader(local_path)` で読み込むため、
オフライン・エアギャップ環境でも外部 Natural Earth ダウンロードを一切行いません。

海岸線アウトライン（陸地マスクの上に重ねる線）は `coastline/` 配下の
バンドル済み海岸線 CSV から `envgeo_utils.plot_bundled_coastline()` で別途描画します。

GEBCO 測深データ（`data_beta/GEBCO_2025_6min.nc`）は別ディレクトリに格納されており、
鉛直断面の深度補間・海底推定という異なる用途に使用するため、このアセットとは無関係です。
