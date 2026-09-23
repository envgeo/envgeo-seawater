# オフライン動作ログ

## 概要

本ドキュメントは、**EnvGeo Seawater**（v1.3.3）における Plotly
インタラクティブ地図のオフライン / 低速ネットワーク対応に関する設計判断と
実装詳細を記録します。

---

## 背景

EnvGeo Seawater は、インタラクティブな 2-D / 3-D 地図に Plotly Mapbox を
使用しています。デフォルトでは Plotly Mapbox は描画時にサードパーティ CDN
サーバーからタイル画像を取得します。そのサーバーに接続できない環境（例：
学術ネットワーク外のエアギャップ環境・船上など）では地図背景が白くなり、
ユーザーが地理的文脈を失います。

ローカル海岸線オーバーレイ（Natural Earth 50 m 解像度、CSV としてバンドル）は、
ネットワーク状態によらず海岸線が常に表示されることを保証するために導入されました。

---

## Map mode 別タイル供給元

| Map mode | タイル供給元 | ネットワーク必要 |
|---|---|---|
| Coastline (offline) | なし（白背景） | 不要 |
| Standard | OpenStreetMap タイル | 必要 |
| Satellite | USGS National Map | 必要 |
| Bathymetry (Sea) | Esri World Ocean Base | 必要 |
| Contour (GSI) | 国土地理院（GSI） | 必要 |

**注記：** すべてのオンラインタイルは各プロバイダーの公開ウェブ利用ライセンスのもとで
使用しています。v1.3.3 時点では API キーは不要です。
以前使用していた CARTO ベースマップは API キーが必要になったため削除しました。

---

## 海岸線オーバーレイ — 常時表示

スプリント #48（2026-09）以降、`apply_map_style()` は返却前に
**すべての** map mode に対して `add_coastline_overlay()` を呼び出します。
これにより：

* **Coastline (offline):** 白背景 ＋ ローカル海岸線 ＋ データ点
* **オンラインモード（Standard、Satellite 等）:** タイル背景 ＋ ローカル海岸線 ＋ データ点
  　ブラウザのタイル取得が失敗した場合でも、ローカル海岸線によって位置関係が保たれます。

オーバーレイは冪等（idempotent）です。`_coastline_overlay` トレースがすでに
存在する場合、2 回目以降の `add_coastline_overlay()` 呼び出しは無操作になります。

---

## オフライン検出とユーザーへのフィードバック

`resolve_map_mode(map_mode)` は軽量なソケット接続確認を行い
`(effective_mode, fell_back)` を返します。`fell_back` が True の場合、
呼び出し元ページは以下のメッセージを表示します：

```
⚠️ Online map tiles are unavailable. Showing the local coastline map.
```

ページは図を構築する前に `resolve_map_mode()` を呼び、その結果の
`effective_mode` を `apply_map_style()` に渡してください。

---

## 自己完結 HTML ダウンロード

`figure_to_self_contained_html(fig) → bytes` は Plotly 図を完全オフライン対応の
HTML ファイルとしてエクスポートします。Plotly.js（約 4.5 MB）がインラインで
埋め込まれ（`include_plotlyjs=True`）、CDN 参照は一切含みません。
ネットワーク接続なしで任意の現代的ブラウザで開くことができます。

エクスポートされた HTML の設定フラグ：

| フラグ | 値 |
|---|---|
| `scrollZoom` | `true` |
| `displayModeBar` | `true` |
| `responsive` | `true` |

**"Download interactive HTML"** ラベルのダウンロードボタンはページ 03・04 に
あります。ページ 05 も `download_figure()` 経由でダウンロードボタンを提供します。

---

## 今回のスプリントで変更しない制約

* ページ 32 の Cartopy 静的コンター地図・Vertical Section・GEBCO ロジックは変更なし
* 恒常的ユーザーデータの保存場所・アップロード共有仕様は変更なし
* Folium / Leaflet / Draw の CDN 資産はローカル同梱しない
* 地震カタログ・プレート境界・白令海日付変更線処理は変更なし
* バージョンは 1.3.3

---

## テストカバレッジ

| テストファイル | スイート | 検証内容 |
|---|---|---|
| `test/test_offline_map.py` | `TestAddCoastlineOverlayIdempotency` | 2 回目呼び出しが無操作・True 返却・トレース名 |
| `test/test_offline_map.py` | `TestApplyMapStyleAddsCoastline` | 全 5 mode で `_coastline_overlay` が存在・二重適用でも 1 本のみ |
| `test/test_offline_map.py` | `TestOfflineFallbackWarningText` | 英語のみ・⚠️ 始まり・日本語なし・`/` 区切りなし |
| `test/test_self_contained_html.py` | `TestFigureToSelfContainedHtmlSource` | helper 定義・`include_plotlyjs=True`・設定定数の存在 |
| `test/test_self_contained_html.py` | `TestFigureToSelfContainedHtmlRuntime` | bytes >1 MB 返却・CDN script src なし・設定フラグの HTML 埋め込み |
| `test/test_self_contained_html.py` | `TestPage03/04/05*` | ページが helper を呼ぶ・ダウンロードボタンあり・CDN 参照なし |
