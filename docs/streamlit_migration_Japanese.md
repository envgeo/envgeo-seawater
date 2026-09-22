# Streamlit 1.63移行記録

この文書には、検証済みのStreamlit 1.42環境から移行する際の環境構成、互換性確認、判断理由を記録します。残作業は非公開のローカル作業メモで管理します。

## 環境構成

| 目的 | Conda環境 | Python | Streamlit | Pandas | NumPy | Plotly |
|---|---|---:|---:|---:|---:|---:|
| 検証済み基準環境 | `envgeo_st142_py310_plotly5` | 3.10.15 | 1.42.0 | 2.3.3 | 1.26.4 | 5.24.1 |
| 以前の中間試験環境 | `envgeo_st155_py312_plotly6` | 3.12.9 | 1.55.0 | 2.3.3 | 2.4.4 | 6.7.0 |
| 将来構成試験 | `envgeo_st163_py312_plotly7` | 3.12.14 | 1.63.0 | 3.0.6 | 2.5.3 | 7.1.0 |
| Plotly互換性試験 | `envgeo_st163_py312_plotly5` | 3.12.14 | 1.63.0 | 3.0.6 | 2.5.3 | 5.24.1 |

移行が完了するまで、検証済み基準環境は削除・変更しません。

## テスト結果

実施日: 2026-09-18

Streamlit 1.63の両環境で、自動テストに合格しました。

- EnvGeo-Seawater: 57件合格。
- EnvGeo-Earthquake: 9件合格、任意テスト4件スキップ。
- `pip check`: 依存関係の破損なし。
- Seawater Homeが正常起動し、HTTP 200を確認。

`envgeo_st163_py312_plotly5` では、AppTestによる初期表示のsmoke testも実施しました。Homeと13ページすべてがStreamlit例外なしで初期表示を完了しました。ただし、この確認にはサイドバーのApply操作、Plotly上の選択、ダウンロード、アップロード、すべての描画分岐は含まれません。ユーザー操作後に生じるエラーは、別途記録して確認します。

Streamlit 1.42では `use_container_width=True`、Streamlit 1.63では `width="stretch"` を自動選択する互換ヘルパーを追加しました。Pandasのfuture optionは、必要なPandas 3未満でのみ設定します。これにより、Streamlit 1.42互換性を残したまま、`use_container_width`、`copy_on_write`、`future.no_silent_downcasting` の反復警告を解消しました。変更後は両環境でSeawaterのテストが合格し、1.63環境では全ページの初期表示も引き続き正常です。

## Plotlyに関する確認

Plotly 7.1環境で、対話的な図の描画エラーを確認しました。Plotly 7では、現在EnvGeoの複数ページが利用しているMapbox系APIが削除されています。

- `px.scatter_mapbox` は `px.scatter_map` へ変更。
- `go.Scattermapbox` は `go.Scattermap` へ変更。
- `layout.mapbox` は `layout.map` へ変更。
- `mapbox_style` は `map_style` へ変更。

現在のアプリは、マッピング、プロファイル、統合、対話的可視化ページで旧APIを使用しています。このため、今回の描画エラーだけでStreamlit 1.63に互換性がないとは判断しません。

Interactive 2D/2.5D VisualizerのBox/Lasso連動には `streamlit-plotly-events==0.0.6` を使用しているため、これは別途確認します。

## 現在の方針

- 最初のStreamlit移行確認中は、各ページをPlotly 7向けに個別修正しない。
- `envgeo_st163_py312_plotly5` を使い、既存のPlotly 5の挙動を保った状態でStreamlit 1.63を確認する。
- `envgeo_st163_py312_plotly7` は、Plotly 7、Pandas 3、NumPy 2を含む将来構成の試験環境として残す。
- 1.3.2テストサイトでは `requirements.txt` のStreamlit対応範囲を1.42〜1.63とする。新規デプロイでは1.63を選択し、1.42基準環境はローカル回帰確認用として維持する。
- 移行試験中はPlotly 5.24を現在のリリース基準として維持する。
- 実行環境をPlotly 7へ変更する前に、Plotly 5.24で導入済みのMapLibre API（`scatter_map`、`Scattermap`、`layout.map`、`map_style`）へコードを移行する。
- 同じMapLibreコードをPlotly 5.24、6.7、7.1で検証する。問題がなければ、バージョン別地図コードを持たず、Plotly `>=5.24,<8`で共通実装することを目標とする。
- MapLibre化は独立した移行作業とし、地図中心、ズーム、範囲、背景、帰属表示、カラーバー、重ね描き、選択機能、書き出しを画面確認する。
- `streamlit-plotly-events` は別途検証し、可能であればStreamlit標準の `st.plotly_chart` 選択イベントへ置き換える。
- 対応範囲内の全組み合わせを検証済みと表記する前に、Python 3.10 / Streamlit 1.63 / Plotly 7の交差環境を追加確認する。

## バージョン更新計画

- EnvGeo-Seawater 1.3.2: Plotly 5.24を検証基準として維持しながら、Python 3.10〜3.12、Streamlit 1.42〜1.63互換を整理する。Streamlit 1.63で変化したタブDOMへの対応も含む。
- その後のマイナー更新: MapLibre地図をPlotly 5.24、6.7、7.1で検証する。

## ローカル比較

- Plotly 7.1将来構成: `http://localhost:8503`
- Plotly 5.24互換性試験: `http://localhost:8504`

互換性試験環境は次のコマンドで起動します。

```bash
conda activate envgeo_st163_py312_plotly5
cd "/Users/toyoho/Documents/study/704-Python/Webアプリ_main134_20230513/Streamlit_EnvGeo2/envgeo_seawater_v130"
python -m streamlit run home.py --server.port 8504
```

Homebrew版のStreamlitを誤って呼び出さないように、単独の `streamlit` コマンドではなく `python -m streamlit` を使用します。

## 残りの画面確認

- HomeとEnvironment Check
- Interactive 2D/2.5Dの散布図、Box/Lasso選択
- Interactive 3D/4Dの3D図、map-depth表示、カラーバー
- Salinity-d18O、T-S、Mapping、Depth Profileの地図
- Vertical SectionのPlotly・Folium表示
- Integrated Visualizerのアップロード、品質確認、Plotly表示
- 画像・データのダウンロードとサイドバーフォーム
- EarthquakeのSimple / Advancedページ
