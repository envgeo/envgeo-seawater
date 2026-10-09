# Streamlit 1.63移行記録（履歴記録）

この文書には、検証済みのStreamlit 1.42環境から移行する際に行った過去のローカル互換性作業を記録します。
現在の安定版の対応範囲を示すものではありません。現行のRelease境界は`release_checklist_Japanese.md`、
`testing_Japanese.md`、および安定版repositoryの`stable_release_publication_notes_Japanese.md`を参照してください。

v1.3.4はPython >=3.10を宣言し、CIでは最終wheelをPython 3.10と3.12で検証しています。検証済みの
アプリ基準は、Python 3.10／Streamlit 1.42／Plotly 5.24、およびPython 3.12／Streamlit 1.63／Plotly 5.24です。
以下のPlotly 7、Pandas 3、NumPy 2に関する結果は将来向けの実験であり、対応を表明するRelease構成ではありません。

## Release後のpackage導入確認（2026-10-04）

Apple Silicon MacStudioで新しいConda環境を作成し、公開済みv1.3.4を
`python -m pip install envgeo-seawater`で導入した。その後の`pip check`と
`envgeo-seawater` launcherの起動は、Python 3.10、3.11、3.12、3.13で成功した。
Python 3.13では固定している`numpy==1.26.4`と`pyproj==3.6.1`がソースからビルドされた。
pip 26.2.1と、`--no-cache-dir`を指定したpip 24.3.1の両方で成功したが、数分かかる場合があり、
ネイティブのビルド環境に依存する。

2026-10-10には、Python 3.13の導入がApple Silicon M1 MacBook Airで失敗した。NumPyのビルド前にネイティブCコンパイラが見つからなかったためである。これはM1固有の失敗例ではない。pipが対応するwheelを取得できず、ネイティブのビルド環境もない場合には、どのプラットフォームでも同じ結果になり得る。したがってPython 3.13は、すべての環境で`pip install`だけによる導入を保証するものではなく、**条件付きサポート**とする。ローカル導入はPython 3.12を推奨し、Python 3.13を使用する場合にはmacOSではAppleのCommand Line Tools、他の環境では同等のネイティブビルド環境が必要である。

Python 3.9は`Requires-Python >=3.10`により意図して拒否される。Python 3.14はv1.3.4で固定した
依存関係では未対応であり、観察した導入試行はSciPyのビルド段階で停止した。ここでの記録はpackageの
導入・launcher起動確認であり、CIによるwheel確認や全対話機能の検証を置き換えるものではない。

## 環境構成

| 目的 | Conda環境 | Python | Streamlit | Pandas | NumPy | Plotly |
|---|---|---:|---:|---:|---:|---:|
| 検証済み基準環境 | `envgeo_st142_py310_plotly5` | 3.10.15 | 1.42.0 | 2.3.3 | 1.26.4 | 5.24.1 |
| 以前の中間試験環境 | `envgeo_st155_py312_plotly6` | 3.12.9 | 1.55.0 | 2.3.3 | 2.4.4 | 6.7.0 |
| 将来構成試験 | `envgeo_st163_py312_plotly7` | 3.12.14 | 1.63.0 | 3.0.6 | 2.5.3 | 7.1.0 |
| Plotly互換性試験 | `envgeo_st163_py312_plotly5` | 3.12.14 | 1.63.0 | 3.0.6 | 2.5.3 | 5.24.1 |

検証済み基準環境は、意図して再作成する場合を除き再現性のため保持します。Releaseに別環境として必須ではありません。

## 当時のテスト結果

実施日: 2026-09-18

Streamlit 1.63の両環境で、自動テストに合格しました。

- EnvGeo-Seawater: 57件合格。
- `pip check`: 依存関係の破損なし。
- Seawater Homeが正常起動し、HTTP 200を確認。

`envgeo_st163_py312_plotly5` では、AppTestによる初期表示のsmoke testも実施しました。当時存在したHomeと13ページすべてがStreamlit例外なしで初期表示を完了しました。ただし、この確認にはサイドバーのApply操作、Plotly上の選択、ダウンロード、アップロード、すべての描画分岐は含まれません。ユーザー操作後に生じるエラーは、別途記録して確認します。

Streamlit 1.42では `use_container_width=True`、Streamlit 1.63では `width="stretch"` を自動選択する互換ヘルパーを追加しました。Pandasのfuture optionは、必要なPandas 3未満でのみ設定します。これにより、Streamlit 1.42互換性を残したまま、`use_container_width`、`copy_on_write`、`future.no_silent_downcasting` の反復警告を解消しました。変更後は両環境でSeawaterのテストが合格し、1.63環境では全ページの初期表示も引き続き正常です。

## Plotlyに関する確認

Plotly 7.1環境で、対話的な図の描画エラーを確認しました。Plotly 7では、現在EnvGeoの複数ページが利用しているMapbox系APIが削除されています。

- `px.scatter_mapbox` は `px.scatter_map` へ変更。
- `go.Scattermapbox` は `go.Scattermap` へ変更。
- `layout.mapbox` は `layout.map` へ変更。
- `mapbox_style` は `map_style` へ変更。

現在のアプリは、マッピング、プロファイル、統合、対話的可視化ページで旧APIを使用しています。このため、今回の描画エラーだけでStreamlit 1.63に互換性がないとは判断しません。

Interactive 2D/2.5D VisualizerのBox/Lasso連動には `streamlit-plotly-events==0.0.6` を使用しているため、これは別途確認します。

## 当時の移行判断と保留作業

- 最初のStreamlit移行確認中は、各ページをPlotly 7向けに個別修正しない。
- `envgeo_st163_py312_plotly5` を使い、既存のPlotly 5の挙動を保った状態でStreamlit 1.63を確認する。
- `envgeo_st163_py312_plotly7` は、Plotly 7、Pandas 3、NumPy 2を含む将来構成の試験環境として残す。
- 当時予定していた1.3.2テストサイトでは `requirements.txt` のStreamlit対応範囲を1.42〜1.63とする。新規デプロイでは1.63を選択し、1.42基準環境はローカル回帰確認用として維持する。
- 移行試験中はPlotly 5.24を現在のリリース基準として維持する。
- 実行環境をPlotly 7へ変更する前に、Plotly 5.24で導入済みのMapLibre API（`scatter_map`、`Scattermap`、`layout.map`、`map_style`）へコードを移行する。
- 同じMapLibreコードをPlotly 5.24、6.7、7.1で検証する。問題がなければ、バージョン別地図コードを持たず、Plotly `>=5.24,<8`で共通実装することを目標とする。
- MapLibre化は独立した移行作業とし、地図中心、ズーム、範囲、背景、帰属表示、カラーバー、重ね描き、選択機能、書き出しを画面確認する。
- `streamlit-plotly-events` は別途検証し、可能であればStreamlit標準の `st.plotly_chart` 選択イベントへ置き換える。
- 対応範囲内の全組み合わせを検証済みと表記する前に、Python 3.10 / Streamlit 1.63 / Plotly 7の交差環境を追加確認する。

## 当時の更新計画（現在は履歴）

- 当時はEnvGeo-Seawater 1.3.2で、Plotly 5.24を検証基準として維持しながら、Python 3.10〜3.12、Streamlit 1.42〜1.63互換を整理する予定だった。Streamlit 1.63で変化したタブDOMへの対応も含む。
- その後のマイナー更新: MapLibre地図をPlotly 5.24、6.7、7.1で検証する。

## 当時のローカル比較

- Plotly 7.1将来構成: `http://localhost:8503`
- Plotly 5.24互換性試験: `http://localhost:8504`

互換性試験環境は次のコマンドで起動します。

```bash
conda activate envgeo_st163_py312_plotly5
cd /path/to/envgeo_seawater_v130
python -m streamlit run home.py --server.port 8504
```

Homebrew版のStreamlitを誤って呼び出さないように、単独の `streamlit` コマンドではなく `python -m streamlit` を使用します。

## 当時の残り画面確認

ここに挙げる項目は移行時のチェックリストであり、現行の安定版受入項目ではありません。現在の手動確認は
`release_checklist_Japanese.md`で管理します。

- HomeとEnvironment Check
- Interactive 2D/2.5Dの散布図、Box/Lasso選択
- Interactive 3D/4Dの3D図、map-depth表示、カラーバー
- Salinity-d18O、T-S、Mapping、Depth Profileの地図
- Vertical SectionのPlotly・Folium表示
- Integrated Visualizerのアップロード、品質確認、Plotly表示
- 画像・データのダウンロードとサイドバーフォーム
