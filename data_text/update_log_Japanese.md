# 更新履歴

新しい項目を上に追加します。`未リリース` 内でも更新日ごとにまとめます。今後のリリースノートを整理しやすくするため、`追加`、`変更`、`改善`、`修正`、`削除`、`準備` などの分類を使います。

## 未リリース

### Version 1.3.0 簡潔まとめ - 2026-09-11

- 公開テストに向けて、ページタイトル、サイドバー見出し、地図案内、図調整まわりのUI文言を整理。
- d18O、dD、d-excess、塩分、水温、水深、緯度、経度など、海水データの可視化パラメーター選択肢を拡充。
- 共通の海域プリセット、APIキー不要の地図背景、cmocean/EnvGeo カラーマップ、色分け操作の整理により、地図・Plotly 可視化を改善。
- 統合 beta ワークフローで、アップロードデータの重ね描き、同一カラーバー利用、表示スタイル調整、品質チェック概要を追加・改善。
- 不適切な水深・水温・塩分値の品質チェック処理を整備し、NaN変換後も元の値を確認できる品質フラグ列を追加。
- d-excess 計算、保存ファイル名生成、UIラベル、地図スタイル、カラーマップ、抽出データ概要などの共通処理を `envgeo_utils.py` に集約。
- 環境診断のCSV/PDF書き出しと、抽出データ統計のCSV書き出しに対応。
- 将来の公開リリースに向けて、README、日本語README、更新履歴、リポジトリ構成を整理。
- 公開構成、データ読み込み、品質ルール、保存ファイル名、主要ユーティリティに関する pytest を拡充。

### 2026-09-14

- 改善: Isotope & Hydrographic Mapping ページで、地図表示パラメーター選択を Map type の横に移動し、重複していた小さなパラメーター caption を削除。
- 改善: 共通の地図案内文を、サイドバーで Map center、表示範囲、カラーマップ、図設定を調整できることが分かる表現へ変更。
- 追加: 3D Visualizer の Plotly 図で、`Color filtered` の横にカラーマップ選択を追加し、散布図と対応する地図の両方に反映。
- 追加: 3D Visualizer の塩分-d18O Plotly 図に、任意表示の近似直線と、式・相関係数を小さく表示する情報ボックスを追加。
- 変更: 試験後、3D Visualizer の Temperature-Salinity 図では近似直線を表示しない構成に戻した。
- 修正: 3D Visualizer で近似直線を追加しても、Box/Lasso 選択による対応採水地点の地図ハイライトが機能するようにした。
- 追加: 既存の Fig.1-Fig.6 を温存したまま、4D Visualizer に `Custom 4D plot beta` を追加。
- 追加: Custom 4D plot beta に、`Salinity-d18O-[custom]-[custom]`、`T-S-[custom]-[custom]`、`Lon-Lat-depth-[custom]` のテンプレートを追加。
- 改善: `Lon-Lat-depth-[custom]` は地図系テンプレートとして扱い、Fig.3-Fig.6 と同様に、地図中心に合わせた経度、海岸線、地理的な縦横比を反映するようにした。
- 削除: 4D Visualizer 画面上の実装説明寄りの custom beta caption を削除。
- 追加: 共通 Data filtering サイドバーに `Area filter preset` を追加し、既存の海域プリセットから Longitude / Latitude フィルタの初期範囲を選び、その後スライダーで微調整できるようにした。
- 準備: 個別ページへのユーザーデータアップロード対応、Streamlit 更新後の submit button key 対応、Integrated Visualizer 中心の公開方針、独立 3D/4D uploader の非公開・開発用候補化を ToDo に記録。

### 2026-09-11

- 変更: 複数の同位体・海洋環境パラメーターを地図表示するページになったため、`32_d18O_mapping.py` を `32_Isotope_Hydrographic_Mapping.py` に変更し、ページタイトルを `Isotope & Hydrographic Mapping` に変更。
- 変更: Depth Profile ページは dD や d-excess など複数パラメーターに対応したため、ファイル名を `37_Depth_Profile_(T,S,d18O).py` から `37_Depth_Profile.py` に変更。
- 追加: Custom Parameter Plot beta ページに `Size contrast` を追加し、マーカーサイズ差をより強調できるようにした。
- 追加: Custom Parameter Plot beta ページで、カラーバー用のカラーマップを選択できるようにした。
- 追加: Custom Parameter Plot beta ページに、凡例表示のオン/オフと回帰線表示のオン/オフ設定を追加。
- 追加: Temperature-Salinity Diagram ページで、背景データ表示の横に凡例表示のオン/オフ設定を追加。
- 改善: README から内部のプロジェクト管理メモを削り、公開向けのセットアップ、使い方、データ、引用案内を中心に整理。
- 修正: テスト公開時に古い `envgeo_utils.py` が残っていてもUIラベルで落ちにくいよう、主要ページにフォールバック文言を追加。
- 変更: Salinity-d18O Relationship ページの背景全データ表示の初期値を `No` に変更。
- 追加: Salinity-d18O Relationship ページに、図サイズ、目盛本数、フォントサイズを数値入力で調整する機能を追加。
- 追加: Salinity-d18O Relationship ページで、フィルタ後データ点を任意パラメーターで色分けし、カラーバー表示できるようにした。
- 変更: 画面上の図・地図見出しから実装寄りの `Auto-Zoom` 表記を削除。地図の連動表示機能自体は維持。
- 変更: 3D Visualizer の Plotly 図の色分け操作をラジオボタンから `Color filtered` セレクターへ変更し、選択可能なパラメーター候補も拡充。
- 改善: サイドバーと地図表示まわりに残っていた装飾つきの古い案内文を、簡潔な共通UI文言へ整理。
- 改善: 図調整まわりの文言を整理し、古い赤字の地図範囲説明を控えめな共通 caption に変更。サイドバー見出しも共通ラベルに統一。
- 改善: 図のダウンロード用ファイル名の整形処理を `envgeo_utils.py` の共通関数に集約し、主要な作図ページに適用。
- 変更: Correlation Overview ページの画面タイトルを `Compiled figs` から `Correlation Overview` に変更。
- 追加: `35_Custom_Parameter_Plot_beta.py` を追加。T-S図をベースに、X軸、Y軸、色、マーカーサイズを任意の数値パラメーターから選べる試験的な2Dプロットページとして作成。
- 変更: Temperature-Salinity Diagram と Depth Profile ページで、フォントサイズと目盛本数の2値スライダーを個別の数値入力に変更。
- 修正: 現在のアプリおよび各ページのバージョン表示を `1.3.0` に統一。
- 追加: Depth Profile ページの横軸候補に dD と d-excess を追加し、欠損除外数の表示と選択パラメーターによる地図色分けに対応。
- 変更: 同位体・海洋環境パラメーターの地図表示ページを整理し、d18O、dD、d-excess、塩分、水温を選択して地図表示できるようにした。
- 追加: Temperature-Salinity Diagram ページで、フィルタ後データを水深、緯度、経度、年、月、d18O、dD、d-excess などで色分けできるようにした。
- 追加: Temperature-Salinity Diagram ページで、選択した色分けパラメーターごとにカラーバーレンジを調整できるようにした。
- 追加: Temperature-Salinity Diagram ページで、図の縦横サイズ、目盛本数、フォントサイズを調整できるようにした。
- 変更: Temperature-Salinity Diagram ページの T-S colormap 選択を外し、色分けパラメーターとカラーレンジだけを調整するシンプルな構成にした。
- 変更: Temperature-Salinity Diagram ページの背景全データ表示の初期値を `No` に変更。

### 2026-09-10

- 追加: 同位体・海洋環境パラメーターの地図表示ページの Matplotlib 図で、カラーバーの太さ、長さ、フォントサイズを調整できるようにした。
- 改善: HomeのMainタブ導線、About本文、Data Sources見出し、Manualの開始案内を、Core Dataset / Reference Datasets の考え方に沿って整理。
- 改善: Home/About/Manual の英語表現を、データセット範囲、推奨端末、図の利用・引用案内が伝わりやすい表現へ修正。
- 改善: 統合 beta ページの Map タブを `st.fragment` 化し、地図設定変更時にページ全体が再実行されにくい構成へ変更。
- 改善: 統合 beta ページの Map タブで、左側に色設定とRegion preset、右側1/3幅にMap Styleを置く二段レイアウトへ調整。
- 改善: 統合 beta ページの Shared-filter beta タブ名とタブCSSを、earthquake Advancedに近い見分けやすい表示へ変更。
- 追加: `Sidebar-filtered dataset (CSV)` の下に、品質フラグの判定基準を小さな注記として表示。
- 追加: `Details and statistics of sidebar-filtered data` に、フィルタ条件・選択データ別件数・行数・品質フラグ数・概要統計をCSVで書き出す機能を追加。
- 変更: CARTO basemapのAPI key必須化に対応するため、共通地図スタイルの標準背景を `carto-positron` からAPIキー不要の `open-street-map` へ変更。
- 追加: 環境診断ツールに、実行環境・依存パッケージ・主要ファイル確認結果をCSV/PDFレポートとして書き出す機能を追加。
- 変更: 環境診断用 Streamlit ツールの実体を `tools/env_check_streamlit.py` に置き、ローカル開発中は `pages/99_Environment_Check.py` からサイドバー表示できる構成に整理。
- 変更: `requirements.txt` を現在の Anaconda `envgeo_streamlit142` 環境に合わせて更新し、Python 3.10系で確認済みの依存関係として整理。
- 追加: 共通海域プリセットを拡充し、日本近海、黒潮・親潮、北太平洋、熱帯太平洋、インド洋、大西洋、地中海、北極海、南大洋セクターなどを選択できるようにした。
- 変更: 同位体・海洋環境パラメーターの地図表示ページの初期カラーマップを元の `Jet` に戻し、EnvGeo と cmocean の選択肢は残した。
- 追加: 同位体・海洋環境パラメーターの地図表示ページで、Matplotlib/Cartopy地図とPlotly Mapbox地図の両方に cmocean / EnvGeo カラーマップ選択を追加。
- 変更: 海洋データ向けの cmocean カラーマップ候補を正式採用しつつ、既存図との連続性のため初期値は `EnvGeo variable default` のままにした。
- 追加: Plotly図で使う共通カラーマップ選択ヘルパーを追加。
- 追加: 最新版作業フォルダから 4D Visualizer と 3D/4D Uploader の改訂版を取り込み、Correlation Overview と Vertical Section Visualizer を採用候補ページとして追加。
- 追加: `lon`, `lat`, `Depth`, `Temp`, `S`, `delta18O`, `delta_D` など、アップロードデータでよくある列名の別名を標準列名へ自動変換する機能を追加。
- 改善: 統合 beta ページの簡易表示で、数値列のPlotlyカラースケールをEnvGeo-Seawater共通関数に揃えた。
- 追加: 統合 beta ページの Summary、Upload、Quality タブに、アップロードデータの品質チェック結果を表示するようにした。
- 改善: 統合 beta ページの塩分-d18O図で、カラーバー比較に使える数値列をカテゴリ列より先に表示するようにした。
- 追加: 統合 beta ページに Map marker offset のヘルプ説明と、アップロードデータ重ね描き用のマーカー枠幅コントロールを追加。
- 追加: 統合 beta ページの Map、T-S図、塩分-d18O図で、選択中の色付け列が数値列の場合に、アップロードデータを同じカラーバーで色付けできるマーカーカラーモードを追加。
- 修正: 統合 beta ページのT-S図と塩分-d18O図で、アップロードデータの重ね描きをWebGLトレースにし、密な参照データより見えやすくした。
- 追加: 統合 beta ページで、アップロードデータのマーカーサイズ、色、形、透明度、Mapboxでの最前面表示、必要時の地図上オフセットを調整できるようにした。
- 追加: 共通の海域地図プリセットを追加し、統合 beta ページの Map 表示から選択できるようにした。
- 改善: 共通サイドバーの抽出データ概要に、行数、品質フラグ件数、d18O、dD、d-excess、塩分、水温、水深のパラメータ統計を追加。
- 追加: `90_Integrated_Visualizer_beta.py` を、既存ページ互換モードと共通フィルタ beta モードを持つ試作統合ページとして追加。
- 追加: 統合 beta ページに、選択した既存可視化ワークフロー、地図、T-S図、塩分-d18O図、任意列の2D/3Dプロットで使えるユーザーデータアップロード機能を追加。
- 削除: 独立版 3D/4D uploader はアップロード起点の別ワークフローであるため、統合 beta ページのワークフロー選択肢から外した。
- 変更: 共通アプリバージョン情報を `1.3.0` に更新。
- 追加: `envgeo_utils.py` に、品質チェックと d-excess 計算の日本語説明を追加。
- 追加: 不適切な水深、水温、塩分値を扱うための再利用可能な品質ルール情報を追加。
- 変更: d-excess 計算を `envgeo_utils.py` に集約し、複数の Streamlit ページで再利用できるようにした。
- 追加: NaN変換後も元の不適切値を確認できるよう、品質フラグ列を追加。
- 準備: 将来の公開リリースに向けてリポジトリ構成を整理。
- 変更: 以前の独立した about ページを `home.py` に統合。
- 変更: 退役したナビゲーションページ、重複ページ、古い beta ページを現在のソースツリーから除外。
- 追加: `.gitignore` を追加し、`.DS_Store`、`__pycache__`、`.pytest_cache` などの生成ファイルを整理。
- 追加: 日本語版 README を追加。
- 改善: README とアプリ更新履歴の公開向け表現を簡潔に整理。

## 1.0.1 - 2026-03-24

- 改善: EnvGeo-Seawater の公開リリースに向けて、安定版 Streamlit アプリ構成を改善。
- 変更: home、about、データソース、マニュアル、更新履歴、日本語情報ページを更新。
- 準備: 将来の公開リリースに向けたリポジトリ資料を整備。

## 1.0.0 - 2026-03-18

- 変更: 海水同位体・水文データ可視化アプリを大幅更新。
- 追加: 3D/4D、2Dマッピング、T-S図、深度プロファイル、塩分-d18O ワークフローを更新。
- 改善: データフィルタリング、図の出力、データソースを意識した表示を改善。

## 0.2.0 - 2026-02-18

- 変更: 統合 Streamlit アプリを pre-1.0 として大幅更新。
- 改善: ページ構成と可視化ページの挙動を改善。

## b20 - 2024-12-14

- 追加: Excelアップロードとカスタムプロット機能を追加。
- 追加: 追加文献に基づくデータセットを拡張。
- 改善: 可視化ページの性能と使いやすさを改善。

## Public release - 2024-05-15

- 公開: Streamlit アプリを公開。

## Maintenance - 2023-07-22

- 修正: 一般的な不具合を修正。

## b03 - 2023-05-22

- 追加: EnvGeo-Seawater アプリの初期 pre-release 版を追加。
