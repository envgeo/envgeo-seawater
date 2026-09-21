# 更新履歴

新しい項目を上に追加します。`未リリース` 内でも更新日ごとにまとめます。今後のリリースノートを整理しやすくするため、`追加`、`変更`、`改善`、`修正`、`削除`、`準備` などの分類を使います。

## 未リリース

### 2026-09-21

- 修正: Integrated Visualizerの`uses_native_upload_overlay`判定が、T-S Diagramだけでなく、独自のアップロードパネルを持つ全ページ（Salinity-d18O Relationship、Isotope & Hydrographic Mapping、T-S Diagram、Custom Parameter Plot beta、Depth Profile）を正しく認識するように修正。修正前は、これら他ページをIntegratedの「Full existing page」モードで開くと、アップロード済みデータが旧`load_isotope_data`差し替え経由で参照データへ無断で混入し（Mappingページではコンター補間にも影響）、さらにページ側の独自アップロードパネルが二重表示されていた。`NATIVE_UPLOAD_OVERLAY_PAGES`が実際に`envgeo_user_data.render_upload_panel`と`INTEGRATED_EMBEDDED_PAGE_KEY`判定を持つページと一致し続けることを確認する回帰テストを追加。
- 追加: Custom Parameter Plot betaをIntegrated VisualizerのFull-existing-pageワークフロー一覧へ登録。実装済みのアップロードオーバーレイが単独ページだけでなくIntegrated経由でも利用できるようにした。
- 修正: Depth Profileで、必須列が対応済みであればアップロードオーバーレイの件数キャプションを常に表示するようにした。アップロード行が全て除外される場合（Xパラメーターまたは水深が欠損・不正）でも、何も表示しないのではなく「0 / N plotted (N excluded due to missing values)」と表示し、他のアップロード対応ページと挙動を揃えた。
- 追加: Isotope & Hydrographic Mapping（ページ32）に共通アップロード位置オーバーレイを追加。経度・緯度列（唯一の必須ロール）を自動認識または手動対応し、現在選択中のパラメーター（d18O・dD・d-excess・Salinity・Temperature）が利用できる場合は既存カラーバーで色付け、なければ固定色でフォールバック。Scatter MapとContour Mapの両方に最前面（zorder=10）の輪郭付きマーカーを描画し、griddata補間には混入しない。Plotly Sampling Location MapにはScattermapboxオーバーレイを追加し、自動表示範囲にアップロード地点を含める。アップロードデータがある場合は品質確認エクスパンダーを表示。
- 追加: Salinity-d18O Relationshipの採水地点地図にも共通アップロード位置オーバーレイを追加。最前面の輪郭付きマーカー、d18O共有色または固定色、地図範囲への反映、位置情報がない場合の説明に対応。
- 修正: T-Sなどの個別ページで自動認識済みアップロード列を再確認した際、既存の品質フラグが消去される問題を修正。既存フラグを保持し、手動列対応後に新しく見つかったフラグと統合するよう変更。
- 追加: 有効な経度・緯度がある場合、T-S Diagramの採水地点地図にもアップロード点を最前面で表示。輪郭付きマーカー、可能な場合のd18O地図カラースケール共有、固定色へのフォールバック、自動表示範囲へのアップロード地点反映に対応。
- 共通化: アップロード、変更可能な列対応、マーカー設定のサイドバーパネルを `envgeo_user_data.py` へ切り出し、各可視化ページにはページ固有の描画処理を残す構成へ移行。
- 追加: Salinity-d18O Relationshipに、Salinity・d18O列の自動/手動対応、既存カラーバー共有または固定色マーカー、品質確認、Matplotlib図の最前面重ね描画を追加。
- 試験: T-S Diagramのアップロードとマーカー設定の間に `Uploaded data columns` パネルを追加。自動認識結果を変更でき、未知の水温・塩分列はユーザーが明示的に選択できる構成にした。
- 追加: 元の試験的パラメーター列を残したまま標準列へ手動対応し、共通の数値化・品質判定を再適用する関数を追加。
- 方針: 各対象ページのアップロードを共通セッションへ登録し、アップロード元にかかわらずUser Data Validatorで同じ共通品質基準を確認できる構成を採用。全対象ページの重ね描画とValidatorの同等機能を確認した後にのみ、Integratedのアップローダーを廃止する。
- 変更: 稼働中の50m・110m海岸線データをExcelからCSVへ置き換え、`envgeo_utils.py` の共通ローダーから読み込む構成に統一。
- 整理: ローカル3D/4Dアップローダーの旧日本海岸線Excel直接読み込みを廃止し、10m・50m・110m・日本海岸線の旧Excelを作業領域の過去パーツへ退避。
- 方針: Shared-filter betaを独立したUser Data Validatorへ移し、個別可視化ページを正式ワークフローとして残す構成に整理。Integrated Visualizerは移行中の代替経路として維持し、移行完了後に非公開の開発アーカイブとする。
- 方針: 1回の変更を共通部品1つまたは個別ページ1つに限定し、代替実装のテストと画面確認が完了するまで現行機能を残す段階的移行原則を追加。
- 設計: `envgeo_utils.py` の巨大化を避け、共通アップロード処理とUIを仮称 `envgeo_user_data.py` へ切り出す方針を記録。

### 2026-09-20

- 追加: 前処理済みのユーザーデータを、同じStreamlitセッション内でIntegrated Visualizerと個別ページ間に共有できるメモリ内限定の保持機能を追加。
- 改善: CSV/Excel読み込み、アップロード前処理、テンプレート生成、品質フラグ行抽出、数値化、品質正規化、d-excess計算を `envgeo_utils.py` の再利用可能な関数へ集約。
- 試験: Temperature-Salinity Diagramに、ユーザーデータのアップロード、品質確認、マーカー調整、既存カラーバーを使う色分け、最前面への重ね描画を試験導入。
- 改善: T-S Diagramのアップロードとアップロード点の表示設定を、参照データのフィルター・図設定と分け、サイドバー最上部の2つの折りたたみに集約。
- 修正: Integrated Visualizer内でT-S Diagramを開いたときのアップロードUI重複と二重描画を防止。ファイル読み込みはIntegrated側、マーカー設定と重ね描画はT-S側が担当する構成に整理。
- 文書: 個別ページを正式ワークフローとして残すIntegrated Visualizerの設計と移行手順を、英語・日本語の専用方針文書に記録。
- 追加: 経度、緯度、水深、水温、塩分の明確な日本語列名を自動認識候補に追加。
- 確認: アップロード関連テストを拡充し、pytest 65件の合格と、共有アップロードデータの有無両条件でT-S DiagramのAppTest成功を確認。

### 2026-09-19

- 変更: Python 3.10〜3.12、Streamlit 1.42〜1.63の互換性改善サイクルとして、開発バージョンを1.3.1へ更新。
- 変更: Plotly 5.24をリリース基準として維持し、テストサイト用のStreamlit対応範囲を1.42〜1.63に設定。
- 修正: Correlation OverviewとSalinity-d18O Relationshipで、GeoAxesへの明示描画、Figureの明示保存・解放を行い、Matplotlib/CartopyのFigure状態が混在する問題を解消。
- 改善: Streamlitサーバーログを読みやすくするため、Correlation Overviewの調査用`print()`出力を無効化。
- 修正: Custom Parameter Plotの軸・カラー範囲の状態をデータソースごとに分離し、データセット切替時の自動範囲設定を復元。
- 修正: Streamlit Cloudで発生したMatplotlib数式ラベルの解析エラーを避けるため、Custom Parameter Plotの同位体ラベルをUnicode表記へ変更。
- 改善: Vertical Section Visualizerのページタイトルサイズを、他の主要可視化ページと統一。
- 方針: Plotly 5.24環境のままMapLibre APIへ移行し、同じコードをPlotly 6.7、7.1で検証する段階的な移行方針を採用。
- 方針: 共通処理と段階的テストを通じて、各個別ページへメモリ内限定のユーザーデータアップロード・重ね描きを追加する長期計画を確定。
- 方針: Correlation Overviewは手書きの探索ワークフローを残すアーカイブ表示ページとし、新機能・アップロード対応の対象から外す。
- 改善: 適用ボタンの押下が必要な設定は赤字、自動反映される設定は青字のキャプションで区別し、サイドバーの更新方法を明確化。
- 変更: Interactive 3D/4Dの図スケール設定とIsotope & Hydrographic Mappingの地図表示設定を自動反映に統一し、同一セクション内に手動適用と自動反映が混在する状態を解消。
- 試験: Interactive 2D/2.5D図にレスポンシブ表示を追加。PCでは最大850 pxを維持し、狭い画面では利用可能な幅まで縮小する。
- 変更: 対話的なPlotly探索ページであることをStreamlitのページ一覧とリポジトリ上で明確にするため、現行の2D/2.5D・3D/4Dページのファイル名に `[Interactive]` を追加。

### 2026-09-18

- 検証: 検証済みのStreamlit 1.42環境を残したまま、Python 3.12.14 / Streamlit 1.63.0の独立したConda環境で互換性テストを開始。
- 確認: 依存関係に破損がなく、Seawaterは57件合格、Earthquakeは9件合格・4件スキップとなり、新環境でSeawater Homeが正常起動することを確認。
- 検証: Plotly 7でMapbox系APIが削除されたことを対話的地図エラーの主因として整理し、Streamlit 1.63 / Plotly 5.24.1の比較環境を追加。移行方針と試験結果を専用の開発メモへ記録。
- 改善: Streamlitの全幅表示とPandas future optionに共通の互換処理を追加し、Streamlit 1.42対応を維持しながら反復する非推奨警告を解消。
- 確認: Streamlit 1.42・1.63の両環境でSeawaterの57テストが合格し、1.63では全ページが初期表示できることを再確認。

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

### 2026-09-17

- 改善: Interactive 2D/2.5Dおよび3D/4D VisualizerはPlotlyによる対話的なデータ探索用であり、論文・プレゼン向けの静的図は対応する個別ページで作成することを画面上に明記。
- 文書: ローカル開発方針を確認しやすいように `TODO_Japanese.md` を追加し、英語版と日本語版を相互参照できるようにした。
- 整理: Earthquakeの稼働中実装は専用アプリで管理する方針に合わせ、`91_EnvGeo_Earthquake.py` を専用サイトへの軽量な案内ページへ整理。

### 2026-09-16

- 変更: 旧3D/4Dページを `Interactive 2D/2.5D Visualizer` と `Interactive 3D/4D Visualizer` に整理し、ページファイル名も短めに変更。
- 追加: Interactive 2D/2.5D Visualizer に `dD-δ18O relationship` と `Custom 2D/2.5D plot beta` を追加し、Box/Lasso 選択と対応する採水地点マップの連動表示を再利用できるようにした。
- 追加: Custom 2D/2.5D plot に単色表示を追加し、純粋な2D散布図としても、色分け付き2.5D散布図としても使えるようにした。
- 変更: 2D/2.5D ページのファイル名を `03_2Dplus_Visualizer.py` に変更し、Custom plot の色分けパラメーターとカラーマップを横並びに整理。初期表示は単色ではなくデータパラメーターによる色分けにした。
- 追加: 4D Visualizer の custom view に `Full custom X-Y-Z-color` を追加し、X/Y/Z軸と色分けパラメーターをすべて選択できるようにした。
- 改善: 3D/4D Visualizer で、map-depth のスケール設定が Fig.3-Fig.6 用であることと、採水地点マップの詳細設定が別物であることが分かるように文言を整理。
- 改善: データ表の用語を整理し、サイドバーで抽出された結果は `Filtered dataset`、Plotly の Box/Lasso で選択した結果は `Box/Lasso-selected dataset` として区別。
- 改善: 外部コードレビューの指摘をもとに、`st.radio()` の不要な `args`、空の `else` ブロック、関数内 import、意味のない uploaded marker colorscale 設定など、挙動を変えにくい範囲を整理。
- 準備: Claudeレビュー由来の今後の整理項目として、データソース選択、地図自動ズーム、月表示、XY散布図、古い変数、アップロードデータ読み込み方式の共通化を `TODO.md` に記録。
- 改善: 全ファイルClaudeレビューをもとに、空データ判定のbool比較、4Dの未使用変数、Custom plotの除外件数、未使用の月表示変数、Vertical Sectionのカラースケール範囲とimportエラー処理など、低リスクな範囲を追加整理。
- 準備: 将来の論文公開・パッケージ化に向けて、研究利用実績の引用、論文図、パッケージ公開、開発用requirements、依存バージョン指定、今後の共通化項目を `TODO.md` に記録。

### 2026-09-14

- 修正: Isotope & Hydrographic Mapping ページで Map Center を変更した際、Streamlit Cloud の Cartopy で全球に近い経度範囲が NaN になって落ちる問題を回避。
- 修正: Isotope & Hydrographic Mapping ページで陸地塗りつぶしの輪郭線と海岸線が重なり、海岸線が二重に見える問題を修正。
- 追加: Isotope & Hydrographic Mapping ページの Map display settings に `Region preset` を追加し、フィルタ済みデータを変えずに図の表示範囲だけ切り替えられるようにした。
- 改善: Isotope & Hydrographic Mapping ページで、地図表示パラメーター選択を Map type の横に移動し、重複していた小さなパラメーター caption を削除。
- 試験: Custom Parameter Plot beta のサイドバー折りたたみ表示を試したが、現在のフィルタUIでは expander の入れ子が適さないため、通常の bordered container 表示に戻した。
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
- 改善: 4D Visualizer のメイン表示選択、custom view 設定、map-depth 設定、カラーバーレンジ、採水地点マップまわりのUI文言を整理。
- 改善: 3D Visualizer、T-S、塩分-d18O、同位体・海洋環境マッピング、Depth Profile、Custom Parameter Plot の画面表示文言を整理し、地図ラベル、Map style、色分けパラメーター、背景データ表示、選択データ表の表現を統一。
- 改善: T-S、塩分-d18O、Depth Profile、同位体・海洋環境マッピング、Custom Parameter Plot で、2D図のダウンロードボタンを対応する図の直下に移動。
- 追加: 色分けパラメーター、背景データ表示、回帰線、Map style、4D view、Profile parameter など、主要な操作項目に短い help テキストを追加。
- 修正: Depth Profile の保存画像で、長いフィルタ条件入りタイトルがはみ出しにくいように、タイトルの折り返し幅と上部余白を調整。
- 改善: Depth Profile の図幅・図高さ設定を、2値スライダーから個別の数値入力へ変更し、より細かく調整できるようにした。
- 改善: 図サイズ、フォントサイズ、目盛数、マッピングページのカラーバーフォントサイズなど、精密な再現性が必要な図設定を数値入力中心に整理。
- 改善: 英語版・日本語版 README を更新し、現在のページ構成、beta/ローカル開発ページの位置づけ、環境診断ツール、ユーザーデータ機能の現状、再現性に関する表現を整理。
- 追加: `docs/README.md` と `docs/release_checklist.md` を追加し、公開前確認、Streamlit公開、GitHubリリース、Zenodo、内部計画メモをトップREADMEとは別枠で管理できるようにした。
- 追加: `docs/testing.md` と `docs/testing_Japanese.md` を追加し、現在の pytest 群の内容、テスト範囲、限界、今後の拡充方針を公開向けに説明。
- 追加: `docs/manual/` と `docs/manual_Japanese/` に、全体概要、共通フィルタ、各ページ別の詳細マニュアル骨組みを英語版・日本語版で追加。
- 改善: `docs/testing.md` と `docs/testing_Japanese.md` を公開向けのテスト説明に整理し、内部計画メモを外した。
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
- 追加: `Filtered dataset (CSV)` の下に、品質フラグの判定基準を小さな注記として表示。
- 追加: `Details and statistics of filtered data` に、フィルタ条件・フィルタ後データ別件数・行数・品質フラグ数・概要統計をCSVで書き出す機能を追加。
- 変更: CARTO basemapのAPI key必須化に対応するため、共通地図スタイルの標準背景を `carto-positron` からAPIキー不要の `open-street-map` へ変更。
- 追加: 環境診断ツールに、実行環境・依存パッケージ・主要ファイル確認結果をCSV/PDFレポートとして書き出す機能を追加。
- 変更: 環境診断用 Streamlit ツールの実体を `tools/env_check_streamlit.py` に置き、ローカル開発中は `pages/99_Environment_Check.py` からサイドバー表示できる構成に整理。
- 変更: `requirements.txt` を現在の Anaconda `envgeo_st142_py310_plotly5` 環境に合わせて更新し、Python 3.10系で確認済みの依存関係として整理。
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
