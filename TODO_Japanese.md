# EnvGeo-Seawater ToDo（日本語版）

英語版: [TODO.md](TODO.md)

このファイルには、ローカルでの整理・改修作業中に忘れないようにするための開発メモを記録します。内容が十分に固まった項目は、README、詳細ドキュメント、またはGitHub Issuesへ移すことができます。

原則として、英語版の `TODO.md` と同じ内容を保ちます。ファイル名、関数名、画面上のUIラベルは、コードとの対応を確認しやすいように英語表記を残します。

プロジェクトの基本方針として、ユーザー向けUI、README、マニュアル、テスト説明、リリースノート、および主要な開発方針は、日本語・英語の両方で利用できる状態を維持します。

## 作業の優先順位

更新日: 2026-09-19

各ページで同じ修正を繰り返さないよう、次の順番で進めます。

1. 互換性パッチ版を完成させる。
   - Python 3.10 / Streamlit 1.42 / Plotly 5.24と、Python 3.12 / Streamlit 1.63 / Plotly 5.24で手動確認を完了する。
   - Plotly選択、Cartopy地図、フォーム、アップロード・ダウンロード、Vertical Section、Earthquake Simple / Advancedの操作を確認する。
   - 再現できる互換性問題だけを修正し、必要なpytestを追加する。
   - Seawater 1.3.2とEarthquake 0.3.2を現在のローカル版の区切りとし、公開・タグ付け前に残りの画面確認を完了する。
2. 地図をMapboxからMapLibreへ移行する。
   - 共通処理と各ページに残るMapbox使用箇所を一覧化する。
   - Plotly 5.24のまま、代表ページ1つを先にMapLibre化する。
   - 再利用できる地図処理を共通ヘルパーへ移す。
   - 同じ実装をPlotly 5.24、6.7、7.1で確認し、Python 3.10 / Streamlit 1.63 / Plotly 7の交差環境も試験する。
   - `streamlit-plotly-events`をStreamlit標準の`st.plotly_chart`選択イベントへ置き換えられるか確認する。
3. 共通ユーザーデータ基盤を作る。
   - CSV/XLSX読込、列名自動認識と手動対応、検証、品質フラグ、d-excess、データソースラベル、メモリ内限定管理を、UIから独立した関数として実装・テストする。
   - Streamlitウィジェットは、再利用可能なデータ処理ロジックと分離する。
4. 現行の個別ページへユーザーデータ重ね描きを展開する。
   - T-S、Salinity-d18O、Mapping、Depth Profileから開始する。
   - 最初のページが安定した後、Custom Parameter Plot、Interactive Visualizer、Vertical Sectionへ展開する。
   - Correlation Overviewはアーカイブ表示として維持し、新機能の対象外とする。
5. 公開構成とドキュメントを仕上げる。
   - Integrated Visualizerの最終的な役割と、公開、Advanced、beta、ローカル専用、非公開にする個別ページを決める。
   - 英語・日本語UI、ヘルプ、詳細マニュアル、スクリーンショット、インストール説明、パッケージ化、JOSS再投稿チェックリストを完成させる。

MapLibre移行、全ページへのアップロード展開、ナビゲーション変更、大規模リファクタリングを同じリリースで一度に行わない。

## JOSS再投稿に向けたドキュメント・テスト方針

採用日: 2026-09-19

外部レビューから有益な提案を取り入れる。ただし、特定のドキュメント作成ツールをJOSS投稿の必須条件とは考えず、継続的な開発方針として扱う。

ドキュメント方針:

- まず既存の英語・日本語Markdownマニュアルの内容を充実させる。インストール、クイックスタート、各ページの操作、ユーザーデータ読込、品質管理、トラブルシューティング、テスト、コントリビューション、サポート情報を対象とする。
- 将来の独立ドキュメントサイトには、GitHub Pagesで公開できるMkDocsを第一候補とする。Markdownと利用者向け説明を中心とする現在のStreamlitプロジェクトでは、現時点でSphinxを導入する必要性は低い。
- マニュアル構成がある程度安定してからMkDocsを導入する。生成サイトは、内容が不足した文書の代わりではなく、整った文書を読みやすく公開するために使用する。
- 将来の共通コア（`envgeo4d` / `envgeo_utils`）で安定した公開関数を対象に、APIリファレンスを追加する。特にデータ読込、検証、品質フラグ、d-excess、海域プリセット、書き出し処理を対象とし、各ページスクリプトのすべてを公開APIとして文書化しない。
- `51_Correlation_Overview.py` は、意図的に保存した探索・アーカイブワークフローとして説明し、API整理の対象にはしない。

テスト方針:

- 純粋関数のpytest、Streamlit `AppTest`による画面ワークフローテスト、短い手動視覚確認の3層を維持する。
- 現在一時的に実行しているStreamlit起動確認を、`streamlit.testing.v1.AppTest`を使った永続的なpytestへ移す。
- AppTestは、Home、Mapping、T-S、Depth Profile、Integrated Visualizerのアップロード・品質確認など、代表的な現行ワークフローから開始し、残りのページへ小さな単位で広げる。
- 通常入力だけでなく、抽出結果0件、必須列不足、不正なCSV/XLSX、NaN、範囲外値、品質警告、ダウンロード操作などの異常系も確認する。
- 数値計算・科学的処理は通常の単体テストで確認し、AppTestはページ起動、ウィジェット、フォーム、メッセージ、セッション状態の確認に使用する。
- PlotlyのBox/Lasso操作、地図タイル、Cartopy描画、カラーバー、ダウンロード図のレイアウトなど、AppTestでブラウザ描画を十分に検証できない項目は手動視覚確認に残す。
- 現在のpytest群を実行する最小構成のGitHub Actionsを先に追加する。代表的なAppTest群とStreamlit・Plotlyの基準環境が安定した後に、AppTestと互換性マトリクスを拡張する。

JOSS向けの注意:

- ドキュメントとテストは重要な作業項目だが、再投稿にはパッケージ化・インストール性、CI実績、コントリビューション・サポート案内、研究利用実績の引用、公開開発履歴、タグ付きリリース、AI支援開発の適切な開示も必要である。

## 優先度: 高

### Claudeレビューのフォローアップ

追加日: 2026-09-17

現在の状況:

- Claudeのレビュー内容を、現在のコードベースと照合した。
- 稼働中の主要ページに、低リスクな整理を反映した。
  - コールバック引数として使われていない `st.radio()` の誤解を招く `args=[1, 0]` を削除した。
  - レビュー対象となった主要描画ページの `else:()` を、明示的な `else: pass` に変更した。
  - T-S、Salinity-d18O、Depth Profileページ内にあった `io` / `textwrap` のローカルimportを、ファイル先頭のimport領域へ移動した。
  - Integrated Visualizerのアップロードデータ地図トレースから、実質的に何も行わない `colorscale=None if use_colorbar else None` を削除した。
- `51_Correlation_Overview.py` には、現在も探索的な旧形式のコードが含まれている。専用の整理作業を開始するまでは、開発時の探索ワークフローを示すページとして、基本的に現在の形を保つ。
- これは単なる放置ではなく、意図的な保存である。このページは、開発中に行われた手書きの探索的ワークフローを記録している。公開を継続する場合は、README、マニュアル、リリース資料などでもその位置づけを説明する。

今後の方向性:

- 複数ページで重複しているデータソースのラジオボタンや出典表示を整理するため、`envgeo_utils.render_data_source_selector()` を追加する。
- `envgeo_utils.auto_zoom_from_extent()` のような共通の地図自動ズーム関数を追加し、各ページに重複するズーム計算を段階的に置き換える。
- 現在 `normalize_lon_to_center()` として重複している経度中心調整処理を、`envgeo_utils.py` の共通関数にする。
- Integrated Visualizerで重複しているT-S / Salinity-d18O散布図の処理を、再利用可能なXY散布図ヘルパーへ切り出すことを検討する。
- 月範囲の表示処理を `envgeo_utils.format_month_selection()` にまとめる。
- `04_[Interactive]_3D_4D_Visualizer.py` で繰り返されている海岸線トレース処理を、`add_coastline_traces()` のような共通関数にする。
- `X_Y = 1`、`sheet_num = [2]`、未使用の色変数などの旧変数をページごとに確認する。まずDepth ProfileとSalinity-d18O Relationshipから着手する。
- `sidebar_filter_and_display()` を段階的に分割する。現在は、UI、フィルタリング、統計表示が混在した大きな関数で、返り値も長いタプルになっている。
- 長期的には、Integrated Visualizerのアップロードデータ用モンキーパッチを、明示的なローダー差し替えまたはデータオブジェクトの注入方式へ変更し、複数ユーザー利用時の挙動を理解・管理しやすくする。

全ファイルレビューから得られたJOSS・パッケージ関連の課題:

- 海水d18Oデータを利用したAono、Sakamoto、Kurokiの研究を含む、研究上の波及効果を示す引用を `paper.md` と `paper.bib` に追加する。
- JOSS論文に、少なくとも1点の図またはスクリーンショットを追加する。
- State of the fieldを見直し、ODVなどの海洋学ツールとの違いを、同位体・水理データのワークフロー、d-excess、EnvGeo-SeawaterのWeb中心の利用方法という観点から具体的に説明する。
- `pyproject.toml`、PyPI、必要に応じてconda-forgeを含むパッケージ公開方針を検討する。パッケージマネージャーからのインストールが求められる場合、JOSS再投稿前に対応が必要となる。
- pytest関連の開発用依存関係をまとめた `requirements-dev.txt` を作成する。
- Streamlit 1.6x移行テスト後に、依存関係を厳密な `==` で固定する方針を再検討する。

### JOSS準備状況の外部評価フォローアップ

追加日: 2026-09-19

外部評価で、リリース前のCritical項目が4点示された。評価時点ではテストが4ファイル・約46件と集計されていたが、現在は`test_*.py`が5ファイル、テストが60件である。`envgeo_utils.py`は現在約78 KBである。

Critical項目は、依存関係を考慮して次の順番で進める。

1. pushおよびpull request時に現在のpytest群を実行する、最小構成のGitHub Actionsを追加する。AppTestの拡充を待たず、基本CIを先に導入する。
2. 現在のプロジェクト情報とバージョンを記載した`CITATION.cff`を作成する。Zenodoから正式な識別子が発行されるまでは、DOIを未記載または明確な保留状態にする。
3. `paper.md`へ正式なリポジトリURLとソフトウェアバージョンを追加し、リリース前にAvailabilityの記述と研究利用実績の引用を整える。
4. 1.3.2の互換性確認を完了し、最終版のタグ付きGitHub Releaseを作成する。
5. そのリリースをZenodoでアーカイブし、発行されたDOIを`paper.md`、`CITATION.cff`、READMEの引用案内、リリース記録へ一貫して追記する。

テストカバレッジは今後の計画資料として測定する。ただし、数値目標だけを追わず、科学的処理と実際のワークフローのリスクに基づいてテストを拡充する。

### 個別ページへのユーザーデータアップロード対応

追加日: 2026-09-13、設計方針確定: 2026-09-20

現在の状況:

- `90_Integrated_Visualizer_beta.py` の既存実装を、ユーザーデータアップロード機能の主な試験場所として維持する。
- 長期方針として、各個別可視化ページでユーザーデータのアップロードと重ね描きに対応する。
- ページごとに別実装を追加する前に、共通の読み込み・列対応処理を完成させる。
- 個別可視化ページを正式な公開ワークフローとして残す。データ検証と既存データとの比較は独立したUser Data Validatorへ移す。Integrated Visualizerは移行中は残し、完了後は非公開の開発アーカイブとする。詳細は `docs/integrated_visualizer_strategy_Japanese.md` を参照。

今後の方向性:

- 個別ページへ展開する前に、共通のアップロード処理を `envgeo_user_data.py` のような専用モジュールへ切り出す。既存の `envgeo_utils.py` 関数は再利用するが、同ファイルをさらに巨大化させない。
- ユーザーデータのアップロードを海水専用機能ではなく、EnvGeoのコア機能として扱う。この基盤は将来的に、地震やその他の地球科学可視化にも利用できるようにする。
- 共通処理には、CSV/XLSXの読み込み、列名の自動認識、手動列修正、必須列の確認、d-excess計算、品質フラグ、データソースラベル、セッション内メモリだけでのデータ管理を含める。
- マーカーサイズ、単色または共通カラーバー、形、枠、透明度、最前面表示を、共通の重ね描き設定として扱う。
- Shared-filter betaを独立したUser Data Validatorへ移し、同等の検証が確認できるまで現行Integratedを残す。
- 1回の作業は共通部品1つまたは個別ページ1つに限定し、段階ごとに利用可能・テスト済みの状態を保つ。
- 共通処理を次の順で段階的に展開・確認する。
  - Temperature-Salinity Diagram
  - Salinity-d18O Relationship
  - Mapping / isotope map pages
  - Depth Profile
  - Custom Parameter Plot
  - Interactive 2D/2.5D、3D/4D Visualizer
  - Vertical Section beta
- Correlation Overviewは開発過程を残すアーカイブページであるため、アップロード機能の展開対象から外す。
- 全ページ展開前に、CSV/XLSX読込、別名・日本語ラベル認識、手動列対応、検証エラー、品質フラグ、メモリ内限定処理のpytestを追加する。
- `User Data Check & Quick Visualizer` を、品質確認と任意列の簡易2D--4D可視化を行う公開のアップロード起点ページとして維持する。セッション内限定のデータ管理と共通Data filteringを保ち、ページ90を移行期間中は残しつつ、専門的な個別解析ページとの役割を明確にする。
- `dataset/91_USER_UPLOAD_UNPUB.xlsx` は、`envgeo_utils.py` の稼働中ローダー参照、`Unpublished dataset` への統合、テスト／サンプル依存、古い文書を監査して置換・削除できることを確認した後にのみ廃止する。ブラウザからのCSV/XLSXアップロードを通常のユーザーデータ運用とし、監査完了前にこのブックを削除しない。

注意事項:

- アップロードされたユーザーデータはメモリ上だけに保持し、ローカルマシンやサーバーへ保存しない。
- ユーザーデータは、参照データと視覚的に区別できるようにする。
- 必要な場所では、アップロードデータの品質フラグを表示し、ダウンロードできるようにする。
- 将来ユーザーが明示的に書き出しを選ぶ場合を除き、アップロード元ファイルや統合データをローカル・サーバーへ保存しない。

### Correlation Overviewの保存方針

- 手書きで研究機能を追加してきた元の探索ワークフローを残す、表示用アーカイブページとして維持する。
- 新機能、ユーザーデータアップロード、構造的リファクタリングは追加しない。
- 保守は、ページを起動できない問題や既存図を表示できない重大な互換性問題への最低限の修正に限定する。

## Streamlitアップグレード後の対応

### Streamlit 1.63 / Python 3.12移行テスト

開始日: 2026-09-18

詳細記録: [docs/streamlit_migration_Japanese.md](docs/streamlit_migration_Japanese.md)

現在の状況:

- Conda環境 `envgeo_st163_py312_plotly7` を新規作成した。既存の `envgeo_st142_py310_plotly5` は、安定版との比較用環境として残している。
- Streamlit 1.63互換性とPlotly 7 / MapLibre移行を分けて確認するため、`envgeo_st163_py312_plotly5` を追加した。
- Python 3.12.14、Streamlit 1.63.0、Cartopy 0.26.0、Pandas 3.0.6、NumPy 2.5.3、Plotly 7.1.0、Matplotlib 3.11.2、SciPy 1.18.1、scikit-learn 1.9.1、gsw 3.6.23を確認した。
- `pip check` では依存関係の破損は検出されなかった。
- EnvGeo-Seawaterのテストは57件合格した。
- EnvGeo-Earthquakeのテストは9件合格、4件スキップだった。
- SeawaterのHomeページはStreamlit 1.63で `http://localhost:8503` に正常起動し、HTTP 200を確認した。
- 現在の `requirements.txt` はStreamlit 1.42-1.63を許容し、描画ライブラリはPlotly 5.24を検証済み基準として維持している。テストサイトの新規デプロイではStreamlit 1.63が選択される。

残りの確認:

- Plotly選択、Cartopy地図、ファイルのアップロード・ダウンロード、フォーム、Vertical Sectionなど、代表的なSeawaterページを画面上で確認する。
- EarthquakeのSimple / Advancedページを画面上で確認する。
- Pandas future optionとStreamlit全幅表示の反復する非推奨警告は、共通互換ヘルパーで解消済み。
- `docs/streamlit_migration_Japanese.md` のPlotly 5.24からMapLibreへ移行する方針に従い、同じコードをPlotly 6.7、7.1でも確認する。
- 公開ナビゲーションを変更する前に、`st.Page` / `st.navigation` を別途試験する。
- 画面確認後、再現可能な移行用requirementsを作成し、この環境を新しい開発基準にするか判断する。

### フォーム送信ボタンのkey

追加日: 2026-09-13

現在の状況:

- 現在のローカル・テスト環境はStreamlit 1.42.0を使用している。
- このバージョンでは、`st.form_submit_button()` が `key` 引数に対応していない。
- そのため `51_Correlation_Overview.py` では、同じフォームの上部と下部の送信ボタンに、`Apply settings` と `Apply settings!` という異なるラベルを使用している。

今後の方向性:

- `st.form_submit_button()` が `key` に対応するStreamlitへ更新した後、2つのボタンを持つフォーム構成を再検討する。
- 上部と下部で別のkeyを指定し、どちらも `Apply settings` のような同じラベルにする。
- 必要に応じて、他の長いサイドバーフォームにも同じ形式を適用する。

### ページナビゲーションの管理

追加日: 2026-09-17

今後の方向性:

- Streamlit 1.6xへの移行テスト時に、`pages/` フォルダによる自動ナビゲーションから、`st.Page` / `st.navigation` を使う方式への移行を試す。
- ソースファイル名は短く安定した状態を保ち、ユーザーに見せるページ名とアイコンを別に設定する。
- サイドバーを、対話的なデータ探索、論文・プレゼン向け静的図、betaツール、開発・診断用ツールなどのグループに整理する。
- 公開画面に必要なページだけを登録する。開発用・診断用ファイルはリポジトリに残しながら、公開サイドバーには自動表示しないようにする。
- Pythonファイル名へUnicode絵文字を直接追加せず、統一感のある控えめなMaterial iconまたは単純な記号を優先する。
- 正式採用前に、ページURL、`st.switch_page` / `st.page_link` の参照、Integrated Visualizerからの呼び出し、およびStreamlit Cloudでの挙動を確認する。

## ユーザー向けUIの改善

### 描画ワークフロー内の小さな混乱を減らす

追加日: 2026-09-14

現在の状況:

- 主要な描画ページでは、`Sampling Location Map`、`Map style`、`Color parameter`、`Show background data` など、より統一されたラベルを使用している。
- 2D図のダウンロードボタンは、それぞれ対応する図の下へ移動した。
- `Data filtering` には、フィルタ条件を変更した後に `Apply settings` を押す必要があることを説明する短い注記を追加した。

今後の方向性:

- ユーザーがまだ操作を見落とす場合は、`Data filtering` の適用手順をさらに明確にする。候補文: `Change filters, then click Apply settings to refresh all figures.`
- 同じ `Color parameter` というラベルでも、T-S図のマーカー色、地図上の表示変数、3D/4D図のカラー軸など、ページごとに意味が異なる場合は短いヘルプを追加する。
- `Map controls` ポップオーバーが見つけやすいか確認する。必要であれば、地図背景を変更できることを示す簡潔なヘルプを追加する。
- 3D・インタラクティブページで、`Sidebar-filtered dataset` と `Box/Lasso-selected dataset` の違いをさらに分かりやすくする。初めて利用するユーザーには、Box/Lasso選択が追加の対話的な部分集合であることを示す短い説明が役立つ可能性がある。
- Plotly地図・図のダウンロード方法を決める。静的なMatplotlib図には図の下に明確な `Download image` ボタンがあるが、Plotly表示では現在、主にPlotlyモードバーのカメラ・書き出し機能を利用している。
- 公開前に、`beta` ページの役割を明確に説明する。一部は現在進行中の研究・開発ツールであり、別のページは将来的に高度な機能または非公開ワークフローとなる可能性がある。
