# 公開前コンテンツ確認チェックリスト

[English version](publication_content_review_checklist.md)

## 目的と状態表示

安定版、GitHub Release、Zenodo archive、JOSS投稿の前に、ファイル単位で内容を確認するための
チェックリストです。これは内容・可読性の確認用であり、build、インストール、CI、Releaseの実行手順は
`release_checklist.md`で管理します。

- `[x]` この確認で内容レビュー済み
- `[ ]` この確認では未着手
- 「安定版のみ」は`seawater_map`、「開発版のみ」は正規作業フォルダおよび／または
  `envgeo-seawater`を指し、安定版repositoryには含めません。

各項目では、科学的・利用者向けの表現、現在の版と範囲、必要な英日注記、ローカルパス・個人データ・
token・内部会話本文がないこと、共通項目では正規作業フォルダと安定版が一致することを確認します。

## 1. 共通Pythonモジュール

- [x] `envgeo_utils.py` — 見出し、コメント、コメントアウト済み旧コード、設定の読みやすさを確認済み。
  機能の分割は保留し、最終回帰試験はRelease時の別作業とする。
- [x] `envgeo_assets.py` — 役割、公開パス安全境界、docstring、見出し書式を確認済み。
  正規作業フォルダのasset-pathテストは成功。
- [x] `envgeo_user_data.py` — session内限定のアップロード境界、列対応、hover表示の上限、
  地図overlayの書式、専用テスト（4 passed）を確認済み。
- [x] `envgeo_launcher.py` — 起点ファイルの解決、インストール後のimport境界、日英見出し、
  package化テスト（7 passed）を確認済み。
- [x] `envgeo_diagnostic_launcher.py` — 明示的なローカル診断起点、package command定義、
  公開ナビゲーションからの分離、package化テスト（7 passed）を確認済み。
- [x] `__init__.py` — package metadataにimport副作用がなく、版1.3.4が
  `pyproject.toml`と一致すること、および版・importの焦点テスト成功を確認済み。
- [x] `home.py` — 公開文言、版表示、同梱文書パス、外部参照データの表記、安全な読込失敗表示、
  公開面の焦点テスト（4 passed）を確認済み。
- [x] `bathymetry/make_lightweight_gebco.py` — 同梱GEBCO格子の生成手順を記録するソース専用ファイル。
  実行時参照はなく、wheelから意図して除外する。

## 2. ページスクリプト

### 安定版ページ（`seawater_map`）

- [x] `pages/03_[Interactive]_2Dplus_Visualizer.py` — import配置、英日併記の
  見出し・注記、旧コメントアウトコード、T–S σ0参照注記、連動地図操作、
  等値線の焦点テスト（32 passed）を確認済み。
- [x] `pages/04_[Interactive]_3D_4D_Visualizer.py` — import配置、英日併記の
  見出し・注記、Fig.1–Fig.6とカスタム表示の構造、地図・深度操作、旧コメントアウトコード、
  焦点テストを確認済み。
- [x] `pages/05_User_Data_Check_Quick_Visualizer.py` — セッション内だけの
  アップロード境界、内部出所列の除外、2D/3D/地図補助処理、英日docstring・見出し、
  焦点テスト（7 + 5 passed）を確認済み。
- [x] `pages/31_Salinity-d18O_Relationship.py` — import配置、英日併記の
  見出し・注記、塩分–δ18O回帰、アップロード重ね表示の境界、図の
  ダウンロード、採取地点地図、焦点テストを確認済み（3 + 52 passed）。
- [x] `pages/32_Isotope_Hydrographic_Mapping.py` — import配置、英日併記の
  見出し・注記、同梱・オフライン陸域マスク、散布図／コンター図の安全処理、
  アップロード重ね表示、Cartopy非依存の焦点テストを確認済み
  （106 passed、Cartopy依存overlayテスト4件はローカルでskip）。
- [x] `pages/34_T-S_diagram.py` — import配置、英日併記の見出し・注記、
  近似σ0参照等値線、アップロード重ね表示、図のダウンロード、採取地点地図、
  焦点テストを確認済み（37 + 11 + 52 passed）。
- [x] `pages/35_Custom_Parameter_Plot.py` — 数値パラメーター補助関数、
  英日併記の見出し・注記、アップロード重ね表示、回帰の対象範囲、図・表出力、
  焦点テストを確認済み（6 + 52 passed）。
- [x] `pages/37_Depth_Profile.py` — import配置、英日併記の見出し・注記、
  深度方向・gap row処理、月帯・アップロード重ね表示、図・表出力、採取地点地図、
  焦点テストを確認済み（4 + 52 passed）。
- [x] `pages/53_Vertical_Section_Visualizer.py` — import配置、英日併記の
  見出し・注記、A–B／軸基準断面の保護、GEBCO・アップロード海底地形の
  フォールバック、補間信頼度overlay、オフライン地図縮退、アップロード
  重ね表示、焦点テストを確認済み（51 + 7 + 52 passed）。
- [x] `pages/80_Correlation_Overview.py` — 手書き開発履歴として維持する方針を確認した。
  実行時のデータ・媒体パスはpackage対応済みで、実行中のデバッグ出力や機微なパスはない。英日併記の
  アーカイブ説明だけを加え、近代化によって開発履歴を消していない。

### 開発版のみのページと診断

- [x] `pages/90_Integrated_Visualizer_beta.py` — 開発版だけでレビュー済み。
  安定版／JOSSからの除外、履歴上の説明、メモリ内アップロード、固定された埋込みページ一覧、
  焦点テストを確認済み（7 + 4 passed）。
- [x] `pages/91_EnvGeo_Earthquake.py` — 開発版のリダイレクトとして確認済み。
  Seawaterのコード・データに依存せず、別のEarthquake監査までSprint 3 packageおよび安定版には入れない
  （4 passed）。
- [x] `pages/99_Environment_Check.py` — ローカル開発用ラッパーとしてだけ保持する。
  安定版clone、公開ナビゲーション、wheel、Release、archiveから除外されることを確認済み。
  診断は`tools/env_check_streamlit.py`と`envgeo-seawater-check`で提供する（11 passed）。

## 3. テスト、ツール、パッケージ設定

- [x] `test/conftest.py` と `test/README.md` / `test/README_Japanese.md` — importパスの理由、テスト・CIの範囲を現行化し、英日で確認済み。
- [x] `test_basic.py`と`test_data_integrity.py` — 重複していたパス設定を`conftest.py`へ集約し、現行v1.3.4の版情報と公開データの完全性検査を確認済み（8 passed）。
- [x] `test_envgeo_assets.py`と`test_envgeo_user_data.py` — 重複していたパス設定を集約し、英日で役割を明確化した。資産パスの安全性とセッション内アップロードを確認済み（29 passed）。
- [x] `test_envgeo_utils.py` — 重複したパス設定と未使用importを除去し、英日で対象範囲を明確化した。共通のデータ、地図、アップロード、海岸線、書出しの回帰テストを確認済み（55 passed）。
- [x] `test_natural_earth_land.py`と`test_offline_map.py` — 同梱陸域資産、ダウンロードなしの縮退表示、オフライン地図の挙動を英日で明確化した。Cartopy未導入時はCartopy必須部分だけskipし、残りを確認済み（67 passed、10 skipped）。
- [x] `test_p03_ts_density_contour.py`、`test_ts_density_contour.py`、`test_packaging_proof.py` — 近似σ0の表記とGSW入力範囲を静的に確認し、package dataと開発専用ページの除外を確認済み（76 passed）。
- [x] `test_public_surface.py`、`test_quick_visualizer.py`、`test_self_contained_html.py` — 公開ページ構成、セッション内アップロードのorigin表示、CDNを使わないPlotly HTML書出しを確認済み（40 passed）。
- [x] `test_repository_health.py`、`test_uploaded_page_overlays.py`、`test_vertical_section_visualizer.py` — source整合性、セッション内アップロードoverlay、Vertical Sectionの安全対策を確認した。全NA列のconcat警告を修正し、回帰テストで保護した（正規115 passed／4 skipped、stable 106 passed／13 skipped、overlay再実行は警告なし）。
- [ ] 残るテスト群：
  なし。
- [x] `tools/env_check_streamlit.py` — ローカル専用の実行環境・package診断。
  ローカルパスとpackage一覧を含む出力を共有記録から除外することを明記した。
  外部通信・永続書込みを行わず、`pip list`は読取り専用（7件＋AppTest成功）。
- [x] `pyproject.toml`、`requirements.txt`、`requirements-dev.txt`、`runtime.txt` — Python >=3.10の宣言、
  Python 3.10/3.12 CI、CloudのPython 3.12 runtime、版1.3.4、依存関係の正本、package data、
  console commandを確認済み（焦点テスト26件成功）。
- [x] `.github/workflows/ci.yml` — Python 3.10/3.12 matrix、local user dataの
  隔離、一時dependency cache、ネットワーク非依存test suite、wheel build、隔離導入、
  公開ページ確認、artifactを確認済み（焦点テスト113件成功、GitHub CI両jobも成功済み）。
- [x] `.gitignore`、`LICENSE`、`CONTRIBUTING.md` /
  `CONTRIBUTING_Japanese.md` — 生成物、secret、private data、診断、内部記録、
  スクリーンショットの除外、MITの適用範囲、安全な英日貢献案内を確認済み（焦点テスト30件成功）。

## 4. 利用者向けroot文書とdata_text

- [x] `README.md` / `README_Japanese.md` — 安定版ページ範囲、約50,000件の引用付きデータ、
  ユーザーデータ境界、英日公開文言、簡潔なデータ・ソフトウェア・地理空間謝辞を確認済み。
- [x] `TODO.md` / `TODO_Japanese.md` — 現在のv1.3.4 Release作業とRelease後の開発を分離し、
  旧レビュー／JOSS記録を履歴として明示した。EnvGeo Data計画を保持し、古い版を現行とする表現を除去した。
- [x] `local_data/README.md` / `local_data/README_Japanese.md` — データ行0件の
  公開sample、外部パス指定、セッション限定ブラウザアップロード、private dataを公開しない
  境界を確認済み。
- [x] `data_text/about.md`、`japanese.md`、`manual.md`、`manual_Japanese.md`、
  `main_references.md`、`other_references.md`、`NASA_references.md`、
  `CoralHydro2_references.md` — Home本文、英日案内、引用付き出典表示、約50,000件の範囲、
  ユーザーデータ境界を確認済み。
- [x] `data_text/update_log.md` / `update_log_Japanese.md` — 現行v1.3.4 Release
  candidateの簡潔な要約を追加し、過去の作業を履歴として明示した。英日対応も確認済み。

## 5. 利用者マニュアル

- [x] `docs/manual/README.md` / `docs/manual_Japanese/README.md` — 安定版の
  公開範囲、navigation、履歴ページの境界を確認済み。
- [x] 概要と絞り込み：`00_overview.md`、`01_data_filtering.md`。
- [x] ページ別manual：`03_2dplus_visualizer.md`、`04_3d_4d_visualizer.md`、
  `05_user_data_check_quick_visualizer.md`、`31_salinity_d18o.md`、`32_isotope_hydrographic_mapping.md`、
  `34_ts_diagram.md`、`35_custom_parameter_plot.md`、`37_depth_profile.md`、
  `53_vertical_section.md`と、それぞれの日本語版。実装上の境界と利用者向け文言を確認済み。
- [x] `90_integrated_visualizer.md`と日本語版 — 開発履歴として残し、安定版および
  静的websiteからの除外を明確化済み。

## 6. 技術・来歴・履歴文書

- [x] 文書一覧：`docs/README.md` / `docs/README_Japanese.md` — 英日リンク、文書一覧、
  開発履歴の表現、安定版の公開範囲、古い重複項目がないことを確認済み（焦点テスト23件成功）。
- [x] Release準備：`release_checklist*`と`stable_release_publication_notes*`（安定版のみ）—
  Release repository、tag／Zenodoの順序、ページ範囲、データ境界、CI artifact境界、英日リンクを確認済み
  （焦点テスト30件成功）。
- [x] `testing.md` / `testing_Japanese.md` — CI matrix、開発版／安定版wheelの範囲、
  実行時ネットワーク非依存の境界、手動QAの境界、Page 90の範囲、英日リンクを確認済み
  （焦点テスト30件成功）。
- [x] `code_guide.md` / `code_guide_Japanese.md` — source／module・ページ一覧、
  安定版／開発版の境界、Page 80のアーカイブ方針、Page 90/91/99の範囲、文書化基準、
  英日リンクを確認済み（焦点テスト23件成功）。
- [ ] 残るレビュー文書：本チェックリスト。
- [x] `citation_and_license_plan.md` / `citation_and_license_plan_Japanese.md`
  — 現行の学術利用データ範囲、プロジェクトと第三者データ所有の区別、
  `CITATION.cff`とZenodo DOIの段階的な順序、Natural Earthの帰属方針、現在のGEBCO利用条件、
  READMEの簡潔な謝辞、英日リンクを確認済み。
- [x] `provenance_inventory*` / `dataset_redistribution_audit*` — 5件の同梱workbook、
  ゼロ値local sample、正規／安定版のchecksum一致、package-data範囲、出典記録、wheel非収録資産の
  残る確認境界を英日で確認済み。
- [x] `external_dataset_workbook_notes*` — NASA GISSとCoralHydro2kについて、元データ対応記録、現行同梱workbook構造、引用、および非観測値変換の境界を英日で確認済み。
- [x] `geospatial_assets*` — 現行wheelのasset範囲、Natural EarthとGEBCOの境界、正規／安定版assetの一致、派生海岸線CSVのsource version記録という残課題を英日で確認済み。
- [x] `sprint3_distribution_design_and_acceptance*` — 現行1.3.4、Python 3.10／3.12 CI、開発版の12ページ範囲、手動QA境界、Release承認ではない状態を英日で確認済み。
- [x] `distribution_foundation_audit_and_plan*` / `wheel_proof_report*` —
  Sprint 2の監査・実証証跡、現行1.3.4検証との区別、英日リンクを確認済み。
- [x] `offline_operation_log*` — v1.3.3の履歴実装メモを現行v1.3.4のrelease根拠から
  区別し、オフライン縮退、HTML出力、タイル提供元の再確認境界を確認済み。
- [x] `streamlit_migration*` — 履歴上の検証結果と将来の移行作業を、現行v1.3.4の
  Python 3.10/3.12 release根拠から区別済み。
- [x] `ts_density_contour_review_and_plan*` — v1.3.4の近似σ0透明化作業と、将来の
  SA–CT／TEOS-10開発を区別済み。
- [x] `integrated_visualizer_strategy*` — Page 90移行履歴、現行v1.3.4の公開範囲、保留した
  共通コア構想を明確に分離済み。
- [x] `development_notes*`：履歴記録として残し、現行v1.3.4の公開状態、完了したパッケージ化、
  Page 90の除外、保留中の共有コア構想を明記済み。
- [x] `paper.md`／`paper.bib` — 正本の原稿・文献表を安定版repositoryと同期し、v1.3.4 Release candidateの
  状態、repository URL、引用、AI開示、近似σ0の表現、DOIを未取得として明記する表現を確認済み。旧
  `paper_revised*`は使用しない。

## 7. データ、資産、公開範囲

- [x] すべての`dataset/*.xlsx` — workbook内容を変更せず、引用、DOI／source URL、変換記録、
  現行の行／列構造、正規版／安定版のSHA-256一致を確認済み。過去のraw取得情報は、残るものを
  維持し、残っていないものは利用不可として明記した。
- [x] `coastline/`、Natural Earth帰属表示、同梱CSV、派生CSVのsource version記録 — Natural
  Earth v4.1.0中間ファイル、数値対応、現行出力チェックサムを確認済み。
- [x] `data/`、`bathymetry/`、公開画像／GIF — 必要なHome animation、プロジェクト地図GIF、GEBCO格子／生成記録、
  package参照を確認した。個人・未公表資料は確認されず、未使用legacy spreadsheetはアプリケーションおよび
  公開release tree外へ移動した。
- [x] `local_data/user_data.xlsx` — 意図した0行・22列の公開sampleであり、研究者個人の測定値を含まないことを確認した。
  ヘッダーと見た目のtemplate書式を維持したまま、ローカル絶対パスmetadataを含まないworkbookへ再生成した。
- [x] stable cloneには`Claude outputs/`、スクリーンショット、診断専用Page 99、内部review log、legacy fileが
  含まれず、`.DS_Store`、cache、build生成物、仮想環境、一時ファイルは追跡せずignoreすることを確認した。
  2026-10-03のlocal wheel検査では、許可した公開assetだけが入り、除外対象のpath classは含まれなかった。
  最終GitHub Release／Zenodo archiveは、review済みrelease commitから改めて作成して確認する。

## 8. 完了記録

完了時には、レビュー対象commit ID、確認者、日付、Python版、CI run URL、wheel SHA-256、最終Release版を、
Release checklistまたはRelease noteへ記録します。ディレクトリ間でコピーしただけでは、完了扱いにしません。
