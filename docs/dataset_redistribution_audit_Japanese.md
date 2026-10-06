# データセット再配布根拠の監査メモ

**状態:** 2026-09-25に初回確認し、2026-09-30にworkbook／package範囲、2026-10-03に派生資産の記録を
再確認した公開根拠監査。
これはリリース計画の記録であり、法的助言ではない。簡潔な
[`provenance_inventory_Japanese.md`](provenance_inventory_Japanese.md)を補足する。

## 目的と公開時の原則

EnvGeo-Seawaterは現在、`dataset/`内の複数の統合workbookを読み込む。出典がダウンロード可能、引用済み、または公開リポジトリで閲覧可能であっても、それだけで派生workbookをGitリポジトリ、Python wheel、Zenodoアーカイブ、その他のリリース成果物として**再配布できる根拠にはならない**。

本プロジェクトでは、現在の学術利用の方針を維持する。すなわち、データセットDOIまたは正規の取得元を示し、求められた形で引用し、取得日と変換内容を記録し、第三者データをプロジェクト所有物として扱わない。この監査は、既に引用して公開しているworkbookの削除を求めるものではない。公開リリースおよび論文投稿まで、データ提供者への追加問い合わせは行わない。将来のインストール型packageやZenodo設計ではこれらの記録を維持し、明示的な制限、査読者からの指摘、または権利者からの要請があった場合にのみ見直す。

### NASA GISSとCoralHydro2kの現行配布判断

プロジェクトは、NASA GISSおよびPAGES CoralHydro2k由来workbookについて、公開利用と論文投稿まで、現行の学術利用としての配布を継続する判断をしている。この判断は、追跡可能な公開出典、記録済みの元データ対workbook照合・変換、およびデータ品質上の発見に関するプロジェクトの既存連絡を踏まえる。出典URL、版、該当するDOI、取得日、求められる引用、変換記録を維持する。いずれもプロジェクト所有データとは表現しない。

私的なメール、連絡先、返信、添付資料はrelease metadataではなく、repository、wheel、release note、Zenodo archive、公開文書へ複写してはならない。明示的な制限、権利者からの要請、または具体的な査読上の懸念が生じた場合だけ、この判断を見直す。

### 現行package dataの判断（2026-09-28）

現行の全`dataset/*.xlsx` workbookをEnvGeo-Seawater packageに含める。これは、現在引用している公開データ群についての、プロジェクトの記録済み学術利用配布判断である。各workbookでは、出典引用、利用可能な場合のDOIまたは正規出典、取得日、判明しているプロジェクト側の共通スキーマまたは表示用変換を維持する。同梱は元レコードをプロジェクト所有にするものではなく、将来追加するデータセットへ自動的に同じ判断を適用するものでもない。

### 検証済みpackageスナップショット（2026-09-30）

読取り専用の比較により、`pyproject.toml`で選択される5つの`dataset/*.xlsx` workbookが、
正規作業フォルダと安定版`seawater_map` cloneでbyte単位で一致することを確認した。ファイル名、
worksheet名、行数・列数、SHA-256は[`provenance_inventory_Japanese.md`](provenance_inventory_Japanese.md)
に記録した。本監査でworkbook内容は変更していない。追跡する`local_data/user_data.xlsx` sampleも、
両方の場所で0行・22列の公開templateである。2026-10-03に、ヘッダー・0行状態・見た目のtemplate書式を
維持したまま、workbook metadataからローカル絶対パスを除去して再生成した。

未使用の旧`data/reference.xlsx`と`data/seawater_data_sample.xlsx`は、2026-10-03に両方のアプリケーション
treeから除外した。byte単位で同一の復旧用copyは、アプリケーションおよび公開release tree外のローカル履歴
archiveだけに保管する。wheel、GitHub Release、Zenodoの入力にはしない。

## 資産・workbookごとの確認結果

| 資産・workbook | 確認した根拠 | 公開時に維持する事項 | 現時点のパッケージ化判断 |
|---|---|---|---|
| `local_data/user_data.xlsx` | プロジェクト作成のゼロ値テンプレート。2026-10-03にヘッダー、0行状態、見た目の書式、ローカル絶対パスmetadataがないことを確認した。 | 公開同期時にゼロ値状態とmetadata安全なworkbook形式を維持する。 | サンプルとして追跡・同梱可能。 |
| Natural Earth陸域・派生海岸線CSV | Natural Earthはデータをパブリックドメインとしている。Natural Earth 50m landは別文書で記録し、50m／110m CSVの中間ファイル、出力チェックサム、数値対応を2026-10-03に検証した。 | credit、中間ファイル記録、出力チェックサムを維持する。元のraw downloadチェックサムは残っていない。 | 記録済みのパブリックドメイン出典に基づき含める。 |
| 派生`bathymetry/GEBCO_2025_6min.nc` | GEBCOはGridをパブリックドメインとし、帰属・免責・非航法条件の下で再配布を認めている。NetCDF履歴、保持する生成スクリプト、格子寸法、出力チェックサムを2026-10-03に検証した。 | GEBCO 2025引用、免責、非航法注意、生成スクリプト、出力チェックサムを維持する。元入力Gridのチェックサムは残っていない。 | 記録済みのGEBCO出典に基づき含める。 |
| `01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx` | 論文DOI／引用、分析記録、現行行数、出力チェックサムを記録している。 | 利用可能な出典位置、版・取得日、引用、変換記録を維持する。 | 記録済み学術利用方針により現行packageへ含める。 |
| `11_AROUND_JAPAN_PUB_20260305.xlsx` | 4つの地域出典label、引用、現行行数、出力チェックサムを記録している。 | 寄与する記録の行／出典対応、引用、変換を維持する。 | 記録済み学術利用方針により現行packageへ含める。 |
| `71_GLOBA_NASA_20260226.xlsx` | `data_text/NASA_references.md`に、GISSの参照URL、database v1.22の引用、取得元URL、アクセス日**2026-03-01**が記録されている。`external_dataset_workbook_notes_Japanese.md`には、25,514行の照合、共通スキーマラベル、`Transect = Nasa_database`を記録した。 | 出典・版・取得日、`Transect`の規約、変更記録を維持する。明示的な制限、査読者からの指摘、権利者からの要請があった場合にのみ見直す。 | 引用を伴う現在の学術利用としての配布を維持し、プロジェクト所有データとは表現しない。 |
| `71_GLOBAL_Atwood_et_al_2026_v02.xlsx` | `data_text/CoralHydro2_references.md`に、NCEI study URL、プロジェクトDOI、Atwood et al.の引用、アクセス日**2026-03-16**が記録されている。`external_dataset_workbook_notes_Japanese.md`には、18,598行の照合、元列の対応、2026-10-06の短縮reference補完、`Transect`、日付形式表現を記録した。 | DOI、取得元・取得日、求められる引用、変更記録を維持する。study固有の明示的な制限が見つかった場合に確認する。 | 引用を伴う現在の学術利用としての配布を維持し、プロジェクト所有データとは表現しない。 |
| `72_GLOBAL_RECENT_REPORTS_20260302.xlsx` | 現在はSakamoto et al. (2022) labelの35行からなるプロジェクト統合で、出力チェックサムを記録している。 | 報告由来の記録について、行／出典対応、引用、変換を維持する。 | 記録済み学術利用方針により現行packageへ含める。 |

## T–S Stage 2への科学的な含意

NASA GISSの文書は、水温が現場水温またはポテンシャル水温のどちらの場合もあり、不足情報の解決には元文献が必要となる場合があると注意している。従って、現行のglobal workbookを、SA/CT変換を一律自動適用できるTEOS-10準備済みデータとは扱えない。Stage 2の計算では、塩分種別、水温種別、圧力／深度、緯度・経度について行ごとの宣言を必要とし、検証済みメタデータがない行は明示的にスキップまたは近似扱いとする。

## 規約確認の結果（2026-09-25）

**NASA GISS。** データベースページはv1.22の指定引用と研究利用のためのダウンロードを示し、独立した出典から統合したコレクションであると説明する。本プロジェクトでは通常の学術利用として、指定引用、取得元URL、取得日、変換記録を維持する。公開リリースおよび論文投稿まで追加連絡は行わない。権利者・査読者からの指摘、または明示的な制限が生じた場合には、該当記録を速やかに見直す。

**判明しているNASAの選択規約。** 元データベースには、アプリのサブデータセット選択に使えるtransectメタデータがない。このため、NASA全行に`Transect = Nasa_database`を設定している。これは元データ由来の海洋学的transectではなく、アプリ側のグループ化ラベルであり、今後のデータ文書でもそのように説明する。

**PAGES / NCEI。** コレクションには公開study page、データセットDOI、指定引用がある。NCEI一般方針は連邦作成データと外部提供データを区別し、ESSD論文のCC BY 4.0は論文のライセンスである。本プロジェクトでは、完全な来歴を維持した既存の引用付きworkbook配布を継続し、公開リリースおよび論文投稿まで追加連絡は行わない。study固有の明示的な制限または要請があれば、それを記録して対応する。

## 将来の追加・変更時に整えるworkbook／出典ごとの根拠一式

新しい出典を追加する場合、または現在同梱するworkbookを置換する場合は、各出典について人間・機械の双方が読める小さな記録を整える。

1. データセット名、所有者／公開機関、正規landing URL、DOIまたは識別子。
2. データ版、取得日、元ファイル名・チェックサム、ローカル出力チェックサム。
3. 表示されている場合のライセンスまたは規約、必要な帰属、判明している制限。
4. 使用した入力行・列、全ての変換、フィルター、単位変換、手作業の補正。
5. 必須の学術引用と謝辞文。
6. 記録済みの学術利用判断と、関連する制限または見直し経路。

## 推奨する公開手順

1. プロジェクト所有・地域統合workbookについて、この根拠一式を維持し、workbookを変更する場合または新規出典を追加する場合に出典単位の詳細を補完する。
2. 第三者データごとにDOI／出典、取得日、引用、変換を維持する。公開リリースおよび論文投稿まで追加問い合わせは行わず、具体的な問題が生じた場合にのみ対応する。
3. 選択した全データの学術利用package範囲を維持する。現行の全`dataset/*.xlsx` workbookを含め、出典ごとの引用・来歴記録を維持する。
4. このモードを`pyproject.toml`のpackage data宣言、READMEの導入説明、更新履歴、Zenodoメタデータ、将来のJOSSアーカイブに記録する。

## 確認した一次情報

- NASA GISS, *Global Seawater Oxygen-18 Database*: <https://data.giss.nasa.gov/o18data/>.
- NASA, *Science Data Licenses*: <https://science.data.nasa.gov/about/license>.
- NOAA NCEI, *CoralHydro2k directory*: <https://www.ncei.noaa.gov/pub/data/paleo/coral/coralhydro2k/>.
- NOAA NCEI, *Open Data Policy*: <https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf>.
- Atwood et al. (2026), *PAGES CoralHydro2k Seawater δ18O Database*: <https://doi.org/10.5194/essd-18-1921-2026>.
- Natural Earth, *Terms of Use*: <https://www.naturalearthdata.com/about/terms-of-use/>.
- GEBCO, *Grid terms of use*: <https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use>.
