# 同梱資産・データ来歴一覧

**状態:** 2026-09-25に出典根拠を監査し、2026-09-30に同梱workbookの
スナップショット、2026-10-03に派生資産の記録、2026-10-05にデータセット間重複の初期スクリーニング、2026-10-06にCoralHydro2k短縮referenceの補完を記録した公開準備用一覧。
ここでは同梱物と不足している根拠を明確にする。「公開されている」ことを、研究ソフトウェアの再配布許可の根拠とは扱わない。

| 資産・データ | 使用箇所 | 出典／派生記録 | 再配布状態 | 必要な次の確認 |
|---|---|---|---|---|
| `coastline/world_coastline_coordinates_50m.csv`・`110m.csv` | オフラインPlotly地図、Cartopy海岸線 | Natural Earth coastline v4.1.0から、保持する50m／110m Excel中間ファイルを経て派生。数値対応と出力チェックサムを2026-10-03に検証済み。 | Natural Earthはパブリックドメイン。出典表示を維持する。 | 中間ファイルとの対応と現行出力チェックサムを維持する。元のraw download取得時チェックサムは残っていない。 |
| `coastline/natural_earth_50m_land/` | Page 32陸域マスク | Natural Earth 50m land。詳細は`LICENSE_OR_SOURCE.md`。 | 来歴・帰属を維持すれば再配布可能。 | 「Made with Natural Earth」とチェックサムを維持する。 |
| `bathymetry/GEBCO_2025_6min.nc` | Page 53の任意測深 | GEBCO_2025の6 arc-minute間引き。NetCDF履歴と保持する生成スクリプトはstride 24を記録し、出力チェックサムを2026-10-03に検証済み。 | GEBCO Gridは帰属・免責・非航法条件付きのパブリックドメイン。 | GEBCO 2025引用、生成スクリプト、出力チェックサム、非航法注意を維持する。元入力Gridの取得時チェックサムは残っていない。 |
| `local_data/user_data.xlsx` | 常時読み込みUser Excel | プロジェクト作成のゼロ値公開テンプレート。2026-10-03にローカル絶対パスmetadataを除去して再生成し、ヘッダー、0行状態、見た目の書式を維持した。 | 公開配布を承認済み。 | 公開同期前にゼロ値かつmetadata安全なtemplateを維持する。個人データは外部パスまたは未commitのローカル編集を使う。 |
| `dataset/01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx` | Japan Sea／Global loader | Kodama et al. (2024)。DOIと分析情報を`data_text/main_references.md`に、現行workbook fingerprintを下表に記録。 | 現行の学術利用packageへ含める。 | 正規出典、引用、利用可能な版・取得日、変換記録を維持する。 |
| `dataset/11_AROUND_JAPAN_PUB_20260305.xlsx` | Around Japan／Global loader | Yamamoto、Sakamoto、Kodaira、Horikawaの地域統合。4つの出典labelと現行workbook fingerprintを検証済み。 | 現行の学術利用packageへ含める。 | 寄与する各出典の行／出典対応、引用、変換を維持する。 |
| `dataset/71_GLOBA_NASA_20260226.xlsx` | Global loader | NASA GISS Global Seawater Oxygen-18 Database。`data_text/NASA_references.md`にv1.22、取得元URL、引用、アクセス日2026-03-01を記録済み。元データ対workbookの25,514行照合と、プロジェクト側の`Transect = Nasa_database`規約も記録済み。 | 現行の学術利用としての配布を承認済み。出典を維持し、プロジェクト所有データとは表現しない。 | 既存の版・アクセス記録、引用、変換記録を維持する。明示的な制限、権利者からの要請、具体的な査読上の懸念がある場合だけ見直す。 |
| `dataset/71_GLOBAL_Atwood_et_al_2026_v02.xlsx` | Global loader | PAGES CoralHydro2k Seawater δ18O Database。`data_text/CoralHydro2_references.md`にstudy URL、DOI、引用、アクセス日2026-03-16を記録済み。元データ対workbookの18,598行照合、引用の保持、`Transect`スキーマ対応、保持されていた`Dataset citation`からプロジェクト側の短縮`reference`を3,258行補完した2026-10-06の記録もある。 | 現行の学術利用としての配布を承認済み。出典を維持し、プロジェクト所有データとは表現しない。 | 次の公開同期では、このv02 snapshotのpackage checksum表とrelease記録を更新する。 |
| `dataset/72_GLOBAL_RECENT_REPORTS_20260302.xlsx` | Global loader | 現在はSakamoto et al. (2022)の35行を含むプロジェクト統合。出典引用と現行workbook fingerprintを検証済み。 | 現行の学術利用packageへ含める。 | 報告由来の記録について、出典対応、引用、変換を維持する。 |
| 退避済み`d18O_upload_data_tmp_seawater.xlsx` | アプリでは未使用。 | コード、テスト、公開クローンから参照されない旧作業用workbook。 | アプリおよび将来のpackage dataには含めない。 | アプリ外の上位ワークスペースのアーカイブ（`過去のパーツ/`）へ保管し、再利用を提案する場合だけ別途レビューする。 |
| `data/`メディア・`data_text/`文書 | Home、アプリ内文書 | `d18O_all.mp4`はGMT作成と表示するHome animationである。`sites_20230515.gif`と`year_20230517.gif`はアプリ／文書で使うプロジェクト地図visualizationである。3件はpackage化する。今回の確認では、別ライセンス記録が必要な第三者メディアは確認されなかった。`data_text/*.md`も同梱する。`data/`にあった未使用legacy spreadsheetは、2026-10-03にアプリケーションおよび公開release tree外のローカル履歴archiveへ移動した。 | プロジェクト文書／メディアは同梱可能。履歴spreadsheetは公開releaseから除外する。 | 今後プロジェクト作成物でないメディアを追加する場合は、作成者・出典・再利用条件を記録する。archive済みspreadsheetを戻す場合は別途reviewする。 |

## データセット間重複の初期スクリーニング（2026-10-05）

これは読取り専用の候補抽出であり、重複除外処理ではない。同梱workbook、出典の帰属、各候補対が同じ物理観測であるかどうかの判断は変更していない。現在のNASAおよびCoralHydro2k workbookでは、試料ID照合に十分な共通の日付フィールドが得られず、近接したプロファイル観測から複数の候補対が生じる場合もある。

| 比較 | 確認用の閾値 | 厳しい閾値 | 初期スクリーニング結果 |
|---|---|---|---|
| NASA GISS × PAGES CoralHydro2k | 水平距離≤30 km、深度差≤10 m、塩分差≤0.20、δ18O差≤0.10‰ | ≤15 km、≤3 m、≤0.10、≤0.05‰ | 確認用の閾値で4,068組、厳しい閾値で2,463組の候補対。対の数は一意の重複観測数ではない。 |
| 日本周辺統合 × NASA GISS | 上記確認用閾値と同じ | 上記厳しい閾値と同じ | 厳しい閾値で139組、419行中131行の日本周辺記録を含む。日本側は主に`Yamamoto et al. (2001)` / `PI=KAWAI`であり、NASA側にはYamamoto et al. (2001)および(2002)のラベルがある。 |
| 日本周辺統合 × PAGES CoralHydro2k | 上記確認用閾値と同じ | 上記厳しい閾値と同じ | この初期スクリーニングでは、いずれの閾値でも候補対なし。 |

現在のアプリは、行レベルの監査根拠（出典行ID、座標、変数差、reference metadata）を出力し、すべての元出典記録と必要な引用を変更せず利用可能なまま保持する。共通フィルターの表示スクリーンは任意であり、既定では全行を保持する。推奨モードは、丸め幅との整合が取れたStrong候補のうち、決定論的に一対一対応した候補だけを表示から外す。残す側はメタデータの充実度で選び、同点の場合は文書化した一定の優先順を使う。より広いStrong候補モードは感度確認だけを目的とし、重複確定の結果ではない。

### v1.3.5設計に向けた時間条件付き候補確認

次の読取り専用の確認では、有効な採水年・月が双方で一致すること、水平距離≤15 km、塩分差≤0.1、δ18O差≤0.1‰を必要とした。`**`などの非数値・保留月は除外している。NASA workbookには利用可能な採水日がないため、これは同日確認ではなく同一年同月の条件である。件数は候補対であり、括弧内は左／右の異なる出典行数である。

| 比較 | 深度差≤1 m | ≤3 m | ≤10 m | ≤50 m |
|---|---:|---:|---:|---:|
| NASA GISS × PAGES CoralHydro2k | 1,730組（1,644 / 1,654） | 1,764組（1,670 / 1,680） | 1,897組（1,704 / 1,712） | 2,389組（1,805 / 1,777） |
| 日本周辺統合 × NASA GISS | 55組（55 / 55） | 55組（55 / 55） | 62組（55 / 55） | 91組（55 / 55） |

日本周辺--NASAの55組は、`Yamamoto et al. (2001)` / `PI=KAWAI`に対応する1996年9月の鉛直プロファイルである。日本側とNASA側の座標は丸めのため最大約11 km異なるが、深度プロファイルの並びと塩分・δ18Oは記録精度の範囲で一致する。非常に強い候補群だが、出典単位の対応を記録するまでは監査上の候補として扱う。

深度許容幅を広げると、単一の鉛直プロファイル観測から複数の候補対が生じる。したがって≤10 m・≤50 mの列は確認・マーク表示用であり、抑制する行数として解釈してはならない。現在のUIは「全データ表示」を既定とし、広いStrong候補モードは感度確認用、推奨モードは決定論的な一対一・丸め整合候補だけに限定する。

## 同梱workbookの検証済みスナップショット（2026-09-30、履歴上の基準）

読取り専用の確認により、次の5ファイルが正規作業フォルダと安定版`seawater_map` cloneの両方にあり、
SHA-256が一致することを確認した。これらは現在の`dataset/*.xlsx` package-data規則で選択される全ファイルである。
本監査でworkbook内容は変更していない。

| 同梱ファイル | Worksheet | 行数 | 列数 | SHA-256 |
|---|---|---:|---:|---|
| `01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx` | `Kodam_et_al_2024` | 2,222 | 22 | `8184016fe53fb3b537b5ca60061b2e3d798f69b63534ffea6d6a11d9904d2962` |
| `11_AROUND_JAPAN_PUB_20260305.xlsx` | `for_streamlit_YSKH_20260227` | 419 | 22 | `ac49cf552e88caff1d294bbcbff978b8ea72faa31c7fc1f698456ea978663dd0` |
| `71_GLOBAL_Atwood_et_al_2026_v02.xlsx` | `CoralHydro2k_SW_1_0_0_20260303` | 18,598 | 58 | `50c7cbc27140edabfac42fa8004725228054931671035285073354b829939514` |
| `71_GLOBA_NASA_20260226.xlsx` | `NASA_20260227` | 25,514 | 22 | `13cccbffa3948a2570fd7c6faa342a888d1e14d3bb076f48d520564a25c85840` |
| `72_GLOBAL_RECENT_REPORTS_20260302.xlsx` | `20260303` | 35 | 23 | `d8027747739247601bfbc9ae8ffc2c9b9036edfaaa83b7f84972e28b9f84a565` |

追跡する`local_data/user_data.xlsx`公開sampleも、両方の場所で0行・22列のtemplateであることを確認した。CoralHydro2kの行は、正規作業フォルダにおけるv02 workbookと2026-10-06時点のchecksumを記録する。まだ公開cloneへ同期していない。上記workbookを変更するreleaseでは、このスナップショット、出典単位の来歴記録、release checksum記録を併せて更新する。

## 派生資産の検証済みスナップショット（2026-10-03）

次の確認は読取り専用であり、元のsource downloadを再構成・置換するものではない。プロジェクト内に
保持されている再現可能な対応関係と、配布する出力の正確なfingerprintを記録する。

| 同梱資産 | 検証した派生記録 | SHA-256 |
|---|---|---|
| `coastline/world_coastline_coordinates_50m.csv` | 保持する`world_coastline_coordinates_50m.xlsx`中間ファイルと数値的に一致（61,844行、NaN位置一致、最大浮動小数差`1.42e-14`）。中間ファイルは別の海岸線作業フォルダでNatural Earth 50m coastline v4.1.0から派生。 | `c3d7bee4fb696b011fa34bb13bed0c335c5250eeaf37d8739d77d29a27fe385c` |
| `coastline/world_coastline_coordinates_110m.csv` | 保持する`world_coastline_coordinates_110m.xlsx`中間ファイルと数値的に一致（5,261行、NaN位置一致、最大浮動小数差`1.42e-14`）。中間ファイルは別の海岸線作業フォルダでNatural Earth 110m coastline v4.1.0から派生。 | `a31df3aeee9dc4195af35a31b0605fdb572c7c7dd7cde17f773c9438f5ec7f3f` |
| `bathymetry/GEBCO_2025_6min.nc` | NetCDF履歴は`GEBCO_2025.nc`、6.0 arc-minute、stride 24を示し、保持する`make_lightweight_gebco.py`が同じ派生を実装する。格子は緯度1,800×経度3,600セル。 | `0afdf1d0e023b0529c56b69a2684e505c7e2ea28d78a3af8814817af59b09030` |

これらの派生資産について、元source downloadのチェックサムと取得日は残っていない。この制約を明示し、
ここでの記録は配布物、その保持する派生経路、公開された上流出典を特定するものとする。過去のraw downloadを
bit-for-bitで再構成できると主張するものではない。

## パッケージ化の判断点

海岸線、Natural Earth、GEBCO、ゼロ値User Excelには、公開用の出典・fingerprint記録がある。現在のwheelは、選択した学術利用packageの範囲として現行の全`dataset/*.xlsx` workbookを含める。各workbookでは出典帰属を維持し、同梱は所有関係を移転するものでも、将来追加するデータセットの取扱いを決めるものでもない。出典単位の来歴記録を維持・整備し、新たな明示的制限、権利者からの要請、または具体的な査読上の懸念が生じた場合だけ見直す。この一覧は、T–S Stage 2のデータ来歴監査の出発点でもある。
