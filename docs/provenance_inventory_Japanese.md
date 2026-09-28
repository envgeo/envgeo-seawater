# 同梱資産・データ来歴一覧

**状態:** 2026-09-25時点の公開準備用の作業一覧。  
ここでは同梱物と不足している根拠を明確にする。「公開されている」ことを、研究ソフトウェアの再配布許可の根拠とは扱わない。

| 資産・データ | 使用箇所 | 出典／派生記録 | 再配布状態 | 必要な次の確認 |
|---|---|---|---|---|
| `coastline/world_coastline_coordinates_50m.csv`・`110m.csv` | オフラインPlotly地図、Cartopy海岸線 | Natural Earthベクタデータから派生。 | Natural Earthはパブリックドメイン。ただし派生CSV固有の出典版・チェックサム記録は未整備。 | CSV固有の取得元URL、取得日、生成手順、チェックサムを記録する。 |
| `coastline/natural_earth_50m_land/` | Page 32陸域マスク | Natural Earth 50m land。詳細は`LICENSE_OR_SOURCE.md`。 | 来歴・帰属を維持すれば再配布可能。 | 「Made with Natural Earth」とチェックサムを維持する。 |
| `data_beta/GEBCO_2025_6min.nc` | Page 53の任意測深 | GEBCO_2025を6 arc-minuteへ間引き。生成: `make_lightweight_gebco.py`。 | GEBCO Gridは帰属・免責・非航法条件付きのパブリックドメイン。 | 入力Gridの取得日・チェックサムとGEBCO 2025引用を記録する。 |
| `local_data/user_data.xlsx` | 常時読み込みUser Excel | プロジェクト作成のゼロ値公開テンプレート。 | 公開配布を承認済み。 | 公開同期前にゼロ値を維持する。個人データは外部パスまたは未commitのローカル編集を使う。 |
| `dataset/01_ECS_JAPAN_SEA_Kodam_et_al_2024.xlsx` | Japan Sea／Global loader | Kodama et al. (2024)。`data_text/main_references.md`に引用。 | 現行の学術利用packageへ含める。ファイル単位の来歴記録は整備中。 | 正規出典、引用、利用可能な版・取得日、変換記録を維持する。 |
| `dataset/11_AROUND_JAPAN_PUB_20260305.xlsx` | Around Japan／Global loader | 地域統合。Yamamoto、Sakamoto、Kodaira、Horikawa等を含む。 | 現行の学術利用packageへ含める。ファイル単位の来歴記録は整備中。 | 寄与する各出典の行／出典対応、引用、変換を維持する。 |
| `dataset/71_GLOBA_NASA_20260226.xlsx` | Global loader | NASA GISS Global Seawater Oxygen-18 Database。`data_text/NASA_references.md`にv1.22、取得元URL、引用、アクセス日2026-03-01を記録済み。元データ対workbookの25,514行照合と、プロジェクト側の`Transect = Nasa_database`規約も記録済み。 | 現行の学術利用としての配布を承認済み。出典を維持し、プロジェクト所有データとは表現しない。 | 既存の版・アクセス記録、引用、変換記録を維持する。明示的な制限、権利者からの要請、具体的な査読上の懸念がある場合だけ見直す。 |
| `dataset/71_GLOBAL_Atwood_et_al_2026.xlsx` | Global loader | PAGES CoralHydro2k Seawater δ18O Database。`data_text/CoralHydro2_references.md`にstudy URL、DOI、引用、アクセス日2026-03-16を記録済み。元データ対workbookの18,598行照合、引用の保持、短縮reference label、`Transect`スキーマ対応も記録済み。 | 現行の学術利用としての配布を承認済み。出典を維持し、プロジェクト所有データとは表現しない。 | 既存のDOI・アクセス記録、引用、変換記録を維持する。明示的な制限、権利者からの要請、具体的な査読上の懸念がある場合だけ見直す。 |
| `dataset/72_GLOBAL_RECENT_REPORTS_20260302.xlsx` | Global loader | 最近の報告のプロジェクト統合。 | 現行の学術利用packageへ含める。ファイル単位の来歴記録は整備中。 | 報告由来の記録について、出典対応、引用、変換を維持する。 |
| 退避済み`d18O_upload_data_tmp_seawater.xlsx` | アプリでは未使用。 | コード、テスト、公開クローンから参照されない旧作業用workbook。 | アプリおよび将来のpackage dataには含めない。 | アプリ外の上位ワークスペースのアーカイブ（`過去のパーツ/`）へ保管し、再利用を提案する場合だけ別途レビューする。 |
| `data/`メディア・`data_text/`文書 | Home、アプリ内文書 | プロジェクトメディア・文書。外部参照はリンクで記録。 | 第三者メディアを同梱する場合は個別確認が必要。 | プロジェクト作成物でないメディアについて、作成者・出典・再利用条件を記録する。 |

## パッケージ化の判断点

海岸線、Natural Earth、GEBCO、ゼロ値User Excelについては、次の資産ローダー検証へ進む方向性が明確である。現在のSprint 3 wheelは、選択した学術利用packageの範囲として現行の全`dataset/*.xlsx` workbookを含める。各workbookでは出典帰属を維持し、同梱は所有関係を移転するものでも、将来追加するデータセットの取扱いを決めるものでもない。出典単位の来歴記録を維持・整備し、新たな明示的制限、権利者からの要請、または具体的な査読上の懸念が生じた場合だけ見直す。この一覧は、T–S Stage 2のデータ来歴監査の出発点でもある。
