# 外部データworkbookの取扱いメモ

**状態:** 2026-09-25時点の元データ対workbook照合記録、2026-09-30時点の同梱workbook構造再確認、および2026-10-06のCoralHydro2k引用補助メタデータの補完記録。この文書は、監査用に提供されたローカル元データスナップショットと同梱workbookを比較し、確認できた差分だけを記録する。英語版は[`external_dataset_workbook_notes.md`](external_dataset_workbook_notes.md)。

## 目的

NASA GISSおよびPAGES CoralHydro2kのworkbookは、比較・可視化に用いる引用付き第三者参照データである。このメモでは、出典の引用と、アプリで利用するプロジェクト側のフィールドを分けて記録する。私的な連絡や連絡先は含めない。

## 現行同梱workbookの再確認（2026-09-30）

この読み取り専用の再確認では、現在同梱しているファイルとworkbook構造を確認した。元データスナップショットは意図的にアプリケーションツリー外で管理しているため、2026-09-25の元ファイル照合自体を繰り返したものではない。2つのworkbookのSHA-256値と安定版cloneとの関係は[`provenance_inventory_Japanese.md`](provenance_inventory_Japanese.md)に記録する。

- NASA GISS：`NASA_20260227`は25,514行・22列である。全行が`Transect = Nasa_database`であり、`Cruise`、`Station`、`remarks by TI`は空欄である。
- PAGES CoralHydro2k：`CoralHydro2k_SW_1_0_0_20260303`は18,598行・58列である。`Transect`は全行に入り93種類あり、プロジェクト側で追加した短縮`reference`は全行に入り、記録済みの共通スキーマ用プレースホルダー6列は空欄のままである。

## NASA GISS workbook

| 項目 | 現行記録 |
|---|---|
| 同梱ファイル | `dataset/71_GLOBA_NASA_20260226.xlsx` |
| worksheet | `NASA_20260227` |
| 出典／引用 | `data_text/NASA_references.md`：Global Seawater Oxygen-18 Database v1.22、GISS URL、アクセス日2026-03-01。 |
| 元データスナップショット | `geto18.cgitab.txt`（アプリケーションツリー外で保持するタブ区切りの元ダウンロード） |
| 現行データ行数 | 25,514（元データ行数と一致） |
| 共通アプリ列 | `Transect`、位置・深度・日時、`Temperature_degC`、`Salinity`、`d18O`、`dD`、`reference`、notes系の列。 |
| 判明しているアプリ側規約 | 全行に`Transect = Nasa_database`を設定する。NASAにはアプリのサブデータセット選択に使えるtransectメタデータがないためである。これは元データ由来の海洋学的transectではなく、アプリ側のグループ化ラベルである。 |

元データの11列は共通スキーマの列名へ対応付けて保持している。
`Longitude` → `Longitude_degE`、`Latitude` → `Latitude_degN`、`Depth` →
`Depth_m`、`pTemperature` → `Temperature_degC`、`Reference` → `reference`であり、
`Salinity`、`d18O`、`dD`、`Year`、`Month`、`Notes`は同名で保持する。25,514行すべてについて、元データと同梱workbookの対応値は数値的に一致するか、元の`**`欠損値マーカーをそのまま保持していた。プロジェクト側の共通スキーマ列は、`Transect`を除き空欄である。この照合では観測値の差異は見つからなかった。

`Cruise`および`Station`は共通スキーマの列として存在するが、現行NASA workbookでは値を持たない。これは現行workbookについての記録であり、元データベースについての主張ではない。

## PAGES CoralHydro2k workbook

| 項目 | 現行記録 |
|---|---|
| 同梱ファイル | `dataset/71_GLOBAL_Atwood_et_al_2026_v02.xlsx` |
| worksheet | `CoralHydro2k_SW_1_0_0_20260303` |
| 出典／引用 | `data_text/CoralHydro2_references.md`：PAGES CoralHydro2k Seawater δ18O Database、NCEI study URL、データセットDOI、Atwood et al. (2026)、アクセス日2026-03-16。 |
| 元データスナップショット | `CoralHydro2k_Seawater_1_0_0.xlsx`（アプリケーションツリー外で保持） |
| 現行データ行数 | 18,598（元データ行数と一致） |
| 共通アプリ列 | `Cruise`、`Station`、`Transect`、位置・深度・日時、`Temperature_degC`、`Salinity`、`d18O`、`dD`、`reference`、および詳細な出典来歴列。 |
| 現行のグループ化列 | `Transect`は全行で値を持ち、このworkbook内には93種類の値がある。元データの`Site name or geographic area`を共通スキーマ名へ対応付けた列である。 |

元データの50列は保持または共通スキーマ名へ対応付けている。例として、`Cruise ID` → `Cruise`、`Station ID` → `Station`、collection year/month/day → `Year`/`Month`/`Day`、位置・深度・水温・塩分・同位体の列は対応する共通列、`Publication citation` → `reference_full`である。現行workbookには、空欄の共通スキーマ用プレースホルダー（`Date`、`TargetDepth_m`、`Bottle`、`d13C`、`PI`、`Vertical`）と、アプリ内の表示・選択に使うプロジェクト側の短縮`reference`列がある。短縮表記は、単著を`Surname (year)`、複数著者を`Surname et al. (year)`とし、団体著者は団体名を著者表記として残す。これは表示・分類用の補助であり、完全な引用を置き換えない。

2026-10-06に、短縮`reference`が空欄だった3,258行を補完した。これらの行では`Publication citation`／`reference_full`が空欄でも、元データの`Dataset citation`には完全な出典情報が残っていた。観測値および出典来歴列は変更せず、7つの出典ごとに短縮表記を確認して追加した：`GEOTRACES Intermediate Data Product Group (2021)`、`Lamb et al. (2014)`、`Dickson (2016)`、`Schmidt et al. (1999)`、`Lo Monaco et al. (2013)`、`Henley et al. (2020)`、`Stoll et al. (2013)`。複合姓や接頭辞を持つ姓は一般的な自動判定に任せない。完全な出典は引き続き`reference_full`または`Dataset citation`に保持し、プロジェクト側の`reference`は簡潔な表示・グループ化だけに使う。

編集済みworkbookは`71_GLOBAL_Atwood_et_al_2026_v02.xlsx`として版を示すファイル名にした。次回の公開repository同期では、loader、ページ側catalogue、来歴文書、package checksum表、release notesを一体として更新する。

確認できたメタデータ形式上の違いは2種類の日付形式に限られる。元データの`Water isotope analysis date` 5,969セルと`Station ID` 9セルは、同梱workbookではExcel日付シリアル値として保存されている。この監査では元の日付を編集せず、形式変更の理由も推測しない。その他の対応付けた元データ列は、数値表記上等価な差を除いて同梱workbookと一致した。

## 文書化の境界

これらのworkbookには、プロジェクト側の共通スキーマ用ラベル・列がある。本メモは、今回提供された2つの元データスナップショットとの間で確認できた対応だけを記録する。他のデータ版・出典へ根拠なく拡張してはならない。将来の再現性のため、DOI／取得元、アクセス日、引用、元データスナップショット、この対応記録を維持する。

## 現行packageの判断

両workbookは、学術利用のための現行`dataset/*.xlsx` package dataへ引き続き含める。照合結果から、観測レコードは保持されていると説明できる。記録した差分は、共通スキーマまたはグループ化ラベル、空欄プレースホルダー、表示用の短縮引用、上記のExcel日付形式表現であり、観測値を置き換えるデータではない。出典引用とこの変更記録を維持し、いずれのworkbookもプロジェクト所有データとは扱わない。
