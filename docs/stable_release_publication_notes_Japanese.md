# 安定版公開時の注意事項

## 目的

`seawater_map`は、EnvGeo-Seawaterの正式な安定版repositoryです。GitHub Release、Zenodo archive、
JOSS向けsoftware recordの正規sourceとして扱います。この文書は公開範囲を短く確認するための
メモです。詳細な確認には[Release checklist](release_checklist.md)を使います。

[English version](stable_release_publication_notes.md)

## 安定版の範囲

安定版packageには、`home.py`、共通アプリmodule、runtime asset、出典を記録したdatasetと、
次の10個のStreamlitページを収録します。

- 03 Interactive 2Dplus Visualizer
- 04 Interactive 3D/4D Visualizer
- 05 User Data Check & Quick Visualizer
- 31 Salinity-d18O Relationship
- 32 Isotope Hydrographic Mapping
- 34 T-S Diagram
- 35 Custom Parameter Plot
- 37 Depth Profile
- 53 Vertical Section Visualizer beta
- 80 Correlation Overview archive

Page 80は、元の手書き探索workflowを残すアーカイブです。見た目の統一だけを理由に、
過去のコメントを削除したりリファクタリングしたりしません。明確な不具合、互換性、安全性、
配布要件に必要な修正だけを行います。Page 35と53はbeta workflowであることを明記して残します。

## 意図して除外するもの

開発repositoryの内容を、このrepositoryへ一括コピーしません。安定版packageと公開deploymentには、
次を含めません。

- `pages/90_Integrated_Visualizer_beta.py`と`pages/91_EnvGeo_Earthquake.py`
- `pages/99_Environment_Check.py`（ローカルでは`envgeo-seawater-check`を使う）
- `Claude outputs/`、内部reviewメモ、private correspondence、local user-data path
- build directory、wheel file、仮想環境、cache、`.DS_Store`、report、screenshotなどの生成物

CIのwheel確認では、Page 90、91、99が存在しないことを検証します。

## データと研究者個人のファイル

現行の学術利用packageには、`dataset/*.xlsx`の各workbookを、出典引用、来歴、出典からworkbookへの
変換記録とともに収録します。同梱は所有権を移転せず、将来のdatasetの扱いを自動的に決めるものでも
ありません。[データセット再配布根拠の監査](dataset_redistribution_audit_Japanese.md)と
[同梱資産・データ来歴一覧](provenance_inventory_Japanese.md)を参照してください。

研究者自身の測定値はcommitしません。同梱する`local_data/user_data.xlsx`はゼロ値の公開sampleです。
private local dataには`ENVGEO_LOCAL_USER_DATA_PATH`を使い、その値を公開文書、log、screenshot、
CI出力に残しません。

## Release／Zenodo archiveの前に行うこと

1. 対象commitのclean checkoutから始め、別の作業folderから無差別にファイルをコピーせず、
   staged changeを一つずつ確認する。
2. 最新CIがPython 3.10と3.12の両方で成功し、test、wheel作成、隔離wheel導入を完了したことを確認する。
3. Release checklistに従い、Home、Page 05、Mapping、T-S Diagram、Depth Profile、Vertical Sectionを
   含む安定版のsmoke testを行う。
4. version、Git tag、commit ID、Python版、解決済み依存関係記録、test結果、wheel SHA-256が、
   同一source revisionに対応することを確認する。CI wheel artifactは確認根拠だけであり、
   Release記録用wheelはclean tagged checkoutから再作成する。
5. そのtagからGitHub Release、続いてZenodo archiveを作成する。発行後にDOIをcitation資料へ追加する。

## 変更管理

Releaseに関係する判断は、公開可能な`docs/`記録または利用者向け更新ログへ残します。内部連絡・
作業調整資料は公開repositoryの外側に置きます。技術的なpackage化の変更に付随して、データ方針、
ページ範囲、Release境界を変更しません。
