# 第三者データ・資産に関する通知

[English version](THIRD_PARTY_NOTICES.md)

この通知は、MITライセンスのEnvGeo-Seawaterソースコードと、同梱する第三者データ・
地理空間資産を区別するための簡潔な公開用記録です。出典ごとの記録や最新の利用条件に
代わるものではありません。

## プロジェクトのソースコード

EnvGeo-Seawaterのソースコードには、repositoryの[`LICENSE`](../LICENSE)（MIT）が適用されます。
このライセンスは第三者データ・地図資産・派生記録の所有権や再配布権を移転または新設するものではありません。

## 同梱データセット

5つの`dataset/*.xlsx` workbookは、プロジェクトが記録した学術利用方針に基づいて同梱します。
出典引用、版または取得日、プロジェクト側の変換記録を維持し、第三者記録をプロジェクト所有とは記載しません。
ファイル別の出典・再検討条件は[`dataset_redistribution_audit_Japanese.md`](dataset_redistribution_audit_Japanese.md)、
[`provenance_inventory_Japanese.md`](provenance_inventory_Japanese.md)、
[`external_dataset_workbook_notes_Japanese.md`](external_dataset_workbook_notes_Japanese.md)を参照してください。

追跡対象の`local_data/user_data.xlsx`は、プロジェクトが作成したゼロ行の公開templateであり、測定データではありません。

## 地理空間資産

- 海岸線CSVとNatural Earth 50m land polygonは、パブリックドメインのNatural Earthデータに由来します。
  “Made with Natural Earth (https://www.naturalearthdata.com/)”の帰属表示を維持してください。
  [`geospatial_assets_Japanese.md`](geospatial_assets_Japanese.md)および
  [`../coastline/natural_earth_50m_land/LICENSE_OR_SOURCE_Japanese.md`](../coastline/natural_earth_50m_land/LICENSE_OR_SOURCE_Japanese.md)を参照してください。
- `bathymetry/GEBCO_2025_6min.nc`は、プロジェクトが間引きしたGEBCO 2025 Gridです。鉛直断面解析の
  文脈用であり、航海用途には使えません。GEBCOに必要な帰属表示と免責を維持してください。詳細は
  [`geospatial_assets_Japanese.md`](geospatial_assets_Japanese.md)を参照してください。

## ソフトウェア依存関係

依存関係はPython packageとは別に、それぞれのライセンスでinstallされます。手法上重要なsoftwareと
データ出典は、README、`paper.bib`、出典ごとの記録に従って引用してください。
