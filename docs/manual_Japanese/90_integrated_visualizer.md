# Integrated Visualizer beta

## このページでできること

既存可視化ページ、共通フィルタ、ユーザーデータ比較、簡易チェックを統合するための beta ページです。

## 基本操作

1. Workflow mode を選ぶ。
2. 必要に応じてデータをアップロードする。
3. 共通フィルタを適用する。
4. 地図、T-S図、塩分-d18O図、Custom plot、Summary、Quality Flags を確認する。
5. 必要に応じて概要CSVや品質フラグ表をダウンロードする。

## 主な設定

- Workflow mode
- Full existing page workflows
- Shared-filter beta workflow
- Uploaded data overlay controls
- Marker style controls
- Map controls
- Quality Flags tab

## 出力

- 統合マップ
- T-S図
- 塩分-d18O図
- Custom 2D/3D quick views
- アップロードデータ重ね描き
- 参照データ・アップロードデータの品質概要
- ダウンロード可能なCSV概要

## 注意点

- このページは将来の統合構想の主な試験場です。
- アップロードデータは Streamlit セッション中のメモリ上のみで扱う方針です。
- 一部のワークフローは、互換性のため既存ページファイルを呼び出します。
- 個別可視化ページは正式な主要ワークフローとして残します。以前に独立User Data Validatorへ移す予定だったアップロード起点の検証、既存データとの比較、簡易診断は、公開ページ **User Data Check & Quick Visualizer** が担当します。
- Full existing page互換モードは移行中の一時機能であり、対象ページのアップロード重ね描画完了後に廃止します。
- Integrated Visualizerは移行中は利用可能な状態を保ち、移行完了後は非公開の開発アーカイブとして残すことができます。
- 決定済みの設計と移行手順は [Integrated Visualizer運用・移行方針](../integrated_visualizer_strategy_Japanese.md) を参照してください。
