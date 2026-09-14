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
- 最終的には Integrated Visualizer を主要入口とし、個別ページを高度利用・保守用に残す構成も検討しています。

