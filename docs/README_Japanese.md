# EnvGeo-Seawater 文書一覧

このディレクトリには、トップレベルREADMEより詳しいプロジェクト文書を収録します。
有用な履歴・開発記録も残しますが、Releaseの承認や安定版の公開範囲を示すものではありません。

[English version](README.md)

利用者向けの操作案内は、サポート対象の安定版workflowだけへ案内する
[利用ガイド](index_Japanese.md)から始める。このページは、公開済みの[GitHub Pages website](https://envgeo.github.io/seawater_map/)の入口でもある。

## 現在の文書

- `ROADMAP.md` / `ROADMAP_Japanese.md`（安定版repositoryのみ）
  `seawater_map`で公開する簡潔な英日ロードマップ。詳細な
  `TODO.md` / `TODO_Japanese.md`の計画記録は開発用repositoryだけに保持し、安定版には意図して含めない。

- `version_update_checklist.md` / `version_update_checklist_Japanese.md`
  版メタデータ、データ来歴、文書、clone同期、検証、GitHub Release、Zenodo DOI更新を確認する再利用用チェックリスト。

- `release_checklist.md` / `release_checklist_Japanese.md`
  ローカル試験、Streamlit deployment、GitHub Release準備、Zenodo／DOI archiveの確認項目。

- `post_release_maintenance_log.md` / `post_release_maintenance_log_Japanese.md`
  tag済みRelease後のJOSS以外の保守作業を、次回Release前に確認するための開発用記録。安定版public repositoryには意図して含めない。

- `testing.md` / `testing_Japanese.md`
  現在のpytest suite、その範囲、限界、今後の拡充方針の公開向け説明。

- `streamlit_migration.md` / `streamlit_migration_Japanese.md`
  Python 3.12、Streamlit 1.63、Plotly互換性、環境比較、視覚確認に関する移行記録。

- `integrated_visualizer_strategy.md` / `integrated_visualizer_strategy_Japanese.md`
  個別ページ、共通ユーザーデータupload、Integrated Visualizerに関する構成・移行方針。
  Page 90のワークフローは開発履歴であり、安定版の公開範囲には含めない。

- `development_notes.md` / `development_notes_Japanese.md`
  オフライン地図設計、ユーザーデータ構成、地理空間asset、保留作業、安全な整理条件に関する1.3.3時点の技術的な履歴。現在のRelease checklistではない。

- `citation_and_license_plan.md` / `citation_and_license_plan_Japanese.md`
  プロジェクトの引用、第三者ソフトウェア、データ・地図asset、帰属表示、再配布条件、Zenodo、JOSSに関する公開準備チェック。

- `code_guide.md` / `code_guide_Japanese.md`
  中心モジュール、ページの役割、資産ディレクトリ、コード文書化の目安を示す英日案内。

- `publication_content_review_checklist.md` /
  `publication_content_review_checklist_Japanese.md`
  正規作業フォルダ、安定版範囲、コード、文書、データ、資産、除外対象をファイル単位で
  確認する公開前コンテンツチェックリスト。

- `sprint3_distribution_design_and_acceptance.md` /
  `sprint3_distribution_design_and_acceptance_Japanese.md`
  1.3.4開発／公開前repositoryの履歴上の配布設計・検証記録。「Sprint 3」の意味は文書内で説明する。

- `distribution_foundation_audit_and_plan.md` /
  `distribution_foundation_audit_and_plan_Japanese.md`
  asset解決、package化、インストール、オフライン動作、deployment検証に関するSprint 2の
  配布基盤監査の履歴記録。現在のRelease構成そのものではない。

- `wheel_proof_report.md` / `wheel_proof_report_Japanese.md`
  Sprint 2B–2Cのローカルwheel作成・インストール実証、受入結果、限界、保留した本番package判断の
  履歴記録。現行v1.3.5の検証記録は`release_checklist*.md`を参照する。

- `provenance_inventory.md` / `provenance_inventory_Japanese.md`
  同梱asset・データ、その既知の出典記録、再配布状態、公開前の確認事項をまとめた英日一覧。

- `dataset_redistribution_audit.md` / `dataset_redistribution_audit_Japanese.md`
  各データセットの公開状況と再配布根拠を分けて記録し、T–S Stage 2の科学的来歴確認も扱う監査記録。

- `external_dataset_workbook_notes.md` /
  `external_dataset_workbook_notes_Japanese.md`
  引用済みNASA GISS・PAGES workbookの現状、出典参照、アプリ内だけのNASA `Transect`規約。

- `offline_operation_log.md` / `offline_operation_log_Japanese.md`
  オフライン／通信不安定時の地図動作、自己完結HTML出力、既知の制約、検証範囲。

- `manual/` / `manual_Japanese/`
  安定版の公開ワークフローを対象とした、ページ別の英日ユーザーマニュアル。公開済みの静的
  ドキュメントサイトの内容正本であり、研究室Webサイトの案内にも再利用できます。

## JOSS投稿準備

現在のJOSS向け資料は`paper.md`、`paper.bib`、`CITATION.cff`、
`release_checklist*.md`で管理します。別のJOSS checklistは、必要になった場合に追加できますが、
現時点の投稿準備記録には必須ではありません。
