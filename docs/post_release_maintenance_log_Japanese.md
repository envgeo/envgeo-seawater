# Release後の保守作業ログ

この開発用ログには、v1.3.5公開後に完了した保守作業を記録します。JOSS原稿関係の作業は含めません。
ここへの記録は、tag済みv1.3.5 GitHub Release、Zenodo archive、PyPI配布物を変更するものではありません。
次回版へ採用する項目は、改めて確認したうえでrelease noteへ反映します。

[English version](post_release_maintenance_log.md)

## v1.3.5後（2026年10月）

- Home動画assetから進捗表示を除去し、見た目を維持したより小さい符号化動画へ更新した。
- ブラウザuploadのサイドバー説明を明確化した。ブラウザuploadはsession単位であり、アプリケーションが
  localまたはserver storageへ保存するものではない。
- Custom Parameter Plotの軸、title、captionで、\(\delta^{18}\mathrm{O}\)の同位体番号を上付き表記へ修正した。
- 開発コピーのroot package `__init__.__version__`をv1.3.5のsource metadataと一致させ、将来のReleaseで
  この値とインストール済みpackage metadataを確認する項目をchecklistへ追加した。
- Custom Parameter Plot、T–S Diagram、Depth Profile、Correlation Overviewのmatplotlib legendを、内容がある場合だけ作成するようにした。
  空のlegend警告を避け、Custom Plotの大規模データで遅い自動legend配置検索を使わないようにした。

## 記録ルール

tag済みRelease後のJOSS以外の変更は、利用者への影響、対象ページ・asset、検証状況、次回Releaseでの採用候補かを
簡潔にここへ追記します。安定版public repositoryへ詳細な開発計画は複写しません。
