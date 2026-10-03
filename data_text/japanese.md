EnvGeo-Seawaterは、海水の安定同位体と水文データを対話的に探索するWebアプリです。現在の中核データセットは、東シナ海から日本海にかけて複数年にわたり採取された、2,000点以上のKodama et al. (2024)のデータです。塩分、水温、δ¹⁸O、δD、採取時期、位置を、地図、3D/4D表示、関係図、深度プロファイルで確認できます。

NASA GISS Global Seawater Oxygen Isotope DatabaseとPAGES CoralHydro2kを含む全球参照データセットを収録しており、約50,000件のデータをインタラクティブに検索できます。

また、ユーザーデータのアップロードとプロット機能も備えています。

## 主な機能

- **中核データセット：** 東シナ海から日本海を中心とする、複数年のKodama et al. (2024)海水同位体データを探索できます。
- **比較用データセット：** 日本周辺の引用付きデータに加え、NASA GISSおよびPAGES CoralHydro2kを含む全球参照データセットと比較できます。
- **可視化：** サイドバーから、地図、2D/3D/4D表示、T--S図、深度プロファイルなどのページを開けます。
- **ブラウザからのアップロード：** **User Data Check Quick Visualizer**では、CSV/XLSXファイルを現在のブラウザセッション内で確認・可視化できます。salinity--d18O、mapping、T--S、custom-parameter、depth-profile、vertical-sectionの各ページでも、アップロードしたデータを重ね描きできます。2Dplusと3D/4Dでは、現時点でブラウザからのアップロードには対応していません。

## はじめに・日本語マニュアル

1. サイドバーからページを選び、同梱された海水同位体・水文データを探索します。
2. CSV/XLSXを確認する場合は、まず **User Data Check Quick Visualizer** を開き、欠損値や品質情報を確認して簡単な2D--4D図を作成します。
3. 対応するページでは、**Data filtering**で参照データセットを選び、条件を変えた後に **Apply settings** を選ぶと図が更新されます。

Homeの **Manual** タブには簡潔な使い方を、[詳細な日本語ユーザーマニュアル](https://github.com/envgeo/seawater_map/tree/main/docs/manual_Japanese)にはページごとの説明を掲載しています。

## データと引用

比較用として、日本周辺の引用付きデータセットに加え、NASA GISS Global Seawater Oxygen Isotope DatabaseとPAGES CoralHydro2kを含む全球参照データセットを収録しています。EnvGeo-Seawaterは可視化・比較のためのツールです。結果を利用する際は、**Data Sources** と **Filtered dataset** に示される元データの出典を引用してください。

全球参照データを含む現行コレクションは、約50,000件の記録で構成されます。

日本周辺の中核データセットは一貫した条件で分析されているため、海域や採取時期をまたぐ比較に利用できます。研究室で分析した追加データセットは、公表と来歴記録が整った後にのみ追加します。

ダウンロード機能があるページでは、図を研究・教育目的で書き出せます。本アプリケーションの引用方法は、リポジトリのREADMEおよびRelease資料を参照してください。

[研究室Webサイト](https://envgeo.h.kyoto-u.ac.jp/sw_jpn/)

開発・公開：石村豊穂（京都大学）。PythonとStreamlitを用いて開発しました。

**AI支援による開発について:** バージョン1.3以降、コードレビュー、実装草案の作成、リファクタリング、テスト設計、バグ調査、文書整備にOpenAI CodexとAnthropic Claude Codeを活用しています。AIツールを著者または共同開発者として位置付けていません。科学的・設計上の判断、最終レビュー、検証、公開内容の責任は人間の著者が負います。
