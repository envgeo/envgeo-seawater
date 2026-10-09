# 概要

## このアプリでできること

EnvGeo-Seawaterは、海水の安定同位体・水文データを探索するためのインタラクティブなWebアプリです。

地図、水温–塩分図、塩分–δ¹⁸O関係図、深度プロファイル、2D/3D/4D可視化に対応しています。

現在のコレクションには、**EnvGeo Dataset [ECS–Japan Sea]**中核コレクション（主要出典：Kodama et al. (2024)）、日本周辺の引用付きデータ、NASA GISSとPAGES CoralHydro2kを含む全球参照データセットが収録され、約50,000件の記録を探索できます。結果を利用する際は、**Data Sources**と**Filtered dataset**に示される元データの出典を引用してください。

## 想定ユーザー

- 海洋地球化学の研究者
- 海洋学を学ぶ学生
- 自分の海水データを参照データセットと比較したいユーザー

## ローカル導入

公開パッケージは、ソースリポジトリをcloneせずに、次のようにローカルアプリを導入・起動できます。

```bash
python -m pip install envgeo-seawater
envgeo-seawater
```

**Python 3.12を推奨**します。Python 3.13は、どのプラットフォームでも条件付きサポートです。固定しているNumPyまたはPyProjに対応する完成済みパッケージ（wheel）が利用できない場合、pipはソースからビルドを試みます。このビルドには数分かかることがあり、ネイティブのビルド環境が必要です。この条件はM1 Mac固有でも、Cartopy固有でもありません。macOSでPython 3.13を使う場合は、AppleのCommand Line Toolsを導入してから再試行してください。導入に失敗した場合は、[安定版README](https://github.com/envgeo/seawater_map/blob/main/README_Japanese.md)にあるPython 3.12のConda fallbackを使ってください。Python 3.14はv1.3.4で固定している依存関係では未対応です。

## 基本的な流れ

1. サイドバーからページを選びます。一時的なファイルを使う場合は、**User Data Check & Quick Visualizer**でCSV/XLSXをアップロードします。ファイルは現在のブラウザセッション内だけで扱われます。
2. 常時使うローカル表を読み込む場合は、アプリ起動前に`ENVGEO_LOCAL_USER_DATA_PATH`で、repository外のCSV、XLSX、XLSファイルを指定します。同梱の`local_data/user_data.xlsx`はゼロ値の公開サンプルです。個人データをrepositoryやアプリフォルダへ入れないでください。
3. **Data filtering**で参照データセットを選び、条件を調整します。常時読み込み表を設定した場合は`User Excel data`、セッション内アップロードが使える場合は`Uploaded data`が表示されます。
4. 条件を変更した後に**Apply settings**を選び、必要に応じて図の設定を調整します。
5. 地図、図、表、品質フラグを確認する。
6. ページに操作がある場合は、必要に応じて図や抽出データの概要をダウンロードします。

## 主なデータ項目

- δ¹⁸O（`d18O`）
- δD（`dD`）
- d-excess
- 塩分
- 水温
- 水深
- 緯度・経度
- 採水年・月

## 注意点

- beta と表示されたページは、ワークフローや科学的な扱いを調整中です。
- 全球データを大きく選択すると、3D/4D表示が重くなることがあります。
- ブラウザからアップロードしたデータは、現在のStreamlitセッションのメモリ内だけで扱い、アプリは保存しません。アップロードは**User Data Check & Quick Visualizer**に加え、salinity--d18O、mapping、T--S、custom-parameter、depth-profile、vertical-sectionの各ページで使えます。2Dplusと3D/4Dでは現時点でブラウザからのアップロードには対応していません。
- `User Excel data`は、repository外のパスから読み込む常時データセットです。セッション限定の`Uploaded data`とは別に扱います。
- Vertical Sectionの補間は実験的な機能です。解析結果として扱う前に、観測点、設定、データ密度を確認してください。
- Custom Parameter PlotとVertical Sectionはbetaワークフローです。Correlation Overviewは、探索的な開発ワークフローを残すアーカイブであり、新機能の追加対象ではありません。
