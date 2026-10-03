# はじめに

サイドバーからページを選び、同梱した海水同位体・水文データを探索します。対話的な図と操作項目を十分に表示するため、デスクトップまたはノートPCのブラウザを推奨します。スマートフォンやタブレットでは表示領域が限られる場合があります。

## 最初の操作例

1. まず **User Data Check Quick Visualizer** でCSV/XLSXファイルを確認し、欠損値と品質フラグを確認して、簡単な2D--4D図を作成します。
2. 同じページおよび対応する専門ページの **Data filtering** で参照データセットを選び、必要に応じて **Uploaded data** を選択します。条件を変更した後、**Apply settings** を選ぶと図が更新されます。
3. 専門ページでは、同梱参照データまたは任意のローカル **User Excel data** を使います。空間分布の概観には **2Dplus Visualizer**、経度・緯度・深度と変数の関係には **3D 4D Visualizer**、より詳しい水文・同位体の確認にはsalinity--d18O、mapping、T--S、depth-profileの各ページを使います。

## ユーザーデータ

`local_data/user_data.xlsx` は、常時読み込む **User Excel data** ワークフローのための、値を含まない公開サンプルです。repositoryの外にある研究者自身のCSV、XLSX、XLSデータを使う場合は、アプリ起動前に `ENVGEO_LOCAL_USER_DATA_PATH` を設定します。ブラウザからアップロードしたファイルは現在のセッションだけで使用され、同梱参照データセットには書き込まれません。ブラウザアップロードは **User Data Check Quick Visualizer** に加え、salinity--d18O、mapping、T--S、custom-parameter、depth-profile、vertical-sectionの各ページで重ね描きに使えます。2Dplusと3D 4Dでは現時点で利用できません。

## ページの状態と解釈上の注意

**Vertical Section Visualizer** は試験的なワークフローです。補間結果を解析結果として扱う前に、観測点とデータ分布を必ず確認してください。**Correlation Overview** は、探索・開発時のワークフローを保存したアーカイブページであり、新機能の追加対象ではありません。

参照データの出典、引用方法、既知の制約は、Homeの **Data Sources**、**About**、**Updates** タブ、およびrepositoryの文書で確認できます。
