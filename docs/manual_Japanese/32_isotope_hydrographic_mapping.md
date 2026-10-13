# 同位体・水文マッピング

## このページでできること

δ¹⁸O、δD、d-excess、塩分、水温などの同位体・海洋環境パラメーターを地図上に表示します。

## 基本操作

1. 参照データソースを選び、サイドバーで絞り込み条件を設定します。
2. 必要に応じて**Uploaded data overlay**からセッション限定データをアップロードします。アップロード地点を地図に表示するには、経度・緯度の列割当が必要です。
3. 地図に表示するパラメーターと、**Scatter Map**または**Contour Map**を選びます。
4. 条件を変更した後に**Apply settings**を選びます。
5. サイドバーで地図中心、表示範囲、カラーマップ、色範囲、カラーバーを調整します。
6. 静的なパラメーター地図と、インタラクティブな採水地点地図を確認します。
7. ページに操作がある場合は、ScatterまたはContourのPNGをダウンロードします。

## 地図の種類

- **Scatter Map**：利用可能な各観測値を、その地点にプロットします。
- **Contour Map**：利用可能な経度・緯度・パラメーター値を格子へ補間し、観測点を重ねて表示します。可能な場合は線形補間を行い、点数不足、重複地点、共線的な地点では最近傍補間へ切り替えます。

## 主な設定

- **Mapped parameter**
- **Map type**
- **Map center**
- **Colormap**
- **Map longitude / latitude range**
- **Colorbar range**
- **Colorbar thickness / length / font size**
- **Map controls / Map style**

## 出力

- Scatter map
- Contour-style map
- Sampling Location Map
- ダウンロード可能な地図画像
- フィルタ後データ表
- アップロードデータの品質確認と前景重ね描き（使用時）

## 注意点

- Contour Mapは探索的な視覚化であり、独立に検証された連続場ではありません。描画された観測点、データ密度、選択された補間方法とあわせて解釈してください。
- Map display settings は Data filtering とは別の表示範囲設定です。
- 静的なScatter／Contour地図は同梱の海岸線・陸地データを使うため、実行時にNatural Earthデータをダウンロードしません。インタラクティブな採水地点地図では、外部タイルを使わない場合に**Coastline (offline)**を選びます。
- ブラウザからのアップロードは現在のStreamlitセッション内だけで扱い、参照データや常時読み込みの**User Excel data**を変更しません。
