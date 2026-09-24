# Temperature-Salinity Diagram

## このページでできること

水温と塩分の関係を、密度等値線とともに表示します。

## 基本操作

1. データセットを選択する。
2. 背景データを表示するか選ぶ。
3. Data filtering を適用する。
4. 必要に応じて Color parameter を選ぶ。
5. 図の見た目を調整する。
6. 必要に応じて図をダウンロードする。

## 主な設定

- **Show background data**
- **Show legend**
- **Color parameter**
- **Colorbar range**
- **Salinity scale**
- **Temperature scale**
- **Density contour interval (approx. σ0)** — 近似 σ0 参照等値線の間隔
  （0.2、0.5、1.0 kg m⁻³ から選択；既定値 0.5 kg m⁻³）
- **Figure width / height**
- **Tick font size / Label font size**
- **X tick count / Y tick count**
- **Map controls / Map style**

## 出力

- Temperature-Salinity 図
- 密度等値線
- Sampling Location Map
- ダウンロード可能な PNG 図
- フィルタ後データ表

## 注意点

- 密度等値線の描画には、有効な水温・塩分値が必要です。
- 欠損値を含む点は図から除外され、caption に件数が表示されます。
- 密度等値線は**近似的な σ0 参照格子**であり、個々の観測値の密度ではありません。
  実用塩分（≈ 絶対塩分）と現場水温（≈ 保存温度）を `gsw.sigma0` に渡して計算しており、
  視覚的な参照用途のみを目的としています。真の TEOS-10 σ0 との差は、データソース、
  地理的位置、深度、海況によって異なります。
- 等値線格子は選択された表示軸範囲を基礎とし、品質チェック済みの入力許容範囲
  （塩分 0–50；水温 −5–45 °C）にクリップします。軸設定で負の塩分が選択されても、
  密度計算には渡されません。
