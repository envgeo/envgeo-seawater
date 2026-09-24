# Interactive 2D/2.5D Visualizer

## このページでできること

Plotly のインタラクティブ散布図と採水地点マップを連動して確認できます。

## 基本操作

1. データセットを選択する。
2. Data filtering を適用する。
3. Plot type を選ぶ。
4. Color parameter と Colormap を選ぶ。
5. Box Select または Lasso Select で点を選択する。
6. 対応する採水地点を地図で確認する。

## 主な設定

- **Plot type**
- **Color parameter**
- **Colormap**
- **Density contour interval (approx. σ0)** — T–S図の近似σ0参照等値線の間隔
  （0.2、0.5、1.0 kg m⁻³ から選択；既定値 1.0 kg m⁻³）。
  Temperature–Salinity 表示のみに適用されます。
- **Show density contours** — 近似σ0参照等値線の表示／非表示
- **Regression line**（塩分-d18O表示）
- **Map controls / Map style**

## 出力

- Temperature-Salinity のインタラクティブ図
- Salinity-d18O のインタラクティブ図
- Sampling Location Map
- Filtered dataset
- Box/Lasso-selected dataset

## 注意点

- 回帰線は探索的な目安として使います。
- Box/Lasso 選択は Plotly の操作仕様に依存します。
- 点数が多い場合、古いPCでは動作が重くなることがあります。
- T–S図の密度等値線は**近似的なσ0参照線**であり、個々の観測値の密度ではありません。
  実用塩分（≈ 絶対塩分）と現場水温（≈ 保存温度）を `gsw.sigma0` に渡して計算しており、
  視覚的な参照用途のみを目的としています。真の TEOS-10 σ0 との差は、データソース、
  地理的位置、深度、海況によって異なります。
- 等値線格子は選択された表示軸範囲を基礎とし、品質チェック済みの入力許容範囲
  （塩分 0–50；水温 −5–45 °C）にクリップします。
