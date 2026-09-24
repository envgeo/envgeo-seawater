# 引用・ライセンス管理方針

これは EnvGeo-Seawater の公開準備チェックリストと来歴管理メモです。公開GitHub
Release、Zenodoアーカイブ、Streamlit配布、JOSS投稿の前に、該当項目を確認します。
個別のデータ提供元の利用条件確認や法的助言に代わるものではありません。

## 基本方針

- 科学成果では、EnvGeo-Seawater本体、実際に使用したデータ、結果に重要な計算標準・
  ソフトウェアを引用する。
- 同梱する各データ・地図資産について、出典URL、版または取得日、ライセンス／利用条件、
  推奨引用、再配布可否、加工内容を記録する。
- READMEには簡潔な謝辞・引用を記載し、詳細は本書または将来の機械可読な来歴表へ残す。
- 依存ライブラリのオープンソースライセンスは、そのライブラリが扱うデータの再配布権を
  意味するものではない。

## 公開前チェックリスト

### プロジェクトとリリース

- [ ] 著者、題名、リポジトリURL、版、公開日、Zenodo DOIを記した `CITATION.cff` を追加する。
- [ ] READMEの暫定引用を、版付きDOIの正式引用へ置き換える。
- [ ] リリースタグを付け、対応する依存関係の記録を保存する。
- [ ] プロジェクトの `LICENSE` と、再配布資産・必要なライセンス文をまとめた第三者通知文書を整備する。

### 科学データ

- [ ] `dataset/` の各ファイルについて、元論文／データDOI、データ版、取得日、ライセンスまたは
      再利用許可、本プロジェクトで行ったフィルタ・列名標準化・統合処理を記録する。
- [ ] CoralHydro2k、NASA GISS、Kodama et al.、各地域データを個別に追跡可能にし、READMEの
      総覧だけに依存しない。
- [ ] サンプルやユーザー向け出力でも、必要に応じて元データの帰属表示を保つかリンクする。
- [ ] 未公表・制限付き・研究者個人の測定データは公開リポジトリへ入れない。
      `local_data/user_data.xlsx` はGit管理するゼロ値の公開サンプルであり、ローカル利用の
      ために編集できるが、commitまたは公開用同期前には元へ戻す。代わりに研究者自身の
      外部ファイルを `ENVGEO_LOCAL_USER_DATA_PATH` で指定してもよい。

### 地図・地理空間資産

- [ ] Natural Earthの帰属表示「Made with Natural Earth
      (https://www.naturalearthdata.com/)」を維持する。Natural Earthはパブリックドメインであり、
      同梱Land shapefileの来歴・チェックサムは
      `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md` に記録している。
- [ ] Vertical Sectionの出力・文書では GEBCO 2025 Gridを引用する：
      `GEBCO Compilation Group (2025) GEBCO 2025 Grid,
      doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29`。
- [ ] `GEBCO_2025_6min.nc` が間引きした派生物であること、生成スクリプト・条件を記録する。
- [ ] 公開前にGEBCOの利用条件記述を公式情報と整合させる。公式条件ではGridはパブリックドメインで
      商用利用も可能だが、帰属表示が必要であり、航海用途には使えない。参照：
      https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use

### 手法とソフトウェア

- [ ] 密度などの熱力学計算を科学結果として使う場合は、TEOS-10／GSW標準を引用する。Absolute
      SalinityとConservative Temperatureへ変換したか、明示した近似かを記載する。
- [ ] cmoceanを科学的な色設計として示す場合は、Thyng et al. (2016), *Oceanography*, 29(3),
      9–13, doi:10.5670/oceanog.2016.66 を引用する。
- [ ] StreamlitとPlotlyは、リリースで用いた版とともに中核フレームワーク・対話可視化ソフトウェア
      として謝辞に記載する。論文では既存の `paper_revised.bib` の項目を再利用できる。
- [ ] Cartopy、Matplotlib、Folium、NumPy、pandas、SciPy、scikit-learn、openpyxlなどは、
      第三者通知に版・ライセンスを記録する。報告するアルゴリズムや結果に本質的な場合は、
      論文でも該当ライブラリまたは手法を引用する。

### 公開・査読準備

- [ ] 図表キャプションに、使用データと、GEBCO・Natural Earth・cmoceanが本質的に寄与する場合の
      帰属を記す。
- [ ] README、アプリ内Data Sources、`data_text/`、文献データベースの引用・URLを一致させる。
- [ ] 上流の利用条件や推奨引用は変わり得るため、タグ付きリリース時に最終確認する。

## 現時点のフォローアップ

- READMEには主要同位体データとNatural Earthがある。GEBCO、TEOS-10/GSW、cmocean、Streamlit、
  Plotlyの簡潔な謝辞を追加する。
- `docs/geospatial_assets.md` のGEBCOを「非商用」とする記述は、公開前に公式利用条件に合わせて更新する。
- `paper_revised.bib` にはStreamlit、Plotly、cmoceanがある。論文準備時にTEOS-10/GSWとGEBCOの項目を追加・検証する。
