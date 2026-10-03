# 引用・ライセンス管理方針

[English version](citation_and_license_plan.md)

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

## 現在のデータ配布範囲

現在の `dataset/*.xlsx` はすべて、プロジェクトが記録した学術利用方針に基づいて同梱を維持する。
各workbookについて、出典引用、利用可能な場合のDOIまたは正規出典、取得日、プロジェクト側の
変換記録を保ち、第三者の記録をプロジェクト所有とは記載しない。これは本プロジェクトにおける
現時点の配布判断であり、引用だけで一般的な再利用権が得られるという主張ではない。ファイル単位の
記録と再検討条件は [`dataset_redistribution_audit.md`](dataset_redistribution_audit.md) と
[`provenance_inventory.md`](provenance_inventory.md) を参照する。

## 公開前チェックリスト

### プロジェクトとリリース

- [x] 著者、題名、repository URL、版、記録済みの1.3.4 Release candidate日を記した`CITATION.cff`を追加した。
      archive公開前は意図してDOIを記さない。`release_metadata_draft_Japanese.md`を参照する。
- [ ] 最終tag日へ更新し、Zenodo archive作成後に版付きDOIを追記する。
- [ ] READMEの暫定引用を、版付きDOIの正式引用へ置き換える。
- [ ] リリースタグを付け、対応する依存関係の記録を保存する。
- [x] プロジェクトの `LICENSE` と`THIRD_PARTY_NOTICES.md`／日本語版を追加した。通知文書では、
      MITのcode licenseと同梱する第三者データ・地理空間資産の記録を区別し、出典ごとの記録を正本とする。

### Package indexでの公開（JOSS再投稿前）

- [ ] review済みの安定版sourceから最終artifactをbuildし、distribution fileに対して
      `twine check`を実行する。
- [ ] そのartifactをTestPyPIへuploadし、新しいmacOS test environmentでinstallする。
      必要なcompiled geospatial prerequisiteはconda-forgeから入れ、その後に`pip`で
      EnvGeo-Seawaterをinstallする。
- [ ] TestPyPI確認後、review済みの最終artifactをPyPIの`envgeo-seawater`として公開する。
      PyPI Trusted Publishingまたは安全な手動uploadを使い、upload tokenをrepositoryへ
      保存しない。
- [ ] 新しいmacOS environmentで`pip install envgeo-seawater`を再現し、起動確認の結果を
      release recordに記録する。
- [ ] conda-forge recipe/feedstockはv1.3.4後の独立した改善作業とする。PyPIからの
      installationが動作すれば、現releaseおよびJOSS再投稿のblockerとはしない。

### 科学データ

- [ ] `dataset/` の各ファイルについて、元論文／データDOI、データ版、取得日、明示されている場合の
      ライセンスまたは利用条件、必要な引用、記録済みの学術利用判断、本プロジェクトで行ったフィルタ・
      列名標準化・統合処理を記録する。
- [ ] CoralHydro2k、NASA GISS、Kodama et al.、各地域データを個別に追跡可能にし、READMEの
      総覧だけに依存しない。
- [ ] サンプルやユーザー向け出力でも、必要に応じて元データの帰属表示を保つかリンクする。
- [ ] 未公表・制限付き・研究者個人の測定データは公開リポジトリへ入れない。
      `local_data/user_data.xlsx` はGit管理するゼロ値の公開サンプルであり、ローカル利用の
      ために編集できるが、commitまたは公開用同期前には元へ戻す。代わりに研究者自身の
      外部ファイルを `ENVGEO_LOCAL_USER_DATA_PATH` で指定してもよい。

### 地図・地理空間資産

- [ ] プロジェクトで採用するNatural Earthの帰属表示「Made with Natural Earth
      (https://www.naturalearthdata.com/)」を、来歴記録と該当する出力で維持する。Natural Earthは
      パブリックドメインであり、同梱Land shapefileの来歴・チェックサムは
      `coastline/natural_earth_50m_land/LICENSE_OR_SOURCE.md` に記録している。
- [ ] Vertical Sectionの出力・文書では GEBCO 2025 Gridを引用する：
      `GEBCO Compilation Group (2025) GEBCO 2025 Grid,
      doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29`。
- [ ] `GEBCO_2025_6min.nc` が間引きした派生物であること、生成スクリプト・条件を記録する。
- [ ] タグ付きリリース時にGEBCOの利用条件を再確認する。公式条件ではGridはパブリックドメインで
      商用利用も可能だが、帰属表示が必要で、誤解を招く公式承認の示唆は禁止され、航海用途には
      使えない。参照：
      https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use

### 手法とソフトウェア

- [ ] 密度などの熱力学計算を科学結果として使う場合は、TEOS-10／GSW標準を引用する。Absolute
      SalinityとConservative Temperatureへ変換したか、明示した近似かを記載する。
- [ ] cmoceanを科学的な色設計として示す場合は、Thyng et al. (2016), *Oceanography*, 29(3),
      9–13, doi:10.5670/oceanog.2016.66 を引用する。
- [ ] StreamlitとPlotlyは、リリースで用いた版とともに中核フレームワーク・対話可視化ソフトウェア
      として謝辞に記載する。論文では `paper.bib` の項目を再利用できる。
- [ ] Cartopy、Matplotlib、Folium、NumPy、pandas、SciPy、scikit-learn、openpyxlなどは、
      第三者通知に版・ライセンスを記録する。報告するアルゴリズムや結果に本質的な場合は、
      論文でも該当ライブラリまたは手法を引用する。

### 公開・査読準備

- [ ] 図表キャプションに、使用データと、GEBCO・Natural Earth・cmoceanが本質的に寄与する場合の
      帰属を記す。
- [ ] README、アプリ内Data Sources、`data_text/`、文献データベースの引用・URLを一致させる。
- [ ] 上流の利用条件や推奨引用は変わり得るため、タグ付きリリース時に最終確認する。

## 現時点のフォローアップ

- README英日版には、GEBCO、TEOS-10/GSW、cmocean、Streamlit、Plotly、Natural Earthの利用範囲を
  簡潔に記載した。この謝辞は、出典ごとの引用、来歴記録、将来の第三者通知に代わるものではない。
- `docs/geospatial_assets.md` は現在のGEBCO利用条件を記録している。タグ付きリリース時に記述と出典引用を再確認する。
- `paper.bib` にはStreamlit、Plotly、cmocean、TEOS-10/GSW、GEBCO 2025 Gridの検証済み項目がある。
  参照するsoftware、データ版、原稿の範囲が変わる場合は、タグ付きリリース時に各引用を再確認する。
