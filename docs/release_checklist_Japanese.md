# EnvGeo-Seawater Release Checklist（日本語版）

[English version](release_checklist.md)

テストサイトの公開、GitHub更新、Release作成、Zenodoでのversion archiveの前に、この確認項目を使います。

この正規作業コピーのchecklistは、開発・公開前確認に用います。正式なGitHub Release、Zenodo archive、
JOSS向けsoftware recordは、この作業コピーにtagを付けるのではなく、安定版`seawater_map` repositoryから作成します。

## 1. ローカル環境

### Release candidate 1.3.4（2026-09-28）

- [x] 最終試験にPython 3.12／Streamlit 1.63／Plotly 5.24環境を使用した。
- [x] `python -m pytest -q test`完了：**302 passed、1 warning**。warningはclass-scoped fixtureに関するPytestの将来非推奨通知であり、アプリ試験の失敗ではない。保守上の対応項目として残す。

### 既に行った手動smoke確認（release commit後に再確認する）

以下はRelease candidateの確認中に完了した探索的な手動確認である。有用な根拠ではあるが、commit済みの
安定版Releaseとdeploymentに対する最終手動確認の代わりにはしない。

- [x] Homeが開き、ローカル専用の`99_Environment_Check.py`が公開sidebarに表示されなかった。
- [x] Mapping、T–S Diagram、Depth Profileがアプリケーションerrorなしで開き、データ選択と
      `Apply settings`で表示結果が更新された。
- [x] Mappingの背景tileが表示された。
- [x] アプリケーションerror画面は出なかった。ローカル絶対pathとcredentialらしき値については、
      別途、公開fileとwheelの静的監査を行った。

- [ ] 想定したPython環境が有効であることを確認する。
- [ ] 検証済み基準のいずれかが有効であることを確認する：Python 3.10.15／Streamlit 1.42、またはPython 3.12.14／Streamlit 1.63（いずれもPlotly 5.24）。
- [ ] `requirements.txt`が検証環境と一致することを確認する。
- [ ] 必要に応じてローカル環境チェッカーを実行する。

```bash
streamlit run tools/env_check_streamlit.py
```

- [ ] Release記録に必要なら、環境レポートをCSVまたはPDFで保存する。

## 2. 基本起動

- [ ] ローカルでアプリを起動する。

```bash
streamlit run home.py
```

- [ ] `home.py`が正常に開くことを確認する。
- [ ] アプリ版が`1.3.4 (2026-09-28)`と表示されることを確認する。
- [ ] HomeのMain、About、Data Sources、Manual、Update History、および該当する日本語情報が開くことを確認する。

## 3. サイドバーとフィルター

- [ ] サイドバーに`Data filtering`が表示される。
- [ ] `Apply settings`を押す説明が表示される。
- [ ] `Area filter preset`でLongitude／Latitudeの初期範囲が変わる。
- [ ] area preset選択後もLongitude／Latitude sliderを手動調整できる。
- [ ] フィルター変更後に`Apply settings`で図が更新される。
- [ ] `Details and statistics of filtered data`が開き、CSV exportできる。

## 4. 主な可視化ページ

各ページを開き、簡単な視覚確認を行う。

- [ ] `03_[Interactive]_2Dplus_Visualizer.py`
- [ ] `04_[Interactive]_3D_4D_Visualizer.py`
- [ ] `05_[Utils]_User_Data_Check_Quick_Visualizer.py`
- [ ] `06_[Utils]_Data_Overlap_Check.py` — 監査を実行し、Strong／Review結果と3種類の監査CSV downloadを確認する。
- [ ] `31_Salinity-d18O_Relationship.py`
- [ ] `32_Isotope_Hydrographic_Mapping.py`
- [ ] `34_T-S_diagram.py`
- [ ] `35_Custom_Parameter_Plot.py`
- [ ] `37_Depth_Profile.py`
- [ ] `53_Vertical_Section_Visualizer.py`
- [ ] `80_Correlation_Overview.py`

各ページで以下を確認する。

- [ ] データソースを選択できる。
- [ ] `Apply settings`後にフィルターが反映される。
- [ ] 主な図がエラーなく描画される。
- [ ] caption、help、labelが理解できる。
- [ ] 該当する場合、`Sampling Location Map`が正しく描画される。
- [ ] 該当する場合、`Map controls`と`Map style`が動作する。
- [ ] 明らかな文字重なりやレイアウト崩れがない。

## 5. 図の出力

- [ ] 2D Matplotlib図の下に`Download image`が表示される。
- [ ] ダウンロードしたPNGが正常に開く。
- [ ] 特にDepth Profileで、図タイトルがダウンロード画像内に収まる。
- [ ] ファイル名が妥当で、危険な文字を含まない。
- [ ] 今回のReleaseで、Plotly図・地図の画像出力をPlotly modebarのcamera/exportに任せるか判断する。

## 6. ユーザーデータのアップロード

- [ ] 同梱するゼロ値サンプル`local_data/user_data.xlsx`が`User Excel data`と表示され、browser uploadのsession stateに入らず、各参照sourceへ1回だけ追加される。
- [ ] 公開サンプルに研究者の測定値が含まれず、研究者所有データは外部path経由で設定する。
- [ ] 統合betaのupload workflowが動作する。
- [ ] アップロードデータが参照データと視覚的に区別できる。
- [ ] アップロードデータがsession内だけに保持され、ディスク・server storageへ保存されない。
- [ ] 対応する一般的な列aliasが標準化される。
- [ ] 対応するページでアップロードデータの品質summaryが見える。
- [ ] User Data Check & Quick VisualizerがCSV/XLSX uploadを受け入れ、session内だけで扱う。

## 7. ローカル専用・開発ページ

- [x] `pages/99_Environment_Check.py`を公開repository・deploymentから除外し、ローカル開発作業コピーにだけ残す。
- [x] `pages/05_[Utils]_User_Data_Check_Quick_Visualizer.py`は、アップロード起点の品質確認と2D/3D/4D探索用の公開User Data Check & Quick Visualizerである。
- [ ] 開発／テストdeploymentを確認する場合は、`pages/90_Integrated_Visualizer_beta.py`が
      実験的機能として明確に表示されることを別途確認する。これは安定版の受入項目ではなく、
      安定版`seawater_map`アプリケーションと利用者向けドキュメントwebsiteから除外する。
- [ ] betaページを公開、非表示、または実験的機能として文書化するか判断する。
- [ ] private note、制限データ、未公開データセットが公開deployment filesに含まれないことを確認する。

## 8. テスト

- [ ] 基本テスト群を実行する。

```bash
pytest -q test/test_envgeo_utils.py test/test_repository_health.py
```

- [ ] GitHub Release準備時は広いテスト群を実行する。

```bash
pytest
```

- [ ] skipまたはexpected-failing testを確認する。
- [ ] 意図したpage list変更でtestが失敗する場合は、testを更新するか理由を記録する。

## 9. 文書

- [x] `README.md`が現在の公開状態を反映することを確認した。
- [x] `README_Japanese.md`が現在の公開状態を反映することを確認した。
- [x] `data_text/update_log.md`に最新の未公開変更があることを確認した。
- [x] `data_text/update_log_Japanese.md`に最新の未公開変更があることを確認した。
- [x] beta、archive、ローカル開発ページが明確に説明されていることを確認した。
- [x] 引用・データソースの案内が理解できることを確認した。
- [x] 精査済みmanualを基に、図付きの英日静的ドキュメントwebsiteを作成した。安定版の公開範囲だけを説明し、private path、個人データ、token、内部記録を含めないことを確認した。
- [x] GitHub Pagesでドキュメントwebsiteを公開し、公開URL、navigation、図、linkを確認した：<https://envgeo.github.io/seawater_map/>。
- [ ] 安定版URL、release version、公開ページ範囲、ドキュメントURL、Zenodo DOIが確定した後に、研究室websiteを更新する。安定版`seawater_map`の説明と一致させ、NASA GISSとPAGES CoralHydro2kを含む引用付き約50,000件のデータ、ユーザーデータのアップロード／プロット機能を記載する。旧いversion番号、DOIの保留表現、安定版から除外したページの説明を残さない。

## 10. GitHub Release準備

- [ ] repositoryの送付先とbranchを確認する。
- [ ] 一時ファイル、private file、ダウンロード済みreport、ローカルcacheがstageされていない。
- [ ] commit前に変更ファイルを確認する。
- [ ] Release準備の範囲を表す明確なcommit messageを使う。
- [ ] ローカル確認と公開テストdeploymentの確認が終わるまでtagを付けない。

## 11. Streamlit Deployment

- [ ] deployment repositoryに必要なファイルがある。
  - `home.py`
  - `envgeo_utils.py`
  - `pages/`
  - `dataset/`
  - `data/`
  - `data_text/`
  - `coastline/`
  - `requirements.txt`
  - deployment platformが使用する場合は`runtime.txt`
- [ ] 必要なら、除外対象のローカル専用ページが実際に除外されている。
- [ ] deploymentしたアプリを開き、短いsmoke testを繰り返す。
  - Homeが開く。
  - T–Sページが開く。
  - Mappingページが開く。
  - Depth Profileが開く。
  - 含める場合はIntegrated Visualizer betaが開く。

## 12. Package indexでの公開（PyPI）

- [ ] 想定した最終distribution artifactをbuildし、`twine check`を実行する。
- [ ] TestPyPI project pageでREADMEの表示を確認する。PyPI upload前に、repository内では
      有効でもpackage index上では切れる相対link・画像参照を、必要に応じて永続的な
      GitHubまたはGitHub Pagesの絶対URLへ置換する。TestPyPIの配布fileはimmutableなので、
      そこで見つかった相対linkの不備は、別サービスであるPyPI upload前に最終sourceで修正する。
- [ ] 想定artifactをTestPyPIへuploadし、新しいmacOS environmentでinstallする。
      必要なcompiled geospatial prerequisiteだけをconda-forgeから入れ、その後に
      `pip`で本packageをinstallする。
- [ ] TestPyPI installationからappを起動し、短いsmoke testを繰り返す。
- [ ] 最終tagの確認後、同じreview済みartifactをPyPIの`envgeo-seawater`として公開する。
      Trusted Publishingまたは安全な手動uploadを使用し、PyPI tokenをcommitしない。
- [ ] 新しいmacOS environmentで`pip install envgeo-seawater`と起動確認を再現する。
- [ ] conda-forge recipe/feedstockは有用な後続改善とする。ただしPyPI経路を確認できれば、
      v1.3.4とJOSS再投稿のblockerとはしない。

## 13. Zenodo／DOI準備

- [ ] GitHub Release作成前に、`seawater_map` GitHub repositoryをZenodoで有効化し、
      tag付きReleaseが自動archiveされるようにする。
- [ ] Zenodo archiveを作る前にGitHub Releaseが最終版であることを確認する。
- [ ] title、author、affiliation、license、descriptionを確認する。
- [ ] archive版がRelease tagと一致する。
- [ ] wheel SHA-256を記録する場合は、clean tagged checkoutから再作成したwheelの値だけを使う。
      CI wheel artifactは確認根拠であり、ReleaseまたはZenodoの配布fileではない。
- [ ] archive作成後に、version DOIとconcept DOIをREADMEとcitation filesへ記録する。
      DOIを事前予約しない限り、このfollow-up document commitはimmutableなtag付きarchiveには含まれない。

## 14. JOSSを見据えた次作業

- [ ] 再投稿前に別途`docs/joss_checklist.md`を準備する。
- [ ] pytest coverageが形式だけでなく意味のあるものか確認する。
- [ ] reviewerに十分な例と利用者文書があることを確認する。
- [ ] clean environmentでinstallation instructionsを再現できることを確認する。
- [ ] citation instructionsにEnvGeo-Seawaterと元データ提供者の両方を含める。
