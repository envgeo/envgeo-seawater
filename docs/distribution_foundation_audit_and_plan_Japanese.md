# 配布基盤の監査と段階的計画（Sprint 2）

「Sprint 2」は、配布基盤の監査とローカルwheel実証を行った開発フェーズのプロジェクト内名称である。以下の監査結果は2026-09-25時点の状態を示す歴史的記録であり、現在のRelease構成そのものではない。

**状態:** 歴史的なSprint 2A設計記録。当時はpackage実装、ディレクトリ移動、データセット変更を許可しなかった。  
**監査日:** 2026-09-25  
**Sprint 2A方針確定:** 2026-09-25

**進捗:** Python 3.12でローカルSprint 2B〜2C実証に合格した。詳細は
`wheel_proof_report_Japanese.md`を参照。この知見をもとにSprint 3の構成を実装した。現行1.3.4の配布・CI検証記録は[`sprint3_distribution_design_and_acceptance_Japanese.md`](sprint3_distribution_design_and_acceptance_Japanese.md)であり、正式Releaseの承認は別途行う。

## 結論

`envgeo-utils`の共通package化やEarthquakeアプリとの基盤統合より先に、EnvGeo-Seawater単体の配布方法を安定させる。

Sprint 1では`envgeo_assets.py`を導入し、主要な同梱資産をCWDに依存しないソースチェックアウト内のパスへ移行した。Sprint 2で最初に確認する範囲は、**アプリを`src/`へ移動せず、現在のStreamlit構成のままwheelへ収録し、ソースチェックアウト外から起動できるか**に限定する。

最初のwheelはローカル技術実証であり、公開用成果物ではない。正式なpackage構成、依存関係方針、対応インストール手順を確定するものではなく、サポート済みの`pip install`手順として案内しない。

## 当時の監査結果（2026-09-25）

| 項目 | 現状 | Sprint 2での扱い |
|---|---|---|
| アプリ構成 | エントリースクリプトは`home.py`のみで、Streamlitは同階層の`pages/`を発見する。 | 実証中はこの物理的な関係を変更しない。 |
| 資産ローダー | `envgeo_assets.asset_path()`はモジュールの場所を基準に解決し、絶対パスとルート外参照を拒否し、起動CWDに依存しない。 | wheel実証中はこのAPIを維持し、`importlib.resources`へ一括置換しない。 |
| 移行済み資産 | 5件の`dataset/*.xlsx`、海岸線CSV、Natural Earth陸域、GEBCO、ローカルユーザーデータサンプル、Home文書・メディア、代表的なページ画像は`asset_path()`を使う。 | package対応を仮定せず、インストール後の各パスを明示的に検証する。 |
| 残るファイル相対処理 | ローカルモジュール／ページ間関係とローカル専用環境診断には少数の`Path(__file__)`処理が残る。 | 実証で棚卸しする。すべてを資産ローダーの欠陥とは扱わない。 |
| package化 | `pyproject.toml`、package宣言、package data定義、インストール後の起動コマンドはない。 | Sprint 2Bでローカルwheel実証に必要な最小限のbuild設定だけを追加できる。 |
| 依存関係・Cloud | `requirements.txt`と`runtime.txt`がソース実行／Streamlit Cloudの現行構成である。 | Sprint 2A〜2CではCloud依存設定を変更せず、新しい正本も決めない。 |
| 書込み先 | Streamlitのメモリ／データキャッシュは使うが、EnvGeo専用のOSキャッシュはない。 | インストール済みpackageの隣へ書き込まないことを確認する。永続cacheはSprint 2Dで設計する。 |
| 同梱地理資産 | 海岸線CSV約2 MB、Natural Earth 50m陸域約1 MB、派生GEBCO約12〜13 MBを同梱する。 | 初期実証ではGEBCO同梱を維持する。downloaderを追加せず、Pythonのoptional extraとも表現しない。 |
| ローカルユーザーデータ | `local_data/user_data.xlsx`はGit管理するゼロ値公開サンプル。個人データは`ENVGEO_LOCAL_USER_DATA_PATH`でも指定できる。 | 公開サンプルだけを含め、個人データをwheelやcacheへコピーしない。 |
| 公開対象外 | `pages/99_Environment_Check.py`と内部レビュー記録はローカル開発用で、公開クローンから除外する。 | wheel実証の収録一覧からも除外する。 |
| 来歴 | Natural EarthとGEBCOの規約は記録済み。データセット来歴・再配布記録は公開準備項目として継続中。 | ローカル実証では現行ファイルを読めるが、公開用フルデータwheelやReleaseを許可するものではない。 |

## Sprint 2の固定境界

### 維持するもの

- `home.py`と同階層の`pages/`。
- 既存UI、アップロード、共通フィルター、ページ名と順序。
- 現行`asset_path()`の動作と、ソースチェックアウトでの起動方法。
- ゼロ値の公開`local_data/user_data.xlsx`。
- 初期実証でのGEBCO同梱。
- 現在のStreamlit Cloud依存ファイルと配布設定。

### まだ変更しないもの

- コードを`src/`や新しいアプリpackageディレクトリへ移動しない。
- `envgeo-utils`共通packageを分離・公開しない。
- SeawaterとEarthquakeの基盤を統合しない。
- `asset_path()`を一度にresource APIへ置換しない。
- OS cache、遠隔資産download、明示的offline modeを追加しない。
- `dataset/`内を編集・削除・変換・再生成しない。
- GEBCOを外部download化せず、optional dependency extraでpackage dataを選択導入できるとも主張しない。
- Streamlit Community Cloudが使う依存ファイルを変更しない。
- 実証wheelを公開、commit、push、または正式版として案内しない。

## 段階計画

| 段階 | 目的 | 変更可能範囲 |
|---|---|---|
| **2A** | 設計文書を訂正し、実証の受入条件を確定する。 | 文書のみ。 |
| **2B** | 現行アプリ配置を維持して最小wheelを作る。 | 最小限の実験用build metadata、package data規則、起動／実証補助、焦点テスト。 |
| **2C** | checkout外へwheelを導入して検証する。 | テスト環境とテスト／報告の調整。配置移動はしない。 |
| **2D** | 実証結果からresource context、cache、明示的offline動作を設計する。 | 先に設計レビュー。実装は別途承認後。 |
| **3** | 合意した正式package、起動コマンド、CI、Cloud構成を導入する。 | Sprint 2レビュー後の実装。 |
| **後続・必要な場合のみ** | packageディレクトリまたは`src/`移動を再検討する。 | 独立した移行設計と回帰計画。 |

## Sprint 2B〜2Cのwheel実証

### 目的

インストール済みコピーでも、現行のStreamlitエントリーと`pages/`の関係を維持し、大規模なソースツリー再編なしに必要ファイルを利用できるか確認する。同時に、どの利用箇所が永続的な実ファイル`Path`を必要とし、どこが将来resource streamやcontext managerを利用できるかを明らかにする。

### 必須確認

1. 現行配置のcleanなコピーからwheelをbuildする。
2. 明示的な収録allowlist／除外listとwheel内容を照合する。
3. リポジトリ外の新規仮想環境へインストールする。
4. プロセスのCWDをcheckout外にし、checkout上のmoduleをimportせずにインストール済みStreamlitアプリを起動する。
5. 予定する公開ページをStreamlitが発見することを確認する。
6. 必須資産の各系統から代表ファイルを読む：`dataset/`、海岸線CSV、Natural Earth shapefile一式、`local_data/user_data.xlsx`、Home文書・メディア、GEBCO。
7. 通常の起動・読込みが`site-packages`へ書き込まないことを確認する。
8. `pages/99_Environment_Check.py`、cache、個人データ、ローカル出力、`.DS_Store`、内部レビュー記録が含まれないことを確認する。
9. 実証設定追加後、関連資産テストと対応環境の全テストをソースチェックアウトで再実行する。
10. 手動確認は別に記録し、UI上の失敗に合わせて自動テストを弱めない。

### 合格条件

以下をすべて満たした場合だけ実証を合格とする。

- 新規環境でwheelを再現可能にbuild・installできる。
- インストール済み起動コマンドがソースcheckoutやそのCWDに依存せず、意図した`home.py`を開始する。
- 予定する公開`pages/`を発見し、ローカル専用ページを表示しない。
- 上記の代表的な同梱資産をインストール先から読み込める。
- ソースチェックアウトでは引き続き`streamlit run home.py`で起動できる。
- 既存UI、アップロード、共通フィルター、科学データ内容が変わらない。
- GEBCOを含む同梱資産に実行時downloadを導入しない。
- 個人データや生成状態を含めず、インストール済みアプリの隣へ書き込まない。
- 対応する自動テストが通過し、手動確認上の制限を記録する。

実証の失敗も有用な結果である。ただし、失敗だけを理由に直ちに`src/`移動を開始しない。失敗した前提を記録して設計レビューへ戻る。

## 実証後のresource API判断

インストール済み資産には`importlib.resources`が有力だが、常に永続的な実ファイルパスを保証するわけではない。`as_file()`で一時実体化したファイルはcontext終了後に消える場合がある。このためSprint 2Dでは、Cartopy、NetCDF、pandas、Streamlitメディア、テキストreaderの要件を先に分類する。

将来はresource object／stream APIと、スコープを持つ`asset_file()` context managerを組み合わせ、`asset_path()`をソースチェックアウト互換層として残す案を検討する。標準ライブラリのディレクトリ実体化はPython 3.12で拡張されたため、Python 3.10互換性も判断材料になる。Sprint 2A〜2CではこのAPI変更を行わない。

## 依存関係・cache・offline mode

- 別途レビューしたテスト要件が不可避でない限り、実証中は`requirements.txt`と`runtime.txt`を変更しない。
- インストール実証とCloud動作を確認するまで、`pyproject.toml`と`requirements.txt`のどちらを最終的な依存関係の正本にするか決めない。
- 将来の永続cacheはOSに適したユーザー書込み可能領域を使い、package資産・個人データと分離する。
- 将来の明示的offline modeは、現行の自動縮退を維持しつつ、接続確認と外部service／tile試行を抑止する。wheel実証には含めない。

## 保留: EnvGeo共通基盤

`envgeo-utils`共通packageの分離とEarthquakeアプリとの統合は、Seawaterの配布方式を実証するまで保留する。将来の候補は資産解決、cache配置、offline可否／縮退メッセージ、ローカル海岸線描画である。海水処理、同位体・GSW・GEBCOの意味付け、地震カタログ、プレート境界の解釈は各アプリに残す。

## package安定後の公開作業

正式package化の後に、対応OS試験、CI、`CITATION.cff`、第三者通知、タグ付きGitHub Release、Zenodo archive／DOI、JOSS準備を進める。ローカルwheel実証だけでは、これらの公開条件を満たしたことにはならない。

## Sprint 2D 確定設計判断

以下の判断はSprint 2B〜2C実証結果のレビュー後に確定した。Sprint 3の設計範囲と今後の実装提案を規定する。Sprint 2Dではコード、テスト、設定、dataset、同梱資産を変更していない。

1. **配布方式。** 正式配布は標準的なpip/wheelインストールで実ファイルを展開する。zip-importやzipapp方式は使わない。`importlib.resources`への移行条件は、非ファイルシステムresource loader、zipapp、組込み配布の場合であり、通常のpipインストールは対象外。

2. **資産ローダー。** `envgeo_assets.asset_path()`を同梱資産の主要公開APIとして維持する。`Path(__file__).resolve().parent`基準の解決方式はwheelインストールで正しく動作し、置換しない。

3. **`importlib.resources`移行なし、`open_asset()`追加なし、OSキャッシュへの実体化なし。** resource streamや一時実体化を必要とする具体的な利用箇所が生じるまでこれらを追加しない。Sprint 2Dの結論（`asset_path()`維持、resource API未導入）は変えない。ただし理由は「現行の通常wheelで安定動作し、移行の具体的要件がない」ことであり、全consumerがPath必須だからではない。pandasやStreamlitメディアは将来stream/bytes利用が検討できるconsumerだが、今回移行しない。

4. **Cartopy shapefile sidecarsとGEBCO NetCDF。** Natural Earth shapefileを読むCartopyのconsumerは`.shp`・`.shx`・`.dbf`を同一ディレクトリで扱う必要があるため、ディレクトリを含む実filesystem pathが必要である。現行GEBCO実装は`scipy.io.netcdf_file`へPathを渡しており、通常wheelの展開済みファイルでは安定して動作する。`scipy.io.netcdf_file`はfile-like objectも受け取れるため、APIが実Pathを必須と断定しない。いずれも引き続き`asset_path()`ベースのパスを使う。context managerやstream APIは導入しない。

5. **`home.py`の`BASE_DIR`。** `BASE_DIR = Path(__file__).resolve().parent`はHomeのMarkdown表示における相対画像の補助基準として維持する。`render_markdown_file()`への引数としてだけでなく、英日READMEを`render_markdown_streamlit(..., base_dir=BASE_DIR)`へ直接渡す箇所でも使用される。`resolve_path()`は`envgeo_assets.asset_path()`に委譲しており、`BASE_DIR`は並行する資産ローダーではない。

6. **EnvGeo専用永続OSキャッシュ不要。** 実証で`site-packages`への書込みがないことを確認した。このsprintでは永続cacheは不要。将来の設計（Sprint 3以降）はOSに適したユーザー書込み可能領域を使い、package資産・個人データと分離する。

7. **明示的offline起動。** 接続確認と外部service／tile試行を抑止する将来機能として維持する。wheel要件ではなく、テストのネットワーク分離の解決にも使わない。ネットワーク分離が必要なテストは`monkeypatch`を使う。`ENVGEO_OFFLINE`はテスト修正ツールではない。これは二つの別問題である。

8. **GEBCOは同梱を維持。** 外部downloadは追加しない。Pythonのoptional extraとも表現しない。Sprint 2Aから変更なし。

9. **`envgeo-utils`分離、Earthquake統合、`src/`移動。** Seawaterの配布方式が確立するまでいずれも保留。

10. **Sprint 3の範囲。** 依存関係の正本（`pyproject.toml`対`requirements.txt`）の決定、既存runtimeに依存しない新規クリーンインストール、package data allowlist監査（`bathymetry/make_lightweight_gebco.py`をSprint 3監査項目として記録）、OS／Cloud／CI互換性試験を設計する。
