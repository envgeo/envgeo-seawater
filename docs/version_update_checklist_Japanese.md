# 版番号更新チェックリスト

EnvGeo-Seawaterの新しい版を準備するたびに使う、再利用用のチェックリストです。
`release_checklist_Japanese.md`を補完します。この文書は版番号、文書、公開状態のずれを防ぎ、
個別リリースの検証根拠はrelease checklist側に記録します。

review済みstable commit、GitHub Release、意図したarchive状態が揃うまで、公開済み安定版とは
表現しません。過去のZenodo固定版DOIを新しい版に流用しません。

## 1. 公開状態を決める

- [ ] 対象版、候補日、対象repository、公開ページ範囲を記録する。
- [ ] 開発候補か公開済みreleaseかを区別する。GitHub Release公開前は「リリース候補」と表現する。
- [ ] 同期前にrepository固有の例外を列挙する（privateな`test2`のデータ・Home差分、安定版だけのページ範囲など）。

## 2. 版メタデータと回帰テストを更新する

- [ ] `pyproject.toml`を更新する。
- [ ] `envgeo_utils.APP_VERSION`と`APP_VERSION_DATE`を更新する。
- [ ] 定義している公開ページの表示用`version`を更新する。
- [ ] `test/test_basic.py`、`test/test_envgeo_utils.py`、その他の固定版テストを更新する。
- [ ] コードとテストから現行版を示す古い文字列を検索する。明示的な履歴記録だけは残す。

## 3. データと再現性の状態を確認する

- [ ] loaderが選ぶ全workbookが対象repositoryに存在し、意図したファイル名であることを確認する。
- [ ] 同梱workbookを変更する場合、loader path、重複・出典catalogue、来歴inventory、再配布監査、workbook notes、checksum記録を一体で更新する。
- [ ] privateまたは未公開データを公開repositoryへ入れない。`test2`固有データは、比較・判断なしに置き換えない。

## 4. 利用者向け文書を更新する

- [ ] 英日`README`の対応Python版、公開ページ範囲、公開状態を更新する。
- [ ] `CITATION.cff`と`data_text/citation*.md`を更新する。Zenodo archive前はconcept DOIを使い、固定版DOIは保留と明記する。
- [ ] Homeの**Update History**と`data_text/update_log*.md`を更新する。
- [ ] 埋め込みmanual（`data_text/manual*.md`）、ページ別manual、GitHub Pages入口`docs/index*.md`を更新する。
- [ ] `release_metadata_draft*.md`、`release_checklist*.md`、現行状態を示す開発ノートを更新する。過去releaseの証跡は保持する。
- [ ] `paper.md`は固定版Zenodo DOIが確定してから更新する。

## 5. repositoryを意図的に同期する

- [ ] 正規作業フォルダから、共通コード・文書を`envgeo-seawater`、`test2`、`seawater_map`へ同期する。
- [ ] 各cloneで`git status`を確認する。コピーしたファイルが追跡対象か確認し、GitHub Pagesを公開しないrepositoryへ入口ファイルを誤って追加しない。
- [ ] テスト専用asset、安定版で除外する開発ページなど、文書化したclone固有差分を保持する。
- [ ] 詳細な`TODO.md` / `TODO_Japanese.md`は開発用repositoryだけに保持する。
  `seawater_map`では、代わりに簡潔な英日`ROADMAP.md` / `ROADMAP_Japanese.md`を同期し、
  詳細ToDoが安定版releaseにstageされていないことを確認する。

## 6. 候補を検証する

- [ ] 変更コードと版番号アサーションのfocused testをローカルで実行する。
- [ ] repository health testを実行し、安定版repositoryからwheelを構築する。
- [ ] 影響する各repositoryでPython 3.10 / 3.12のCIが緑であることを確認する。
- [ ] Home、フィルター、地図、アップロード、変更ページをCloudで短く確認する。文書サイト更新後はGitHub Pagesも確認する。

## 7. 公開し、archiveメタデータを確定する

- [ ] review済みの安定版`seawater_map`をcommit・pushし、正確なcommit SHAを記録する。
- [ ] そのcommitからGitHub Releaseとtag（例：`v1.3.5`）を作成する。
- [ ] PyPI公開は、その配布を今回のrelease計画に含める場合だけ行う。
- [ ] Zenodoがtag付きreleaseをarchiveするのを待ち、全版共通concept DOIとは別に、新しい固定版DOIを記録する。
- [ ] `paper.md`、引用文、README、更新履歴、release metadataへDOIだけの最終更新を行い、CI・デプロイ済みアプリ・GitHub Pagesを再確認する。

## 完了記録

release notesまたはmetadata draftへ、次を記録します。

- 対象版と公開日
- stable commit SHAとGit tag
- GitHub Release URL
- Zenodo固定版DOIとconcept DOI
- 必要に応じたCI結果・URL、wheel名・checksum
- repository固有の除外項目・保留項目
