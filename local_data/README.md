# Optional local user data

Place a researcher-owned file at `local_data/user_data.xlsx` to use it as the
always-loaded local user table when EnvGeo-Seawater runs locally. CSV, XLSX, and XLS files can
also be selected with the `ENVGEO_LOCAL_USER_DATA_PATH` environment variable:

```bash
ENVGEO_LOCAL_USER_DATA_PATH=/absolute/path/to/user_data.xlsx streamlit run home.py
```

The data pass through the same column standardization and quality checks as a
browser upload. The loader assigns every row the dataset name `User Excel data`
and combines the table with each selected reference source. It is available in
Data filtering immediately after app startup. Browser `Uploaded data` remain a
separate session-only category; selecting both categories uses both datasets.
Restart Streamlit or clear its data cache after replacing or editing the local
table.

Spreadsheet values copied from other sources can contain invisible Unicode
whitespace. The loader removes common regular, non-breaking, narrow
non-breaking, and full-width spaces before numeric conversion. Other invalid
values remain missing and are reported through the usual quality handling.

Files in this directory are ignored by Git. Do not commit measurement data.
The reference-data loader reads this table only when it exists or an explicit
path is configured. Public installations without a local table continue with
the bundled reference data only.
Automated tests set `ENVGEO_DISABLE_LOCAL_USER_DATA=1` so results never depend
on a researcher's local workbook.

## ローカルユーザーデータ

EnvGeo-Seawaterをローカル実行するとき、研究者自身のデータを
`local_data/user_data.xlsx` に置くと、ローカルユーザーの常時読み込み用表として利用します。別のCSV、
XLSX、XLSを使う場合は、`ENVGEO_LOCAL_USER_DATA_PATH` でファイルを
指定してください。

ローカルデータにはブラウザアップロードと同じ列名標準化と品質検査を
適用し、全行に `User Excel data` というデータセット名を付けます。
選択した参照データに結合され、アプリ起動時からData filteringで選択できます。
ブラウザの `Uploaded data` は別のセッション限定データです。
ローカル表を置き換えたり編集した後は、Streamlitを再起動するかデータキャッシュを
クリアしてください。

表計算ソフトからコピーした値には、ノーブレークスペースや全角空白などの
見えないUnicode空白が混入することがあります。ローダーは主な空白を除去してから
数値化します。その他の不正値は欠損値とし、通常の品質処理の対象にします。

`local_data` 内の測定データはGit管理対象外です。ローカル表がない
公開環境では、同梱の参照データだけを読み込みます。
自動テストでは `ENVGEO_DISABLE_LOCAL_USER_DATA=1` を設定し、個人の
ローカルブックに結果が左右されないようにしています。
