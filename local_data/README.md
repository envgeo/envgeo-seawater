# Bundled User Excel data sample

`user_data.xlsx` is a public, zero-row sample workbook bundled with
EnvGeo-Seawater. It demonstrates the always-loaded `User Excel data` workflow
without containing researcher measurements. Unless an external data path is
configured, the app reads this workbook at startup.

For the Japanese version of this guide, see `README_Japanese.md`.

## Use researcher-owned data

Choose one of the following methods.

1. Enter data into `local_data/user_data.xlsx` for local use.
2. Set `ENVGEO_LOCAL_USER_DATA_PATH` to another CSV, XLSX, or XLS file.

```bash
ENVGEO_LOCAL_USER_DATA_PATH=/absolute/path/to/user_data.xlsx streamlit run home.py
```

With either method, the shared loader applies the same column
standardization and quality checks used for browser uploads. It labels rows as
`User Excel data` and combines them with the selected reference source.
Browser `Uploaded data` remains a separate, session-only category; selecting
both categories includes both datasets.

## Local use and public release

`user_data.xlsx` is tracked as a public sample. You may edit it for local
analysis. Before committing or synchronizing a public clone, restore the
zero-row sample and do not commit researcher-owned, unpublished, restricted,
or otherwise non-redistributable data.

After editing either the bundled workbook or an externally configured file,
restart Streamlit or clear its data cache before reloading the table.

## Data cleaning and tests

The loader removes common regular, non-breaking, narrow non-breaking, and
full-width whitespace before numeric conversion. Values that remain invalid are
handled as missing by the usual quality checks.

Automated tests set `ENVGEO_DISABLE_LOCAL_USER_DATA=1`, so test results do not
depend on the local workbook or any external path.
