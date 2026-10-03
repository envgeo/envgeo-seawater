# Test Notes

[日本語版](README_Japanese.md)

This folder contains small `pytest` checks for EnvGeo-Seawater.

In simple terms, each test is an automatic checklist item. For example:

- Can Python import the main utility module?
- Can the bundled datasets be loaded?
- Are important numeric columns really numeric after loading?
- Do loaded data columns stay within broad physical ranges?
- Are placeholder strings such as `**` removed before analysis?
- Are known invalid values converted to `NaN` while keeping their original values visible?
- Does `insert_gap_rows()` add blank rows only where plot lines should break?
- Do the released stable Streamlit pages still compile as valid Python?
- In the development repository, do optional local/development pages still
  compile without changing the stable public-page scope?
- Do README image links point to files that actually exist?
- Do public-facing README/update files avoid internal submission-status wording?
- Does the stable public surface contain only its declared visualization pages?

Run tests from the `envgeo_seawater_v130` directory:

```bash
python -m pytest -q
```

GitHub Actions repeats the suite on Python 3.10 and 3.12, builds a wheel, and
checks the isolated installed wheel. See `docs/testing.md` and its Japanese
counterpart for the full CI and manual-QA boundary.

`pytest` may create `.pytest_cache/`, and Python may create `__pycache__/`.
Those files are local generated files and are ignored by `.gitignore`.

## How To Read A Test

A line such as:

```python
assert not df.empty
```

means "the loaded table must not be empty." If the condition is false, pytest
stops and reports which check failed.

The goal is to grow these tests from simple checks into a reliable safety net
for the main scientific workflow: load data, clean data, filter data, prepare
figures, and handle user-supplied data.
