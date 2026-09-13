# Test Notes

This folder contains small `pytest` checks for EnvGeo-Seawater.

In simple terms, each test is an automatic checklist item. For example:

- Can Python import the main utility module?
- Can the bundled datasets be loaded?
- Are important numeric columns really numeric after loading?
- Do loaded data columns stay within broad physical ranges?
- Are placeholder strings such as `**` removed before analysis?
- Are known invalid values converted to `NaN` while keeping their original values visible?
- Does `insert_gap_rows()` add blank rows only where plot lines should break?
- Do the stable Streamlit pages still compile as valid Python?
- Do README image links point to files that actually exist?
- Do public-facing README/update files avoid internal submission-status wording?
- Does `pages/` contain only the stable visualization pages shown in Streamlit?

Run tests from the `envgeo_seawater_v130` directory:

```bash
pytest
```

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
