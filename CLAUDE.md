# CLAUDE.md — cane_cycle

## Project context

Portfolio project: Sentinel-2 time series analysis of two sugarcane fields
(`field_id` 1440 and 1460), 2020–2026.

Final goals:
- detect harvest (cut) dates;
- split cycles and seasons (Centre-South season: April to March);
- identify ratoon number (plant cane, ratoons, field renewal / reforma);
- phenology and relationship with climate.

Work proceeds one stage at a time. Stage 1 = harvest date detection.

Data: `data/raw/s2_fields_example.csv` — one row per (date, field), date as
`YYYYMMDD`, `field_id`, `cloudyPerc`, `granuleID`, `pixelTotal`, and S2 L2A
bands B2–B12 (B10 absent). Never modify files in `data/raw/`.

## About the user

Experienced in remote sensing and agriculture, beginner in Python.
**All explanations and discussions with the user must be in Portuguese.**
All code, variable names, docstrings, comments and README must be in English.

## WORK RULE (applies to the whole project)

Before implementing ANY function, discuss it with the user first:
- what the function will do and why;
- the theory or logic behind it (agronomic, remote sensing or statistical);
- inputs, outputs and parameters, with proposed default values;
- possible alternatives, with pros and cons, when there are any.

Only write the code after the user approves. After implementing, explain
what each part does and STOP for the user to review before moving on.

## Repository layout

- `src/cane_cycle/` — reusable package (`io.py`, `indices.py`, `plotting.py`, `harvest.py`)
- `notebooks/` — notebooks that import the package
- `data/raw/`, `data/processed/` — data (git-ignored)
- `figures/` — saved plots (file names include the index names)
- Virtual env: `.venv` (Windows: `.venv\Scripts\python`); package installed with `pip install -e .`
- Thresholds/parameters: module-level constants with defaults at the top of each module.
