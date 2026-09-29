# cane_cycle

Time series analysis of Sentinel-2 imagery over two sugarcane fields
(`field_id` 1440 and 1460), 2020–2026.

## Goals

1. Detect harvest dates.
2. Split the series into crop cycles and seasons (Centre-South season: April to March).
3. Identify the ratoon number (plant cane, ratoons, sugarcane field renovation).
4. Describe phenology and its relationship with climate.

## Project layout

```
cane_cycle/
├── src/cane_cycle/     # reusable package (io, indices, plotting, harvest)
├── notebooks/          # exploratory and analysis notebooks
├── data/raw/           # original input data (not versioned)
├── data/processed/     # derived tables (not versioned)
├── figures/            # saved plots
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Data

A CSV with one row per (date, field): acquisition date (`YYYYMMDD`),
`field_id`, cloud percentage and Sentinel-2 L2A band reflectances.
