# Minimum Wage Effects on Leisure & Hospitality Employment
## Staggered Difference-in-Differences, 2018–2024

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Dingyi Zhuang** · MIT JTL Urban Mobility Lab · March 2026

This repository contains the full replication package for the paper:

> **"Minimum Wage Increases and Employment in Leisure & Hospitality: Evidence from Staggered State-Level Adoption, 2018–2024"**

---

## Key Results

| Estimator | ATT | SE | p-value | |
|-----------|----:|---:|--------:|---|
| TWFE | −0.0092 | 0.0020 | <0.001 | *** |
| Callaway-Sant'Anna (equal weights) | −0.0132 | 0.0067 | 0.048 | ** |
| Callaway-Sant'Anna (sample weights) | −0.0159 | 0.0048 | 0.001 | *** |
| Sun-Abraham (2021) | +0.0024 | 0.0012 | 0.056 | * |

**Pre-trend test:** Pre-treatment ATT coefficients (t = −6 to −1) all insignificant ✓  
**Interpretation:** TWFE and C&S estimates suggest a ~1–1.6% reduction in L&H employment from minimum wage increases; Sun-Abraham reveals heterogeneous treatment effects across adoption cohorts.

---

## Data

All data are from publicly available sources. **Raw data is NOT included** (re-downloadable via `code/01_download_data.py`). Cleaned panel data is included for convenience.

| Dataset | Source | Download |
|---------|--------|----------|
| L&H Employment by State | BLS State & Metro Area Employment Statistics (SAE) | [BLS SAE](https://www.bls.gov/sae/) |
| State Minimum Wage History | U.S. DOL Wage & Hour Division | [DOL WHD](https://www.dol.gov/agencies/whd/state/minimum-wage/history) |
| State Unemployment Rates | BLS Local Area Unemployment Statistics (LAUS) | [BLS LAUS](https://www.bls.gov/lau/) |

**Panel summary:** 51 states/DC × 28 quarters (2018Q1–2024Q4) = 1,428 state-quarter obs.  
21 never-treated states (federal floor $7.25); 30 ever-treated states across 9 treatment cohorts.

---

## Replication

### Requirements

```bash
python >= 3.10
pip install pandas numpy statsmodels scipy matplotlib linearmodels
```

Or install from the provided requirements file:

```bash
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# Step 1: Download raw data (requires internet; ~2 min)
python code/01_download_data.py

# Step 2: Clean data and construct panel
python code/02_clean_data.py

# Step 3: Run all estimators (TWFE, C&S 2021, Sun-Abraham 2021)
python code/03_analysis_main.py

# Step 4: Robustness checks and placebo tests
python code/04_robustness.py

# Step 5: Generate publication-quality figures (PDF + PNG)
python code/05_figures.py
```

Output: `data/cleaned/main_did_results.csv`, `data/cleaned/event_study_att.csv`, `paper/figures/*.pdf`

### Expected Runtime

| Script | Time |
|--------|------|
| 01_download_data.py | ~2 min (API calls) |
| 02_clean_data.py | <30 sec |
| 03_analysis_main.py | ~1 min |
| 04_robustness.py | ~1 min |
| 05_figures.py | ~30 sec |
| **Total** | **~5 min** |

---

## Repository Structure

```
auto-econ-research/
├── code/
│   ├── 01_download_data.py    # BLS SAE + DOL data download via API
│   ├── 02_clean_data.py       # Panel construction, treatment coding
│   ├── 03_analysis_main.py    # TWFE, Callaway-Sant'Anna, Sun-Abraham
│   ├── 04_robustness.py       # Placebo tests, sample restrictions
│   └── 05_figures.py          # Publication figures (PDF + PNG)
├── data/
│   ├── raw/                   # Downloaded raw data (git-ignored; re-download via script)
│   └── cleaned/               # Processed panel + results CSVs (included)
│       ├── panel.csv          # Main balanced panel (state × quarter)
│       ├── main_did_results.csv
│       ├── event_study_att.csv
│       └── treatment_timing.csv
├── paper/
│   └── figures/               # Generated figures (PDF + PNG)
│       ├── fig1_pretrends_timing.pdf
│       └── fig2_event_study_robustness.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

---


## Citation

```bibtex
@article{zhuang2026minwage,
  title   = {Minimum Wage Increases and Employment in Leisure \& Hospitality:
             Evidence from Staggered State-Level Adoption, 2018--2024},
  author  = {Zhuang, Dingyi},
  year    = {2026},
  note    = {Working Paper, MIT JTL Urban Mobility Lab},
  url     = {https://github.com/ZhuangDingyi/auto-econ-research}
}
```

---

## License

MIT License. See `LICENSE` for details.
