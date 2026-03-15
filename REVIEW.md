# Reproducibility Audit Report

**Project:** Minimum Wage Effects on Leisure & Hospitality Employment  
**Reviewer:** OpenClaw Code Review Agent  
**Date:** 2026-03-15  
**Audit scope:** Full end-to-end pipeline execution + code quality review

---

## Summary

The replication package **passes** end-to-end reproducibility. All four pipeline scripts execute without errors, all ATT point estimates match README exactly, all expected figures are generated, and pre-trend tests support the parallel trends assumption. Three bugs were found and fixed.

---

## Checklist

### 1. Setup & Dependencies

| Check | Status | Notes |
|-------|--------|-------|
| README.md present | ✅ | Clear pipeline instructions |
| requirements.txt present | ✅ | 8 packages listed with minimum versions |
| venv present and functional | ✅ | `source venv/bin/activate` works |
| All imports resolvable | ✅ | pandas, numpy, statsmodels, scipy, matplotlib, linearmodels, patsy, requests all available |
| Raw data present | ✅ | `data/raw/` has 6 files (employment, min wages, unemployment) |
| Cleaned data present | ✅ | 13 output CSVs in `data/cleaned/` |

### 2. Code Quality

| Check | Status | Notes |
|-------|--------|-------|
| No hardcoded paths | ✅ | All scripts use `Path(__file__).parent.parent` correctly |
| Pipeline order documented | ✅ | Steps 01–05 clearly numbered |
| Scripts match README pipeline | ⚠️ | Two undocumented scripts: `01_build_panel.py` and `02_analysis.py` exist in `code/` but not in README (appear to be legacy drafts) |
| Code style | ✅ | Well-commented, logical section headers, docstrings |
| Import organization | ✅ | All imports at top of each file; no missing imports |
| Data assertions | ✅ | `02_clean_data.py` includes 5 data quality assertions |

### 3. Pipeline Execution

| Script | Status | Runtime | Output |
|--------|--------|---------|--------|
| `02_clean_data.py` | ✅ | <5s | panel.csv (1,428 obs), treatment_timing.csv, 2 figures |
| `03_analysis_main.py` | ✅ | <10s | main_did_results.csv, att_gt_estimates.csv, event_study_att.csv, sa_event_study.csv, 1 figure |
| `04_robustness.py` | ✅ | ~15s (200 placebo simulations) | placebo_results.csv, threshold_robustness.csv, preperiod_robustness.csv, 2 figures |
| `05_figures.py` | ✅ | <5s | 4 publication figures (PDF + PNG) |

### 4. Result Verification

**main_did_results.csv — ATT values vs. README:**

| Estimator | README ATT | Reproduced ATT | Match? |
|-----------|-----------|----------------|--------|
| TWFE | −0.0092 | −0.0092 | ✅ |
| C&S equal weights | −0.0132 | −0.0132 | ✅ |
| C&S sample weights | −0.0159 | −0.0159 | ✅ |
| Sun-Abraham (2021) | +0.0024 | +0.0024 | ✅ |

**Figures generated (paper/figures/):**

| Figure | Status |
|--------|--------|
| fig1_event_study_polished.pdf | ✅ |
| fig2_data_overview.pdf | ✅ |
| fig3_summary_stats.pdf | ✅ |
| figA1_raw_trends.pdf | ✅ |
| fig_placebo_distribution.pdf | ✅ |
| fig_robustness_coefplot.pdf | ✅ |
| fig_pretrends_descriptive.pdf | ✅ |
| fig_cohort_distribution.pdf | ✅ |
| (+ PNG copies of all above) | ✅ |

**Pre-trend validity:**
All pre-treatment ATT(g,t) estimates (event times −6 to −1) are statistically insignificant and economically small, supporting the parallel trends assumption. ✅

**Placebo test:** Placebo p-value = 0.0000 — true ATT is far in the tail of the permutation distribution. ✅

**Robustness:** Results stable across sample restrictions (excl. CA/NY/TX: −0.0100; excl. COVID: −0.0088) and alternative wage thresholds ($7.75–$10.25 all negative). ✅

### 5. Bugs Found & Fixed

#### Bug 1: Incorrect panel obs count in quality check print (cosmetic) — **FIXED**
- **File:** `code/02_clean_data.py`, Step 5 Check 3
- **Issue:** `expected = df["state"].nunique() * 24` — used 24 quarters, but the panel covers 28 quarters (2018Q1–2024Q4 = 7 years × 4 quarters)
- **Fix:** Changed `24` → `28`. The assertion was a print statement only (no crash), but the displayed "expected" count was misleading.

#### Bug 2: Calendar year labels off by one for Q4 observations — **FIXED**
- **File:** `code/03_analysis_main.py`, lines 589–590
- **Issue:** `cal_year = t // 4 + 2018` is wrong. With `time = (year-2018)*4 + quarter`, this formula gives year+1 for all Q4 periods and reports 2025 for 2024Q4.
  - t=4 (2018Q4): `4//4 + 2018 = 2019` ✗ (should be 2018)  
  - t=28 (2024Q4): `28//4 + 2018 = 2025` ✗ (should be 2024)
- **Fix:** Changed to `(t - 1) // 4 + 2018`. Also fixed `treat_year` and `g_year`/`g_quarter` labels in the `callaway_santanna()` function with the same correction. **Note:** This bug did NOT affect any ATT point estimates or SEs — only the calendar-year axis labels in `att_by_year.csv`.

#### Bug 3: README panel description had wrong counts — **FIXED**
- **File:** `README.md`
- **Issue:** "47 states × 28 quarters = 1,316 obs; 21 never-treated + 26 ever-treated = 47 total" — all wrong. The panel includes DC + 50 states = 51 total; 30 are ever-treated.
- **Fix:** Updated to "51 states/DC × 28 quarters = 1,428 obs; 21 never-treated + 30 ever-treated."

---

## Potential Concerns (Not Bugs, But Worth Noting)

### Pre-period sensitivity (Robustness Check 5)
The TWFE estimate flips sign depending on how many pre-treatment quarters are included:
- 2Q pre-period: ATT = +0.0378 (positive, significant)
- 3Q pre-period: ATT = +0.0214
- 4Q pre-period: ATT = +0.0054
- 6Q pre-period: ATT = −0.0084
- 8Q pre-period: ATT = −0.0156 (= baseline direction)

This sign instability for short windows is likely because short windows exclude earlier-treated states, creating sample composition issues. It is worth a sentence of explanation in the paper since a reader could flag this.

### Undocumented legacy scripts
`code/01_build_panel.py` and `code/02_analysis.py` exist but are not in the README pipeline or `.gitignore`. They appear to be earlier drafts. Recommend either documenting or removing them to avoid confusion.

### C-S estimator is simplified
The implemented `callaway_santanna()` function is a hand-coded approximation (mean-differences DiD), not the full doubly-robust IPW estimator from the original paper. SEs use an analytical formula based on variance decomposition rather than the bootstrap. This is appropriate for a replication exercise but should be noted in the paper's methods section.

---

## Reproducibility Verdict

✅ **REPRODUCIBLE** — Full pipeline runs end-to-end, all claimed ATT values match, all figures are generated, and three bugs (none affecting point estimates) have been corrected.
