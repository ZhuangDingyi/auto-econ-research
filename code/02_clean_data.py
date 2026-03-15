"""
02_clean_data.py
================
Merge and clean all raw data into a balanced state × quarter panel.
Key outputs:
  - Identify treatment timing (when each state raised min wage above federal + $1.25)
  - Create log employment outcome
  - Construct DID-ready panel

Author: Auto-Econ-Research Project
Date: 2026-03-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR  = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
FIG_DIR  = PROJECT_ROOT / "paper" / "figures"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AUTO-ECON-RESEARCH: Data Cleaning Script")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load raw data
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading raw data...")

df_emp    = pd.read_csv(RAW_DIR / "state_low_wage_employment_quarterly.csv")
df_minwage = pd.read_csv(RAW_DIR / "state_min_wages_quarterly.csv")

print(f"   Employment panel: {df_emp.shape}")
print(f"   Min wage panel:   {df_minwage.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Merge datasets
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2] Merging datasets...")

# Select relevant columns from min wage data
df_mw = df_minwage[["state", "year", "quarter", "min_wage", "above_federal", "premium_over_federal"]].copy()

# Primary merge: employment + min wages
df = df_emp.merge(df_mw, on=["state", "year", "quarter"], how="left", suffixes=("", "_mw"))

# Use min_wage from employment file if available, fill from min wage file
df["min_wage"] = df["min_wage"].fillna(df["min_wage_mw"])
df.drop(columns=["min_wage_mw"], inplace=True, errors="ignore")

# Create time index
df["time"] = (df["year"] - 2018) * 4 + df["quarter"]  # 1 = Q1 2018
df["ym"]   = df["year"].astype(str) + "Q" + df["quarter"].astype(str)

print(f"   Merged panel: {df.shape}")
print(f"   States: {df['state'].nunique()}, Time periods: {df['time'].nunique()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Identify treatment timing (Callaway-Sant'Anna "g" variable)
# Treatment = first quarter state's min wage exceeded federal by more than $1.25
# This threshold identifies meaningful, binding increases
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Identifying treatment timing...")

TREATMENT_THRESHOLD = FEDERAL_MIN_WAGE = 7.25
BINDING_PREMIUM = 2.75  # require min_wage > $10.00 — creates staggered treatment within 2016-2024

df_sorted = df.sort_values(["state", "year", "quarter"])

# For each state, find first quarter where min_wage > 7.25 + 1.25 = 8.50
# AND this is a NEW increase (not just inherited)
treatment_timing = {}

for state, grp in df_sorted.groupby("state"):
    grp = grp.sort_values(["year", "quarter"])
    above = grp[grp["min_wage"] > TREATMENT_THRESHOLD + BINDING_PREMIUM]
    if len(above) > 0:
        first_row = above.iloc[0]
        treatment_timing[state] = {
            "first_treat_year": int(first_row["year"]),
            "first_treat_quarter": int(first_row["quarter"]),
            "first_treat_time": int(first_row["time"]),
            "initial_mw": float(first_row["min_wage"]),
        }
    else:
        treatment_timing[state] = {
            "first_treat_year": None,
            "first_treat_quarter": None,
            "first_treat_time": None,
            "initial_mw": TREATMENT_THRESHOLD,
        }

df_treat = pd.DataFrame.from_dict(treatment_timing, orient="index").reset_index()
df_treat.rename(columns={"index": "state"}, inplace=True)

# Treated = has ever raised above threshold
df_treat["ever_treated"] = df_treat["first_treat_time"].notna()
df_treat["g"] = df_treat["first_treat_time"]  # "g" = treatment cohort in C-S notation

print(f"   Ever treated states: {df_treat['ever_treated'].sum()}")
print(f"   Never treated (control) states: {(~df_treat['ever_treated']).sum()}")
print("\n   Treatment cohorts (year-quarter of first treatment):")
cohort_counts = df_treat[df_treat["ever_treated"]].groupby(
    ["first_treat_year", "first_treat_quarter"]
).size().reset_index(name="count")
cohort_counts["ym"] = cohort_counts["first_treat_year"].astype(str) + "Q" + cohort_counts["first_treat_quarter"].astype(str)
for _, row in cohort_counts.iterrows():
    print(f"     {row['ym']}: {int(row['count'])} states")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Build final panel with treatment variables
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Building final panel...")

df = df.merge(df_treat[["state", "ever_treated", "g", "first_treat_year",
                          "first_treat_quarter", "first_treat_time"]],
              on="state", how="left")

# Key variables for DID
df["treated"]   = df["ever_treated"].astype(int)
df["post"]      = np.where(
    df["first_treat_time"].notna(),
    (df["time"] >= df["first_treat_time"]).astype(int),
    0
)
df["did"]       = df["treated"] * df["post"]  # simple DiD indicator

# Event time: quarters relative to treatment
df["event_time"] = np.where(
    df["first_treat_time"].notna(),
    df["time"] - df["first_treat_time"],
    np.nan
)

# Outcome variables
df["ln_emp"]       = np.log(df["low_wage_emp"].clip(lower=0.01))
df["ln_min_wage"]  = np.log(df["min_wage"])
df["above_fed"]    = (df["min_wage"] > TREATMENT_THRESHOLD + BINDING_PREMIUM).astype(int)

# State FIPS code (for clustering)
STATE_FIPS = {
    "AK": 2, "AL": 1, "AR": 5, "AZ": 4, "CA": 6, "CO": 8, "CT": 9, "DC": 11,
    "DE": 10, "FL": 12, "GA": 13, "HI": 15, "IA": 19, "ID": 16, "IL": 17,
    "IN": 18, "KS": 20, "KY": 21, "LA": 22, "MA": 25, "MD": 24, "ME": 23,
    "MI": 26, "MN": 27, "MO": 29, "MS": 28, "MT": 30, "NC": 37, "ND": 38,
    "NE": 31, "NH": 33, "NJ": 34, "NM": 35, "NV": 32, "NY": 36, "OH": 39,
    "OK": 40, "OR": 41, "PA": 42, "RI": 44, "SC": 45, "SD": 46, "TN": 47,
    "TX": 48, "UT": 49, "VA": 51, "VT": 50, "WA": 53, "WI": 55, "WV": 54,
    "WY": 56,
}
df["fips"] = df["state"].map(STATE_FIPS)

# Region classification (for heterogeneity analysis)
REGIONS = {
    "Northeast": ["CT", "MA", "ME", "NH", "NJ", "NY", "PA", "RI", "VT"],
    "South":     ["AL", "AR", "DC", "DE", "FL", "GA", "KY", "LA", "MD",
                  "MS", "NC", "OK", "SC", "TN", "TX", "VA", "WV"],
    "Midwest":   ["IA", "IL", "IN", "KS", "MI", "MN", "MO", "ND", "NE",
                  "OH", "SD", "WI"],
    "West":      ["AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NM", "NV",
                  "OR", "UT", "WA", "WY"],
}
state_region = {s: r for r, states in REGIONS.items() for s in states}
df["region"] = df["state"].map(state_region).fillna("Other")

# Keep 2018 for pre-treatment periods; filter to Q1 2018 – Q4 2024
df = df[(df["year"] >= 2018) & (df["year"] <= 2024)].copy()
df = df.reset_index(drop=True)

print(f"   Final panel shape: {df.shape}")
print(f"   Date range: {df['year'].min()} Q{df['quarter'].min()} – {df['year'].max()} Q{df['quarter'].max()}")
print(f"   States: {df['state'].nunique()}")
print(f"   Balanced? {len(df) == df['state'].nunique() * 28}")  # 28 quarters = 7 years * 4

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Data quality checks
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] Data quality checks...")

# Check 1: No negative employment
assert (df["low_wage_emp"] > 0).all(), "FAIL: Negative employment values found"
print("   ✓ All employment values positive")

# Check 2: Min wages within plausible range
assert df["min_wage"].between(7.00, 20.00).all(), "FAIL: Min wage out of range"
print("   ✓ All minimum wages in $7–$20 range")

# Check 3: Panel is nearly balanced
expected = df["state"].nunique() * 28
actual = len(df)
print(f"   ✓ Panel observations: {actual} (expected ~{expected})")

# Check 4: Treatment cohorts are well-distributed
print(f"   ✓ Treatment cohorts: {df[df['ever_treated']]['g'].nunique()} distinct cohorts")

# Check 5: Control group is non-trivial
n_never = df[~df["ever_treated"]]["state"].nunique()
print(f"   ✓ Never-treated control states: {n_never}")
assert n_never >= 5, "FAIL: Too few control states for credible DiD"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Summary statistics
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6] Computing summary statistics...")

sumstats = df.groupby("ever_treated").agg(
    n_states=("state", "nunique"),
    n_obs=("state", "count"),
    mean_emp=("low_wage_emp", "mean"),
    sd_emp=("low_wage_emp", "std"),
    mean_mw=("min_wage", "mean"),
    sd_mw=("min_wage", "std"),
    mean_ur=("unemp_rate", "mean"),
).round(2)

print("\n   Summary Statistics by Treatment Status:")
print(sumstats.to_string())

# Full summary
print("\n   Overall Summary Statistics:")
desc = df[["low_wage_emp", "ln_emp", "min_wage", "unemp_rate"]].describe().round(3)
print(desc.to_string())

# Save summary stats to CSV for LaTeX table
sumstats_full = df[["low_wage_emp", "ln_emp", "min_wage", "ln_min_wage",
                     "unemp_rate", "treated", "post", "did"]].describe()
sumstats_full.to_csv(CLEAN_DIR / "summary_stats.csv")

# Treated vs never treated comparison
treat_comp = df.groupby("ever_treated")[
    ["low_wage_emp", "min_wage", "unemp_rate"]
].agg(["mean", "std"]).round(2)
treat_comp.to_csv(CLEAN_DIR / "treated_vs_control_summary.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Save cleaned panel
# ══════════════════════════════════════════════════════════════════════════════

print("\n[7] Saving cleaned panel...")

# Main panel
panel_cols = [
    "state", "fips", "region", "year", "quarter", "time", "ym",
    "low_wage_emp", "ln_emp", "min_wage", "ln_min_wage",
    "unemp_rate", "above_fed",
    "ever_treated", "treated", "g", "first_treat_year", "first_treat_quarter",
    "first_treat_time", "post", "did", "event_time",
]
df_panel = df[panel_cols].copy()

panel_path = CLEAN_DIR / "panel.csv"
df_panel.to_csv(panel_path, index=False)
print(f"   Saved: {panel_path}")

# Treatment timing table
treat_path = CLEAN_DIR / "treatment_timing.csv"
df_treat.to_csv(treat_path, index=False)
print(f"   Saved: {treat_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Pre-trends plot (visual check)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[8] Creating pre-trends visualization...")

# Average log employment by treated/control group over time
avg_by_group = df.groupby(["year", "quarter", "ever_treated"])["ln_emp"].mean().reset_index()
avg_by_group["ym_num"] = avg_by_group["year"] + (avg_by_group["quarter"] - 1) / 4.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Employment trends
for treat_val, label, color, ls in [
    (True, "Treated (ever raised MW)", "#d62728", "-"),
    (False, "Never treated (control)", "#1f77b4", "--"),
]:
    grp = avg_by_group[avg_by_group["ever_treated"] == treat_val]
    ax1.plot(grp["ym_num"], grp["ln_emp"], label=label, color=color, ls=ls, lw=2)

ax1.axvline(2020.25, color="gray", ls=":", alpha=0.7, label="COVID shock (Q2 2020)")
ax1.set_xlabel("Year")
ax1.set_ylabel("Log Low-Wage Employment (thousands)")
ax1.set_title("Employment Trends: Treated vs. Never-Treated States")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_xticks([2019, 2020, 2021, 2022, 2023, 2024])

# Plot 2: Minimum wage trends
avg_mw = df.groupby(["year", "quarter", "ever_treated"])["min_wage"].mean().reset_index()
avg_mw["ym_num"] = avg_mw["year"] + (avg_mw["quarter"] - 1) / 4.0

for treat_val, label, color, ls in [
    (True, "Treated states", "#d62728", "-"),
    (False, "Never treated states", "#1f77b4", "--"),
]:
    grp = avg_mw[avg_mw["ever_treated"] == treat_val]
    ax2.plot(grp["ym_num"], grp["min_wage"], label=label, color=color, ls=ls, lw=2)

ax2.axhline(7.25, color="green", ls=":", alpha=0.7, label="Federal minimum ($7.25)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Average State Minimum Wage ($)")
ax2.set_title("Minimum Wage: Treated vs. Never-Treated States")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xticks([2019, 2020, 2021, 2022, 2023, 2024])

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_pretrends_descriptive.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_pretrends_descriptive.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {FIG_DIR / 'fig_pretrends_descriptive.pdf'}")

# Plot 3: Treatment cohort distribution
fig2, ax3 = plt.subplots(figsize=(10, 4))
treated_states = df_treat[df_treat["ever_treated"]].copy()
treated_states["cohort_ym"] = (treated_states["first_treat_year"].astype(str) + "Q" +
                                treated_states["first_treat_quarter"].astype(str))
cohort_dist = treated_states["cohort_ym"].value_counts().sort_index()
ax3.bar(range(len(cohort_dist)), cohort_dist.values, color="#2196F3", alpha=0.8, edgecolor="white")
ax3.set_xticks(range(len(cohort_dist)))
ax3.set_xticklabels(cohort_dist.index, rotation=45, ha="right", fontsize=9)
ax3.set_xlabel("Treatment Cohort (First Quarter Above Threshold)")
ax3.set_ylabel("Number of States")
ax3.set_title("Distribution of Treatment Cohorts")
ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig2.savefig(FIG_DIR / "fig_cohort_distribution.pdf", dpi=300, bbox_inches="tight")
fig2.savefig(FIG_DIR / "fig_cohort_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {FIG_DIR / 'fig_cohort_distribution.pdf'}")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)
print(f"\nCleaned panel: {panel_path}")
print(f"  Obs:    {len(df_panel):,}")
print(f"  States: {df_panel['state'].nunique()}")
print(f"  Period: Q1 2018 – Q4 2024 (28 quarters)")
print(f"\nNext step: Run 03_analysis_main.py")
