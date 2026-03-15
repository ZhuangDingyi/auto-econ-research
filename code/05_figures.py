"""
05_figures.py
=============
Publication-quality figures for the paper.

Generates:
- Figure 1: Event study (already in 03_analysis_main.py, but polished here)
- Figure 2: Treatment timing map / cohort distribution
- Figure 3: Minimum wage trends by state
- Figure A1: Bacon decomposition weights (appendix)

Author: Auto-Econ-Research Project
Date: 2026-03-15
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Publication style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
FIG_DIR   = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AUTO-ECON-RESEARCH: Figures Script")
print("=" * 60)

# Load data
df       = pd.read_csv(CLEAN_DIR / "panel.csv")
df_event = pd.read_csv(CLEAN_DIR / "event_study_att.csv")
df_treat = pd.read_csv(CLEAN_DIR / "treatment_timing.csv")
df_main  = pd.read_csv(CLEAN_DIR / "main_did_results.csv")

try:
    df_sa = pd.read_csv(CLEAN_DIR / "sa_event_study.csv")
except FileNotFoundError:
    df_sa = None

print(f"Panel: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Polished Event Study
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Fig 1] Event study plot...")

cs_pre  = df_event[df_event["event_time"] < 0].copy()
cs_post = df_event[df_event["event_time"] >= 0].copy()

fig, ax = plt.subplots(figsize=(10, 6))

# Shaded regions
ax.axvspan(-6.5, -0.5, alpha=0.04, color="gray", label=None)
ax.axvspan(-0.5, 8.5,  alpha=0.04, color="steelblue", label=None)

# Pre-treatment estimates
ax.errorbar(cs_pre["event_time"], cs_pre["att"],
            yerr=1.96 * cs_pre["se"],
            fmt="o-", color="#455A64", ecolor="#90A4AE",
            elinewidth=1.5, capsize=4, markersize=6, lw=1.5,
            label="Pre-treatment (placebo)")

# Post-treatment estimates
ax.errorbar(cs_post["event_time"], cs_post["att"],
            yerr=1.96 * cs_post["se"],
            fmt="s-", color="#1565C0", ecolor="#90CAF9",
            elinewidth=1.5, capsize=4, markersize=7, lw=2,
            label="Post-treatment ATT")

# Zero line
ax.axhline(0, color="black", lw=1.2, ls="-", alpha=0.5)
# Treatment line
ax.axvline(-0.5, color="#C62828", lw=2, ls="--", alpha=0.9, label="Treatment (t=0)")

# Reference text
ax.text(-3.2, cs_pre["att"].min() * 0.7, "Pre-treatment\nperiod", ha="center",
        fontsize=8.5, color="gray", style="italic")
ax.text(4, cs_post["att"].max() * 0.8 if cs_post["att"].max() != 0 else 0.02,
        "Post-treatment\nperiod", ha="center", fontsize=8.5, color="#1565C0", style="italic")

ax.set_xlabel("Quarters Relative to Minimum Wage Increase (Event Time)", fontsize=11)
ax.set_ylabel(r"$\widehat{ATT}(g,t)$: Log Low-Wage Employment", fontsize=11)
ax.set_title("Figure 1: Event Study — Effect of State Minimum Wage Increases\n"
             "on Low-Wage Employment (Callaway–Sant'Anna 2021 Estimator)", fontsize=11)
ax.legend(loc="lower right", framealpha=0.9)
ax.set_xticks(range(-6, 9))
ax.set_xlim(-6.8, 8.8)

# Add overall ATT annotation
cs_att = df_main[df_main["estimator"] == "CS-2021 (equal weights)"]
if len(cs_att) > 0:
    att_val = cs_att["att"].values[0]
    se_val  = cs_att["se"].values[0]
    p_val   = cs_att["p_val"].values[0]
    sig_stars = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
    ax.text(0.02, 0.97,
            f"Overall ATT = {att_val:.4f}{sig_stars}\n(SE = {se_val:.4f})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#bbb"))

plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_event_study_polished.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig1_event_study_polished.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {FIG_DIR / 'fig1_event_study_polished.pdf'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Employment and Minimum Wage Trends
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Fig 2] Employment and minimum wage trends...")

# Select a few illustrative states
highlight_treated = ["CA", "WA", "NJ", "IL", "FL"]
highlight_control = ["TX", "GA", "MS", "ND"]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# ── Panel (a): Log employment by treatment status ──
ax = axes[0, 0]
treated_avg = df[df["ever_treated"]].groupby(["year", "quarter"])["ln_emp"].mean().reset_index()
control_avg = df[~df["ever_treated"]].groupby(["year", "quarter"])["ln_emp"].mean().reset_index()

for df_agg, label, color, ls in [
    (treated_avg, "Treated states (ever raised MW)", "#D32F2F", "-"),
    (control_avg, "Never-treated states",           "#1565C0", "--"),
]:
    x = df_agg["year"] + (df_agg["quarter"] - 1) / 4
    ax.plot(x, df_agg["ln_emp"], label=label, color=color, ls=ls, lw=2.2)

ax.axvspan(2020.25, 2021.0, alpha=0.12, color="orange", label="COVID disruption")
ax.set_xlabel("Year")
ax.set_ylabel("Avg. Log Low-Wage Employment")
ax.set_title("(a) Employment Trends by Treatment Status")
ax.legend(fontsize=8)
ax.set_xticks([2019, 2020, 2021, 2022, 2023, 2024])

# ── Panel (b): Min wage trends for selected states ──
ax = axes[0, 1]
colors_treat = plt.cm.Reds(np.linspace(0.4, 0.9, len(highlight_treated)))
colors_ctrl  = plt.cm.Blues(np.linspace(0.4, 0.9, len(highlight_control)))

for state, color in zip(highlight_treated, colors_treat):
    s_df = df[df["state"] == state].sort_values(["year", "quarter"])
    x = s_df["year"] + (s_df["quarter"] - 1) / 4
    ax.plot(x, s_df["min_wage"], color=color, lw=1.8, label=state)

for state, color in zip(highlight_control, colors_ctrl):
    s_df = df[df["state"] == state].sort_values(["year", "quarter"])
    x = s_df["year"] + (s_df["quarter"] - 1) / 4
    ax.plot(x, s_df["min_wage"], color=color, lw=1.8, ls="--", label=state)

ax.axhline(7.25, color="black", lw=1.5, ls=":", alpha=0.8, label="Federal min ($7.25)")
ax.set_xlabel("Year")
ax.set_ylabel("State Minimum Wage ($)")
ax.set_title("(b) Minimum Wage Trends: Selected States")
ax.legend(ncol=2, fontsize=7)
ax.set_xticks([2019, 2020, 2021, 2022, 2023, 2024])

# ── Panel (c): Distribution of treatment cohorts ──
ax = axes[1, 0]
df_treat_ever = df_treat[df_treat["ever_treated"]].copy()
df_treat_ever["cohort_label"] = (df_treat_ever["first_treat_year"].astype(int).astype(str) + "Q" +
                                  df_treat_ever["first_treat_quarter"].astype(int).astype(str))
cohort_counts = df_treat_ever["cohort_label"].value_counts().sort_index()

bars = ax.bar(range(len(cohort_counts)), cohort_counts.values,
              color="#42A5F5", edgecolor="white", linewidth=1.2, alpha=0.85)
ax.set_xticks(range(len(cohort_counts)))
ax.set_xticklabels(cohort_counts.index, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Treatment Cohort")
ax.set_ylabel("Number of States")
ax.set_title("(c) Distribution of Treatment Cohorts")
for bar, count in zip(bars, cohort_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            str(count), ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# ── Panel (d): ATT by post-event year ──
ax = axes[1, 1]
try:
    df_att_year = pd.read_csv(CLEAN_DIR / "att_by_year.csv")
    x_pos = range(len(df_att_year))
    ax.bar(x_pos, df_att_year["att_mean"], yerr=1.96*df_att_year["att_se"],
           color=["#EF5350" if v < 0 else "#66BB6A" for v in df_att_year["att_mean"]],
           alpha=0.8, edgecolor="white", capsize=5, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_att_year["cal_year"].astype(int), fontsize=9)
    ax.axhline(0, color="black", lw=1.5, alpha=0.7)
    ax.set_xlabel("Calendar Year")
    ax.set_ylabel("Average ATT(g,t)")
    ax.set_title("(d) Heterogeneous ATT by Post-Treatment Year")
except FileNotFoundError:
    ax.text(0.5, 0.5, "ATT by year\nnot available", transform=ax.transAxes,
            ha="center", va="center")

plt.suptitle("Figure 2: Data Overview — Employment, Minimum Wages, and Treatment Timing",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_data_overview.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig2_data_overview.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {FIG_DIR / 'fig2_data_overview.pdf'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Summary statistics visualization
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Fig 3] Summary statistics visualization...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# (a) Distribution of minimum wages
ax = axes[0]
mw_treat  = df[df["ever_treated"]]["min_wage"]
mw_ctrl   = df[~df["ever_treated"]]["min_wage"]
bins = np.linspace(7, 18, 25)
ax.hist(mw_treat, bins=bins, alpha=0.7, label="Treated states", color="#EF5350", density=True)
ax.hist(mw_ctrl,  bins=bins, alpha=0.7, label="Control states", color="#42A5F5", density=True)
ax.axvline(7.25, color="black", ls="--", lw=1.5, label="Federal ($7.25)")
ax.set_xlabel("Minimum Wage ($)")
ax.set_ylabel("Density")
ax.set_title("(a) Min Wage Distribution")
ax.legend(fontsize=8)

# (b) Distribution of log employment
ax = axes[1]
for ever_treat, label, color in [(True, "Treated", "#EF5350"), (False, "Control", "#42A5F5")]:
    emp_vals = df[df["ever_treated"] == ever_treat]["ln_emp"].dropna()
    ax.hist(emp_vals, bins=25, alpha=0.7, label=label, color=color, density=True)
ax.set_xlabel("Log Low-Wage Employment")
ax.set_ylabel("Density")
ax.set_title("(b) Employment Distribution")
ax.legend(fontsize=8)

# (c) Unemployment rate over time
ax = axes[2]
ur_treated = df[df["ever_treated"]].groupby(["year", "quarter"])["unemp_rate"].mean()
ur_control = df[~df["ever_treated"]].groupby(["year", "quarter"])["unemp_rate"].mean()

def to_float_time(idx):
    years  = [i[0] for i in idx]
    quarts = [i[1] for i in idx]
    return [y + (q - 1) / 4 for y, q in zip(years, quarts)]

ax.plot(to_float_time(ur_treated.index), ur_treated.values,
        color="#EF5350", lw=2, label="Treated states")
ax.plot(to_float_time(ur_control.index), ur_control.values,
        color="#42A5F5", lw=2, ls="--", label="Control states")
ax.set_xlabel("Year")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_title("(c) Unemployment Rate Trends")
ax.legend(fontsize=8)
ax.set_xticks([2019, 2020, 2021, 2022, 2023, 2024])

plt.suptitle("Figure 3: Summary Statistics", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_summary_stats.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig3_summary_stats.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {FIG_DIR / 'fig3_summary_stats.pdf'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A1 (Appendix): Parallel trends raw visualization
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Fig A1] Parallel trends visualization...")

# Show raw employment trends for a subset of states
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: treated states (2021 cohort)
ax = axes[0]
cohort_2021_states = df_treat[
    (df_treat["ever_treated"]) &
    (df_treat["first_treat_year"] == 2021)
]["state"].tolist()

# If no 2021 cohort, use any treated
if not cohort_2021_states:
    cohort_2021_states = df_treat[df_treat["ever_treated"]]["state"].head(6).tolist()

colors_set = plt.cm.tab10(np.linspace(0, 0.9, min(6, len(cohort_2021_states))))
for state, color in zip(cohort_2021_states[:6], colors_set):
    s_df = df[df["state"] == state].sort_values(["year", "quarter"])
    x = s_df["year"] + (s_df["quarter"] - 1) / 4
    treat_time = s_df["first_treat_time"].iloc[0]
    treat_year = s_df["first_treat_year"].iloc[0] + (s_df["first_treat_quarter"].iloc[0] - 1) / 4 if not pd.isna(s_df["first_treat_time"].iloc[0]) else None
    ax.plot(x, s_df["ln_emp"], color=color, lw=1.8, label=state)
    if treat_year:
        ax.axvline(treat_year, color=color, lw=0.8, ls=":", alpha=0.5)

ax.set_xlabel("Year")
ax.set_ylabel("Log Low-Wage Employment")
ax.set_title("(a) Treated States: Individual Employment Trends\n(dashed vertical lines = treatment time)")
ax.legend(ncol=2, fontsize=8)

# Right: never-treated control states
ax = axes[1]
control_states = df_treat[~df_treat["ever_treated"]]["state"].tolist()[:6]
colors_ctrl = plt.cm.tab20b(np.linspace(0, 0.9, len(control_states)))

for state, color in zip(control_states, colors_ctrl):
    s_df = df[df["state"] == state].sort_values(["year", "quarter"])
    x = s_df["year"] + (s_df["quarter"] - 1) / 4
    ax.plot(x, s_df["ln_emp"], color=color, lw=1.8, label=state)

ax.set_xlabel("Year")
ax.set_ylabel("Log Low-Wage Employment")
ax.set_title("(b) Never-Treated Control States: Employment Trends")
ax.legend(ncol=2, fontsize=8)

plt.suptitle("Appendix Figure A1: Raw Employment Trends by State", fontsize=11)
plt.tight_layout()
fig.savefig(FIG_DIR / "figA1_raw_trends.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "figA1_raw_trends.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {FIG_DIR / 'figA1_raw_trends.pdf'}")

print("\n" + "=" * 60)
print("FIGURES COMPLETE")
print("=" * 60)
print(f"\nFigures saved to: {FIG_DIR}")
for f in sorted(FIG_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<50} {size_kb:>8.1f} KB")

print("\nAll analysis code complete. Next: LaTeX paper draft in paper/main.tex")
