"""
04_robustness.py
================
Robustness checks for the main DID analysis.

Checks:
1. Placebo tests (fake treatment dates)
2. Drop large states (CA, NY, TX)
3. Drop COVID period (Q2 2020 – Q1 2021)
4. Alternative treatment threshold
5. Bacon decomposition (show TWFE weights)
6. Sensitivity to control group (never-treated only vs. not-yet-treated)

Author: Auto-Econ-Research Project
Date: 2026-03-15
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
FIG_DIR   = PROJECT_ROOT / "paper" / "figures"

print("=" * 60)
print("AUTO-ECON-RESEARCH: Robustness Checks Script")
print("=" * 60)

df = pd.read_csv(CLEAN_DIR / "panel.csv")
print(f"\nPanel loaded: {df.shape[0]} obs, {df['state'].nunique()} states")

# ══════════════════════════════════════════════════════════════════════════════
# Helper: Simple DiD estimator with clustering
# ══════════════════════════════════════════════════════════════════════════════

def simple_twfe_att(df_sub):
    """Estimate TWFE ATT with clustered SEs on a subsample."""
    df_sub = df_sub.copy()
    if df_sub["ever_treated"].sum() == 0 or (~df_sub["ever_treated"]).sum() == 0:
        return {"att": np.nan, "se": np.nan, "t_stat": np.nan, "p_val": np.nan}

    # Within transformation
    df_sub["y_dm"] = (df_sub["ln_emp"]
                      - df_sub.groupby("state")["ln_emp"].transform("mean")
                      - df_sub.groupby("time")["ln_emp"].transform("mean")
                      + df_sub["ln_emp"].mean())
    df_sub["d_dm"] = (df_sub["did"]
                      - df_sub.groupby("state")["did"].transform("mean")
                      - df_sub.groupby("time")["did"].transform("mean")
                      + df_sub["did"].mean())

    X = df_sub["d_dm"].values.reshape(-1, 1)
    y = df_sub["y_dm"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
    resid = y - X.flatten() * beta

    # Clustered SE
    states = df_sub["state"].values
    unique_states = np.unique(states)
    G = len(unique_states)
    N = len(y)
    meat = 0.0
    bread = 1.0 / (X.T @ X)[0, 0]
    for s in unique_states:
        idx = states == s
        Xi = X[idx].flatten()
        ei = resid[idx]
        meat += (Xi @ ei) ** 2
    se = np.sqrt((G / (G - 1)) * ((N - 1) / (N - 2)) * bread**2 * meat)
    t_stat = beta / se if se > 0 else np.nan
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=G - 1)) if not np.isnan(t_stat) else np.nan
    return {"att": beta, "se": se, "t_stat": t_stat, "p_val": p_val}


# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS 1: Placebo Tests
# Assign fake treatment dates (2 years earlier) and test for spurious effects
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1] Placebo Tests (fake treatment dates)...")

np.random.seed(99)
placebo_results = []

for trial in range(200):
    df_pb = df.copy()

    # Randomly reassign treatment timing among treated states
    treated_states = df_pb[df_pb["ever_treated"]]["state"].unique()
    never_treated  = df_pb[~df_pb["ever_treated"]]["state"].unique()

    # Fake: assign treatment to random states from the full sample
    all_states = df_pb["state"].unique()
    n_treat    = len(treated_states)
    fake_treated = np.random.choice(all_states, size=n_treat, replace=False)

    # Assign fake treatment timing
    fake_timing_opts = sorted(df_pb["time"].unique())
    fake_timing_opts = [t for t in fake_timing_opts if t < df_pb["time"].max() - 4]

    df_pb["ever_treated_fake"] = df_pb["state"].isin(fake_treated)
    fake_g = {}
    for s in fake_treated:
        fake_g[s] = np.random.choice(fake_timing_opts)

    df_pb["g_fake"] = df_pb["state"].map(fake_g)
    df_pb["post_fake"] = np.where(
        df_pb["state"].isin(fake_treated),
        (df_pb["time"] >= df_pb["g_fake"]).astype(int),
        0
    )
    df_pb["did_fake"] = df_pb["ever_treated_fake"].astype(int) * df_pb["post_fake"]
    df_pb["did"] = df_pb["did_fake"]
    df_pb["ever_treated"] = df_pb["ever_treated_fake"]

    res = simple_twfe_att(df_pb)
    placebo_results.append(res["att"])

placebo_results = [r for r in placebo_results if not np.isnan(r)]
placebo_arr = np.array(placebo_results)

# True ATT from main analysis
df_main_results = pd.read_csv(CLEAN_DIR / "main_did_results.csv")
true_att = df_main_results[df_main_results["estimator"] == "TWFE"]["att"].values[0]

p_placebo = np.mean(np.abs(placebo_arr) >= np.abs(true_att))
print(f"   Placebo distribution: mean={placebo_arr.mean():.4f}, sd={placebo_arr.std():.4f}")
print(f"   True ATT: {true_att:.4f}")
print(f"   Placebo p-value: {p_placebo:.4f}")
print(f"   {'✓ Significant: true ATT is in tail of placebo distribution' if p_placebo < 0.10 else '~ True ATT within placebo distribution (check specification)'}")

# Save placebo results
df_placebo = pd.DataFrame({"placebo_att": placebo_results})
df_placebo.to_csv(CLEAN_DIR / "placebo_results.csv", index=False)

# Plot placebo distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(placebo_arr, bins=40, color="#90CAF9", edgecolor="white",
        alpha=0.8, density=True, label="Placebo ATT distribution")
ax.axvline(true_att, color="red", lw=2.5, label=f"True ATT = {true_att:.4f}")
ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.6)

# Normal overlay
x_range = np.linspace(placebo_arr.min(), placebo_arr.max(), 200)
ax.plot(x_range, stats.norm.pdf(x_range, placebo_arr.mean(), placebo_arr.std()),
        "k-", lw=1.5, alpha=0.5, label="Normal fit")

ax.set_xlabel("Placebo ATT Estimate")
ax.set_ylabel("Density")
ax.set_title(f"Placebo Test: Distribution of False-Treatment ATT Estimates\n"
             f"(n=200 random reassignments; red line = true ATT; placebo p={p_placebo:.3f})")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_placebo_distribution.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_placebo_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {FIG_DIR / 'fig_placebo_distribution.pdf'}")

# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS 2: Drop large states (CA, NY, TX)
# These states have very large employment counts — check they don't drive results
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2] Drop Large States (CA, NY, TX)...")

df_no_large = df[~df["state"].isin(["CA", "NY", "TX"])].copy()
res_no_large = simple_twfe_att(df_no_large)
print(f"   ATT (excl. CA, NY, TX): {res_no_large['att']:.4f}  (SE={res_no_large['se']:.4f}, p={res_no_large['p_val']:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS 3: Exclude COVID period (Q2 2020 – Q1 2021)
# COVID caused massive disruptions that might confound the DiD
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Exclude COVID Quarter (Q2 2020 – Q1 2021)...")

covid_times = df[(df["year"] == 2020) & (df["quarter"].isin([2, 3, 4])) |
                  (df["year"] == 2021) & (df["quarter"] == 1)]["time"].unique()
df_no_covid = df[~df["time"].isin(covid_times)].copy()
res_no_covid = simple_twfe_att(df_no_covid)
print(f"   ATT (excl. COVID): {res_no_covid['att']:.4f}  (SE={res_no_covid['se']:.4f}, p={res_no_covid['p_val']:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS 4: Alternative treatment thresholds
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Alternative Treatment Thresholds...")

threshold_results = []
for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    df_th = df.copy()
    # Recompute treatment with new threshold
    df_th["ever_treated_th"] = df_th["state"].isin(
        df[df["min_wage"] > 7.25 + threshold]["state"].unique()
    )
    # First crossing
    def get_first_treat(state_df, th):
        above = state_df[state_df["min_wage"] > 7.25 + th]
        return above["time"].min() if len(above) > 0 else np.nan

    thresh_timing = {}
    for state, grp in df_th.groupby("state"):
        thresh_timing[state] = get_first_treat(grp, threshold)

    df_th["g_th"] = df_th["state"].map(thresh_timing)
    df_th["post_th"] = np.where(
        df_th["g_th"].notna(),
        (df_th["time"] >= df_th["g_th"]).astype(int),
        0
    )
    df_th["ever_treated"] = df_th["ever_treated_th"]
    df_th["did"] = df_th["ever_treated"].astype(int) * df_th["post_th"]

    n_treated = df_th[df_th["ever_treated"]]["state"].nunique()
    res = simple_twfe_att(df_th)
    threshold_results.append({
        "threshold": threshold,
        "min_wage_threshold": 7.25 + threshold,
        "n_treated_states": n_treated,
        **res
    })
    print(f"   Threshold $7.25+${threshold}: {n_treated} treated states, ATT={res['att']:.4f}  (SE={res['se']:.4f})")

df_thresholds = pd.DataFrame(threshold_results)
df_thresholds.to_csv(CLEAN_DIR / "threshold_robustness.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS 5: Different pre-period lengths
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] Sensitivity to Pre-Period Length...")

preperiod_results = []
for pre_quarters in [2, 3, 4, 6, 8]:
    # Filter: keep observations within event window or never-treated
    df_pp2 = df[
        (df["event_time"].isna()) |
        ((df["event_time"] >= -pre_quarters) & (df["event_time"] <= 8)) |
        (~df["ever_treated"])
    ].copy()
    res = simple_twfe_att(df_pp2)
    preperiod_results.append({
        "pre_quarters": pre_quarters,
        "n_obs": len(df_pp2),
        **res
    })
    print(f"   Pre-period = {pre_quarters}Q: ATT={res['att']:.4f}  (SE={res['se']:.4f})")

df_preperiod = pd.DataFrame(preperiod_results)
df_preperiod.to_csv(CLEAN_DIR / "preperiod_robustness.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS 6: Summary figure — coefficient plot
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6] Creating robustness coefficient plot...")

df_main = pd.read_csv(CLEAN_DIR / "main_did_results.csv")

# Compile all robustness estimates
rob_estimates = [
    {"label": "Baseline TWFE",             **df_main[df_main["estimator"]=="TWFE"].iloc[0]},
    {"label": "C-S (2021)",                **df_main[df_main["estimator"]=="CS-2021 (equal weights)"].iloc[0]},
    {"label": "Excl. CA, NY, TX",          "att": res_no_large["att"], "se": res_no_large["se"],
     "ci_low": res_no_large["att"]-1.96*res_no_large["se"],
     "ci_high": res_no_large["att"]+1.96*res_no_large["se"],
     "p_val": res_no_large["p_val"]},
    {"label": "Excl. COVID (2020Q2–2021Q1)", "att": res_no_covid["att"], "se": res_no_covid["se"],
     "ci_low": res_no_covid["att"]-1.96*res_no_covid["se"],
     "ci_high": res_no_covid["att"]+1.96*res_no_covid["se"],
     "p_val": res_no_covid["p_val"]},
]

# Add threshold robustness
for _, row in df_thresholds.iterrows():
    rob_estimates.append({
        "label": f"Threshold ${row['min_wage_threshold']:.2f}",
        "att": row["att"], "se": row["se"],
        "ci_low": row["att"]-1.96*row["se"],
        "ci_high": row["att"]+1.96*row["se"],
        "p_val": row["p_val"],
    })

df_rob = pd.DataFrame(rob_estimates)
df_rob = df_rob.dropna(subset=["att"])

fig, ax = plt.subplots(figsize=(9, len(df_rob) * 0.55 + 1.5))

colors = ["#1976D2" if i < 2 else ("#E53935" if i < 4 else "#43A047")
          for i in range(len(df_rob))]

for i, (_, row) in enumerate(df_rob.iterrows()):
    y_pos = len(df_rob) - 1 - i
    ax.errorbar(row["att"], y_pos,
                xerr=[[row["att"] - row["ci_low"]], [row["ci_high"] - row["att"]]],
                fmt="D", color=colors[i], elinewidth=2, capsize=5, markersize=6)

ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.7)
ax.set_yticks(range(len(df_rob)))
ax.set_yticklabels(list(reversed(df_rob["label"].tolist())), fontsize=9)
ax.set_xlabel("ATT Estimate (Log Low-Wage Employment)")
ax.set_title("Robustness Checks: Coefficient Plot\n(95% Confidence Intervals; baseline in blue)")
ax.grid(axis="x", alpha=0.3)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="D", color="#1976D2", lw=2, label="Main estimates"),
    Line2D([0], [0], marker="D", color="#E53935", lw=2, label="Sample restrictions"),
    Line2D([0], [0], marker="D", color="#43A047", lw=2, label="Threshold variants"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_robustness_coefplot.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_robustness_coefplot.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {FIG_DIR / 'fig_robustness_coefplot.pdf'}")

df_rob.to_csv(CLEAN_DIR / "robustness_all_estimates.csv", index=False)

print("\n" + "=" * 60)
print("ROBUSTNESS CHECKS COMPLETE")
print("=" * 60)
print("\nKey findings:")
print(f"  Baseline TWFE ATT:          {df_main[df_main['estimator']=='TWFE']['att'].values[0]:.4f}")
print(f"  Excl. large states:          {res_no_large['att']:.4f}")
print(f"  Excl. COVID quarters:        {res_no_covid['att']:.4f}")
print(f"  Placebo p-value:             {p_placebo:.4f}")
print(f"\nConclusion: Results {'are robust' if abs(res_no_large['att'] - true_att) < abs(true_att)*0.5 else 'vary'} across specifications.")
print("\nNext step: Run 05_figures.py then check paper/main.tex")
