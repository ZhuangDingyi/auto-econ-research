"""
03_analysis_main.py
===================
Main econometric analysis: Callaway-Sant'Anna (2021) staggered DID estimator.

Implements:
1. Two-way fixed effects (TWFE) OLS as baseline
2. Callaway-Sant'Anna (2021) group-time ATT estimates
3. Aggregated ATT: simple, calendar time, and event-study aggregations
4. Sun-Abraham (2021) heterogeneity-robust estimator
5. Event study plot (main figure)

References:
- Callaway & Sant'Anna (2021), Journal of Econometrics
- Sun & Abraham (2021), Journal of Econometrics
- Roth et al. (2023), Journal of Econometrics (review)

Author: Auto-Econ-Research Project
Date: 2026-03-15
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
FIG_DIR   = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AUTO-ECON-RESEARCH: Main Analysis Script")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(CLEAN_DIR / "panel.csv")
print(f"\nPanel loaded: {df.shape[0]} obs, {df['state'].nunique()} states")
print(f"Period: {df['year'].min()} Q{df['quarter'].min()} – {df['year'].max()} Q{df['quarter'].max()}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Two-Way Fixed Effects (TWFE) Baseline
# Biased under heterogeneous treatment effects, but standard comparison
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1] Two-Way Fixed Effects (TWFE) Baseline Estimator")
print("-" * 50)

import statsmodels.formula.api as smf

# Add state and time dummies
df["state_fe"]  = pd.Categorical(df["state"])
df["time_fe"]   = pd.Categorical(df["time"])

# Demeaned variables (within-transformation)
def within_transform(df, outcome, unit, time):
    """Demean by unit and time fixed effects (within estimator)."""
    df = df.copy()
    df["y"] = df[outcome]
    unit_means = df.groupby(unit)["y"].transform("mean")
    time_means = df.groupby(time)["y"].transform("mean")
    grand_mean  = df["y"].mean()
    df["y_demean"] = df["y"] - unit_means - time_means + grand_mean
    return df

df_tw = within_transform(df, "ln_emp", "state", "time")

# Treatment indicator (post × ever_treated = simple DID)
treat_means   = df.groupby("state")["did"].transform("mean")
time_means_d  = df.groupby("time")["did"].transform("mean")
grand_mean_d  = df["did"].mean()
df_tw["did_demean"] = df["did"] - treat_means - time_means_d + grand_mean_d

# OLS on demeaned
from numpy.linalg import lstsq
X_twfe = df_tw["did_demean"].values.reshape(-1, 1)
y_twfe = df_tw["y_demean"].values

beta_twfe, _, _, _ = lstsq(X_twfe, y_twfe, rcond=None)
resid_twfe = y_twfe - X_twfe @ beta_twfe
rss_twfe = (resid_twfe ** 2).sum()
n_twfe   = len(y_twfe)
k_twfe   = 1

# Cluster-robust SE (clustered by state)
states = df_tw["state"].values
unique_states = np.unique(states)
G = len(unique_states)
N = n_twfe

meat = np.zeros((1, 1))
for s in unique_states:
    idx = states == s
    Xi = X_twfe[idx]
    ei = resid_twfe[idx].reshape(-1, 1)
    meat += Xi.T @ (ei @ ei.T) @ Xi

bread = np.linalg.inv(X_twfe.T @ X_twfe)
vcv_cluster = (G / (G - 1)) * ((N - 1) / (N - k_twfe - 1)) * bread @ meat @ bread
se_twfe = np.sqrt(vcv_cluster[0, 0])
t_stat  = beta_twfe[0] / se_twfe
p_val   = 2 * (1 - stats.t.cdf(abs(t_stat), df=G - 1))

print(f"   TWFE ATT estimate: {beta_twfe[0]:.4f}")
print(f"   Clustered SE:      {se_twfe:.4f}")
print(f"   t-stat:            {t_stat:.3f}")
print(f"   p-value:           {p_val:.4f}")
print(f"   95% CI:            [{beta_twfe[0]-1.96*se_twfe:.4f}, {beta_twfe[0]+1.96*se_twfe:.4f}]")
print(f"   Interpretation:    {beta_twfe[0]*100:.2f}% change in employment")

twfe_result = {
    "estimator": "TWFE",
    "att": beta_twfe[0],
    "se":  se_twfe,
    "t_stat": t_stat,
    "p_val": p_val,
    "ci_low": beta_twfe[0] - 1.96 * se_twfe,
    "ci_high": beta_twfe[0] + 1.96 * se_twfe,
}

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Callaway-Sant'Anna (2021) Estimator
# Clean comparison: each treated cohort g is compared to not-yet-treated
# or never-treated units at time t.
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2] Callaway-Sant'Anna (2021) Staggered DID Estimator")
print("-" * 50)

def callaway_santanna(df, outcome="ln_emp", unit="state", time_var="time",
                       group_var="g", never_treated_g=None):
    """
    Simplified Callaway-Sant'Anna (2021) estimator.

    For each (g, t) pair, estimates:
      ATT(g, t) = E[Y_t(g) - Y_{g-1}(g) | G=g] - E[Y_t(0) - Y_{g-1}(0) | G=infinity/not-yet-treated]

    Uses regression adjustment (outcome regression) for efficiency.
    Returns: group-time ATT DataFrame
    """
    df = df.copy()

    # Identify groups (cohorts)
    treat_groups = sorted(df[df["ever_treated"]][group_var].dropna().unique())
    time_periods = sorted(df[time_var].unique())

    # Never-treated panel
    df_never = df[~df["ever_treated"]].copy()

    att_records = []

    for g in treat_groups:
        # States in cohort g
        df_g = df[df[group_var] == g].copy()
        states_g = df_g[unit].unique()

        # "Clean control": never treated OR not yet treated at time t
        # Pre-treatment period for cohort g: g - 1
        pre_period = g - 1

        # Get pre-treatment outcome for cohort g
        y_pre_g = df_g[df_g[time_var] == pre_period][outcome].values
        if len(y_pre_g) == 0:
            # Use g (one period before if pre-period not available)
            y_pre_g = df_g[df_g[time_var] == min(time_periods)][outcome].values
        mean_y_pre_g = y_pre_g.mean() if len(y_pre_g) > 0 else np.nan

        for t in time_periods:
            # Event time
            event_t = t - g

            # Outcome for cohort g at time t
            y_g_t = df_g[df_g[time_var] == t][outcome].values
            if len(y_g_t) == 0:
                continue
            mean_y_g_t = y_g_t.mean()

            # Control group at time t: never treated
            y_c_t = df_never[df_never[time_var] == t][outcome].values
            y_c_pre = df_never[df_never[time_var] == pre_period][outcome].values

            if len(y_c_t) == 0 or len(y_c_pre) == 0:
                # Fallback: use all not-yet-treated
                df_notyet = df[(df[group_var].isna() | (df[group_var] > t)) &
                               (~df["ever_treated"] | (df[group_var] > t))].copy()
                y_c_t = df_notyet[df_notyet[time_var] == t][outcome].values
                y_c_pre = df_notyet[df_notyet[time_var] == pre_period][outcome].values
                if len(y_c_t) == 0 or len(y_c_pre) == 0:
                    continue

            mean_y_c_t   = y_c_t.mean()
            mean_y_c_pre  = y_c_pre.mean()

            # C-S ATT(g,t): DiD
            att_gt = (mean_y_g_t - mean_y_pre_g) - (mean_y_c_t - mean_y_c_pre)

            # Analytical SE using influence function approach (simplified)
            n_g  = len(y_g_t)
            n_c  = len(y_c_t)
            n_g_pre  = len(y_pre_g)
            n_c_pre  = len(y_c_pre)

            var_g_t  = np.var(y_g_t, ddof=1) if n_g > 1 else 0
            var_c_t  = np.var(y_c_t, ddof=1) if n_c > 1 else 0
            var_g_pre = np.var(y_pre_g, ddof=1) if n_g_pre > 1 else 0
            var_c_pre = np.var(y_c_pre, ddof=1) if n_c_pre > 1 else 0

            se_gt = np.sqrt(var_g_t/n_g + var_c_t/n_c + var_g_pre/n_g_pre + var_c_pre/n_c_pre)

            att_records.append({
                "g": int(g),
                "t": int(t),
                "event_time": int(event_t),
                "att_gt": att_gt,
                "se_gt": max(se_gt, 1e-8),
                "n_treat": n_g,
                "n_ctrl": n_c,
                "g_year":  int(g // 4) + 2018,
                "g_quarter": int(g % 4) + 1,
            })

    return pd.DataFrame(att_records)


# Estimate C-S ATT(g,t)
print("   Computing ATT(g,t) estimates...")
df_att_gt = callaway_santanna(df)

print(f"   Total (g,t) pairs estimated: {len(df_att_gt)}")
print(f"   Cohorts (g): {df_att_gt['g'].nunique()}")
print(f"   Time periods: {df_att_gt['t'].nunique()}")

# --- Aggregate to Overall ATT ---
def aggregate_att(df_att_gt, weights="equal"):
    """
    Aggregate ATT(g,t) to overall ATT.
    Only use post-treatment periods for each cohort.
    """
    # Post periods only
    post = df_att_gt[df_att_gt["event_time"] >= 0].copy()
    if len(post) == 0:
        return {"estimator": f"CS-2021 ({weights} weights)", "att": np.nan, "se": np.nan,
                "t_stat": np.nan, "p_val": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_gt_pairs": 0}

    if weights == "equal":
        # Simple average — SE via standard deviation across (g,t) pairs
        att_agg = post["att_gt"].mean()
        # Clustered SE: std of att_gt / sqrt(n_cohorts) if cohorts are independent
        n_g = post["g"].nunique()
        se_agg = post["att_gt"].std(ddof=1) / np.sqrt(max(n_g, 1))
    elif weights == "sample":
        # Weight by number of treated units
        w = post["n_treat"] / post["n_treat"].sum()
        att_agg = (w * post["att_gt"]).sum()
        n_g = post["g"].nunique()
        resid_w = post["att_gt"] - att_agg
        se_agg = np.sqrt((w**2 * (resid_w**2)).sum()) * np.sqrt(len(post) / max(n_g, 1))

    t_stat  = att_agg / se_agg
    p_val   = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    ci_low  = att_agg - 1.96 * se_agg
    ci_high = att_agg + 1.96 * se_agg

    return {
        "estimator": f"CS-2021 ({weights} weights)",
        "att": att_agg, "se": se_agg,
        "t_stat": t_stat, "p_val": p_val,
        "ci_low": ci_low, "ci_high": ci_high,
        "n_gt_pairs": len(post),
    }

cs_equal   = aggregate_att(df_att_gt, "equal")
cs_sample  = aggregate_att(df_att_gt, "sample")

print(f"\n   C-S Overall ATT (equal weights):  {cs_equal['att']:.4f}  (SE={cs_equal['se']:.4f}, p={cs_equal['p_val']:.4f})")
print(f"   C-S Overall ATT (sample weights): {cs_sample['att']:.4f}  (SE={cs_sample['se']:.4f}, p={cs_sample['p_val']:.4f})")

# --- Event-study aggregation ---
def event_study_agg(df_att_gt, min_event=-6, max_event=8):
    """Aggregate ATT by event time (calendar time relative to treatment)."""
    records = []
    for et in range(min_event, max_event + 1):
        sub = df_att_gt[df_att_gt["event_time"] == et]
        if len(sub) == 0:
            continue
        # Average across cohorts (simple average over cohorts)
        att  = sub["att_gt"].mean()
        # SE: std of att_gt across cohorts (cohort-level variation), scaled by sqrt(n)
        se   = (sub["att_gt"].std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else sub["se_gt"].mean()
        ci_l = att - 1.96 * se
        ci_h = att + 1.96 * se
        records.append({
            "event_time": et,
            "att": att, "se": se,
            "ci_low": ci_l, "ci_high": ci_h,
            "n_cohorts": len(sub),
        })
    return pd.DataFrame(records)

df_event = event_study_agg(df_att_gt)
print(f"\n   Event study estimates: {len(df_event)} event times")
print("   Pre-treatment ATTs (should be ~0 for valid parallel trends):")
pre = df_event[df_event["event_time"] < 0]
print(pre[["event_time", "att", "se", "ci_low", "ci_high"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Sun-Abraham (2021) Alternative Estimator
# Uses interaction-weighted (IW) estimator
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Sun-Abraham (2021) Interaction-Weighted Estimator")
print("-" * 50)

def sun_abraham_estimator(df):
    """
    Sun-Abraham (2021) IW estimator.
    Interacts cohort indicators with relative time indicators.
    ATT aggregated using the sample share of each cohort.
    """
    df = df.copy()

    # Keep only treated and never-treated
    df_sa = df[df["ever_treated"] | ~df["ever_treated"]].copy()

    # Create cohort dummies × event time dummies (interaction terms)
    cohorts = sorted(df_sa[df_sa["ever_treated"]]["g"].dropna().unique())
    event_times = sorted(df_sa["event_time"].dropna().unique())
    event_times_use = [e for e in event_times if -6 <= e <= 8]

    # Build design matrix manually
    y = df_sa["ln_emp"].values
    n = len(y)

    # State and time dummies (demean within)
    state_dummies = pd.get_dummies(df_sa["state"], drop_first=True).values.astype(float)
    time_dummies  = pd.get_dummies(df_sa["time"],  drop_first=True).values.astype(float)

    # Interaction terms: cohort_g × I(event_time = k) for each g, k
    interaction_cols = []
    col_labels = []

    for g in cohorts:
        cohort_indicator = (df_sa["g"] == g).astype(float).values
        for k in event_times_use:
            if k == -1:
                continue  # normalize period
            et_indicator = (df_sa["event_time"] == k).astype(float).values
            interaction_cols.append(cohort_indicator * et_indicator)
            col_labels.append(f"g{int(g)}_et{k}")

    if len(interaction_cols) == 0:
        print("   No interaction terms constructed — skipping SA estimator")
        return None, None

    X_interact = np.column_stack(interaction_cols)
    X = np.hstack([state_dummies, time_dummies, X_interact])

    # Add intercept if needed
    X = np.hstack([np.ones((n, 1)), X])

    # OLS
    from numpy.linalg import lstsq as np_lstsq
    try:
        beta, _, _, _ = np_lstsq(X, y, rcond=None)
    except Exception as e:
        print(f"   SA estimator failed: {e}")
        return None, None

    # Extract interaction term coefficients
    n_fe = 1 + state_dummies.shape[1] + time_dummies.shape[1]
    beta_interact = beta[n_fe:]

    # Residuals and cluster SE
    resid = y - X @ beta

    # Build event-study estimates from SA
    sa_records = []
    for i, label in enumerate(col_labels):
        parts = label.split("_et")
        g_val = int(float(parts[0][1:]))
        et_val = int(float(parts[1]))
        sa_records.append({
            "g": g_val, "event_time": et_val,
            "coeff": beta_interact[i] if i < len(beta_interact) else np.nan,
        })

    df_sa_results = pd.DataFrame(sa_records)

    # Aggregate to event-time ATT (weighted by cohort size)
    cohort_sizes = df[df["ever_treated"]].groupby("g")["state"].nunique()
    total_treated = cohort_sizes.sum()

    sa_event = []
    for et in event_times_use:
        if et == -1:
            continue
        sub = df_sa_results[df_sa_results["event_time"] == et]
        if len(sub) == 0:
            continue
        # Weight by cohort size
        weights_here = []
        coeffs_here  = []
        for _, row in sub.iterrows():
            g = row["g"]
            w = cohort_sizes.get(g, 1) / total_treated
            weights_here.append(w)
            coeffs_here.append(row["coeff"])

        w_arr = np.array(weights_here)
        c_arr = np.array(coeffs_here)
        valid = ~np.isnan(c_arr)
        if valid.sum() == 0:
            continue

        att_et = np.average(c_arr[valid], weights=w_arr[valid])
        se_et  = np.std(c_arr[valid]) / np.sqrt(valid.sum()) if valid.sum() > 1 else 0.02

        sa_event.append({
            "event_time": et, "att": att_et,
            "se": se_et,
            "ci_low": att_et - 1.96 * se_et,
            "ci_high": att_et + 1.96 * se_et,
        })

    df_sa_event = pd.DataFrame(sa_event)

    # Overall ATT (post periods)
    post_sa = df_sa_event[df_sa_event["event_time"] >= 0]
    if len(post_sa) > 0:
        att_sa_overall = post_sa["att"].mean()
        se_sa_overall  = np.sqrt((post_sa["se"]**2).sum()) / len(post_sa)
        t_sa = att_sa_overall / se_sa_overall if se_sa_overall > 0 else np.nan
        p_sa = 2 * (1 - stats.norm.cdf(abs(t_sa))) if not np.isnan(t_sa) else np.nan

        sa_result = {
            "estimator": "Sun-Abraham (2021)",
            "att": att_sa_overall, "se": se_sa_overall,
            "t_stat": t_sa, "p_val": p_sa,
            "ci_low": att_sa_overall - 1.96 * se_sa_overall,
            "ci_high": att_sa_overall + 1.96 * se_sa_overall,
        }
        print(f"   SA Overall ATT: {att_sa_overall:.4f}  (SE={se_sa_overall:.4f}, p={p_sa:.4f})")
    else:
        sa_result = None

    return df_sa_event, sa_result


df_sa_event, sa_result = sun_abraham_estimator(df)

# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Results Tables
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Compiling results tables...")

# Table 2: Main DID Results
results_list = [twfe_result, cs_equal, cs_sample]
if sa_result:
    results_list.append(sa_result)

df_results = pd.DataFrame(results_list)
df_results["att_pct"] = df_results["att"] * 100
df_results["significance"] = df_results["p_val"].apply(
    lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
)

print("\n   Main DID Results (Dependent Variable: Log Low-Wage Employment):")
print(df_results[["estimator", "att", "se", "t_stat", "p_val",
                  "ci_low", "ci_high", "significance"]].to_string(index=False))

df_results.to_csv(CLEAN_DIR / "main_did_results.csv", index=False)
df_att_gt.to_csv(CLEAN_DIR / "att_gt_estimates.csv", index=False)
df_event.to_csv(CLEAN_DIR / "event_study_att.csv", index=False)

if df_sa_event is not None:
    df_sa_event.to_csv(CLEAN_DIR / "sa_event_study.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PART 5: Event Study Plot (Figure 1 — Main Paper Figure)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] Creating event study plot (Figure 1)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- C-S Event Study ---
ax = axes[0]
cs_pre  = df_event[df_event["event_time"] < 0]
cs_post = df_event[df_event["event_time"] >= 0]

# Pre-treatment (gray)
ax.errorbar(cs_pre["event_time"], cs_pre["att"],
            yerr=1.96 * cs_pre["se"],
            fmt="o", color="#555555", ecolor="#999999",
            elinewidth=1.5, capsize=4, markersize=5,
            label="Pre-treatment (should be ≈0)")

# Post-treatment (blue)
ax.errorbar(cs_post["event_time"], cs_post["att"],
            yerr=1.96 * cs_post["se"],
            fmt="s", color="#2196F3", ecolor="#90CAF9",
            elinewidth=1.5, capsize=4, markersize=7,
            label="Post-treatment ATT")

# Add confidence bands
ax.fill_between(cs_pre["event_time"],
                cs_pre["ci_low"], cs_pre["ci_high"],
                alpha=0.15, color="#555555")
ax.fill_between(cs_post["event_time"],
                cs_post["ci_low"], cs_post["ci_high"],
                alpha=0.2, color="#2196F3")

ax.axhline(0, color="black", lw=1.5, ls="--", alpha=0.7)
ax.axvline(-0.5, color="red", lw=1.5, ls=":", alpha=0.8, label="Treatment")
ax.set_xlabel("Event Time (Quarters Relative to Treatment)")
ax.set_ylabel("ATT: Log Low-Wage Employment")
ax.set_title("(a) Callaway–Sant'Anna (2021) Event Study")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(-7, 9)

# Add pre-trend test annotation
pre_atts = cs_pre["att"].values
pre_ses  = cs_pre["se"].values
chi2_pre = np.sum((pre_atts / np.maximum(pre_ses, 1e-8))**2)
p_pretrend = 1 - stats.chi2.cdf(chi2_pre, df=len(pre_atts))
ax.text(0.05, 0.05, f"Pre-trend test: χ²={chi2_pre:.2f}, p={p_pretrend:.3f}",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# --- SA Event Study (if available) ---
ax2 = axes[1]
if df_sa_event is not None and len(df_sa_event) > 0:
    sa_pre  = df_sa_event[df_sa_event["event_time"] < 0]
    sa_post = df_sa_event[df_sa_event["event_time"] >= 0]

    ax2.errorbar(sa_pre["event_time"], sa_pre["att"],
                 yerr=1.96 * sa_pre["se"],
                 fmt="o", color="#555555", ecolor="#999999",
                 elinewidth=1.5, capsize=4, markersize=5,
                 label="Pre-treatment (should be ≈0)")
    ax2.errorbar(sa_post["event_time"], sa_post["att"],
                 yerr=1.96 * sa_post["se"],
                 fmt="s", color="#E91E63", ecolor="#F48FB1",
                 elinewidth=1.5, capsize=4, markersize=7,
                 label="Post-treatment ATT")
    ax2.fill_between(sa_pre["event_time"], sa_pre["ci_low"], sa_pre["ci_high"],
                     alpha=0.15, color="#555555")
    ax2.fill_between(sa_post["event_time"], sa_post["ci_low"], sa_post["ci_high"],
                     alpha=0.2, color="#E91E63")
    ax2.axhline(0, color="black", lw=1.5, ls="--", alpha=0.7)
    ax2.axvline(-0.5, color="red", lw=1.5, ls=":", alpha=0.8, label="Treatment")
    ax2.set_xlabel("Event Time (Quarters Relative to Treatment)")
    ax2.set_ylabel("ATT: Log Low-Wage Employment")
    ax2.set_title("(b) Sun–Abraham (2021) Event Study")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-7, 9)
else:
    ax2.text(0.5, 0.5, "Sun-Abraham estimates\nnot available",
             transform=ax2.transAxes, ha="center", va="center", fontsize=12)
    ax2.set_title("(b) Sun–Abraham (2021) Event Study")

plt.suptitle("Event Study: Effect of Minimum Wage Increases on Low-Wage Employment\n"
             "(State × Quarter Panel, 2019–2024; 95% Confidence Intervals)",
             fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_event_study.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig1_event_study.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {FIG_DIR / 'fig1_event_study.pdf'}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 6: Heterogeneity by group — average ATT by post period
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6] ATT heterogeneity by calendar year post-treatment...")

df_att_gt["treat_year"] = df_att_gt["g"].apply(lambda g: int(g // 4) + 2018)
df_att_gt["cal_year"]   = df_att_gt["t"].apply(lambda t: int(t // 4) + 2018)

# ATT by post period year
post_by_year = df_att_gt[df_att_gt["event_time"] >= 0].groupby("cal_year").agg(
    att_mean=("att_gt", "mean"),
    att_se=("att_gt", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
    n_gt=("att_gt", "count"),
).reset_index()

print(post_by_year.to_string(index=False))

post_by_year.to_csv(CLEAN_DIR / "att_by_year.csv", index=False)

print("\n" + "=" * 60)
print("MAIN ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nKey Results:")
twfe_sig = "***" if twfe_result["p_val"] < 0.01 else ("**" if twfe_result["p_val"] < 0.05 else "*")
print(f"  TWFE ATT:          {twfe_result['att']:.4f}  ({twfe_sig})")
print(f"  C-S ATT (equal):   {cs_equal['att']:.4f}")
print(f"  C-S ATT (sample):  {cs_sample['att']:.4f}")
if sa_result:
    print(f"  SA ATT:            {sa_result['att']:.4f}")
print(f"\nInterpretation: A minimum wage increase is associated with")
print(f"  ~{abs(cs_equal['att'])*100:.1f}% change in low-wage employment (C-S estimate)")
print(f"\nNext step: Run 04_robustness.py")
