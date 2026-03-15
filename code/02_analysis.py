"""
02_analysis.py
Main DID analysis: TWFE + Event Study
Outcome: log(L&H employment)
Treatment: state min wage > federal floor ($7.25)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import os, warnings
warnings.filterwarnings('ignore')

proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
panel = pd.read_csv(f"{proj}/data/cleaned/panel.csv")
print(f"Panel: {panel.shape}, states: {panel['state'].nunique()}")

# ── 1. Two-Way Fixed Effects (TWFE) Baseline ─────────────────────────────────
# log_emp = β·treated + α_state + γ_year + ε
panel['log_lh_emp'] = np.log(panel['lh_emp'])
panel['treated'] = (panel['g'] > 0).astype(int)
panel['post'] = (panel['year'] >= panel['g']).astype(int)
panel.loc[panel['g'] == 0, 'post'] = 0
panel['did'] = panel['treated'] * panel['post']

# TWFE via statsmodels with C() fixed effects
twfe = smf.ols(
    "log_lh_emp ~ did + C(state) + C(year)",
    data=panel
).fit(cov_type='cluster', cov_kwds={'groups': panel['state']})

print("\n" + "="*60)
print("TWFE DID RESULTS (clustered SE by state)")
print("="*60)
print(f"DID coefficient: {twfe.params['did']:.4f}")
print(f"Std Error:       {twfe.bse['did']:.4f}")
print(f"t-statistic:     {twfe.tvalues['did']:.3f}")
print(f"p-value:         {twfe.pvalues['did']:.4f}")
print(f"N obs:           {twfe.nobs:.0f}")
print(f"R-squared:       {twfe.rsquared:.4f}")
ci = twfe.conf_int().loc['did']
print(f"95% CI:          [{ci[0]:.4f}, {ci[1]:.4f}]")
pct_effect = (np.exp(twfe.params['did']) - 1) * 100
print(f"Implied % effect: {pct_effect:.1f}%")

# ── 2. Event Study ────────────────────────────────────────────────────────────
# Relative time dummies: t = year - g (years since treatment)
# Use only treatment states + never-treated as comparison
panel_es = panel[panel['g'].isin([0, 2021])].copy()
# For g=2021: relative time = year - 2021
panel_es['rel_time'] = np.where(panel_es['g'] == 0, -99, panel_es['year'] - panel_es['g'])
# Bin relative time: -4 to +3
panel_es['rel_time'] = panel_es['rel_time'].clip(-3, 3)
panel_es.loc[panel_es['g'] == 0, 'rel_time'] = -99  # never treated reference

# Also do event study using TWFE approach with year-relative dummies
# Use ALL states with post-2017 treatment timing (g >= 2018) for cleaner study
panel_late = panel[panel['g'].isin([0, 2021])].copy()  # FL 2021, VA 2021
panel_late['rel_time'] = panel_late['year'] - 2021
panel_late.loc[panel_late['g'] == 0, 'rel_time'] = -99

# Relative time dummies (omit t=-1 as baseline)
for t in range(-3, 4):
    if t != -1:
        panel_late[f"rt_{+'n' if t < 0 else 'p'}{abs(t)}"] = ((panel_late['rel_time'] == t) & (panel_late['g'] > 0)).astype(int)

rt_vars = [f"rt_{+'n' if t < 0 else 'p'}{abs(t)}" for t in range(-3, 4) if t != -1]
formula = f"log_lh_emp ~ {' + '.join(rt_vars)} + C(state) + C(year)"
es_model = smf.ols(formula, data=panel_late).fit(
    cov_type='cluster', cov_kwds={'groups': panel_late['state']}
)

# Extract event study coefficients
es_coefs = {}
es_ses = {}
for t in range(-3, 4):
    if t == -1:
        es_coefs[t] = 0.0
        es_ses[t] = 0.0
    elif f"rt_{+'n' if t < 0 else 'p'}{abs(t)}" in es_model.params.index:
        es_coefs[t] = es_model.params[f"rt_{+'n' if t < 0 else 'p'}{abs(t)}"]
        es_ses[t] = es_model.bse[f"rt_{+'n' if t < 0 else 'p'}{abs(t)}"]
    else:
        es_coefs[t] = np.nan
        es_ses[t] = np.nan

print("\n" + "="*60)
print("EVENT STUDY (Virginia 2021 treatment, never-treated control)")
print("="*60)
print(f"{'t':>4}  {'Coef':>8}  {'SE':>8}  {'sig':>5}")
for t, coef in sorted(es_coefs.items()):
    se = es_ses[t]
    sig = '*' if abs(coef) > 1.96*se else ''
    print(f"{t:>4}  {coef:>8.4f}  {se:>8.4f}  {sig:>5}")

# ── 3. Main TWFE Event Study (all treated states) ─────────────────────────────
# Use treatment intensity: log(min_wage) as running variable
panel['log_minwage'] = np.log(panel['min_wage'].clip(lower=7.25))
panel['treated_post'] = panel['did']

# Panel FE with log min wage as treatment variable
twfe2 = smf.ols(
    "log_lh_emp ~ log_minwage + C(state) + C(year)",
    data=panel
).fit(cov_type='cluster', cov_kwds={'groups': panel['state']})

print("\n" + "="*60)
print("TWFE WITH LOG MIN WAGE (continuous treatment)")
print("="*60)
print(f"log(min_wage) coef: {twfe2.params['log_minwage']:.4f}")
print(f"Std Error:          {twfe2.bse['log_minwage']:.4f}")
print(f"t-stat:             {twfe2.tvalues['log_minwage']:.3f}")
print(f"p-value:            {twfe2.pvalues['log_minwage']:.4f}")
print(f"Elasticity:         {twfe2.params['log_minwage']:.4f}")
print(f"(1% increase in min wage → {twfe2.params['log_minwage']*100:.2f}% change in L&H emp)")

# ── 4. Robustness: Drop large states ─────────────────────────────────────────
big_states = ['CA', 'NY', 'TX', 'FL']
panel_small = panel[~panel['state'].isin(big_states)]
twfe_small = smf.ols(
    "log_lh_emp ~ log_minwage + C(state) + C(year)",
    data=panel_small
).fit(cov_type='cluster', cov_kwds={'groups': panel_small['state']})

print("\n" + "="*60)
print(f"ROBUSTNESS: Drop {big_states}")
print("="*60)
print(f"log(min_wage) coef: {twfe_small.params['log_minwage']:.4f} (vs {twfe2.params['log_minwage']:.4f} full sample)")
print(f"p-value:            {twfe_small.pvalues['log_minwage']:.4f}")

# ── 5. COVID robustness: Exclude 2020 ────────────────────────────────────────
panel_no2020 = panel[panel['year'] != 2020]
twfe_no2020 = smf.ols(
    "log_lh_emp ~ log_minwage + C(state) + C(year)",
    data=panel_no2020
).fit(cov_type='cluster', cov_kwds={'groups': panel_no2020['state']})

print(f"\nROBUSTNESS: Exclude 2020 (COVID)")
print(f"log(min_wage) coef: {twfe_no2020.params['log_minwage']:.4f}")
print(f"p-value:            {twfe_no2020.pvalues['log_minwage']:.4f}")

# ── 6. Figures ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fig 1: Pre/post employment by treatment status (TWFE-style)
# Remove state and year FEs manually using partial R2 approach
panel_plot = panel.copy()
state_means = panel_plot.groupby('state')['log_lh_emp'].mean()
year_means = panel_plot.groupby('year')['log_lh_emp'].mean()
grand_mean = panel_plot['log_lh_emp'].mean()
panel_plot['lh_demeaned'] = (panel_plot['log_lh_emp'] 
    - panel_plot['state'].map(state_means)
    - panel_plot['year'].map(year_means) 
    + grand_mean)

for g_val, label, color in [(0, 'Never Treated', 'blue'), (2014, 'Early Treated (pre-2017)', 'orange'), (2021, 'Late Treated (2021)', 'green')]:
    grp = panel_plot[panel_plot['g'] == g_val]
    if len(grp) == 0:
        continue
    avg = grp.groupby('year')['lh_demeaned'].mean()
    axes[0].plot(avg.index, avg.values, label=label, marker='o', markersize=5, linewidth=2, color=color)

axes[0].axvline(x=2021, color='red', linestyle='--', alpha=0.7, label='Treatment 2021')
axes[0].axvline(x=2020, color='gray', linestyle=':', alpha=0.5, label='COVID 2020')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Log L&H Employment (demeaned)')
axes[0].set_title('Employment Trends by Treatment Cohort\n(State & Year FE Removed)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Fig 2: Min wage vs employment scatter (cross-state variation)
avg_panel = panel.groupby('state').agg(
    avg_minwage=('min_wage', 'mean'),
    avg_log_emp=('log_lh_emp', 'mean'),
    g=('g', 'first')
).reset_index()

colors = avg_panel['g'].map({0: 'blue', 2014: 'orange', 2021: 'green'})
axes[1].scatter(avg_panel['avg_minwage'], avg_panel['avg_log_emp'], c=colors, alpha=0.7, s=50)
for _, row in avg_panel.iterrows():
    axes[1].annotate(row['state'], (row['avg_minwage'], row['avg_log_emp']),
                     textcoords='offset points', xytext=(3, 3), fontsize=6)

# Add regression line
z = np.polyfit(avg_panel['avg_minwage'], avg_panel['avg_log_emp'], 1)
p = np.poly1d(z)
x_range = np.linspace(avg_panel['avg_minwage'].min(), avg_panel['avg_minwage'].max(), 100)
axes[1].plot(x_range, p(x_range), 'r--', alpha=0.7, label=f'OLS slope: {z[0]:.3f}')
axes[1].set_xlabel('Average Minimum Wage ($)')
axes[1].set_ylabel('Average Log L&H Employment')
axes[1].set_title('Cross-State: Min Wage vs Employment Level\n(2017-2024 averages)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(color='blue', label='Never Treated'),
    Patch(color='orange', label='Early Treated'),
    Patch(color='green', label='Late Treated 2021')
]
axes[1].legend(handles=legend_elements, fontsize=8)

plt.tight_layout()
plt.savefig(f"{proj}/paper/figures/fig1_main_results.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: fig1_main_results.png")
plt.close()

# ── 7. Save results table ────────────────────────────────────────────────────
results_summary = {
    'TWFE (binary treatment)': {
        'coef': twfe.params['did'],
        'se': twfe.bse['did'],
        'pval': twfe.pvalues['did'],
        'n': twfe.nobs
    },
    'TWFE (log min wage, full)': {
        'coef': twfe2.params['log_minwage'],
        'se': twfe2.bse['log_minwage'],
        'pval': twfe2.pvalues['log_minwage'],
        'n': twfe2.nobs
    },
    'TWFE (drop CA/NY/TX/FL)': {
        'coef': twfe_small.params['log_minwage'],
        'se': twfe_small.bse['log_minwage'],
        'pval': twfe_small.pvalues['log_minwage'],
        'n': twfe_small.nobs
    },
    'TWFE (exclude 2020)': {
        'coef': twfe_no2020.params['log_minwage'],
        'se': twfe_no2020.bse['log_minwage'],
        'pval': twfe_no2020.pvalues['log_minwage'],
        'n': twfe_no2020.nobs
    }
}

import json
with open(f"{proj}/data/cleaned/results.json", 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Specification':<35} {'Coef':>8} {'SE':>8} {'p':>8} {'N':>6}")
print("-"*70)
for spec, r in results_summary.items():
    stars = '***' if r['pval']<0.01 else '**' if r['pval']<0.05 else '*' if r['pval']<0.10 else ''
    print(f"{spec:<35} {r['coef']:>8.4f} {r['se']:>8.4f} {r['pval']:>8.4f} {r['n']:>6.0f} {stars}")

print("\n✅ Analysis complete!")
