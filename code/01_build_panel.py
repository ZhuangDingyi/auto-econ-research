"""
01_build_panel.py
Build state × year panel for staggered DID analysis
Outcome: log(Leisure & Hospitality employment) 
Treatment: state minimum wage above federal floor ($7.25)
"""
import json, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Project: {proj}")

# ── 1. Load employment data ──────────────────────────────────────────────────
with open(f"{proj}/data/raw/bls_leisure_hospitality_employment.json") as f:
    emp_raw = json.load(f)

state_data = emp_raw['state_data']

# Aggregate monthly → annual average (M01-M12 only, skip M13 if absent)
records = []
for state, obs_list in state_data.items():
    if not obs_list:
        continue
    df_s = pd.DataFrame(obs_list)
    df_s['month_num'] = df_s['period'].str.replace('M','').astype(int)
    df_s = df_s[df_s['month_num'].between(1, 12)]   # exclude M13 annual avg
    df_s['value'] = pd.to_numeric(df_s['value'], errors='coerce')
    df_s['year'] = df_s['year'].astype(int)
    annual = df_s.groupby('year')['value'].mean().reset_index()
    annual['state'] = state
    records.append(annual)

emp_df = pd.concat(records, ignore_index=True)
emp_df.columns = ['year', 'lh_emp', 'state']
emp_df = emp_df[emp_df['year'].between(2017, 2024)]

print(f"Employment panel: {emp_df.shape}, states: {emp_df['state'].nunique()}")
print(emp_df.head())

# ── 2. Load minimum wage data ─────────────────────────────────────────────────
with open(f"{proj}/data/raw/state_minwage.json") as f:
    mw_raw = json.load(f)

mw_records = []
for state, wages in mw_raw['minwage_by_state_year'].items():
    for year_str, wage in wages.items():
        mw_records.append({'state': state, 'year': int(year_str), 'min_wage': wage})

mw_df = pd.DataFrame(mw_records)
mw_df = mw_df[mw_df['year'].between(2017, 2024)]

# Treatment indicator: min_wage > federal floor ($7.25)
mw_df['treated'] = (mw_df['min_wage'] > 7.25).astype(int)

# Treatment group timing (first year above federal floor, after 2017)
treatment_years = mw_raw['treatment_years']
mw_df['g'] = mw_df['state'].map(treatment_years)   # g = first year of treatment
# For C&S estimator: g=0 means never treated
mw_df['g'] = mw_df['g'].fillna(0).astype(int)
# If treated before sample starts (2017), set g=2014 (early treated cohort)
# IMPORTANT: only recode states that WERE treated (g>0), not never-treated (g=0)
mw_df.loc[(mw_df['g'] > 0) & (mw_df['g'] < 2017), 'g'] = 2014

print(f"\nMin wage panel: {mw_df.shape}")
print(mw_df[mw_df['state'] == 'CA'].sort_values('year'))

# ── 3. Merge into panel ──────────────────────────────────────────────────────
panel = emp_df.merge(mw_df, on=['state', 'year'], how='inner')
panel['log_lh_emp'] = np.log(panel['lh_emp'])

# Add state and year fixed effects dummies (for TWFE)
panel = panel.sort_values(['state', 'year']).reset_index(drop=True)

# Add population data for normalization (optional - use emp level for now)
print(f"\nMerged panel: {panel.shape}")
print(f"States: {panel['state'].nunique()}")
print(f"Years: {sorted(panel['year'].unique())}")
print(f"Treatment groups (g):")
print(panel.groupby('g')['state'].nunique().rename('n_states'))

# ── 4. Summary statistics ─────────────────────────────────────────────────────
treated_states = panel[panel['g'] > 0]['state'].unique()
control_states = panel[panel['g'] == 0]['state'].unique()

print(f"\nTreatment states: {len(treated_states)}")
print(f"Control states (never treated): {len(control_states)}: {sorted(control_states)}")

summary = panel.groupby(['g']).agg(
    n_states=('state', 'nunique'),
    mean_wage=('min_wage', 'mean'),
    mean_emp=('lh_emp', 'mean')
).round(2)
print(f"\nBy treatment cohort:\n{summary}")

# ── 5. Save cleaned panel ─────────────────────────────────────────────────────
os.makedirs(f"{proj}/data/cleaned", exist_ok=True)
panel.to_csv(f"{proj}/data/cleaned/panel.csv", index=False)
print(f"\nSaved: {proj}/data/cleaned/panel.csv")

# ── 6. Pre-trends visualization ──────────────────────────────────────────────
os.makedirs(f"{proj}/paper/figures", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Min wage over time by treatment group
for g, grp in panel.groupby('g'):
    avg = grp.groupby('year')['min_wage'].mean()
    label = f"Never treated" if g == 0 else f"Cohort {g}"
    lw = 2 if g == 0 else 1
    alpha = 0.9 if g == 0 else 0.6
    axes[0].plot(avg.index, avg.values, label=label, linewidth=lw, alpha=alpha)
axes[0].axhline(7.25, color='red', linestyle='--', label='Federal floor ($7.25)')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Average Minimum Wage ($)')
axes[0].set_title('State Minimum Wages by Treatment Cohort')
axes[0].legend(fontsize=7, ncol=2)
axes[0].grid(True, alpha=0.3)

# Plot 2: Log employment by treated/control
panel['treated_ever'] = (panel['g'] > 0).astype(int)
for label, grp in panel.groupby('treated_ever'):
    avg = grp.groupby('year')['log_lh_emp'].mean()
    axes[1].plot(avg.index, avg.values,
                 label='Treated' if label else 'Never Treated',
                 linewidth=2, marker='o', markersize=4)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Log(L&H Employment, thousands)')
axes[1].set_title('Log Employment: Treated vs. Never Treated')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{proj}/paper/figures/fig0_pretrends.png", dpi=150, bbox_inches='tight')
print(f"Saved figure: fig0_pretrends.png")
plt.close()

print("\n✅ Panel built successfully!")
