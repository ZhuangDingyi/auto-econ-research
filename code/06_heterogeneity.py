#!/usr/bin/env python3
"""
06_heterogeneity.py
Spatial-temporal heterogeneity and equity/fairness analysis for the
minimum wage study.  Creates Figures 3–5 and saves regression results.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
FIG_DIR  = ROOT / 'paper' / 'figures'
DATA_DIR = ROOT / 'data'  / 'cleaned'

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':         'serif',
    'font.size':           10,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.grid':           True,
    'grid.alpha':          0.3,
    'figure.dpi':          150,
})

# ── Hardcoded geographic / demographic data ────────────────────────────────────
URBAN_PCT = {
    'CA':95,'NY':88,'MA':92,'NJ':95,'FL':91,'IL':88,'TX':85,'WA':84,
    'CO':86,'OR':81,'NV':94,'MD':87,'CT':88,'AZ':90,'GA':75,'NC':66,
    'VA':76,'MI':74,'OH':77,'PA':79,'MN':73,'WI':70,'IN':72,'MO':70,
    'TN':66,'KY':58,'AL':59,'LA':73,'OK':66,'KS':61,'NE':74,'IA':64,
    'AR':57,'MS':50,'WV':49,'SC':66,'ID':70,'UT':91,'NM':77,'MT':56,
    'SD':57,'ND':60,'WY':65,'AK':66,'HI':91,'ME':62,'NH':60,'VT':39,
    'RI':91,'DC':100,'DE':83,
}
PCT_MINORITY = {
    'AL':33,'AK':8,'AZ':47,'AR':23,'CA':62,'CO':38,'CT':30,'DE':36,
    'FL':48,'GA':43,'HI':83,'ID':14,'IL':41,'IN':22,'IA':14,'KS':23,
    'KY':16,'LA':44,'ME':7,'MD':52,'MA':34,'MI':29,'MN':22,'MS':45,
    'MO':21,'MT':12,'NE':20,'NV':56,'NH':9,'NJ':52,'NM':70,'NY':46,
    'NC':37,'ND':12,'OH':22,'OK':28,'OR':25,'PA':26,'RI':34,'SC':38,
    'SD':17,'TN':26,'TX':61,'UT':24,'VT':7,'VA':39,'WA':34,'WV':8,
    'WI':19,'WY':16,'DC':70,
}
STATE_COORDS = {
    'AL':(32.8,-86.8),'AK':(64,-153),'AZ':(34,-111),'AR':(34.8,-92.4),
    'CA':(37,-120),'CO':(39,-105.5),'CT':(41.6,-72.7),'DE':(39,-75.5),
    'FL':(28.1,-81.6),'GA':(32.9,-83.4),'HI':(20.9,-157),'ID':(44.4,-114.6),
    'IL':(40,-89.2),'IN':(40,-86.1),'IA':(42.1,-93.5),'KS':(38.5,-98.4),
    'KY':(37.5,-85),'LA':(31,-91.8),'ME':(45.4,-69.2),'MD':(39,-76.8),
    'MA':(42.3,-71.8),'MI':(44,-85.5),'MN':(46.4,-93.1),'MS':(32.7,-89.7),
    'MO':(38.4,-92.3),'MT':(47,-110),'NE':(41.5,-99.9),'NV':(38.5,-117),
    'NH':(43.7,-71.6),'NJ':(40.1,-74.5),'NM':(34.5,-106),'NY':(42.9,-75.6),
    'NC':(35.6,-79.8),'ND':(47.5,-100.5),'OH':(40.4,-82.8),'OK':(35.5,-97.5),
    'OR':(44,-120.6),'PA':(40.6,-77.3),'RI':(41.7,-71.5),'SC':(33.8,-80.9),
    'SD':(44.4,-100.3),'TN':(35.9,-86.4),'TX':(31.1,-97.6),'UT':(39.3,-111.1),
    'VT':(44,-72.7),'VA':(37.8,-78.2),'WA':(47.4,-120.6),'WV':(38.6,-80.6),
    'WI':(44.3,-89.8),'WY':(43,-107.6),'DC':(38.9,-77),
}
CENSUS_REGION = {
    'ME':'Northeast','NH':'Northeast','VT':'Northeast','MA':'Northeast',
    'RI':'Northeast','CT':'Northeast','NY':'Northeast','NJ':'Northeast','PA':'Northeast',
    'OH':'Midwest','MI':'Midwest','IN':'Midwest','IL':'Midwest','WI':'Midwest',
    'MN':'Midwest','IA':'Midwest','MO':'Midwest','ND':'Midwest','SD':'Midwest',
    'NE':'Midwest','KS':'Midwest',
    'DE':'South','MD':'South','DC':'South','VA':'South','WV':'South','NC':'South',
    'SC':'South','GA':'South','FL':'South','KY':'South','TN':'South','AL':'South',
    'MS':'South','AR':'South','LA':'South','OK':'South','TX':'South',
    'MT':'West','ID':'West','WY':'West','CO':'West','NM':'West','AZ':'West',
    'UT':'West','NV':'West','WA':'West','OR':'West','CA':'West','AK':'West','HI':'West',
}

# ── Helper functions ───────────────────────────────────────────────────────────
def sig_stars(p):
    if pd.isna(p): return ''
    if p < 0.01:   return '***'
    if p < 0.05:   return '**'
    if p < 0.10:   return '*'
    return ''


def run_twfe(df_sub, treatments, dep='ln_emp', cluster='state'):
    """OLS with state + time FE and state-clustered SEs."""
    needed = [dep] + treatments + [cluster, 'time']
    df_sub = df_sub.dropna(subset=needed).copy()
    if df_sub[cluster].nunique() < 3 or df_sub['time'].nunique() < 2:
        return None
    df_sub['time'] = df_sub['time'].astype(int)
    rhs = ' + '.join(treatments) + ' + C(state) + C(time)'
    formula = f'{dep} ~ {rhs}'
    try:
        model = smf.ols(formula, data=df_sub).fit(
            cov_type='cluster',
            cov_kwds={'groups': df_sub[cluster]},
        )
        return model
    except Exception as e:
        print(f'  [warn] run_twfe failed: {e}')
        return None


def extract(model, var):
    """Return coef/SE/pval dict; NaN if unavailable."""
    if model is None:
        return dict(coef=np.nan, se=np.nan, pval=np.nan)
    if var not in model.params.index:
        return dict(coef=np.nan, se=np.nan, pval=np.nan)
    return dict(
        coef=float(model.params[var]),
        se=float(model.bse[var]),
        pval=float(model.pvalues[var]),
    )

# ── Load and enrich data ───────────────────────────────────────────────────────
print('Loading panel data ...')
df = pd.read_csv(DATA_DIR / 'panel.csv')

df['urban_pct']     = df['state'].map(URBAN_PCT)
df['pct_minority']  = df['state'].map(PCT_MINORITY)
df['census_region'] = df['state'].map(CENSUS_REGION)
df['high_urban']    = (df['urban_pct']    >= 75).astype(int)
df['high_minority'] = (df['pct_minority'] >= 40).astype(int)
df['low_minority']  = (df['pct_minority'] <  25).astype(int)

# ── (a) Regional TWFE ─────────────────────────────────────────────────────────
print('(a) Regional TWFE ...')
for r in ['Northeast', 'Midwest', 'South', 'West']:
    df[f'did_{r}'] = df['did'] * (df['census_region'] == r).astype(int)

regional_vars = ['did_Northeast', 'did_Midwest', 'did_South', 'did_West']
model_reg = run_twfe(df, regional_vars)
reg_results = {r: extract(model_reg, f'did_{r}') for r in ['Northeast', 'Midwest', 'South', 'West']}

# ── (b) Urban split ───────────────────────────────────────────────────────────
print('(b) Urban-rural split ...')
model_hi_urban = run_twfe(df[df['high_urban'] == 1], ['did'])
model_lo_urban = run_twfe(df[df['high_urban'] == 0], ['did'])
urban_results = {
    'High-Urban':  extract(model_hi_urban, 'did'),
    'Low-Urban':   extract(model_lo_urban, 'did'),
}

# ── (c) Minority split ────────────────────────────────────────────────────────
print('(c) Minority split ...')
model_hi_min = run_twfe(df[df['high_minority'] == 1], ['did'])
model_lo_min = run_twfe(df[df['low_minority']  == 1], ['did'])
minority_results = {
    'High-Minority': extract(model_hi_min, 'did'),
    'Low-Minority':  extract(model_lo_min, 'did'),
}

# ── (d) Temporal periods ───────────────────────────────────────────────────────
print('(d) Temporal-period regressions ...')
model_pre  = run_twfe(df[df['year'] <= 2019],          ['did'])
model_cov  = run_twfe(df[df['year'].isin([2020, 2021])], ['did'])
model_post = run_twfe(df[df['year'] >= 2022],          ['did'])
temporal_results = {
    'Pre-COVID\n(2018–19)':  extract(model_pre,  'did'),
    'COVID\n(2020–21)':      extract(model_cov,  'did'),
    'Post-COVID\n(2022–24)': extract(model_post, 'did'),
}

# ── (e) Wage bite ─────────────────────────────────────────────────────────────
print('(e) Wage-bite interaction ...')
df['did_unemp'] = df['did'] * df['unemp_rate']
model_bite = run_twfe(df, ['did', 'unemp_rate', 'did_unemp'])
bite_results = {
    'DID main':         extract(model_bite, 'did'),
    'DID × Unemp Rate': extract(model_bite, 'did_unemp'),
}

# ── Collect all results → CSV ──────────────────────────────────────────────────
all_results = {
    **{f'Region: {k}': v for k, v in reg_results.items()},
    **{f'Urban: {k}':  v for k, v in urban_results.items()},
    **{f'Min: {k}':    v for k, v in minority_results.items()},
    **{f'Temp: {k.replace(chr(10), " ")}': v for k, v in temporal_results.items()},
    **{f'Bite: {k}':   v for k, v in bite_results.items()},
}
rows = [{'label': lbl, **vals} for lbl, vals in all_results.items()]
results_df = pd.DataFrame(rows)
results_df['stars'] = results_df['pval'].apply(sig_stars)
out_path = DATA_DIR / 'heterogeneity_results.csv'
results_df.to_csv(out_path, index=False)
print(f'Saved → {out_path}  ({len(results_df)} rows)')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Spatial Distribution of Treatment Cohorts
# ══════════════════════════════════════════════════════════════════════════════
print('\nFigure 3: spatial map ...')

state_sum = (
    df.groupby('state')
    .agg(mean_ln_emp=('ln_emp', 'mean'),
         ever_treated=('ever_treated', 'first'),
         first_treat_year=('first_treat_year', 'first'))
    .reset_index()
)
state_sum['ever_treated']    = state_sum['ever_treated'].astype(str).isin(['True', '1', 'true'])
state_sum['first_treat_year'] = pd.to_numeric(state_sum['first_treat_year'], errors='coerce')


def classify_state(row):
    if not row['ever_treated']:
        return 'Never-Treated'
    return 'Early-Treated (g\u202f<\u202f2020)' if row['first_treat_year'] < 2020 else 'Late-Treated (g\u2009\u2265\u20092020)'


state_sum['cohort'] = state_sum.apply(classify_state, axis=1)

COHORT_COLOR = {
    'Never-Treated':               '#888888',
    'Early-Treated (g\u202f<\u202f2020)': '#2166ac',
    'Late-Treated (g\u2009\u2265\u20092020)':  '#d6604d',
}

fig3, ax3 = plt.subplots(figsize=(12, 7))
ax3.set_facecolor('#f8f8f8')

for cohort, color in COHORT_COLOR.items():
    sub = state_sum[state_sum['cohort'] == cohort]
    lons, lats, sizes = [], [], []
    for _, row in sub.iterrows():
        if row['state'] not in STATE_COORDS:
            continue
        lat, lon = STATE_COORDS[row['state']]
        lons.append(lon); lats.append(lat)
        sizes.append(max(30, row['mean_ln_emp'] ** 2 * 8))
    ax3.scatter(lons, lats, s=sizes, c=color, alpha=0.85, zorder=4,
                label=cohort, edgecolors='white', linewidths=0.5)

for _, row in state_sum.iterrows():
    if row['state'] not in STATE_COORDS:
        continue
    lat, lon = STATE_COORDS[row['state']]
    ax3.annotate(row['state'], (lon, lat), fontsize=5.5, ha='center',
                 va='bottom', xytext=(0, 5), textcoords='offset points', zorder=5)

ax3.legend(loc='lower right', fontsize=9, framealpha=0.9, frameon=True)
ax3.set_xlim(-170, -60)
ax3.set_ylim(16, 72)
ax3.set_xlabel('Longitude', fontsize=10)
ax3.set_ylabel('Latitude', fontsize=10)
ax3.set_title('Spatial Distribution of Minimum Wage Treatment Cohorts',
              fontsize=13, fontweight='bold', pad=10)
ax3.annotate('Note: Point size ∝ mean log L&H employment. State centroids shown.',
             xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7.5,
             color='#555555', style='italic')

fig3.tight_layout()
for ext in ('pdf', 'png'):
    fig3.savefig(FIG_DIR / f'fig3_spatial_map.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig3)
print('  → fig3_spatial_map saved')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Temporal Dynamics and Regional Heterogeneity
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 4: temporal + regional heterogeneity ...')

fig4, (ax4l, ax4r) = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: ATT by time period ──────────────────────────────────────────────────
t_labels  = list(temporal_results.keys())
t_coefs   = [temporal_results[k]['coef'] for k in t_labels]
t_ses     = [temporal_results[k]['se']   for k in t_labels]
t_pvals   = [temporal_results[k]['pval'] for k in t_labels]
t_colors  = ['#d6604d' if (c > 0 or np.isnan(c)) else '#2166ac' for c in t_coefs]

x_pos = np.arange(len(t_labels))
ax4l.bar(x_pos, t_coefs, color=t_colors, alpha=0.85, zorder=3,
         yerr=np.where(np.isnan(t_ses), 0, [1.96 * s for s in t_ses]),
         capsize=6, error_kw=dict(elinewidth=1.5, ecolor='#333333'))
ax4l.axhline(0, color='black', linewidth=0.9, linestyle='--', zorder=2)
for i, (c, s, p) in enumerate(zip(t_coefs, t_ses, t_pvals)):
    if np.isnan(c):
        continue
    stars = sig_stars(p)
    top = c + 1.96 * s if not np.isnan(s) else c
    ax4l.text(i, top + 0.001, stars, ha='center', va='bottom', fontsize=13)
ax4l.set_xticks(x_pos)
ax4l.set_xticklabels(t_labels, fontsize=8.5)
ax4l.set_ylabel('ATT (log points)', fontsize=10)
ax4l.set_title('ATT by Time Period', fontsize=11, fontweight='bold')

# ── Right: ATT by Census region ───────────────────────────────────────────────
r_labels = list(reg_results.keys())
r_coefs  = [reg_results[k]['coef'] for k in r_labels]
r_ses    = [reg_results[k]['se']   for k in r_labels]
r_pvals  = [reg_results[k]['pval'] for k in r_labels]
r_colors = ['#d6604d' if (c > 0 or np.isnan(c)) else '#2166ac' for c in r_coefs]

ax4r.bar(np.arange(len(r_labels)), r_coefs, color=r_colors, alpha=0.85, zorder=3,
         yerr=np.where(np.isnan(r_ses), 0, [1.96 * s for s in r_ses]),
         capsize=6, error_kw=dict(elinewidth=1.5, ecolor='#333333'))
ax4r.axhline(0, color='black', linewidth=0.9, linestyle='--', zorder=2)
for i, (c, s, p) in enumerate(zip(r_coefs, r_ses, r_pvals)):
    if np.isnan(c):
        continue
    stars = sig_stars(p)
    top = c + 1.96 * s if not np.isnan(s) else c
    ax4r.text(i, top + 0.001, stars, ha='center', va='bottom', fontsize=13)
ax4r.set_xticks(np.arange(len(r_labels)))
ax4r.set_xticklabels(r_labels, fontsize=9)
ax4r.set_ylabel('ATT (log points)', fontsize=10)
ax4r.set_title('ATT by Census Region', fontsize=11, fontweight='bold')

# common legend
legend_els = [
    mpatches.Patch(color='#2166ac', alpha=0.85, label='Negative effect'),
    mpatches.Patch(color='#d6604d', alpha=0.85, label='Positive effect'),
]
fig4.legend(handles=legend_els, loc='lower center', ncol=2, fontsize=9,
            frameon=False, bbox_to_anchor=(0.5, -0.04))
fig4.suptitle('Temporal and Regional Heterogeneity in Minimum Wage Effects',
              fontsize=12, fontweight='bold', y=1.01)
fig4.tight_layout()
for ext in ('pdf', 'png'):
    fig4.savefig(FIG_DIR / f'fig4_temporal_heterogeneity.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig4)
print('  → fig4_temporal_heterogeneity saved')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Equity / Fairness Coefficient Plot
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 5: equity/fairness plot ...')

equity_items = [
    ('High-Minority (\u226540%)',   minority_results['High-Minority']),
    ('Low-Minority (<25%)',         minority_results['Low-Minority']),
    ('High-Urban (\u226575%)',      urban_results['High-Urban']),
    ('Low-Urban (<75%)',            urban_results['Low-Urban']),
    ('Northeast',                   reg_results['Northeast']),
    ('South',                       reg_results['South']),
    ('Midwest',                     reg_results['Midwest']),
    ('West',                        reg_results['West']),
]

labels_eq = [it[0] for it in equity_items]
coefs_eq  = [it[1]['coef'] for it in equity_items]
ses_eq    = [it[1]['se']   for it in equity_items]
pvals_eq  = [it[1]['pval'] for it in equity_items]

# diverging palette: blue = negative, red = positive
bar_colors = ['#d6604d' if c > 0 else '#2166ac' for c in coefs_eq]

n = len(equity_items)
fig5, ax5 = plt.subplots(figsize=(9, 6))

for i, (c, s, p, col) in enumerate(zip(coefs_eq, ses_eq, pvals_eq, bar_colors)):
    if np.isnan(c):
        ax5.barh(i, 0, color='#cccccc', alpha=0.5, height=0.6)
        ax5.text(0.001, i, 'n/a', va='center', fontsize=8, color='gray')
        continue
    ci = 1.96 * s if not np.isnan(s) else 0
    ax5.barh(i, c, color=col, alpha=0.85, height=0.6)
    ax5.errorbar(c, i, xerr=ci, fmt='none', color='black',
                 capsize=4, linewidth=1.5, zorder=5)
    stars = sig_stars(p)
    x_text = c + ci + 0.0005 if c >= 0 else c - ci - 0.0005
    ha = 'left' if c >= 0 else 'right'
    ax5.text(x_text, i, stars, va='center', ha=ha, fontsize=13, color='black')

ax5.axvline(0, color='black', linewidth=0.9, linestyle='--')
ax5.set_yticks(range(n))
ax5.set_yticklabels(labels_eq, fontsize=9.5)
ax5.set_xlabel('DID Coefficient (log points)', fontsize=10)
ax5.set_title('Heterogeneous Employment Effects: Equity Dimensions',
              fontsize=11, fontweight='bold')

# Separator line between urban and region groups
ax5.axhline(3.5, color='#999999', linewidth=0.8, linestyle=':')
ax5.axhline(1.5, color='#999999', linewidth=0.8, linestyle=':')
ax5.text(-0.001, 0.5, 'Minority\ncomposition', ha='right', va='center',
         fontsize=7.5, color='#666666', style='italic',
         transform=ax5.get_yaxis_transform())

legend_els = [
    mpatches.Patch(color='#2166ac', alpha=0.85, label='Negative employment effect'),
    mpatches.Patch(color='#d6604d', alpha=0.85, label='Positive employment effect'),
]
ax5.legend(handles=legend_els, loc='lower right', fontsize=9, frameon=False)
ax5.annotate('Error bars = 95% CI.  *** p<0.01,  ** p<0.05,  * p<0.10',
             xy=(0.02, -0.08), xycoords='axes fraction', fontsize=7.5,
             color='#555555', style='italic')

fig5.tight_layout()
for ext in ('pdf', 'png'):
    fig5.savefig(FIG_DIR / f'fig5_equity_heterogeneity.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig5)
print('  → fig5_equity_heterogeneity saved')

print('\nAll done.  Figures 3–5 and heterogeneity_results.csv written.')
