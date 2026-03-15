#!/usr/bin/env python3
"""
07_new_figures.py
New spatial, temporal, and comparative figures for minimum wage paper.
Run from project root: python3 code/07_new_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── paths ──────────────────────────────────────────────────────────────────
DATA  = "data/cleaned/panel.csv"
FIGS  = "paper/figures"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "serif"

# ── geographic lookups ─────────────────────────────────────────────────────
STATE_COORDS = {"AL":(32.8,-86.8),"AK":(64,-153),"AZ":(34,-111),"AR":(34.8,-92.4),
  "CA":(37,-120),"CO":(39,-105.5),"CT":(41.6,-72.7),"DE":(39,-75.5),
  "FL":(28.1,-81.6),"GA":(32.9,-83.4),"HI":(20.9,-157),"ID":(44.4,-114.6),
  "IL":(40,-89.2),"IN":(40,-86.1),"IA":(42.1,-93.5),"KS":(38.5,-98.4),
  "KY":(37.5,-85),"LA":(31,-91.8),"ME":(45.4,-69.2),"MD":(39,-76.8),
  "MA":(42.3,-71.8),"MI":(44,-85.5),"MN":(46.4,-93.1),"MS":(32.7,-89.7),
  "MO":(38.4,-92.3),"MT":(47,-110),"NE":(41.5,-99.9),"NV":(38.5,-117),
  "NH":(43.7,-71.6),"NJ":(40.1,-74.5),"NM":(34.5,-106),"NY":(42.9,-75.6),
  "NC":(35.6,-79.8),"ND":(47.5,-100.5),"OH":(40.4,-82.8),"OK":(35.5,-97.5),
  "OR":(44,-120.6),"PA":(40.6,-77.3),"RI":(41.7,-71.5),"SC":(33.8,-80.9),
  "SD":(44.4,-100.3),"TN":(35.9,-86.4),"TX":(31.1,-97.6),"UT":(39.3,-111.1),
  "VT":(44,-72.7),"VA":(37.8,-78.2),"WA":(47.4,-120.6),"WV":(38.6,-80.6),
  "WI":(44.3,-89.8),"WY":(43,-107.6),"DC":(38.9,-77)}

CENSUS_REGION = {"ME":"Northeast","NH":"Northeast","VT":"Northeast","MA":"Northeast",
  "RI":"Northeast","CT":"Northeast","NY":"Northeast","NJ":"Northeast","PA":"Northeast",
  "OH":"Midwest","MI":"Midwest","IN":"Midwest","IL":"Midwest","WI":"Midwest",
  "MN":"Midwest","IA":"Midwest","MO":"Midwest","ND":"Midwest","SD":"Midwest",
  "NE":"Midwest","KS":"Midwest","DE":"South","MD":"South","DC":"South","VA":"South",
  "WV":"South","NC":"South","SC":"South","GA":"South","FL":"South","KY":"South",
  "TN":"South","AL":"South","MS":"South","AR":"South","LA":"South","OK":"South",
  "TX":"South","MT":"West","ID":"West","WY":"West","CO":"West","NM":"West",
  "AZ":"West","UT":"West","NV":"West","WA":"West","OR":"West","CA":"West",
  "AK":"West","HI":"West"}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")

def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure A: Map of minimum wage levels, Q4 2024 ─────────────────────────
def figA_map_minwage_2024():
    df = pd.read_csv(DATA)
    df24 = df[(df["year"] == 2024) & (df["quarter"] == 4)].copy()
    wage_map = df24.groupby("state")["min_wage"].mean().to_dict()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")

    lats, lons, wages, states_plot = [], [], [], []
    for st, coord in STATE_COORDS.items():
        if st in wage_map:
            lats.append(coord[0])
            lons.append(coord[1])
            wages.append(wage_map[st])
            states_plot.append(st)

    wages_arr = np.array(wages)
    sc = ax.scatter(lons, lats, c=wages_arr, cmap="viridis", s=400,
                    zorder=5, edgecolors="white", linewidths=0.5)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Min Wage ($/hr)", fontsize=11)

    for lon, lat, st, w in zip(lons, lats, states_plot, wages):
        color = "white" if w > 12 else "#222222"
        ax.annotate(st, (lon, lat), fontsize=6, ha="center", va="center",
                    color=color, zorder=6, fontweight="bold")

    ax.text(0.98, 0.05, "21 states at federal floor ($7.25)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                      edgecolor="gray"))

    ax.set_title("State Minimum Wage Levels, Q4 2024", fontsize=13, fontweight="bold")
    ax.set_xlim(-175, -65)
    ax.set_ylim(22, 72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#e8f4f8")

    save(fig, "figA_map_minwage_2024")


# ── Figure B: Map of ATT by state ─────────────────────────────────────────
def figB_map_att_by_state():
    df = pd.read_csv(DATA)

    # overall control group averages
    ctrl = df[df["ever_treated"] == False]
    post_control = ctrl[ctrl["post"] == 1]["ln_emp"].mean()
    pre_control  = ctrl[ctrl["post"] == 0]["ln_emp"].mean()

    treated_states = df[df["ever_treated"] == True]["state"].unique()
    att_map = {}
    for s in treated_states:
        ds = df[df["state"] == s]
        post_t = ds[ds["post"] == 1]["ln_emp"].mean()
        pre_t  = ds[ds["post"] == 0]["ln_emp"].mean()
        att_map[s] = (post_t - pre_t) - (post_control - pre_control)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")

    # never-treated states first (gray)
    never_lats, never_lons, never_states = [], [], []
    for st, coord in STATE_COORDS.items():
        if st in df["state"].values and st not in att_map:
            never_lats.append(coord[0])
            never_lons.append(coord[1])
            never_states.append(st)

    ax.scatter(never_lons, never_lats, c="lightgray", s=400, zorder=5,
               edgecolors="white", linewidths=0.5, label="Never treated")

    for lon, lat, st in zip(never_lons, never_lats, never_states):
        ax.annotate(st, (lon, lat), fontsize=6, ha="center", va="center",
                    color="#555555", zorder=6, fontweight="bold")

    # ever-treated states
    t_lats, t_lons, t_atts, t_states = [], [], [], []
    for st, coord in STATE_COORDS.items():
        if st in att_map:
            t_lats.append(coord[0])
            t_lons.append(coord[1])
            t_atts.append(att_map[st])
            t_states.append(st)

    t_atts_arr = np.array(t_atts)
    sc = ax.scatter(t_lons, t_lats, c=t_atts_arr, cmap="RdBu_r",
                    vmin=-0.05, vmax=0.05, s=400, zorder=6,
                    edgecolors="white", linewidths=0.5)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("ATT (log pts)", fontsize=11)

    for lon, lat, st, att in zip(t_lons, t_lats, t_states, t_atts):
        color = "white" if abs(att) > 0.025 else "#222222"
        ax.annotate(st, (lon, lat), fontsize=6, ha="center", va="center",
                    color=color, zorder=7, fontweight="bold")

    ax.set_title("State-Level Employment Effects of Minimum Wage Increase",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(-175, -65)
    ax.set_ylim(22, 72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#e8f4f8")

    gray_patch = mpatches.Patch(color="lightgray", label="Never treated")
    ax.legend(handles=[gray_patch], loc="lower left", fontsize=9)

    save(fig, "figB_map_att_by_state")


# ── Figure C: Case studies for 4 states ───────────────────────────────────
def figC_case_studies():
    df = pd.read_csv(DATA)
    case_states = ["CA", "NY", "VA", "TX"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for idx, state in enumerate(case_states):
        ax = axes[idx]
        df_s = df[df["state"] == state].sort_values("time").reset_index(drop=True)

        # x-axis positions and labels
        ym_list = df_s["ym"].tolist()
        x = np.arange(len(ym_list))
        tick_positions = x[::4]
        tick_labels = [ym_list[i] for i in range(0, len(ym_list), 4)]

        # Left axis: min_wage
        color_left = "steelblue"
        color_right = "crimson"

        ax2 = ax.twinx()

        line1, = ax.plot(x, df_s["min_wage"].values, color=color_left,
                         linewidth=2, label="Min Wage ($/hr)")
        line2, = ax2.plot(x, df_s["ln_emp"].values, color=color_right,
                          linewidth=2, linestyle="--", label="Log Employment")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Min Wage ($/hr)", color=color_left, fontsize=9)
        ax2.set_ylabel("Log Employment", color=color_right, fontsize=9)
        ax.tick_params(axis="y", labelcolor=color_left)
        ax2.tick_params(axis="y", labelcolor=color_right)

        # Vertical line if ever treated
        ever_t = df_s["ever_treated"].iloc[0]
        if ever_t:
            fty = df_s["first_treat_year"].iloc[0]
            ftt = df_s["first_treat_time"].iloc[0]
            # find index in df_s where time == first_treat_time
            match = df_s[df_s["time"] == ftt]
            if len(match) > 0:
                vline_x = match.index[0]
            else:
                vline_x = None
            if vline_x is not None:
                ax.axvline(x=vline_x, color="gray", linestyle="--",
                           alpha=0.7, linewidth=1.5)
            title_str = f"{state} (treated: {int(fty)})"
        else:
            title_str = f"{state} (never treated)"

        ax.set_title(title_str, fontsize=11, fontweight="bold")
        remove_spines(ax)

        # Legend only for first subplot
        if idx == 0:
            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper left", fontsize=8)

    fig.suptitle("Minimum Wage and L\u0026H Employment: State Case Studies",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    save(fig, "figC_case_studies")


# ── Figure D: Wage trends by Census region ────────────────────────────────
def figD_wage_trend_by_region():
    df = pd.read_csv(DATA)
    df["region"] = df["state"].map(CENSUS_REGION)

    reg_ym = df.groupby(["region", "ym"])["min_wage"].mean().reset_index()
    ym_ordered = df.drop_duplicates("ym").sort_values("time")["ym"].tolist()

    region_colors = {
        "Northeast": "navy",
        "South":     "darkorange",
        "Midwest":   "forestgreen",
        "West":      "crimson",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")

    for region, color in region_colors.items():
        sub = reg_ym[reg_ym["region"] == region].copy()
        sub["ym_idx"] = sub["ym"].apply(lambda y: ym_ordered.index(y) if y in ym_ordered else np.nan)
        sub = sub.dropna(subset=["ym_idx"]).sort_values("ym_idx")
        ax.plot(sub["ym_idx"].values, sub["min_wage"].values,
                color=color, linewidth=2, label=region)

    # Federal floor line
    ax.axhline(y=7.25, color="gray", linestyle="--", linewidth=1.5,
               label="Federal Floor ($7.25)")

    # COVID shading
    if "2020Q1" in ym_ordered and "2021Q4" in ym_ordered:
        idx_start = ym_ordered.index("2020Q1")
        idx_end   = ym_ordered.index("2021Q4")
        ax.axvspan(idx_start, idx_end, alpha=0.15, color="gray", label="COVID-19")

    # x-axis ticks
    tick_positions = list(range(0, len(ym_ordered), 4))
    tick_labels = [ym_ordered[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Average Min Wage ($/hr)", fontsize=11)
    ax.set_title("State Minimum Wage Trends by Census Region, 2018\u20132024",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    remove_spines(ax)
    fig.tight_layout()

    save(fig, "figD_wage_trend_by_region")


# ── Figure E: Policy comparison ───────────────────────────────────────────
def figE_policy_comparison():
    labels = [
        "This Study\n(Sun-Abraham)",
        "Dube et al.\n(2010, ReStat)",
        "This Study\n(C-S 2021)",
        "This Study\n(TWFE)",
        "CBO (2021)\nFederal $15",
        "Neumark & Wascher\n(2007, meta)",
    ]
    ests   = [+0.0024, +0.0010, -0.0132, -0.0092, -0.0140, -0.0200]
    ses    = [ 0.0012,  0.0040,  0.0067,  0.0020,  0.0070,  0.0050]
    colors = ["steelblue", "gray", "steelblue", "steelblue", "darkorange", "gray"]

    # Sort descending by estimate
    sort_idx = np.argsort(ests)[::-1]
    labels_s = [labels[i] for i in sort_idx]
    ests_s   = np.array(ests)[sort_idx]
    ses_s    = np.array(ses)[sort_idx]
    colors_s = [colors[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(labels_s))
    xerr  = 1.96 * ses_s

    for i, (y, est, err, col) in enumerate(zip(y_pos, ests_s, xerr, colors_s)):
        ax.barh(y, est, xerr=err, color=col, alpha=0.75, height=0.5,
                error_kw=dict(ecolor="black", capsize=4, linewidth=1.2))

    ax.axvline(x=0, color="black", linewidth=1.5, linestyle="--")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_s, fontsize=9)
    ax.set_xlabel("ATT on Log L\u0026H Employment", fontsize=11)
    ax.set_title("Our Estimates vs. Prior Literature", fontsize=13, fontweight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color="steelblue",   alpha=0.75, label="This Study"),
        mpatches.Patch(color="darkorange",  alpha=0.75, label="Policy Report (CBO)"),
        mpatches.Patch(color="gray",        alpha=0.75, label="Prior Literature"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    ax.text(0.01, 0.01, "Note: Error bars show 95% CI.",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    remove_spines(ax)
    fig.tight_layout()

    save(fig, "figE_policy_comparison")


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    figA_map_minwage_2024()
    figB_map_att_by_state()
    figC_case_studies()
    figD_wage_trend_by_region()
    figE_policy_comparison()
    print("All 5 figures saved to paper/figures/")
