"""
Analysis of the name-halo experiment.

Reads the per-condition ratings produced by name_halo.py and reports:
  1. Mean rating per condition (bootstrap CI over companies)
  2. Paired effects between conditions (bootstrap CI + Wilcoxon test)
  3. Whether each condition's rating separates real winners from failures:
     raw AUC, and a test that controls for batch year and industry
  4. The famous-name effect by actual outcome, industry and famous name
Saves summary CSVs and a figure.

Works with runs that have conditions A-E only or A-G.

Usage:
  python analyze_halo.py data/name_halo_qwen_v2.csv
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, wilcoxon
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
path = sys.argv[1] if len(sys.argv) > 1 else "data/name_halo_qwen_v2.csv"
tag = os.path.splitext(os.path.basename(path))[0]
rng = np.random.default_rng(519)
B = 2000

LABELS = {
    "A_original": "Original\nlogo",
    "B_masked": "Text\nhidden",
    "C_masked_real": "+ real\nname",
    "E_masked_madeup": "+ made-up\nname",
    "G_masked_lookalike": "+ look-alike\nname",
    "F_masked_failed": "+ famous\nfailure",
    "D_masked_famous": "+ famous\nsuccess",
}

res = pd.read_csv(path)
conds = [c for c in LABELS if c in set(res["condition"])]
wide = res.pivot_table(index="id", columns="condition", values="score")
meta = res.drop_duplicates("id").set_index("id")[["name", "success", "industry", "year"]]
shown = res.pivot_table(index="id", columns="condition", values="name_shown", aggfunc="first")
wide = wide.join(meta)
wide["famous_name"] = shown.get("D_masked_famous")
wide["failed_name"] = shown.get("F_masked_failed")
wide = wide.dropna(subset=conds)
n = len(wide)
print(f"Model: {res['model'].iloc[0]}")
print(f"Companies with all {len(conds)} conditions: {n} (successes: {int(wide['success'].sum())})")
print(f"Mean digit-token probability mass: {res['digit_mass'].mean():.3f} "
      f"(lowest condition: {res.groupby('condition')['digit_mass'].mean().min():.3f})\n")


def boot_mean(x):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


print("1. Mean rating by condition (0-9, 95% bootstrap CI)")
cond_rows = []
for c in conds:
    m, lo, hi = boot_mean(wide[c])
    cond_rows.append({"condition": c, "mean": m, "ci_low": lo, "ci_high": hi})
    print(f"   {c:20s} {m:.3f}  [{lo:.3f}, {hi:.3f}]")

print("\n2. Paired differences (same company, same logo)")
pairs = [
    ("D_masked_famous", "E_masked_madeup", "Famous success vs made-up (name halo)"),
    ("F_masked_failed", "E_masked_madeup", "Famous failure vs made-up"),
    ("D_masked_famous", "F_masked_failed", "Famous success vs famous failure"),
    ("G_masked_lookalike", "E_masked_madeup", "Look-alike vs made-up"),
    ("D_masked_famous", "G_masked_lookalike", "Famous success vs its look-alike"),
    ("C_masked_real", "E_masked_madeup", "Real name vs made-up"),
    ("A_original", "B_masked", "Text visible vs text hidden"),
    ("E_masked_madeup", "B_masked", "Made-up name vs no name"),
]
pair_rows = []
for a, b, label in pairs:
    if a not in conds or b not in conds:
        continue
    d = wide[a] - wide[b]
    m, lo, hi = boot_mean(d)
    p = wilcoxon(wide[a], wide[b]).pvalue if (d != 0).any() else 1.0
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    pair_rows.append({"comparison": label, "a": a, "b": b, "mean_diff": m,
                      "ci_low": lo, "ci_high": hi, "wilcoxon_p": p,
                      "effect_size_dz": dz, "share_a_higher": (d > 0).mean()})
    print(f"   {label:40s} diff={m:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.1e}  "
          f"dz={dz:+.2f}  a>b in {(d > 0).mean():.0%}")

print("\n3. Does the rating separate real winners from failures?")
print("   AUC: raw, 0.5 = chance.  Controlled: logistic regression")
print("   success ~ rating + batch year + industry, likelihood-ratio test for the rating")
y = wide["success"].values
yr = (wide["year"] - wide["year"].mean()) / wide["year"].std()
ind = wide["industry"].fillna("Other").astype(str)
top = ind.value_counts()
ind = ind.where(ind.isin(top[top >= 15].index), "Other")
base_X = pd.concat([yr.rename("year_z"),
                    pd.get_dummies(ind, prefix="ind", drop_first=True, dtype=float)], axis=1)
base = sm.Logit(y, sm.add_constant(base_X)).fit(disp=0)
auc_rows = []
for c in conds:
    s = wide[c].values
    auc = roc_auc_score(y, s)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    z = (wide[c] - wide[c].mean()) / wide[c].std()
    full = sm.Logit(y, sm.add_constant(pd.concat([base_X, z.rename("rating_z")], axis=1))).fit(disp=0)
    lr = 2 * (full.llf - base.llf)
    p_ctrl = chi2.sf(lr, 1)
    or_ = np.exp(full.params["rating_z"])
    auc_rows.append({"condition": c, "auc": auc, "ci_low": lo, "ci_high": hi,
                     "odds_ratio_per_sd_controlled": or_, "p_controlled": p_ctrl})
    print(f"   {c:20s} AUC={auc:.3f} [{lo:.3f}, {hi:.3f}]   "
          f"controlled OR per SD={or_:.2f}, p={p_ctrl:.3f}")
print(f"   (batch year + industry alone: AUC={roc_auc_score(y, base.predict()):.3f}, in-sample)")

wide["halo"] = wide["D_masked_famous"] - wide["E_masked_madeup"]

print("\n4. Name halo (famous success - made-up) by actual outcome")
for s, g in wide.groupby("success"):
    m, lo, hi = boot_mean(g["halo"])
    print(f"   {'succeeded' if s == 1 else 'failed':10s} n={len(g):3d}  halo={m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\n5. Name halo by industry (15+ companies)")
for name_, g in wide.groupby(ind):
    if len(g) >= 15:
        m, lo, hi = boot_mean(g["halo"])
        print(f"   {name_:22s} n={len(g):3d}  halo={m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\n6. Rating shift by name used (vs made-up, same company)")
for col, cond in [("famous_name", "D_masked_famous"), ("failed_name", "F_masked_failed")]:
    if cond not in conds or wide[col].isna().all():
        continue
    shift = wide[cond] - wide["E_masked_madeup"]
    for nm, g in shift.groupby(wide[col]):
        print(f"   {nm:10s} n={len(g):3d}  shift={g.mean():+.3f}")

print("\n7. Companies where showing the logo text raised the rating most (A - B)")
wide["text_effect"] = wide["A_original"] - wide["B_masked"]
print(wide.sort_values("text_effect", ascending=False)
      [["name", "success", "text_effect"]].head(10).round(3).to_string())

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
pd.DataFrame(pair_rows).to_csv(f"data/{tag}_pairs.csv", index=False)
pd.DataFrame(auc_rows).to_csv(f"data/{tag}_auc.csv", index=False)
pd.DataFrame(cond_rows).to_csv(f"data/{tag}_means.csv", index=False)

order = conds
cr = pd.DataFrame(cond_rows).set_index("condition").loc[order]
ar = pd.DataFrame(auc_rows).set_index("condition").loc[order]
colors = {"D_masked_famous": "#2f6fb0", "F_masked_failed": "#c0504d",
          "G_masked_lookalike": "#e6a23c"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
ax.bar(range(len(order)), cr["mean"], color=[colors.get(c, "#9aa5b1") for c in order],
       yerr=[cr["mean"] - cr["ci_low"], cr["ci_high"] - cr["mean"]], capsize=4)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([LABELS[c] for c in order], fontsize=8.5)
ax.set_ylabel("Mean success rating (0-9)")
ax.set_title("Same logo, different name")
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
ax.bar(range(len(order)), ar["auc"], color="#9aa5b1",
       yerr=[ar["auc"] - ar["ci_low"], ar["ci_high"] - ar["auc"]], capsize=4)
ax.axhline(0.5, color="black", lw=1, ls="--")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([LABELS[c] for c in order], fontsize=8.5)
ax.set_ylabel("AUC for real outcome")
ax.set_ylim(0.3, 0.8)
ax.set_title("Does the rating predict real success?")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"figures/{tag}.png", dpi=200)
print(f"\nSaved data/{tag}_pairs.csv, data/{tag}_auc.csv, data/{tag}_means.csv, figures/{tag}.png")
