"""
Bouba/kiki in startup logos: does a logo's shape match the sound of the name?

1. Sound score of each company name (-1 = sharp/"kiki", +1 = round/"bouba"),
   from letter classes used in sound-symbolism research:
     round : voiced stops and sonorants b d g l m n, rounded vowels o u
     sharp : voiceless stops and sharp fricatives p t k z x, front vowels i e
2. Shape of each logo DESIGN, measured on the text-masked image so the letters
   of the name are not counted as shape:
     circularity  4*pi*area / perimeter^2 of the foreground (1 = perfect circle)
     corner_rate  sharp corners (turning angle > 45 deg) per 100 px of outline
     roundness    z(circularity) - z(corner_rate), the main shape score
3. Tests:
     a. Spearman correlation between name sound and logo roundness
     b. Regression roundness ~ sound + batch year + industry (+ text/no text)
     c. Permutation test (shuffle names across logos)
     d. Robustness on logos that never had text (pure symbols, no masking needed)
     e. Does a good sound-shape match predict success? (logit with controls)
Saves data/bouba_kiki.csv and figures/bouba_kiki.png

Usage:
  pip install opencv-python-headless   (once)
  python bouba_kiki.py
"""
import os
import re
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

ROUND_CONS, SHARP_CONS = set("bdglmn"), set("ptkzx")
ROUND_VOW, SHARP_VOW = set("ou"), set("ie")
warnings.filterwarnings("ignore")
rng = np.random.default_rng(519)


def sound_scores(name):
    s = re.sub(r"[^a-z]", "", str(name).lower())
    r_c = sum(ch in ROUND_CONS for ch in s)
    s_c = sum(ch in SHARP_CONS for ch in s)
    r_v = sum(ch in ROUND_VOW for ch in s)
    s_v = sum(ch in SHARP_VOW for ch in s)

    def sc(r, k):
        return (r - k) / (r + k) if (r + k) > 0 else np.nan
    return sc(r_c + r_v, s_c + s_v), sc(r_c, s_c), sc(r_v, s_v)


def shape_scores(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.nan, np.nan, np.nan
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    alpha = img[:, :, 3].astype(float) / 255
    rgb = img[:, :, :3].astype(float) * alpha[..., None] + 255 * (1 - alpha[..., None])
    rgb = cv2.resize(rgb.astype(np.uint8), (300, 300), interpolation=cv2.INTER_CUBIC)
    corners = np.concatenate([rgb[:6, :6].reshape(-1, 3), rgb[:6, -6:].reshape(-1, 3),
                              rgb[-6:, :6].reshape(-1, 3), rgb[-6:, -6:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    diff = np.abs(rgb.astype(float) - bg).max(axis=2)
    mask = (diff > 40).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = mask.mean() / 255
    if fg < 0.02:
        return np.nan, np.nan, fg
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 30]
    if not contours:
        return np.nan, np.nan, fg
    area = sum(cv2.contourArea(c) for c in contours)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    circularity = 4 * np.pi * area / perim ** 2 if perim > 0 else np.nan
    n_corners = 0
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        v1 = np.roll(approx, 1, axis=0) - approx
        v2 = np.roll(approx, -1, axis=0) - approx
        cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
        interior = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        n_corners += int((interior < 135).sum())
    corner_rate = 100 * n_corners / perim if perim > 0 else np.nan
    return circularity, corner_rate, fg


df = pd.read_csv("data/yc/clean.csv")
feat_path = "data/yc_all_features.csv"
if os.path.exists(feat_path):
    feats = pd.read_csv(feat_path)
    keep = [c for c in ["id", "has_text", "logo_type", "text_share"] if c in feats.columns]
    if "id" in keep:
        df = df.merge(feats[keep], on="id", how="left")
df["year"] = df["batch"].astype(str).str.extract(r"(\d{4})").astype(float)

rows = []
total = len(df)
for k, r in enumerate(df.itertuples(index=False), 1):
    masked = os.path.join("data/yc/masked", f"{r.id}.png")
    orig = os.path.join("data/yc/logos", f"{r.id}.png")
    path = masked if os.path.exists(masked) else orig
    circ, corn, fg = shape_scores(path)
    snd, snd_c, snd_v = sound_scores(r.name)
    rows.append({"id": r.id, "circularity": circ, "corner_rate": corn, "foreground": fg,
                 "sound": snd, "sound_cons": snd_c, "sound_vowel": snd_v})
    if k % 1000 == 0:
        print(f"{k}/{total} logos measured", flush=True)
d = df.merge(pd.DataFrame(rows), on="id")
d = d.dropna(subset=["circularity", "corner_rate", "sound", "year"]).copy()
z = lambda x: (x - x.mean()) / x.std()
d["roundness"] = z(d["circularity"]) - z(d["corner_rate"])
d["industry5"] = d["industry"].where(
    d["industry"].isin(["B2B", "Consumer", "Healthcare", "Fintech", "Industrials"]), "Other")
os.makedirs("data", exist_ok=True)
d.to_csv("data/bouba_kiki.csv", index=False)
print(f"\nLogos with a usable shape and a scorable name: {len(d)}")
print(f"Shape measures agree with each other: Spearman(circularity, corner_rate) = "
      f"{spearmanr(d['circularity'], d['corner_rate']).correlation:+.2f} (expected negative)")

print("\na. Correlation between name sound (+ = round) and logo roundness")
for col in ["sound", "sound_cons", "sound_vowel"]:
    sub = d.dropna(subset=[col])
    rho, p = spearmanr(sub[col], sub["roundness"])
    print(f"   {col:12s} n={len(sub):5d}  rho={rho:+.3f}  p={p:.4f}")

print("\nb. Regression: roundness ~ sound + batch year + industry")
has_text_col = "has_text" in d.columns and d["has_text"].notna().any()
formula = "roundness ~ sound + year + C(industry5)" + (" + has_text" if has_text_col else "")
fit = smf.ols(formula, data=d.assign(has_text=d.get("has_text", 0)).fillna({"has_text": 0})).fit(cov_type="HC3")
print(f"   sound coefficient = {fit.params['sound']:+.4f}  "
      f"(95% CI {fit.conf_int().loc['sound', 0]:+.4f} to {fit.conf_int().loc['sound', 1]:+.4f}), "
      f"p = {fit.pvalues['sound']:.4f}")

print("\nc. Permutation test (names shuffled across logos, 2000 times)")
obs = spearmanr(d["sound"], d["roundness"]).correlation
perm = np.array([spearmanr(rng.permutation(d["sound"].values), d["roundness"]).correlation
                 for _ in range(2000)])
p_perm = (np.abs(perm) >= abs(obs)).mean()
print(f"   observed rho = {obs:+.3f}, permutation p = {p_perm:.4f}")

if has_text_col:
    sym = d[d["has_text"] == 0]
    rho, p = spearmanr(sym["sound"], sym["roundness"])
    print(f"\nd. Robustness: logos that never had text (pure symbols), n={len(sym)}")
    print(f"   rho = {rho:+.3f}, p = {p:.4f}")

print("\ne. Does a sound-shape match predict success? (2015-2024, Acquired/Public vs Inactive)")
s = d[(d["year"] >= 2015) & (d["year"] <= 2024) & d["status"].isin(["Acquired", "Public", "Inactive"])].copy()
s["success"] = s["status"].isin(["Acquired", "Public"]).astype(int)
s["sound_z"], s["round_z"] = z(s["sound"]), z(s["roundness"])
s["match"] = s["sound_z"] * s["round_z"]
lg = smf.logit("success ~ match + sound_z + round_z + year + C(industry5)", data=s).fit(disp=0)
print(f"   n={len(s)}, successes={int(s['success'].sum())}")
print(f"   match odds ratio = {np.exp(lg.params['match']):.3f} "
      f"(95% CI {np.exp(lg.conf_int().loc['match', 0]):.3f} to {np.exp(lg.conf_int().loc['match', 1]):.3f}), "
      f"p = {lg.pvalues['match']:.4f}")

os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
d["sound_bin"] = pd.cut(d["sound"], [-1.01, -0.6, -0.2, 0.2, 0.6, 1.01],
                        labels=["very sharp", "sharp", "neutral", "round", "very round"])
g = d.groupby("sound_bin", observed=True)["roundness"]
m, se = g.mean(), g.sem()
ax = axes[0]
ax.errorbar(range(len(m)), m, yerr=1.96 * se, fmt="o-", color="#2f6fb0", capsize=4)
ax.set_xticks(range(len(m)))
ax.set_xticklabels(m.index, fontsize=9)
ax.set_xlabel("Sound of the company name")
ax.set_ylabel("Logo roundness (z-score)")
ax.set_title(f"Name sound vs logo shape (rho = {obs:+.3f})")
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
ax.hist(perm, bins=40, color="#9aa5b1")
ax.axvline(obs, color="#c0504d", lw=2, label="observed")
ax.set_xlabel("Correlation with names shuffled")
ax.set_title(f"Permutation test (p = {p_perm:.4f})")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/bouba_kiki.png", dpi=200)

d.sort_values("roundness").to_csv("data/bouba_kiki_sorted.csv", index=False)
print("\nMost angular logos:", ", ".join(d.nsmallest(8, "roundness")["name"].astype(str)))
print("Most round logos:  ", ", ".join(d.nlargest(8, "roundness")["name"].astype(str)))
print("\nSaved data/bouba_kiki.csv and figures/bouba_kiki.png")
