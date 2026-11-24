"""
Logo distinctiveness: how far each firm's logo is from the logos of its own sector (same base year).

Inputs  (data/embed/ for 2019, data/embed_<year>/ otherwise; made by logo_embed.py)
  index.csv, clip_orig.npy, clip_masked.npy, dino_orig.npy, dino_masked.npy,
  validation_by_sector.csv (sector "norm strength")

For each embedding (model x original/masked) and each firm:
  knn{k}      1 - mean cosine similarity to its k most similar same-sector logos (k = 5, 10, 20)
  centroid    1 - cosine similarity to the sector's average logo, leaving the firm itself out
  *_z         the same score standardized within the sector (mean 0, sd 1), so a "distinctive bank"
              and a "distinctive biotech" are on the same scale
Masked scores are left empty for text-only logos (nothing left after removing the text).

Pre-registered main measures (z-scores, larger = more distinctive):
  D_visual    average of dino_masked_knn10_z and clip_masked_knn10_z   (the design, words removed)
  D_overall   clip_orig_knn10_z                                         (design + wording)
Plus:
  sector_norm_strength   how strongly the sector's logos resemble each other (lift from the
                         sector check on masked logos); used to test whether standing out matters
                         more where a visual convention exists

Outputs
  distinctiveness.csv            one row per firm
  distinctiveness_examples.html  per sector: the 5 most typical and 5 most distinctive logos
  distinctiveness_summary.txt    correlations between the measures, by-sector means

Usage:
  python3 distinctiveness.py              # 2019
  python3 distinctiveness.py --year 2022
"""
import argparse
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd

EMBS = ["clip_orig", "clip_masked", "dino_orig", "dino_masked"]
KS = (5, 10, 20)
MIN_SECTOR = 25          # smaller sectors are pooled into "Other" for the peer comparison


def peer_scores(X, groups):
    """kNN and leave-one-out centroid distances within each group. X rows are L2-normalized."""
    n = len(X)
    out = {f"knn{k}": np.full(n, np.nan) for k in KS}
    out["centroid"] = np.full(n, np.nan)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        m = len(idx)
        if m < 3:
            continue
        Z = X[idx]
        S = Z @ Z.T
        np.fill_diagonal(S, -np.inf)
        srt = -np.sort(-S, axis=1)                   # descending similarities to peers
        for k in KS:
            kk = min(k, m - 1)
            out[f"knn{k}"][idx] = 1 - srt[:, :kk].mean(axis=1)
        total = Z.sum(axis=0)
        loo = total[None, :] - Z                     # sector sum without the firm itself
        loo /= np.linalg.norm(loo, axis=1, keepdims=True)
        out["centroid"][idx] = 1 - (Z * loo).sum(axis=1)
    return out


def zscore_within(values, groups):
    s = pd.Series(values)
    g = pd.Series(groups)
    mu = s.groupby(g).transform("mean")
    sd = s.groupby(g).transform("std")
    return ((s - mu) / sd).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()
    Y = args.year
    E = Path("data/embed" if Y == 2019 else f"data/embed_{Y}")
    idx = pd.read_csv(E / "index.csv")
    n = len(idx)

    sector = idx["sector"].fillna("Other").to_numpy()
    counts = pd.Series(sector).value_counts()
    peer_group = np.where(pd.Series(sector).map(counts).to_numpy() >= MIN_SECTOR, sector, "Other")
    has_design = (idx["symbol_type"] != "text_only").to_numpy()

    res = idx[["cik", "base_year", "name_base", "sector", "symbol_type", "logo_path", "masked_path"]].copy()
    res["peer_group"] = peer_group
    res["peer_group_n"] = pd.Series(peer_group).map(pd.Series(peer_group).value_counts()).to_numpy()

    for emb in EMBS:
        f = E / f"{emb}.npy"
        if not f.exists():
            print(f"missing {f}, skipped")
            continue
        X = np.load(f).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        keep = has_design if emb.endswith("masked") else np.ones(n, bool)
        sc = {k: np.full(n, np.nan) for k in [f"knn{k}" for k in KS] + ["centroid"]}
        part = peer_scores(X[keep], peer_group[keep])
        for k, v in part.items():
            sc[k][keep] = v
        for k, v in sc.items():
            res[f"{emb}_{k}"] = v
            res[f"{emb}_{k}_z"] = zscore_within(v, peer_group)

    res["D_visual"] = res[["dino_masked_knn10_z", "clip_masked_knn10_z"]].mean(axis=1, skipna=False)
    res["D_overall"] = res["clip_orig_knn10_z"]
    res["D_visual_dino_orig"] = res["dino_orig_knn10_z"]      # robustness: design incl. lettering shapes

    vfile = E / "validation_by_sector.csv"
    if vfile.exists():
        v = pd.read_csv(vfile)
        lift = v[v["run"].str.startswith("dino masked")].set_index("sector")["lift"]
        lift_c = v[v["run"].str.startswith("clip masked")].set_index("sector")["lift"]
        strength = pd.concat([lift, lift_c], axis=1).mean(axis=1)
        res["sector_norm_strength"] = res["sector"].map(strength)

    res.to_csv(E / "distinctiveness.csv", index=False)

    # ---------------- summary
    cols = ["D_visual", "D_overall", "D_visual_dino_orig", "clip_masked_knn10_z", "dino_masked_knn10_z",
            "clip_orig_centroid_z", "dino_masked_centroid_z", "clip_orig_knn5_z", "clip_orig_knn20_z"]
    cols = [c for c in cols if c in res]
    lines = [f"Distinctiveness, base year {Y}: {n} firms, peer groups {res['peer_group'].nunique()}",
             f"D_visual available for {res['D_visual'].notna().sum()} (text-only logos excluded), "
             f"D_overall for {res['D_overall'].notna().sum()}", "",
             "Spearman correlations between measures (do they agree?):",
             res[cols].corr(method="spearman").round(2).to_string(), "",
             "Raw (not z) mean distinctiveness by sector, clip_orig_knn10 and dino_masked_knn10:",
             res.groupby("sector")[["clip_orig_knn10", "dino_masked_knn10"]].mean().round(3)
                .sort_values("clip_orig_knn10").to_string()]
    if "sector_norm_strength" in res:
        lines += ["", "Sector norm strength (lift of masked logos):",
                  res.groupby("sector")["sector_norm_strength"].first().sort_values(ascending=False).round(2).to_string()]
    by_type = res.groupby("symbol_type")["D_overall"].describe()[["count", "mean", "std"]].round(2)
    lines += ["", "D_overall by logo type:", by_type.to_string()]
    (E / "distinctiveness_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # ---------------- examples page
    sec_html = []
    for g, d in res.dropna(subset=["D_overall"]).groupby("peer_group"):
        d = d.sort_values("D_overall")
        def row(sub, label):
            cells = "".join(
                f'<div class=c><img src="{os.path.relpath(p, E)}"><span>{html.escape(str(nm))[:28]}<br>'
                f'z={z:+.2f}</span></div>' for p, nm, z in zip(sub["logo_path"], sub["name_base"], sub["D_overall"]))
            return f"<div class=row><b>{label}</b>{cells}</div>"
        sec_html.append(f"<h3>{html.escape(g)} (n={len(d)})</h3>" + row(d.head(5), "most typical") +
                        row(d.tail(5)[::-1], "most distinctive"))
    (E / "distinctiveness_examples.html").write_text(
        "<!doctype html><meta charset=utf-8><style>body{font-family:sans-serif;margin:16px}.row{display:flex;"
        "gap:10px;align-items:center;margin:6px 0}.row b{width:120px}.c{width:150px;font-size:11px;color:#555}"
        ".c img{width:140px;height:70px;object-fit:contain;background:#fff;border:1px solid #ddd;display:block}"
        f"</style><h2>Logo distinctiveness {Y} (overall, CLIP): most typical vs most distinctive per sector</h2>"
        + "".join(sec_html))
    print(f"\nSaved {E / 'distinctiveness.csv'} and {E / 'distinctiveness_examples.html'}")


if __name__ == "__main__":
    main()
