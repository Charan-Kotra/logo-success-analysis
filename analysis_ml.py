"""
Exploratory machine-learning and data-science analyses, every one run for 2015, 2019 and 2022
side by side (and pooled where it makes sense). Labelled exploratory: they were not part of the
pre-registered confirmatory tests.

  1. Sector classifier     can a model tell a firm's industry from its logo alone?
                           logistic regression on CLIP / DINOv2 embeddings, original vs text-masked,
                           5-fold cross-validated balanced accuracy vs chance, per year
  2. Design features       colour count, colourfulness, main hue, complexity (edge density), symmetry,
                           aspect ratio, text share, logo type, computed from every logo image;
                           trends 2015 -> 2019 -> 2022 (are logos getting simpler / more alike?),
                           both across all firms and within the same firms
  3. Gradient boosting     do logo features add predictive power for valuation (R^2) and failure (AUC)
                           beyond the financial controls? grouped cross-validation by firm,
                           permutation importance of the logo feature group
  4. Survival analysis     Kaplan-Meier curves by distinctiveness quartile and Cox proportional hazards
                           for time to failure (acquisition = censored), per base year
  5. Clustering            k-means "style families" on CLIP embeddings of all three years;
                           family shares by year and by sector, a 2-D PCA map

Inputs : data/analysis/firms_master_<year>.csv, data/embed[_<year>]/*, the logo images
Outputs: data/results_ml/  (CSV tables, PNG figures, clusters.html, summary.txt)

Usage:  pip install lifelines    (once)
        python3 analysis_ml.py
"""
import html as htmlmod
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import analysis_main as A

warnings.filterwarnings("ignore")
YEARS = (2015, 2019, 2022)
OUT = Path("data/results_ml")
LAST = pd.Timestamp("2024-12-31")
SEED = 519


# ------------------------------------------------------------------ loading
def edir(y):
    return Path("data/embed" if y == 2019 else f"data/embed_{y}")


def load():
    years = {}
    for y in YEARS:
        f = Path(f"data/analysis/firms_master_{y}.csv")
        if not f.exists():
            continue
        m = pd.read_csv(f, low_memory=False)
        if "D_overall" not in m:
            continue
        d = A.prepare(m)
        d["base_year"] = y
        d["failed_p2"] = (d["outcome_reconciled"].isin(A.FAILED) &
                          (d["last_10k_date"].astype(str) < f"{y + 3}-01-01")).astype(int)
        idx = pd.read_csv(edir(y) / "index.csv")
        pos = dict(zip(idx["cik"].astype(int), range(len(idx))))
        d["emb_row"] = d["cik"].map(pos)
        years[y] = d
    return years


def emb(y, name):
    f = edir(y) / f"{name}.npy"
    if not f.exists():
        return None
    X = np.load(f).astype(np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ------------------------------------------------------------------ 1. sector classifier
def sector_classifier(years, lines):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    rows, confusions = [], {}
    for y in years:
        idx = pd.read_csv(edir(y) / "index.csv")
        sec = idx["sector"].fillna("Other")
        keep_s = sec.map(sec.value_counts()) >= 20
        for name in ("clip_orig", "clip_masked", "dino_orig", "dino_masked"):
            X = emb(y, name)
            if X is None:
                continue
            k = keep_s.to_numpy().copy()
            if name.endswith("masked"):
                k &= (idx["symbol_type"] != "text_only").to_numpy()
            Xk, yk = X[k], sec[k].to_numpy()
            clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
            pred = cross_val_predict(clf, Xk, yk, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
            acc = balanced_accuracy_score(yk, pred)
            rows.append({"year": y, "embedding": name, "n": int(k.sum()), "n_sectors": len(set(yk)),
                         "balanced_accuracy": acc, "chance": 1 / len(set(yk)), "lift": acc * len(set(yk))})
            if name == "clip_orig":
                labs = sorted(set(yk))
                cm = pd.DataFrame(confusion_matrix(yk, pred, labels=labs, normalize="true"), index=labs, columns=labs)
                confusions[y] = cm
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "1_sector_classifier.csv", index=False)
    piv = R.pivot(index="embedding", columns="year", values="balanced_accuracy")
    lines += ["", "1. SECTOR FROM LOGO (5-fold balanced accuracy; chance = 1 / number of sectors):",
              piv.round(3).to_string(), f"chance: {R['chance'].iloc[0]:.3f}"]
    for y, cm in confusions.items():
        diag = pd.Series(np.diag(cm), index=cm.index).sort_values(ascending=False)
        cm.to_csv(OUT / f"1_confusion_clip_orig_{y}.csv")
        lines.append(f"  {y} best-recognised sectors: " + ", ".join(f"{s} {v:.0%}" for s, v in diag.head(3).items())
                     + " | worst: " + ", ".join(f"{s} {v:.0%}" for s, v in diag.tail(3).items()))
    return R


# ------------------------------------------------------------------ 2. design features
HUES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]


def design_features(path):
    try:
        im = Image.open(path)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        im.thumbnail((256, 256))
        a = np.asarray(im).astype(np.int16)
    except Exception:
        return None
    h, w = a.shape[:2]
    ring = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(ring, axis=0)
    ink = np.abs(a - bg).sum(axis=2) > 60
    n_ink = int(ink.sum())
    if n_ink < 30:
        return None
    px = a[ink].astype(np.float64)
    q = (px // 32).astype(int)
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    _, cnt = np.unique(keys, return_counts=True)
    n_colors = int((cnt >= 0.05 * n_ink).sum())   # colour bins holding >= 5% of the ink (ignores anti-aliasing)
    rg = px[:, 0] - px[:, 1]
    yb = 0.5 * (px[:, 0] + px[:, 1]) - px[:, 2]
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    hsv = np.asarray(Image.fromarray(a.astype(np.uint8)).convert("HSV")).astype(np.float64)[ink]
    sat = hsv[:, 1] / 255
    chroma = sat > 0.25
    if chroma.mean() < 0.15:
        hue = "black/grey"
    else:
        hdeg = hsv[chroma, 0] / 255 * 360
        bins = np.array([15, 45, 70, 160, 200, 260, 300, 340])
        lab = np.digitize(hdeg, bins) % 8
        hue = HUES[np.bincount(lab, minlength=8).argmax()]
    gray = a.mean(axis=2)
    gx, gy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    edges = (gx[:-1, :] > 40) | (gy[:, :-1] > 40)
    complexity = float(edges.sum() / n_ink)
    sym = float((ink & ink[:, ::-1]).sum() / max(1, (ink | ink[:, ::-1]).sum()))
    return {"n_colors": n_colors, "colorfulness": colorful, "main_hue": hue, "saturation": float(sat.mean()),
            "complexity": complexity, "symmetry": sym, "aspect": w / h, "ink_share": n_ink / (h * w)}


def design_trends(years, lines):
    from scipy import stats
    feats = []
    for y, d in years.items():
        for r in d[["cik", "logo_path", "logo_symbol_type", "logo_text_share", "sector"]].itertuples(index=False):
            f = design_features(r.logo_path) if isinstance(r.logo_path, str) else None
            if f:
                feats.append({"cik": r.cik, "base_year": y, "sector": r.sector, "symbol_type": r.logo_symbol_type,
                              "text_share": r.logo_text_share, **f})
    F = pd.DataFrame(feats)
    F.to_csv(OUT / "2_design_features.csv", index=False)
    num = ["n_colors", "colorfulness", "saturation", "complexity", "symmetry", "aspect", "ink_share", "text_share"]
    F["text_only"] = (F["symbol_type"] == "text_only").astype(int)
    tab = F.groupby("base_year")[num + ["text_only"]].mean().T
    # same firms in every year present: separates real redesign from a changing mix of firms
    counts = F.groupby("cik")["base_year"].nunique()
    panel = F[F["cik"].isin(counts[counts == F["base_year"].nunique()].index)]
    tab_same = panel.groupby("base_year")[num + ["text_only"]].mean().T
    tests = {}
    yrs = sorted(F["base_year"].unique())
    for c in num + ["text_only"]:
        a, b = F.loc[F["base_year"] == yrs[0], c].dropna(), F.loc[F["base_year"] == yrs[-1], c].dropna()
        tests[c] = stats.mannwhitneyu(a, b).pvalue if len(a) > 10 and len(b) > 10 else np.nan
    tab[f"p ({yrs[0]} vs {yrs[-1]})"] = pd.Series(tests)
    tab.to_csv(OUT / "2_design_trends_all_firms.csv")
    tab_same.to_csv(OUT / "2_design_trends_same_firms.csv")
    hue = pd.crosstab(F["main_hue"], F["base_year"], normalize="columns")
    hue.to_csv(OUT / "2_main_hue_by_year.csv")
    sector_prof = F.groupby("sector")[["n_colors", "colorfulness", "complexity", "symmetry", "text_only"]].mean()
    sector_prof.to_csv(OUT / "2_design_by_sector.csv")
    lines += ["", f"2. DESIGN FEATURES BY YEAR (all firms; n = {F.groupby('base_year').size().to_dict()}):",
              tab.round(3).to_string(), "",
              f"   same {panel['cik'].nunique():,} firms present in every year:", tab_same.round(3).to_string(), "",
              "   main hue share by year:", hue.round(3).to_string(), "",
              "   design by sector (all years):", sector_prof.round(2).sort_values("colorfulness").to_string()]
    return F


# ------------------------------------------------------------------ 3. gradient boosting
def boosting(years, F, lines):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    fin = ["log_assets_base", "roa_w", "lev_w", "yl", "yl_missing", "ad_w", "ad_reported_i"]
    rows, imps = [], []
    sets = {y: d for y, d in years.items()}
    sets["pooled"] = pd.concat(years.values(), ignore_index=True)
    for label, d in sets.items():
        d = d.copy()
        d = d.merge(F.drop(columns=["sector", "symbol_type", "text_share"]), on=["cik", "base_year"], how="left")
        # 10 principal components of each firm's CLIP logo embedding (fit within each year)
        pcs = np.full((len(d), 10), np.nan)
        for y in d["base_year"].unique():
            X = emb(y, "clip_orig")
            sel = (d["base_year"] == y).to_numpy() & d["emb_row"].notna().to_numpy()
            if X is None or sel.sum() < 50:
                continue
            Z = X[d.loc[sel, "emb_row"].astype(int).to_numpy()]
            pcs[sel] = PCA(10, random_state=SEED).fit_transform(Z)
        for j in range(10):
            d[f"pc{j + 1}"] = pcs[:, j]
        d["sector_code"] = pd.factorize(d["sector"])[0]
        d["year_code"] = d["base_year"]
        d["D_visual_f"] = d["D_visual"].fillna(0)
        d["D_visual_missing"] = d["D_visual"].isna().astype(int)
        d["type_code"] = pd.factorize(d["logo_symbol_type"])[0]
        logo = ["D_overall", "D_visual_f", "D_visual_missing", "type_code", "n_colors", "colorfulness", "saturation",
                "complexity", "symmetry", "aspect", "ink_share"] + [f"pc{j + 1}" for j in range(10)]
        base = fin + ["sector_code"] + (["year_code"] if label == "pooled" else [])
        groups = d["cik"].to_numpy()
        for target, kind in (("log_q", "reg"), ("failed_p2", "clf")):
            if kind == "clf" and d[target].sum() < 40:
                continue
            yv = d[target].to_numpy()
            scores = {"financial only": [], "financial + logo": []}
            imp_logo = []
            for rep in range(3):
                perm = np.random.default_rng(SEED + rep).permutation(len(d))
                for tr, te in GroupKFold(5).split(perm, groups=groups[perm]):
                    tr, te = perm[tr], perm[te]
                    for name, cols in (("financial only", base), ("financial + logo", base + logo)):
                        mdl = (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                             l2_regularization=1.0, random_state=SEED)
                               if kind == "reg" else
                               HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                              l2_regularization=1.0, class_weight="balanced",
                                                              random_state=SEED))
                        mdl.fit(d.iloc[tr][cols], yv[tr])
                        if kind == "reg":
                            s = r2_score(yv[te], mdl.predict(d.iloc[te][cols]))
                        else:
                            s = roc_auc_score(yv[te], mdl.predict_proba(d.iloc[te][cols])[:, 1])
                        scores[name].append(s)
                        if name == "financial + logo" and rep == 0:
                            pi = permutation_importance(mdl, d.iloc[te][cols], yv[te], n_repeats=3,
                                                        random_state=SEED,
                                                        scoring="r2" if kind == "reg" else "roc_auc")
                            imp = pd.Series(pi.importances_mean, index=cols)
                            imp_logo.append(imp)
            a, b = np.array(scores["financial only"]), np.array(scores["financial + logo"])
            diff = b - a
            rows.append({"sample": str(label), "target": target, "metric": "R2" if kind == "reg" else "AUC",
                         "financial_only": a.mean(), "with_logo": b.mean(), "gain": diff.mean(),
                         "gain_sd": diff.std(ddof=1), "folds_with_gain": f"{(diff > 0).sum()}/{len(diff)}",
                         "n": len(d)})
            imp = pd.concat(imp_logo, axis=1).mean(axis=1)
            imps.append(pd.DataFrame({"sample": str(label), "target": target, "feature": imp.index,
                                      "importance": imp.values,
                                      "group": ["logo" if c in logo else "financial/sector" for c in imp.index]}))
    R = pd.DataFrame(rows)
    I = pd.concat(imps, ignore_index=True)
    R.to_csv(OUT / "3_boosting.csv", index=False)
    I.to_csv(OUT / "3_permutation_importance.csv", index=False)
    grp = I.groupby(["sample", "target", "group"])["importance"].sum().unstack()
    lines += ["", "3. GRADIENT BOOSTING: does adding logo features improve prediction? (5-fold grouped CV x 3)",
              R.round(4).to_string(index=False), "", "   summed permutation importance by feature group:",
              grp.round(4).to_string(), "", "   top 8 features (pooled, log Q):",
              I[(I["sample"] == "pooled") & (I["target"] == "log_q")].nlargest(8, "importance")[
                  ["feature", "group", "importance"]].round(4).to_string(index=False)]
    return R


# ------------------------------------------------------------------ 4. survival
def survival(years, lines):
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
    except ImportError:
        lines += ["", "4. SURVIVAL: skipped (pip install lifelines)"]
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    fig, axes = plt.subplots(1, len(years), figsize=(4.2 * len(years), 3.4), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (y, d) in zip(axes, years.items()):
        d = d.copy()
        start = pd.Timestamp(f"{y}-12-31")
        failed = d["outcome_reconciled"].isin(A.FAILED)
        exited = ~d["outcome_reconciled"].eq("survived")
        delist = pd.to_datetime(d.get("crsp_delist_date"), errors="coerce")
        last10k = pd.to_datetime(d["last_10k_date"], errors="coerce") + pd.Timedelta(days=365)
        end = delist.fillna(last10k).where(exited, LAST).clip(upper=LAST)
        d["T"] = ((end - start).dt.days / 365.25).clip(lower=0.05)
        d["E"] = failed.astype(int)
        d = d[d["T"].notna()]
        d["D_quartile"] = pd.qcut(d["D_overall"], 4, labels=["Q1 most typical", "Q2", "Q3", "Q4 most distinctive"])
        km = KaplanMeierFitter()
        for i, (q, g) in enumerate(d.groupby("D_quartile")):
            km.fit(g["T"], g["E"], label=str(q))
            km.plot_survival_function(ax=ax, ci_show=False, color=["#2b6cb0", "#7aa6d8", "#c7a26a", "#c05621"][i], lw=1.6)
        lr = multivariate_logrank_test(d["T"], d["D_quartile"], d["E"])
        ax.set_title(f"{y}: {int(d['E'].sum())} failures, log-rank p = {lr.p_value:.2f}", fontsize=9)
        ax.set_xlabel("years after base year", fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
        cox = d[["T", "E", "D_overall", "log_assets_base", "roa_w", "lev_w", "yl", "yl_missing", "ad_w", "sector"]].dropna()
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(cox, "T", "E", strata=["sector"])
        s = cph.summary.loc["D_overall"]
        rows.append({"year": y, "n": len(cox), "failures": int(cox["E"].sum()), "hazard_ratio_per_SD": s["exp(coef)"],
                     "ci_low": s["exp(coef) lower 95%"], "ci_high": s["exp(coef) upper 95%"], "p": s["p"],
                     "logrank_p_quartiles": lr.p_value, "max_follow_up_years": round(float(d["T"].max()), 1)})
    axes[0].set_ylabel("share not failed", fontsize=8)
    fig.suptitle("Time to failure by logo distinctiveness quartile (Kaplan-Meier)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "4_kaplan_meier.png")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "4_cox.csv", index=False)
    lines += ["", "4. SURVIVAL (Cox, stratified by sector, acquisition censored): hazard ratio per 1 SD of D_overall",
              R.round(3).to_string(index=False)]
    return R


# ------------------------------------------------------------------ 5. clustering
def clustering(years, F, lines, k=8):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = []
    for y, d in years.items():
        X = emb(y, "clip_masked")
        dd = d[d["emb_row"].notna() & (d["logo_symbol_type"] != "text_only")].copy()
        if X is None or len(dd) < 50:
            continue
        dd["vec"] = list(X[dd["emb_row"].astype(int).to_numpy()])
        parts.append(dd[["cik", "base_year", "sector", "name_base", "logo_path", "D_overall", "log_q", "failed_p2", "vec"]])
    P = pd.concat(parts, ignore_index=True)
    V = np.vstack(P["vec"].to_numpy())
    km = KMeans(k, n_init=10, random_state=SEED).fit(V)
    P["cluster"] = km.labels_
    P = P.drop(columns="vec").merge(F[["cik", "base_year", "n_colors", "colorfulness", "complexity", "symmetry",
                                       "aspect", "main_hue"]], on=["cik", "base_year"], how="left")
    prof = P.groupby("cluster").agg(n=("cik", "size"), n_colors=("n_colors", "mean"),
                                    colorfulness=("colorfulness", "mean"), complexity=("complexity", "mean"),
                                    symmetry=("symmetry", "mean"), aspect=("aspect", "median"),
                                    main_hue=("main_hue", lambda s: s.mode().iat[0] if len(s.dropna()) else ""),
                                    top_sector=("sector", lambda s: s.mode().iat[0]),
                                    mean_log_q=("log_q", "mean"), failed_2y=("failed_p2", "mean"))
    share = pd.crosstab(P["cluster"], P["base_year"], normalize="columns")
    from scipy.stats import chi2_contingency
    chi_p = chi2_contingency(pd.crosstab(P["cluster"], P["base_year"]))[1]
    prof.to_csv(OUT / "5_cluster_profiles.csv")
    share.to_csv(OUT / "5_cluster_share_by_year.csv")
    pd.crosstab(P["sector"], P["cluster"], normalize="index").to_csv(OUT / "5_cluster_by_sector.csv")
    xy = PCA(2, random_state=SEED).fit_transform(V)
    fig, ax = plt.subplots(figsize=(6, 4.6), dpi=150)
    cmap = plt.get_cmap("tab10")
    for c in range(k):
        s = P["cluster"].to_numpy() == c
        ax.scatter(xy[s, 0], xy[s, 1], s=3, alpha=0.4, color=cmap(c), label=f"family {c}")
    ax.set_title("Logo style families (k-means on text-masked CLIP embeddings, PCA map)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=3, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "5_cluster_map.png")
    cells = []
    for c in range(k):
        g = P[P["cluster"] == c]
        cen = km.cluster_centers_[c]
        near = g.assign(dist=[float(np.linalg.norm(v - cen)) for v in V[P["cluster"].to_numpy() == c]]).nsmallest(12, "dist")
        imgs = "".join(f"<img src='../../{p}' title='{htmlmod.escape(str(n))}'>" for p, n in zip(near["logo_path"], near["name_base"]))
        cells.append(f"<h3>Family {c} (n={len(g)}; top sector {htmlmod.escape(str(prof.at[c, 'top_sector']))})</h3><div>{imgs}</div>")
    (OUT / "5_clusters.html").write_text("<style>body{font-family:sans-serif}img{width:120px;height:60px;object-fit:"
                                         "contain;border:1px solid #ddd;margin:3px;background:#fff}</style>" + "".join(cells))
    lines += ["", f"5. STYLE FAMILIES (k-means, k={k}, logos with a symbol, all years):", prof.round(3).to_string(), "",
              "   share of each family by year:", share.round(3).to_string(),
              f"   do the family shares change across years? chi-square p = {chi_p:.3f}"]
    return P


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    years = load()
    if not years:
        raise SystemExit("No firms_master_<year>.csv with distinctiveness scores found.")
    lines = [f"Exploratory ML analyses. Years: " + ", ".join(f"{y} (n={len(d):,})" for y, d in years.items())]
    print("1/5 sector classifier...", flush=True)
    sector_classifier(years, lines)
    print("2/5 design features (reads every logo image)...", flush=True)
    F = design_trends(years, lines)
    print("3/5 gradient boosting...", flush=True)
    boosting(years, F, lines)
    print("4/5 survival...", flush=True)
    survival(years, lines)
    print("5/5 clustering...", flush=True)
    clustering(years, F, lines)
    (OUT / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
