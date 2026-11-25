"""
Course-toolkit analyses (CSE 519 lectures), every one for 2015, 2019 and 2022 side by side.
Exploratory: not part of the pre-registered confirmatory tests.

  A. Lookalike valuation      (L18 nearest neighbours, L13 baselines)
       predict a firm's unexplained valuation from the 10 firms with the most similar logos,
       against a sector-average baseline and a permutation null
  B. Ridge / LASSO            (L15 linear regression, L16 gradient descent and regularization)
       can any direction in the 512-d logo embedding predict valuation? cross-validated R^2 vs a
       shuffled-target null
  C. Logo similarity network  (L19 graphs and networks)
       kNN graph of logos: communities vs sectors (ARI / NMI), sector assortativity,
       PageRank "prototype" logo of each sector
  D. Clustering diagnostics   (L20 clustering)
       choosing k (inertia elbow, silhouette); hierarchical clustering of sector centroids (dendrogram)
  E. Distributions and tests  (L9 distributions, L10 significance)
       Tobin's Q raw vs log (skew, normality, QQ plot); KS test, Cohen's d and bootstrap CI for the
       distinctiveness of failed vs surviving firms
  F. Score and rank stability (L5 correlation, L8 scores and rankings)
       Spearman correlation of the same firm's distinctiveness and valuation across years,
       for firms that kept vs changed their logo
  G. Data funnel and logo map (L6 assembling data, L7 cleaning, L11-12 visualization)
       counts at every cleaning step per year; a t-SNE map of 2019 logos drawn with the logos themselves

Inputs : data/analysis/firms_master_<year>.csv, data/embed[_<year>]/, data/edgar[_<year>]/, data/logos[_<year>]/
Outputs: data/results_course/  (CSV tables, PNG figures, summary.txt)

Usage:  uv pip install networkx     (once)
        python3 analysis_course.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_main as A

warnings.filterwarnings("ignore")
YEARS = (2015, 2019, 2022)
OUT = Path("data/results_course")
SEED = 519
RNG = np.random.default_rng(SEED)


def edir(y):
    return Path("data/embed" if y == 2019 else f"data/embed_{y}")


def ydir(kind, y):
    return Path(f"data/{kind}" if y == 2019 else f"data/{kind}_{y}")


def emb(y, name):
    f = edir(y) / f"{name}.npy"
    if not f.exists():
        return None
    X = np.load(f).astype(np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def load():
    import statsmodels.formula.api as smf
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
        d["failed"] = d["outcome_reconciled"].isin(A.FAILED).astype(int)
        # valuation left unexplained by size, profitability, leverage, age, advertising and sector
        ctrl = A.CONTROLS.replace(" + C(logo_symbol_type) + log_short_side + log_sharpness + conf_high", "")
        d["q_resid"] = smf.ols(f"log_q ~ {ctrl}", data=d).fit().resid                      # sector removed too
        d["q_ctrl"] = smf.ols(f"log_q ~ {ctrl.replace(' + C(sector)', '')}", data=d).fit().resid  # sector kept
        idx = pd.read_csv(edir(y) / "index.csv")
        d["emb_row"] = d["cik"].map(dict(zip(idx["cik"].astype(int), range(len(idx)))))
        d = d[d["emb_row"].notna()].reset_index(drop=True)
        d["emb_row"] = d["emb_row"].astype(int)
        years[y] = d
    return years


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# ------------------------------------------------------------------ A. lookalike valuation
def lookalike(years, lines, k=10, perms=200):
    """Valuation net of size/profit/leverage/age/advertising (sector NOT removed), predicted by
    (1) the sector average, (2) the 10 most similar logos anywhere, (3) the 10 most similar logos
    in the same sector; and whether same-sector lookalikes add anything beyond the sector average."""
    rows = []
    for y, d in years.items():
        X = emb(y, "clip_orig")[d["emb_row"].to_numpy()]
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        nn = np.argpartition(-S, k, axis=1)[:, :k]
        q = d["q_ctrl"].to_numpy()
        sec = pd.factorize(d["sector"])[0]
        tot = pd.Series(q).groupby(sec).transform("sum").to_numpy()
        cnt = pd.Series(q).groupby(sec).transform("size").to_numpy()
        pred_sec = (tot - q) / np.maximum(cnt - 1, 1)               # leave-one-out sector average
        pred_nn = q[nn].mean(axis=1)
        pred_ws = np.full(len(d), np.nan)
        nn_ws = {}
        for s_ in np.unique(sec):
            ix = np.flatnonzero(sec == s_)
            if len(ix) <= k:
                continue
            Ss = S[np.ix_(ix, ix)]
            nns = np.argpartition(-Ss, k, axis=1)[:, :k]
            nn_ws[s_] = (ix, nns)
            pred_ws[ix] = q[ix][nns].mean(axis=1)
        ok = ~np.isnan(pred_ws)
        # beyond the sector average: deviation of the firm vs deviation of its same-sector lookalikes
        dev_q, dev_l = (q - pred_sec)[ok], (pred_ws - pred_sec)[ok]
        beyond = np.corrcoef(dev_q, dev_l)[0, 1]
        null = []
        for _ in range(perms):                                     # shuffle firms within sector
            qp = q.copy()
            for s_ in np.unique(sec):
                ix = np.flatnonzero(sec == s_)
                qp[ix] = q[RNG.permutation(ix)]
            pw = np.full(len(d), np.nan)
            for s_, (ix, nns) in nn_ws.items():
                pw[ix] = qp[ix][nns].mean(axis=1)
            tp = pd.Series(qp).groupby(sec).transform("sum").to_numpy()
            ps = (tp - qp) / np.maximum(cnt - 1, 1)
            null.append(np.corrcoef((qp - ps)[ok], (pw - ps)[ok])[0, 1])
        null = np.array(null)
        rows.append({"year": y, "n": len(d), "r2_sector_average": r2(q, pred_sec),
                     "r2_lookalikes_anywhere": r2(q, pred_nn), "r2_lookalikes_same_sector": r2(q[ok], pred_ws[ok]),
                     "corr_beyond_sector": beyond,
                     "perm_p_beyond_sector": (1 + (np.abs(null) >= abs(beyond)).sum()) / (perms + 1)})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "A_lookalike.csv", index=False)
    lines += ["", "A. LOOKALIKE VALUATION (L18 nearest neighbours, L13 baselines)",
              "   target: log Q net of size, profitability, leverage, age, advertising (sector effects kept)",
              R.round(4).to_string(index=False),
              "   corr_beyond_sector ~ 0: once you know the sector, the valuation of firms with similar-looking",
              "   logos tells you nothing more about a firm's own valuation."]
    return R


# ------------------------------------------------------------------ B. ridge / lasso
def make_lasso():
    from sklearn.linear_model import LassoCV
    try:
        return LassoCV(n_alphas=30, cv=5, random_state=SEED, max_iter=5000)       # scikit-learn < 1.7
    except TypeError:
        return LassoCV(alphas=30, cv=5, random_state=SEED, max_iter=5000)         # newer versions


def regularized(years, lines):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    rows = []
    for y, d in years.items():
        q = d["q_resid"].to_numpy()
        for name in ("clip_orig", "dino_orig"):
            X = emb(y, name)
            if X is None:
                continue
            X = X[d["emb_row"].to_numpy()]
            cv = KFold(5, shuffle=True, random_state=SEED)
            for label, mdl in (("ridge", RidgeCV(alphas=np.logspace(-2, 4, 25))),
                               ("lasso", make_lasso())):
                p = cross_val_predict(mdl, X, q, cv=cv)
                null = [r2(qs, cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 25)), X, qs, cv=cv))
                        for qs in (q[RNG.permutation(len(q))] for _ in range(5))] if label == "ridge" else [np.nan]
                fit = mdl.fit(X, q)
                rows.append({"year": y, "embedding": name, "model": label, "cv_r2": r2(q, p),
                             "null_r2_mean": float(np.nanmean(null)),
                             "alpha": float(fit.alpha_),
                             "nonzero_coefs": int((np.abs(fit.coef_) > 1e-8).sum()) if label == "lasso" else X.shape[1]})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "B_ridge_lasso.csv", index=False)
    lines += ["", "B. RIDGE / LASSO ON THE FULL EMBEDDING (L15, L16): out-of-sample R2 for residual log Q",
              R.round(4).to_string(index=False),
              "   cv_r2 <= 0 means no direction in logo space predicts valuation better than the mean;",
              "   LASSO keeping 0 coefficients means cross-validation chose to use none of the 512 dimensions."]
    return R


# ------------------------------------------------------------------ C. network
def network(years, lines, k=10):
    try:
        import networkx as nx
    except ImportError:
        lines += ["", "C. NETWORK: skipped (uv pip install networkx)"]
        return None
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rows, protos = [], []
    for y, d in years.items():
        X = emb(y, "clip_orig")[d["emb_row"].to_numpy()]
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        nn = np.argpartition(-S, k, axis=1)[:, :k]
        G = nx.Graph()
        G.add_nodes_from(range(len(d)))
        for i in range(len(d)):
            for j in nn[i]:
                G.add_edge(i, int(j), weight=float(S[i, j]))
        sec = d["sector"].to_numpy()
        nx.set_node_attributes(G, {i: sec[i] for i in range(len(d))}, "sector")
        comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
        lab = np.empty(len(d), dtype=int)
        for c, members in enumerate(comms):
            lab[list(members)] = c
        same = np.mean([sec[u] == sec[v] for u, v in G.edges()])
        counts = pd.Series(sec).value_counts(normalize=True)
        expected = float((counts ** 2).sum())
        rows.append({"year": y, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                     "communities": len(comms), "modularity": nx.community.modularity(G, comms, weight="weight"),
                     "ARI_vs_sector": adjusted_rand_score(sec, lab), "NMI_vs_sector": normalized_mutual_info_score(sec, lab),
                     "same_sector_edges": same, "expected_if_random": expected,
                     "sector_assortativity": nx.attribute_assortativity_coefficient(G, "sector")})
        pr = nx.pagerank(G, weight="weight")
        d2 = d.assign(pagerank=[pr[i] for i in range(len(d))])
        for s, g in d2.groupby("sector"):
            # prototype = the most central logo among the sector's logos within the sector's own subgraph
            sub = G.subgraph(g.index.tolist())
            if sub.number_of_edges() == 0:
                continue
            prs = nx.pagerank(sub, weight="weight")
            top = max(prs, key=prs.get)
            protos.append({"year": y, "sector": s, "prototype": d.at[top, "name_base"], "logo_path": d.at[top, "logo_path"]})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "C_network.csv", index=False)
    P = pd.DataFrame(protos)
    P.to_csv(OUT / "C_sector_prototypes.csv", index=False)
    lines += ["", f"C. LOGO SIMILARITY NETWORK (L19): {k}-nearest-neighbour graph on CLIP embeddings",
              R.round(3).to_string(index=False),
              "   prototype (PageRank-central) logo per sector, 2019 if present:",
              P[P["year"] == (2019 if 2019 in years else list(years)[0])][["sector", "prototype"]].to_string(index=False)]
    return R


# ------------------------------------------------------------------ D. clustering diagnostics
def clustering_diag(years, lines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    V = np.vstack([emb(y, "clip_masked")[d.loc[d["logo_symbol_type"] != "text_only", "emb_row"].to_numpy()]
                   for y, d in years.items()])
    sub = V[RNG.choice(len(V), size=min(4000, len(V)), replace=False)]
    ks, inert, sil = list(range(3, 15)), [], []
    for k in ks:
        km = KMeans(k, n_init=5, random_state=SEED).fit(sub)
        inert.append(km.inertia_)
        sil.append(silhouette_score(sub, km.labels_, metric="cosine", sample_size=2000, random_state=SEED))
    K = pd.DataFrame({"k": ks, "inertia": inert, "silhouette": sil})
    K.to_csv(OUT / "D_choose_k.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), dpi=150)
    ax[0].plot(ks, inert, "o-", color="#2b6cb0"); ax[0].set_title("Elbow: within-cluster inertia", fontsize=9)
    ax[1].plot(ks, sil, "o-", color="#2b6cb0"); ax[1].set_title("Silhouette (cosine)", fontsize=9)
    for a in ax:
        a.set_xlabel("k", fontsize=8); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "D_choose_k.png")

    fig, axes = plt.subplots(1, len(years), figsize=(5.2 * len(years), 5), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, (y, d) in zip(axes, years.items()):
        X = emb(y, "clip_masked")
        dd = d[d["logo_symbol_type"] != "text_only"]
        cents = dd.groupby("sector")["emb_row"].apply(lambda r: X[r.to_numpy()].mean(axis=0))
        C = np.vstack(cents.to_numpy())
        Z = linkage(C, method="average", metric="cosine")
        dendrogram(Z, labels=[s[:22] for s in cents.index], orientation="left", ax=ax, color_threshold=0,
                   above_threshold_color="#2b6cb0", leaf_font_size=7)
        ax.set_title(f"{y}: which sectors' logo designs look alike", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "D_sector_dendrogram.png")
    best = K.loc[K["silhouette"].idxmax()]
    lines += ["", "D. CLUSTERING DIAGNOSTICS (L20): choosing k on text-masked logos (4,000 sampled)",
              K.round(4).to_string(index=False),
              f"   best silhouette at k = {int(best['k'])} ({best['silhouette']:.3f}); values near 0 mean logo styles form a "
              "continuum rather than well-separated groups.", "   sector dendrograms: D_sector_dendrogram.png"]
    return K


# ------------------------------------------------------------------ E. distributions and significance
def distributions(years, lines, boots=2000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    rows, tests = [], []
    fig, axes = plt.subplots(2, len(years), figsize=(4 * len(years), 5.4), dpi=150)
    axes = np.atleast_2d(axes)
    for j, (y, d) in enumerate(years.items()):
        q = d["tobins_q_base"].clip(lower=1e-3)
        lq = np.log(q)
        lqw = d["log_q"]                                            # log of Q winsorized at 1% / 99% (the analysis variable)
        rows.append({"year": y, "n": len(q), "Q_median": q.median(), "Q_mean": q.mean(),
                     "share_Q_over_20": float((q > 20).mean()), "Q_max": q.max(),
                     "skew_Q": stats.skew(q), "skew_logQ": stats.skew(lq), "skew_logQ_winsorized": stats.skew(lqw),
                     "kurtosis_Q": stats.kurtosis(q), "kurtosis_logQ_winsorized": stats.kurtosis(lqw)})
        axes[0, j].hist(q.clip(upper=q.quantile(0.99)), bins=60, color="#9aa5b1")
        axes[0, j].set_title(f"{y}: Tobin's Q (skew {stats.skew(q):.1f})", fontsize=9)
        stats.probplot(lqw, dist="norm", plot=axes[1, j])
        axes[1, j].set_title(f"{y}: winsorized log Q vs normal (skew {stats.skew(lqw):.2f})", fontsize=9)
        for a in axes[:, j]:
            a.spines[["top", "right"]].set_visible(False)
        a_, b_ = d.loc[d["failed"] == 1, "D_overall"], d.loc[d["outcome_reconciled"] == "survived", "D_overall"]
        if len(a_) > 20:
            ks = stats.ks_2samp(a_, b_)
            pooled_sd = np.sqrt(((len(a_) - 1) * a_.var() + (len(b_) - 1) * b_.var()) / (len(a_) + len(b_) - 2))
            diffs = [RNG.choice(a_.to_numpy(), len(a_)).mean() - RNG.choice(b_.to_numpy(), len(b_)).mean()
                     for _ in range(boots)]
            tests.append({"year": y, "failed_n": len(a_), "survived_n": len(b_),
                          "mean_D_failed": a_.mean(), "mean_D_survived": b_.mean(),
                          "cohens_d": (a_.mean() - b_.mean()) / pooled_sd,
                          "boot_ci_low": np.quantile(diffs, 0.025), "boot_ci_high": np.quantile(diffs, 0.975),
                          "KS_stat": ks.statistic, "KS_p": ks.pvalue})
    fig.tight_layout(); fig.savefig(OUT / "E_q_distribution.png")
    R, T = pd.DataFrame(rows), pd.DataFrame(tests)
    R.to_csv(OUT / "E_q_distribution.csv", index=False)
    T.to_csv(OUT / "E_failed_vs_survived.csv", index=False)
    lines += ["", "E. DISTRIBUTIONS (L9) and OUTLIERS (L7): raw Q is extremely skewed (shell firms with tiny assets);",
              "   the log helps, winsorizing at 1% / 99% then makes it close to symmetric",
              R.round(3).to_string(index=False), "",
              "   SIGNIFICANCE (L10): distinctiveness of firms that later failed vs survived",
              T.round(4).to_string(index=False)]
    return T


# ------------------------------------------------------------------ F. stability
def stability(years, lines, same_cut=0.95):
    from scipy import stats
    rows = []
    yrs = sorted(years)
    for i, a in enumerate(yrs):
        for b in yrs[i + 1:]:
            da, db = years[a].set_index("cik"), years[b].set_index("cik")
            common = sorted(set(da.index) & set(db.index))
            if len(common) < 50:
                continue
            Xa, Xb = emb(a, "clip_orig"), emb(b, "clip_orig")
            sim = np.array([Xa[da.at[c, "emb_row"]] @ Xb[db.at[c, "emb_row"]] for c in common])
            kept = sim >= same_cut
            for label, sel in (("all firms", np.ones(len(common), bool)), ("kept logo", kept), ("changed logo", ~kept)):
                if sel.sum() < 30:
                    continue
                cs = np.array(common)[sel]
                rows.append({"years": f"{a}-{b}", "firms": label, "n": int(sel.sum()),
                             "spearman_D": stats.spearmanr(da.loc[cs, "D_overall"], db.loc[cs, "D_overall"])[0],
                             "spearman_logQ": stats.spearmanr(da.loc[cs, "log_q"], db.loc[cs, "log_q"])[0],
                             "rank_change_D_median_pct": float(np.median(np.abs(
                                 da.loc[cs, "D_overall"].rank(pct=True).values - db.loc[cs, "D_overall"].rank(pct=True).values)) * 100)})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "F_stability.csv", index=False)
    lines += ["", "F. SCORE AND RANK STABILITY (L5, L8): same firm, two base years",
              R.round(3).to_string(index=False),
              "   kept-logo firms should keep their distinctiveness rank (a check that the score is reliable);",
              "   changed-logo firms show how much a redesign moves a firm's position."]
    return R


# ------------------------------------------------------------------ G. funnel and logo map
def funnel_and_map(years, lines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image
    rows = []
    for y in YEARS:
        e = ydir("edgar", y)
        fa, ft = e / f"firms_{y}_all.csv", e / f"firms_{y}.csv"
        cand = ydir("logos", y) / ("candidates_v3.csv" if y == 2019 else "candidates.csv")
        mf = Path(f"data/analysis/firms_master_{y}.csv")
        if not (fa.exists() and ft.exists() and mf.exists()):
            continue
        m = pd.read_csv(mf, low_memory=False, usecols=lambda c: c in ("cik", "in_logo_sample", "in_main_sample"))
        n_img = pd.read_csv(cand, usecols=["cik", "phash"]).dropna()["cik"].nunique() if cand.exists() else np.nan
        rows.append({"year": y, "10-K filers (operating, after SIC exclusions)": len(pd.read_csv(fa, usecols=["cik"])),
                     "publicly traded": len(pd.read_csv(ft, usecols=["cik"])), "any image in filings": n_img,
                     "usable logo (high/medium, not a slide)": int(m["in_logo_sample"].sum()),
                     "main sample (logo + Tobin's Q)": int(m["in_main_sample"].sum())})
    if rows:
        F = pd.DataFrame(rows).set_index("year").T
        F.to_csv(OUT / "G_funnel.csv")
        lines += ["", "G. DATA FUNNEL (L6, L7): firms left after each assembly and cleaning step", F.to_string()]
    else:
        F = None
        lines += ["", "G. DATA FUNNEL: firm lists not found, skipped"]

    y = 2019 if 2019 in years else list(years)[0]
    d = years[y]
    from sklearn.manifold import TSNE
    X = emb(y, "clip_orig")[d["emb_row"].to_numpy()]
    xy = TSNE(2, perplexity=30, init="pca", random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
    ax.scatter(xy[:, 0], xy[:, 1], s=2, color="#c8ccd2")
    pick = RNG.choice(len(d), size=min(170, len(d)), replace=False)
    for i in pick:
        try:
            im = Image.open(d.at[i, "logo_path"]).convert("RGBA")
            im.thumbnail((110, 55))
            ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=0.5), xy[i], frameon=False))
        except Exception:
            continue
    ax.set_title(f"Logo map {y}: t-SNE of CLIP embeddings ({len(d):,} logos; 170 shown)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / f"G_logo_map_{y}.png")
    lines.append(f"   logo map: G_logo_map_{y}.png")
    return F


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    years = load()
    if not years:
        raise SystemExit("No firms_master_<year>.csv with distinctiveness scores found.")
    lines = ["Course-toolkit analyses. Years: " + ", ".join(f"{y} (n={len(d):,})" for y, d in years.items())]
    steps = [("A lookalike valuation", lookalike), ("B ridge / lasso", regularized), ("C network", network),
             ("D clustering diagnostics", clustering_diag), ("E distributions and tests", distributions),
             ("F stability", stability), ("G funnel and logo map", funnel_and_map)]
    for i, (name, fn) in enumerate(steps, 1):
        print(f"{i}/{len(steps)} {name}...", flush=True)
        try:
            fn(years, lines)
        except Exception as e:                    # one failing part never stops the others
            lines += ["", f"{name}: FAILED - {type(e).__name__}: {e}"]
            print(f"   failed: {type(e).__name__}: {e}")
    (OUT / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
