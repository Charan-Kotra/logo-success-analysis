"""
Pooled analysis across base years (2015, 2019, 2022) and the rebrand (within-firm) analysis.

Part 1  Pooled tests (pre-registered: same models, year fixed effects, standard errors clustered
        by firm, outcomes over the same 2-year window for every year)
          H1  log Tobin's Q ~ D_overall
          H2  + D_overall^2
          H3  D_visual beyond D_overall
          H4  D_overall x sector norm strength
          H5  failed within 2 years (bankrupt or delisted for performance) ~ D_overall  (linear probability)
        plus each year's H1 estimate side by side and a test of whether they differ.

Part 2  Rebrands (exploratory): firms with a usable logo in two base years
          logo similarity between the two years (CLIP and DINOv2, original logos)
          same logo / refreshed / rebranded
          within-firm change: change in log Q ~ change in distinctiveness (+ change in controls);
          comparing a firm with itself removes everything about it that does not change
          do rebrands move toward the sector norm (less distinctive) or away from it?

Inputs : data/analysis/firms_master_<year>.csv and data/embed[_<year>]/ for each year present
Outputs: data/results_pooled/
           pooled_confirmatory.csv, by_year.csv, rebrand_pairs.csv, rebrand_results.csv,
           fig_by_year.png, summary.txt

Usage:  python3 analysis_pooled.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import analysis_main as A

warnings.filterwarnings("ignore")
YEARS = (2015, 2019, 2022)
OUT = Path("data/results_pooled")
SAME, REFRESH = 0.95, 0.85          # CLIP cosine similarity cut-offs between a firm's two logos


def load_year(y):
    f = Path(f"data/analysis/firms_master_{y}.csv")
    if not f.exists():
        return None
    m = pd.read_csv(f, low_memory=False)
    if "D_overall" not in m:
        print(f"  {y}: no distinctiveness scores yet (run distinctiveness.py, then merge_all.py) - skipped")
        return None
    d = A.prepare(m)                      # winsorizing and controls within the year
    d["base_year"] = y
    if "listed_p2" in d:
        ok = d["outcome_reconciled"].isin(A.FAILED) & (d["last_10k_date"].astype(str) < f"{y + 3}-01-01")
        d["failed_p2"] = ok.astype(int)
    return d


def clustered(formula, d):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["cik"])[0]})


def row(name, r, var, n=None, note=""):
    lo, hi = r.conf_int().loc[var]
    return {"test": name, "variable": var, "estimate": r.params[var], "se": r.bse[var], "ci_low": lo,
            "ci_high": hi, "p": r.pvalues[var], "n": int(n or r.nobs), "note": note}


def embeddings(y, model="clip_orig"):
    E = Path("data/embed" if y == 2019 else f"data/embed_{y}")
    if not (E / f"{model}.npy").exists():
        return None
    idx = pd.read_csv(E / "index.csv")
    X = np.load(E / f"{model}.npy").astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return dict(zip(idx["cik"].astype(int), X))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {y: load_year(y) for y in YEARS}
    parts = {y: d for y, d in parts.items() if d is not None}
    if len(parts) < 2:
        raise SystemExit("Need at least two base years with distinctiveness scores.")
    P = pd.concat(parts.values(), ignore_index=True)
    P["D2"] = P["D_overall"] ** 2
    P["Dxnorm"] = P["D_overall"] * P["sector_norm_strength"]
    ctrl = A.CONTROLS + " + C(base_year)"
    lines = [f"Pooled sample: {len(P):,} firm-years from {P['cik'].nunique():,} firms; years "
             + ", ".join(f"{y}: {len(d):,}" for y, d in parts.items())]

    # ---------------- Part 1: pooled confirmatory
    res = []
    r1 = clustered(f"log_q ~ D_overall + {ctrl}", P)
    res.append(row("H1 pooled: distinctiveness -> valuation", r1, "D_overall"))
    r2 = clustered(f"log_q ~ D_overall + D2 + {ctrl}", P)
    b1, b2 = r2.params["D_overall"], r2.params["D2"]
    res.append(row("H2 pooled: squared term", r2, "D2",
                   note=f"peak at D = {(-b1 / (2 * b2)) if b2 else float('nan'):.2f}"))
    pv = P[P["D_visual"].notna()]
    r3 = clustered(f"log_q ~ D_visual + D_overall + {ctrl}", pv)
    res.append(row("H3 pooled: design (D_visual) beyond wording", r3, "D_visual", n=len(pv)))
    r4 = clustered(f"log_q ~ D_overall + Dxnorm + {ctrl}", P)
    res.append(row("H4 pooled: D x sector norm strength", r4, "Dxnorm"))
    if "failed_p2" in P:
        r5 = clustered(f"failed_p2 ~ D_overall + {ctrl}", P)
        res.append(row("H5 pooled: failed within 2 years (linear probability)", r5, "D_overall",
                       note=f"failure rate {P['failed_p2'].mean():.1%}"))
    R = pd.DataFrame(res)
    R["p_holm"] = A.holm(R["p"])
    R["verdict"] = np.where(R["p_holm"] < 0.05, "supported", "not supported")
    R.to_csv(OUT / "pooled_confirmatory.csv", index=False)

    # each year side by side + heterogeneity
    by = []
    for y, d in parts.items():
        r = A.ols(f"log_q ~ D_overall + {A.CONTROLS}", d)
        by.append({"year": str(y), **{k: v for k, v in row(f"H1 {y}", r, "D_overall").items() if k != "test"}})
    by.append({"year": "pooled", **{k: v for k, v in res[0].items() if k != "test"}})
    B = pd.DataFrame(by)
    B.to_csv(OUT / "by_year.csv", index=False)
    rh = clustered(f"log_q ~ D_overall * C(base_year) + {A.CONTROLS}", P)
    inter = [c for c in rh.params.index if c.startswith("D_overall:")]
    het_p = float(rh.f_test(", ".join(f"{c} = 0" for c in inter)).pvalue) if inter else float("nan")

    # ---------------- Part 2: rebrands
    emb = {y: embeddings(y) for y in parts}
    emb_d = {y: embeddings(y, "dino_orig") for y in parts}
    pairs = []
    yrs = sorted(parts)
    for a, b in zip(yrs[:-1], yrs[1:]):
        da = parts[a].set_index("cik")
        db = parts[b].set_index("cik")
        common = sorted(set(da.index) & set(db.index))
        for cik in common:
            ea, eb = emb[a].get(cik), emb[b].get(cik)
            if ea is None or eb is None:
                continue
            fa, fb = emb_d[a].get(cik), emb_d[b].get(cik)
            pairs.append({
                "cik": cik, "from": a, "to": b, "name": db.at[cik, "name_base"], "sector": db.at[cik, "sector"],
                "clip_sim": float(ea @ eb), "dino_sim": float(fa @ fb) if fa is not None and fb is not None else np.nan,
                "D_from": da.at[cik, "D_overall"], "D_to": db.at[cik, "D_overall"],
                "logq_from": da.at[cik, "log_q"], "logq_to": db.at[cik, "log_q"],
                "assets_from": da.at[cik, "log_assets_base"], "assets_to": db.at[cik, "log_assets_base"],
                "roa_from": da.at[cik, "roa_w"], "roa_to": db.at[cik, "roa_w"],
                "lev_from": da.at[cik, "lev_w"], "lev_to": db.at[cik, "lev_w"],
                "logo_from": da.at[cik, "logo_path"], "logo_to": db.at[cik, "logo_path"]})
    Q = pd.DataFrame(pairs)
    rb_lines = []
    if len(Q):
        Q["change"] = np.select([Q["clip_sim"] >= SAME, Q["clip_sim"] >= REFRESH], ["same", "refreshed"], "rebranded")
        Q["dD"] = Q["D_to"] - Q["D_from"]
        Q["dlogq"] = Q["logq_to"] - Q["logq_from"]
        for c in ("assets", "roa", "lev"):
            Q[f"d_{c}"] = Q[f"{c}_to"] - Q[f"{c}_from"]
        Q["pair"] = Q["from"].astype(str) + "-" + Q["to"].astype(str)
        Q.to_csv(OUT / "rebrand_pairs.csv", index=False)
        rr = []
        f_w = "dlogq ~ dD + d_assets + d_roa + d_lev + C(pair) + C(sector)"
        rw = clustered(f_w, Q)
        rr.append(row("within-firm: change in log Q ~ change in distinctiveness (all pairs)", rw, "dD"))
        ch = Q[Q["change"] != "same"]
        if len(ch) > 50:
            rc = clustered(f_w, ch)
            rr.append(row("within-firm, logo changed only", rc, "dD"))
        Q["rebranded"] = (Q["change"] == "rebranded").astype(int)
        rr2 = clustered("dlogq ~ rebranded + d_assets + d_roa + d_lev + C(pair) + C(sector)", Q)
        rr.append(row("rebranded vs kept logo: change in log Q", rr2, "rebranded"))
        RR = pd.DataFrame(rr)
        RR.to_csv(OUT / "rebrand_results.csv", index=False)
        reb = Q[Q["change"] == "rebranded"]
        toward = (reb["dD"] < 0).mean() if len(reb) else np.nan
        from scipy import stats
        t_p = stats.ttest_1samp(reb["dD"], 0).pvalue if len(reb) > 10 else np.nan
        rb_lines = ["", "REBRANDS (exploratory):",
                    f"firms with a usable logo in two consecutive base years: {len(Q):,} pairs",
                    Q.groupby("pair")["change"].value_counts().unstack(fill_value=0).to_string(),
                    f"rebrands that became LESS distinctive (closer to the sector norm): {toward:.0%} "
                    f"(mean change in D {reb['dD'].mean():+.2f}, t-test p = {t_p:.3f})" if len(reb) else "",
                    RR[["test", "estimate", "ci_low", "ci_high", "p", "n"]].round(4).to_string(index=False)]
        # unique logo designs: same firm + similarity >= SAME counts once
        same_pairs = int((Q["change"] == "same").sum())
        rb_lines.insert(2, f"unique logo designs: about {len(P) - same_pairs:,} "
                           f"({len(P):,} firm-year logos minus {same_pairs:,} repeats of the same logo)")
        # page to eyeball the cut-offs
        import html
        rows = []
        for r in Q.sort_values("clip_sim").groupby("change").head(25).itertuples():
            rows.append(f"<tr><td>{html.escape(str(r.name))}<br>{r.pair} · {r.change} · sim {r.clip_sim:.2f}</td>"
                        f"<td><img src='../../{r.logo_from}'></td><td><img src='../../{r.logo_to}'></td></tr>")
        (OUT / "rebrand_review.html").write_text(
            "<style>body{font-family:sans-serif}td{padding:6px;border-bottom:1px solid #ddd}img{max-width:220px;"
            "max-height:90px}</style><table>" + "".join(rows) + "</table>")

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=150)
    yy = list(range(len(B)))[::-1]
    ax.errorbar(B["estimate"], yy, xerr=[B["estimate"] - B["ci_low"], B["ci_high"] - B["estimate"]],
                fmt="o", color="#2b6cb0", ecolor="#90a4c0", capsize=3)
    ax.axvline(0, color="#888", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{y} (n={int(n):,})" for y, n in zip(B["year"], B["n"])], fontsize=8)
    ax.set_xlabel("change in log Tobin's Q per 1 SD of distinctiveness (95% CI)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_by_year.png")

    lines += ["", "POOLED CONFIRMATORY TESTS (year fixed effects, SE clustered by firm, Holm across H1-H5):",
              R[["test", "estimate", "ci_low", "ci_high", "p", "p_holm", "verdict", "n"]].round(4).to_string(index=False),
              "", "H1 BY YEAR:", B[["year", "estimate", "ci_low", "ci_high", "p", "n"]].round(4).to_string(index=False),
              f"do the yearly H1 estimates differ? F-test p = {het_p:.3f}",
              f"pooled effect size: {100 * (np.exp(r1.params['D_overall']) - 1):+.1f}% Q per 1 SD "
              f"(95% CI {100 * (np.exp(r1.conf_int().loc['D_overall', 0]) - 1):+.1f}% to "
              f"{100 * (np.exp(r1.conf_int().loc['D_overall', 1]) - 1):+.1f}%)"] + rb_lines
    (OUT / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
