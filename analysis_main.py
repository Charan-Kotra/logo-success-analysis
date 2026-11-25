"""
Confirmatory analysis, exactly as fixed in the pre-registration (H1-H5), plus its robustness
and placebo checks. One base year per run.

Input : data/analysis/firms_master_<year>.csv (merge_all.py, after distinctiveness.py)
        data/embed[_<year>]/*.npy and index.csv (for the shuffled-peer placebo)
Output: data/results_<year>/
          confirmatory.csv     H1-H5: estimate, 95% CI, p, Holm-adjusted p, verdict
          robustness.csv       every robustness / placebo re-run of H1 (and H3)
          models.txt           full regression tables
          fig_binscatter.png   log Q (after controls) vs distinctiveness, 20 bins, with the quadratic fit
          fig_coefs.png        D_overall coefficient across all robustness runs
          summary.txt          plain-language summary

Usage:  python3 analysis_main.py              # 2019
        python3 analysis_main.py --year 2022  # confirmation sample
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

CONTROLS = ("log_assets_base + roa_w + lev_w + yl + yl_missing + ad_w + ad_reported_i + "
            "C(logo_symbol_type) + log_short_side + log_sharpness + conf_high + C(sector)")
FAILED = {"bankrupt", "distressed_delist"}


def winsor(s, lo=0.01, hi=0.99):
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def prepare(m):
    d = m[m["in_main_sample"].astype(bool) & m["D_overall"].notna()].copy()
    d = d[d["tobins_q_base"] > 0]
    d["log_q"] = np.log(winsor(d["tobins_q_base"]))
    d["roa_w"] = winsor(d["roa_base"]).fillna(0)
    d["roa_missing"] = d["roa_base"].isna().astype(int)
    lev_med = d["leverage_base"].median()
    d["lev_w"] = winsor(d["leverage_base"]).fillna(0 if pd.isna(lev_med) else lev_med)
    d["yl"] = d["years_listed_base"].fillna(0)
    d["yl_missing"] = d["years_listed_base"].isna().astype(int)
    d["ad_w"] = winsor(d["ad_intensity_base"]).fillna(0)
    d["ad_reported_i"] = d["ad_reported"].astype(bool).astype(int)
    d["log_short_side"] = np.log(d["logo_short_side"].clip(lower=1))
    d["log_sharpness"] = np.log1p(d["logo_sharpness"].clip(lower=0))
    d["conf_high"] = (d["logo_confidence"] == "high").astype(int)
    la_med = d["log_assets_base"].median()
    d["log_assets_base"] = d["log_assets_base"].fillna(0 if pd.isna(la_med) else la_med)
    for c in ("roa_w", "lev_w", "ad_w", "log_short_side", "log_sharpness"):
        d[c] = d[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"  sample after preparing controls: {len(d)} firms")
    d["sector"] = d["sector"].fillna("Other")
    d["D2"] = d["D_overall"] ** 2
    return d.reset_index(drop=True)


def ols(formula, d):
    return smf.ols(formula, data=d).fit(cov_type="HC3")


def wild_cluster_p(d, formula, var, cluster="sector", reps=999, seed=519):
    """Wild cluster bootstrap-t (Rademacher, null imposed) for one coefficient."""
    rng = np.random.default_rng(seed)
    full = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    t_obs = full.tvalues[var]
    restricted = smf.ols(formula.replace(f"{var} + ", "").replace(f" + {var}", ""), data=d).fit()
    y_hat, e = restricted.fittedvalues.to_numpy(), restricted.resid.to_numpy()
    yname = formula.split("~")[0].strip()
    g = pd.factorize(d[cluster])[0]
    G = g.max() + 1
    X = full.model.exog
    t_b = []
    for _ in range(reps):
        w = rng.choice([-1.0, 1.0], size=G)[g]
        y_star = y_hat + e * w
        b = sm.OLS(y_star, X).fit(cov_type="cluster", cov_kwds={"groups": g})
        t_b.append(b.tvalues[full.model.exog_names.index(var)])
    t_b = np.abs(np.array(t_b))
    return float((1 + (t_b >= abs(t_obs)).sum()) / (reps + 1))


def row(name, res, var, n=None, note=""):
    b, se = res.params[var], res.bse[var]
    lo, hi = res.conf_int().loc[var]
    return {"test": name, "variable": var, "estimate": b, "se": se, "ci_low": lo, "ci_high": hi,
            "p": res.pvalues[var], "n": int(n or res.nobs), "note": note}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def peer_D(X, groups, k=10):
    """Within-group kNN distinctiveness, z-scored within group (same as distinctiveness.py)."""
    n = len(X)
    d = np.full(n, np.nan)
    for gname in np.unique(groups):
        idx = np.flatnonzero(groups == gname)
        if len(idx) < 3:
            continue
        S = X[idx] @ X[idx].T
        np.fill_diagonal(S, -np.inf)
        kk = min(k, len(idx) - 1)
        d[idx] = 1 - (-np.sort(-S, axis=1))[:, :kk].mean(axis=1)
    s = pd.Series(d)
    gs = pd.Series(groups)
    return ((s - s.groupby(gs).transform("mean")) / s.groupby(gs).transform("std")).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--reps", type=int, default=999, help="wild cluster bootstrap repetitions")
    ap.add_argument("--placebo_reps", type=int, default=200)
    args = ap.parse_args()
    Y = args.year
    out = Path(f"data/results_{Y}")
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(f"data/analysis/firms_master_{Y}.csv", low_memory=False)
    d = prepare(m)
    text = [f"Base year {Y}: main sample n = {len(d)}"]
    conf = []

    # ---------------- H1: linear
    f1 = f"log_q ~ D_overall + {CONTROLS}"
    r1 = ols(f1, d)
    p_wild = wild_cluster_p(d, f1, "D_overall", reps=args.reps)
    conf.append(row("H1 distinctiveness -> valuation", r1, "D_overall",
                    note=f"wild cluster bootstrap p (sector) = {p_wild:.3f}"))
    text += ["", "=== H1 ===", r1.summary().tables[1].as_text()[:4000]]

    # ---------------- H2: sweet spot
    f2 = f"log_q ~ D_overall + D2 + {CONTROLS}"
    r2 = ols(f2, d)
    b1, b2 = r2.params["D_overall"], r2.params["D2"]
    peak = -b1 / (2 * b2) if b2 != 0 else np.nan
    p5, p95 = d["D_overall"].quantile([0.05, 0.95])
    xc = peak if (b2 < 0 and p5 <= peak <= p95) else d["D_overall"].median()
    d["x_lo"] = np.where(d["D_overall"] <= xc, d["D_overall"] - xc, 0.0)
    d["x_hi"] = np.where(d["D_overall"] > xc, d["D_overall"] - xc, 0.0)
    d["hi"] = (d["D_overall"] > xc).astype(int)
    r2b = ols(f"log_q ~ x_lo + x_hi + hi + {CONTROLS}", d)
    two_lines = (r2b.pvalues["x_lo"] < 0.05 and r2b.pvalues["x_hi"] < 0.05
                 and np.sign(r2b.params["x_lo"]) != np.sign(r2b.params["x_hi"]))
    h2_ok_shape = (b2 < 0) and (p5 <= peak <= p95) and two_lines
    conf.append(row("H2 sweet spot (squared term)", r2, "D2",
                    note=f"peak at D = {peak:.2f} (5th-95th pct {p5:.2f} to {p95:.2f}); two-lines: "
                         f"slope below {r2b.params['x_lo']:+.3f} (p {r2b.pvalues['x_lo']:.3f}), "
                         f"above {r2b.params['x_hi']:+.3f} (p {r2b.pvalues['x_hi']:.3f}); full shape criteria met: {h2_ok_shape}"))
    text += ["", "=== H2 ===", r2.summary().tables[1].as_text()[:1500], "", r2b.summary().tables[1].as_text()[:1200]]

    # ---------------- H3: design vs wording
    dv = d[d["D_visual"].notna()].copy()
    r3 = ols(f"log_q ~ D_visual + D_overall + {CONTROLS}", dv)
    comp = {}
    for c in ("clip_masked_knn10_z", "dino_masked_knn10_z"):
        rc = ols(f"log_q ~ {c} + D_overall + {CONTROLS}", dv)
        comp[c] = (rc.params[c], rc.pvalues[c])
    same_sign = np.sign(comp["clip_masked_knn10_z"][0]) == np.sign(comp["dino_masked_knn10_z"][0])
    conf.append(row("H3 design (D_visual) beyond wording", r3, "D_visual", n=len(dv),
                    note="components: " + ", ".join(f"{k.split('_')[0]} {v[0]:+.3f} (p {v[1]:.3f})" for k, v in comp.items())
                         + f"; same sign: {same_sign}"))
    text += ["", "=== H3 ===", r3.summary().tables[1].as_text()[:1500]]

    # ---------------- H4: norm moderation
    d["Dxnorm"] = d["D_overall"] * d["sector_norm_strength"]
    r4 = ols(f"log_q ~ D_overall + Dxnorm + {CONTROLS}", d)
    grows = np.sign(r4.params["Dxnorm"]) == np.sign(r1.params["D_overall"])
    conf.append(row("H4 D x sector norm strength", r4, "Dxnorm",
                    note=f"interaction has the same sign as the main effect (effect grows with norm): {grows}"))
    text += ["", "=== H4 ===", r4.summary().tables[1].as_text()[:1500]]

    # ---------------- H5: exit by 2024 (multinomial logit)
    e = d[~d["outcome_reconciled"].eq("other_exit")].copy()
    e["y"] = np.select([e["outcome_reconciled"].isin(FAILED), e["outcome_reconciled"].eq("acquired")], [2, 1], 0)
    # sector dummies only for sectors with failures and acquisitions present (avoids perfect separation)
    ctr5 = CONTROLS.replace(" + C(sector)", "")
    sect = pd.get_dummies(e["sector"], prefix="s", drop_first=True, dtype=float)
    sect = sect.loc[:, sect.sum() >= 15]
    Xf = smf.ols(f"log_q ~ D_overall + {ctr5}", data=e).fit().model.exog
    names = smf.ols(f"log_q ~ D_overall + {ctr5}", data=e).fit().model.exog_names
    X5 = pd.concat([pd.DataFrame(Xf, columns=names, index=e.index), sect], axis=1)
    X5 = X5.loc[:, X5.std() > 0].assign(Intercept=1.0)
    r5 = sm.MNLogit(e["y"], X5).fit(disp=0, maxiter=300, method="bfgs")
    pr = r5.params
    se5 = r5.bse
    j = 1  # column for y=2 (failed) vs 0
    b, s = pr.loc["D_overall"].iloc[j], se5.loc["D_overall"].iloc[j]
    from scipy import stats
    p5v = 2 * (1 - stats.norm.cdf(abs(b / s)))
    conf.append({"test": "H5 distinctiveness -> failure by 2024 (vs survived)", "variable": "D_overall",
                 "estimate": b, "se": s, "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s, "p": p5v, "n": len(e),
                 "note": f"failed {int((e['y'] == 2).sum())}, acquired {int((e['y'] == 1).sum())}, survived "
                         f"{int((e['y'] == 0).sum())}; acquired-vs-survived coef {pr.loc['D_overall'].iloc[0]:+.3f} "
                         f"(p {2 * (1 - stats.norm.cdf(abs(pr.loc['D_overall'].iloc[0] / se5.loc['D_overall'].iloc[0]))):.3f}); "
                         "log-odds per 1 SD"})
    text += ["", "=== H5 (multinomial logit; columns: acquired, failed) ===",
             pd.concat([pr.loc[["D_overall"]], se5.loc[["D_overall"]]]).to_string()]

    CT = pd.DataFrame(conf)
    CT["p_holm"] = holm(CT["p"])
    CT["verdict"] = np.where(CT["p_holm"] < 0.05, "supported (pending 2022 confirmation)", "not supported")
    CT.loc[CT["test"].str.startswith("H2") & ~pd.Series([h2_ok_shape] * len(CT)), "verdict"] = \
        np.where(CT.loc[CT["test"].str.startswith("H2"), "p_holm"] < 0.05,
                 "squared term significant but shape criteria not met", "not supported")
    CT.to_csv(out / "confirmatory.csv", index=False)

    # ---------------- robustness (H1 re-runs)
    rob = []

    def rerun(label, data, var="D_overall", formula=None):
        try:
            r = ols(formula or f"log_q ~ {var} + {CONTROLS}", data)
            rob.append(row(label, r, var))
        except Exception as ex:
            rob.append({"test": label, "variable": var, "note": f"failed: {ex}"})

    rerun("main (k=10)", d)
    for v in ("clip_orig_knn5_z", "clip_orig_knn20_z", "clip_orig_centroid_z", "dino_orig_knn10_z"):
        if v in d:
            rerun(f"measure: {v}", d, var=v)
    for v in ("clip_masked_knn10_z", "dino_masked_knn10_z", "D_visual"):
        rerun(f"visual only: {v}", dv, var=v)
    fin = d["sector"].isin(["Banks & financial services", "Insurance"])
    rerun("excluding banks and insurers", d[~fin])
    rerun("high-confidence logos only", d[d["conf_high"] == 1])
    if "logo_source" in d:
        rerun("EDGAR logos only", d[d["logo_source"].fillna("edgar") == "edgar"])
    rerun("US-headquartered only", d[d["us_firm"].astype(bool)])
    dq = d[d["q_source"] == "compustat"]
    rerun("Tobin's Q from Compustat only", dq)
    # placebo 1: pre-period returns
    if "bhar_pre" in d:
        dp = d[d["bhar_pre"].notna()].copy()
        dp["bhar_pre_w"] = winsor(dp["bhar_pre"])
        rerun("PLACEBO: returns in the 2 years before (should be ~0)", dp,
              formula=f"bhar_pre_w ~ D_overall + {CONTROLS}")
    # secondary outcomes (not confirmatory)
    for yv, lab in [("dlog_q_p2", "secondary: change in log Q, 2 years"),
                    ("bhar_p2", "secondary: 2-year return minus market")]:
        if yv in d and d[yv].notna().sum() > 200:
            ds = d[d[yv].notna()].copy()
            ds[yv + "_w"] = winsor(ds[yv])
            rerun(lab, ds, formula=f"{yv}_w ~ D_overall + {CONTROLS}")

    # placebo 2: distinctiveness measured against a random peer group
    E = Path("data/embed" if Y == 2019 else f"data/embed_{Y}")
    idx = pd.read_csv(E / "index.csv")
    X = np.load(E / "clip_orig.npy").astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pos = idx.reset_index().set_index("cik")["index"]
    sel = d["cik"].map(pos).to_numpy()
    Xs = X[sel]
    groups = d["sector"].to_numpy()
    rng = np.random.default_rng(519)
    real_b = r1.params["D_overall"]
    fake = []
    for _ in range(args.placebo_reps):
        dd = d.copy()
        dd["D_fake"] = peer_D(Xs, rng.permutation(groups))
        fake.append(ols(f"log_q ~ D_fake + {CONTROLS}", dd).params["D_fake"])
    fake = np.array(fake)
    p_fake = (1 + (np.abs(fake) >= abs(real_b)).sum()) / (len(fake) + 1)
    rob.append({"test": "PLACEBO: D vs a random peer group (real coef vs shuffled)", "variable": "D_fake",
                "estimate": float(np.mean(fake)), "ci_low": float(np.quantile(fake, 0.025)),
                "ci_high": float(np.quantile(fake, 0.975)), "p": p_fake, "n": len(d),
                "note": f"real coefficient {real_b:+.4f}; share of shuffled peer sets at least as large: p = {p_fake:.3f}"})
    R = pd.DataFrame(rob)
    R.to_csv(out / "robustness.csv", index=False)

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    resid_y = smf.ols(f"log_q ~ {CONTROLS}", d).fit().resid + d["log_q"].mean()
    resid_x = smf.ols(f"D_overall ~ {CONTROLS}", d).fit().resid + d["D_overall"].mean()
    bins = pd.qcut(resid_x, 20, labels=False, duplicates="drop")
    bx = resid_x.groupby(bins).mean()
    by = resid_y.groupby(bins).mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.scatter(bx, by, s=28, color="#2b6cb0", zorder=3, label="20 equal-size bins")
    xs = np.linspace(bx.min(), bx.max(), 100)
    q = np.polyfit(resid_x, resid_y, 2)
    ax.plot(xs, np.polyval(q, xs), color="#c05621", lw=2, label="quadratic fit (all firms)")
    ax.set_xlabel("Logo distinctiveness within sector (D_overall, after controls)")
    ax.set_ylabel("log Tobin's Q (after controls)")
    ax.set_title(f"Distinctiveness and firm valuation, {Y} (n = {len(d)})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_binscatter.png")

    Rp = R[R["estimate"].notna() & ~R["test"].str.startswith(("PLACEBO", "secondary"))].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(Rp) + 1.0), dpi=150)
    ax.errorbar(Rp["estimate"], range(len(Rp)), xerr=[Rp["estimate"] - Rp["ci_low"], Rp["ci_high"] - Rp["estimate"]],
                fmt="o", color="#2b6cb0", ecolor="#90a4c0", capsize=2)
    ax.axvline(0, color="#888", lw=1)
    ax.set_yticks(range(len(Rp)))
    ax.set_yticklabels(Rp["test"], fontsize=7)
    ax.set_xlabel("change in log Tobin's Q per 1 SD (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_coefs.png")

    # ---------------- summary
    pd.set_option("display.width", 200)
    mde = 2.8 * r1.bse["D_overall"]
    lines = [f"Base year {Y}: main sample {len(d)} firms (D_visual available for {len(dv)})", "",
             "CONFIRMATORY TESTS (Holm-corrected across H1-H5):",
             CT[["test", "estimate", "ci_low", "ci_high", "p", "p_holm", "verdict"]].round(4).to_string(index=False), "",
             "Notes:"] + [f"  {t}: {n}" for t, n in zip(CT["test"], CT["note"])] + [
             "", f"H1 effect size: 1 SD more distinctive logo -> {100 * (np.exp(r1.params['D_overall']) - 1):+.1f}% Tobin's Q "
             f"(95% CI {100 * (np.exp(r1.conf_int().loc['D_overall', 0]) - 1):+.1f}% to "
             f"{100 * (np.exp(r1.conf_int().loc['D_overall', 1]) - 1):+.1f}%)",
             f"Smallest effect this sample could detect (80% power, 5%): about {100 * mde:.1f}% of Q per 1 SD", "",
             "ROBUSTNESS AND PLACEBO (H1 re-runs):",
             R[["test", "estimate", "ci_low", "ci_high", "p", "n"]].round(4).to_string(index=False)]
    (out / "summary.txt").write_text("\n".join(lines))
    (out / "models.txt").write_text("\n".join(text))
    print("\n".join(lines))
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
