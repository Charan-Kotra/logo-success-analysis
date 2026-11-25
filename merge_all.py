"""
Build the master analysis file for one base year: one row per firm, all sources merged.
Column names do not depend on the year (suffix _base = base year, _p1/_p2/_p3 = 1/2/3 years
later, _prev = the year before), so the 2015, 2019 and 2022 files stack into one pooled file.

Sources (Y = base year)
  EDGAR      data/edgar[_Y]/firms_Y.csv, financials_wide.csv      (the firm list = the base)
  Logos      data/logos[_Y]/logo_choice.csv
  Masking    data/embed[_Y]/index.csv                              (if logo_embed.py has run)
  Wikidata   every data/wikidata*/firm_links.csv                   (domain, founding date)
  Compustat  data/wrds/company.csv.gz, funda.csv.gz                (matched on CIK)
  CRSP       data/wrds/ccm_link, msf, msedelist, msenames          (gvkey -> permno)
  ([_Y] is empty for 2019: data/edgar, data/logos, data/embed)

Outputs (data/analysis/)
  firms_master_Y.csv     one row per firm
  merge_report_Y.txt     coverage of every block, overall and by outcome; EDGAR vs CRSP outcomes
  data_dictionary.csv    every column, its source and meaning
  firms_pooled.csv       with --pool: all firms_master_*.csv stacked (one row per firm-year)

Run:
  python3 merge_all.py                 # 2019
  python3 merge_all.py --year 2022
  python3 merge_all.py --pool          # stack the years already built
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
OUT = D / "analysis"
LAST = 2024
DICT = []


def note(cols, source, meaning):
    for c in ([cols] if isinstance(cols, str) else cols):
        DICT.append((c, source, meaning))


def clean_domain(s):
    s = s.astype("string").str.lower().str.strip()
    s = s.str.replace(r"^https?://", "", regex=True).str.replace(r"^www\d?\.", "", regex=True)
    return s.str.split("/").str[0].str.split("?").str[0].replace("", pd.NA)


def ydir(kind, Y):
    return D / (kind if Y == 2019 else f"{kind}_{Y}")


def build(Y):
    base_date = pd.Timestamp(f"{Y}-12-31")

    # ---------------------------------------------------------- 1. EDGAR base
    firms = pd.read_csv(ydir("edgar", Y) / f"firms_{Y}.csv")
    firms["cik"] = firms["cik"].astype(int)
    firms = firms.rename(columns={f"name_{Y}": "name_base", f"filed_10k_{Y}": "filed_10k_base",
                                  f"public_float_{Y}": "public_float_base"})
    keep = ["cik", "name_base", "name", "sic", "sic_description", "sector", "tickers", "exchanges",
            "state_of_incorporation", "fiscal_year_end", "outcome_2024", "last_10k_date",
            "outcome_evidence", "public_float_base", "website_edgar"]
    m = firms[[c for c in keep if c in firms.columns]].copy()
    m.insert(1, "base_year", Y)
    note(["cik", "base_year", "name_base", "name", "sic", "sic_description", "sector", "tickers", "exchanges",
          "state_of_incorporation", "fiscal_year_end", "public_float_base", "website_edgar"],
         "EDGAR", "identity; name_base = name on the base-year 10-K, name = current SEC name")
    note(["outcome_2024", "last_10k_date", "outcome_evidence"], "EDGAR",
         "outcome by 2024: survived / acquired / bankrupt / other_exit, and the filing evidence")

    fw = pd.read_csv(ydir("edgar", Y) / "financials_wide.csv")
    fin_cols = ["assets", "equity", "liabilities", "revenue", "net_income", "sga", "advertising",
                "public_float", "tobins_q", "sga_to_revenue"]
    for yr, tag in [(Y - 1, "prev"), (Y, "base")]:
        part = fw[fw["year"] == yr].drop_duplicates("cik")[["cik"] + [c for c in fin_cols if c in fw]]
        m = m.merge(part.rename(columns={c: f"e_{c}_{tag}" for c in fin_cols}), on="cik", how="left")
    note([c for c in m.columns if c.startswith("e_")], "EDGAR XBRL",
         "reported financials (USD); tobins_q = (assets - equity + public float) / assets")

    # ---------------------------------------------------------- 2. logos
    alt = ydir("logos", Y) / "logo_choice_all.csv"          # EDGAR + Wayback, if wayback_logos.py ran
    base_choice = ydir("logos", Y) / "logo_choice.csv"
    if alt.exists():
        lc = pd.read_csv(alt)
    elif base_choice.exists():
        lc = pd.read_csv(base_choice)
    else:                                                  # logos not scored yet: build everything else
        print(f"  note: no logos for {Y} yet ({base_choice}); logo columns left empty, rerun after logo_score.py")
        lc = pd.DataFrame({"cik": m["cik"], "confidence": "none", "logo_in_slide": False})
    lc["cik"] = lc["cik"].astype(int)
    if "logo_source" not in lc:
        lc["logo_source"] = "edgar"
    lcols = ["logo_source", "confidence", "n_filings", "name_match", "logo_prob", "quality", "short_side",
             "sharpness", "logo_in_slide", "window", "form", "logo_path", "ocr_text"]
    for c in lcols:
        if c not in lc:
            lc[c] = np.nan
    lg = lc[["cik"] + lcols].rename(
        columns={c: (c if c.startswith("logo_") else f"logo_{c}") for c in lcols})
    m = m.merge(lg, on="cik", how="left")
    slide = m["logo_in_slide"].astype(str).str.lower().eq("true")
    m["in_logo_sample"] = m["logo_confidence"].isin(["high", "medium"]) & ~slide
    note([c for c in m.columns if c.startswith("logo_")], "EDGAR / Wayback logo extraction",
         "chosen base-year logo, its source (edgar = SEC filings, wayback = archived company homepage) "
         "and its confidence / quality scores")
    note("in_logo_sample", "derived", "high/medium confidence and not a slide (the analysis sample)")

    idx = ydir("embed", Y) / "index.csv"
    if idx.exists():
        ei = pd.read_csv(idx)[["cik", "symbol_type", "n_boxes", "text_share", "ink_left", "masked_path"]]
        ei = ei.rename(columns={"symbol_type": "logo_symbol_type", "n_boxes": "logo_text_boxes",
                                "text_share": "logo_text_share", "ink_left": "logo_ink_left",
                                "masked_path": "logo_masked_path"})
        m = m.merge(ei, on="cik", how="left")
        note(["logo_symbol_type", "logo_text_boxes", "logo_text_share", "logo_ink_left", "logo_masked_path"],
             "logo_embed.py", "text masking: text_only / symbol_and_text / symbol_only")

    dfile = ydir("embed", Y) / "distinctiveness.csv"
    if dfile.exists():
        dcols = ["D_visual", "D_overall", "D_visual_dino_orig", "sector_norm_strength", "peer_group", "peer_group_n"]
        dd = pd.read_csv(dfile)
        extra = [c for c in dd.columns if c.endswith("_z") or c.startswith(("clip_", "dino_"))]
        m = m.merge(dd[["cik"] + [c for c in dcols + extra if c in dd]], on="cik", how="left")
        note([c for c in dcols if c in dd], "distinctiveness.py",
             "D_visual = design distinctiveness (masked logos, DINOv2 + CLIP), D_overall = design + wording "
             "(original logo, CLIP); within-sector z-scores, larger = more distinctive")
        note(extra, "distinctiveness.py", "component scores: <model>_<orig|masked>_<knn5|knn10|knn20|centroid>[_z]")

    m = m.copy()

    # ---------------------------------------------------------- 3. Wikidata (firm-level facts, any year's file)
    wfiles = sorted(glob.glob(str(D / "wikidata*" / "firm_links.csv")))
    if wfiles:
        wd = pd.concat([pd.read_csv(f) for f in wfiles], ignore_index=True)
        wd["cik"] = wd["cik"].astype(int)
        wd = wd.sort_values("qid", na_position="last").drop_duplicates("cik")
        wd = wd[["cik", "qid", "domain", "inception", "dissolved"]].rename(
            columns={"qid": "wd_qid", "domain": "wd_domain", "inception": "wd_inception", "dissolved": "wd_dissolved"})
        m = m.merge(wd, on="cik", how="left")
        note(["wd_qid", "wd_domain", "wd_inception", "wd_dissolved"], "Wikidata", "linked item, website, dates")

    # ---------------------------------------------------------- 4. Compustat company
    comp = pd.read_csv(D / "wrds/company.csv.gz", dtype=str)
    comp["cik_i"] = pd.to_numeric(comp["cik"], errors="coerce")
    comp = comp.dropna(subset=["cik_i"]).drop_duplicates("cik_i")
    comp["cik_i"] = comp["cik_i"].astype(int)
    comp = comp.rename(columns={"conm": "c_name", "conml": "c_legal_name", "weburl": "c_weburl", "sic": "c_sic",
                                "naics": "c_naics", "ipodate": "c_ipodate", "loc": "c_hq_country",
                                "fic": "c_inc_country", "state": "c_hq_state"})
    m = m.merge(comp.drop(columns="cik"), left_on="cik", right_on="cik_i", how="left").drop(columns="cik_i")
    note(["gvkey", "c_name", "c_legal_name", "c_weburl", "c_sic", "c_naics", "c_ipodate", "c_hq_country",
          "c_inc_country", "c_hq_state"], "Compustat company", "identity, current website, HQ / incorporation country")

    m["domain"] = clean_domain(m["c_weburl"])
    m["domain_source"] = np.where(m["domain"].notna(), "compustat", None)
    for col, src in [("wd_domain", "wikidata"), ("website_edgar", "edgar")]:
        if col in m:
            alt = clean_domain(m[col])
            fill = m["domain"].isna() & alt.notna()
            m.loc[fill, "domain"] = alt[fill]
            m.loc[fill, "domain_source"] = src
    m["us_firm"] = m["c_hq_country"].eq("USA") | (m["c_hq_country"].isna() & m["state_of_incorporation"].notna())
    note(["domain", "domain_source"], "derived", "website domain (Compustat > Wikidata > EDGAR); the CURRENT site")
    note("us_firm", "derived", "HQ in the US per Compustat (fallback: has a US state of incorporation)")

    # ---------------------------------------------------------- 5. Compustat fundamentals
    fa = pd.read_csv(D / "wrds/funda.csv.gz", dtype={"gvkey": str})
    fa["gvkey"] = fa["gvkey"].str.zfill(6)
    fa = fa.sort_values("datadate").drop_duplicates(["gvkey", "fyear"], keep="last")
    if fa["fyear"].min() > Y - 1:
        print(f"  WARNING: funda starts in {fa['fyear'].min()}; rerun wrds_pull.py for base year {Y}")
    fa["mv"] = fa["csho"] * fa["prcc_f"]
    fa["q"] = ((fa["at"] - fa["ceq"] + fa["mv"]) / fa["at"]).where(fa["at"] > 0)
    rel = {Y - 1: "prev", Y: "base", Y + 1: "p1", Y + 2: "p2", Y + 3: "p3"}
    f2 = fa[fa["fyear"].isin(rel)].assign(t=lambda d: d["fyear"].map(rel))
    piv = f2.pivot(index="gvkey", columns="t", values=["at", "ceq", "mv", "q", "sale", "ni", "xad", "lt"])
    piv.columns = [f"c_{v}_{t}" for v, t in piv.columns]
    m = m.merge(piv, left_on="gvkey", right_index=True, how="left")
    for c in [f"c_{v}_{t}" for v in ["at", "ceq", "mv", "q", "sale", "ni", "xad", "lt"] for t in rel.values()]:
        if c not in m:
            m[c] = np.nan
    m = m.merge(fa.groupby("gvkey")["fyear"].max().rename("c_last_fyear"), left_on="gvkey",
                right_index=True, how="left")
    note([c for c in m.columns if c.startswith(("c_at_", "c_ceq_", "c_mv_", "c_q_", "c_sale_", "c_ni_",
                                                "c_xad_", "c_lt_"))],
         "Compustat funda", "USD millions; c_q = (at - ceq + csho*prcc_f)/at; c_xad = advertising; "
                            "_prev/_base/_p1/_p2/_p3 = fiscal year Y-1 .. Y+3")
    note("c_last_fyear", "Compustat funda", "last fiscal year with a Compustat record")

    m["log_assets_base"] = np.log(m["c_at_base"].where(m["c_at_base"] > 0)).fillna(
        np.log(m["e_assets_base"].where(m["e_assets_base"] > 0) / 1e6))
    m["tobins_q_base"] = m["c_q_base"].fillna(m["e_tobins_q_base"])
    m["q_source"] = np.where(m["c_q_base"].notna(), "compustat", np.where(m["e_tobins_q_base"].notna(), "edgar", None))
    m["log_q_base"] = np.log(m["tobins_q_base"].where(m["tobins_q_base"] > 0))
    lo, hi = m["tobins_q_base"].quantile([0.01, 0.99])
    m["q_base_w"] = m["tobins_q_base"].clip(lo, hi)
    # Compustat first; the same figures from the firm's own SEC filings (XBRL) when Compustat is missing
    e_at = m["e_assets_base"].where(m["e_assets_base"] > 0)
    e_rev = m["e_revenue_base"].where(m["e_revenue_base"] > 0)
    c_ad = (m["c_xad_base"] / m["c_sale_base"]).where(m["c_sale_base"] > 0)
    e_ad = m["e_advertising_base"] / e_rev
    m["ad_intensity_base"] = c_ad.fillna(e_ad)
    m["ad_reported"] = m["c_xad_base"].notna() | m["e_advertising_base"].notna()
    m["roa_base"] = ((m["c_ni_base"] / m["c_at_base"]).where(m["c_at_base"] > 0)
                     .fillna(m["e_net_income_base"] / e_at))
    e_liab = m["e_liabilities_base"].fillna(m["e_assets_base"] - m["e_equity_base"])
    m["leverage_base"] = ((m["c_lt_base"] / m["c_at_base"]).where(m["c_at_base"] > 0)
                          .fillna(e_liab / e_at))
    m["controls_source"] = np.where(m["c_at_base"].notna(), "compustat",
                                    np.where(m["e_assets_base"].notna(), "edgar", None))
    for t in ("p1", "p2", "p3"):
        m[f"dlog_q_{t}"] = np.log(m[f"c_q_{t}"].where(m[f"c_q_{t}"] > 0)) - np.log(m["c_q_base"].where(m["c_q_base"] > 0))
        m[f"sales_growth_{t}"] = np.log(m[f"c_sale_{t}"].where(m[f"c_sale_{t}"] > 0) /
                                        m["c_sale_base"].where(m["c_sale_base"] > 0))
    ipo_year = pd.to_datetime(m["c_ipodate"], errors="coerce").dt.year
    inc_year = (pd.to_datetime(m["wd_inception"].astype("string").str[:10], errors="coerce").dt.year
                if "wd_inception" in m else pd.Series(np.nan, index=m.index))
    m["firm_age_base"] = (Y - inc_year).where(inc_year.between(1700, Y))
    m["years_listed_base"] = (Y - ipo_year).where(ipo_year <= Y)
    note("controls_source", "derived", "where ROA, leverage and advertising came from: compustat or edgar (SEC XBRL)")
    note(["log_assets_base", "tobins_q_base", "q_source", "log_q_base", "q_base_w", "ad_reported",
          "ad_intensity_base", "roa_base", "leverage_base", "firm_age_base", "years_listed_base"],
         "derived", "base-year controls; Compustat first, EDGAR fallback; q_base_w winsorized 1%/99% within the year")
    note([f"{v}_{t}" for v in ("dlog_q", "sales_growth") for t in ("p1", "p2", "p3")], "derived",
         "change from the base year to 1/2/3 years later (log); empty when that year is after 2024")

    # fixed-horizon survival from EDGAR filing history (same length for every base year)
    for k in (2, 5):
        if Y + k <= LAST:
            m[f"listed_p{k}"] = m["last_10k_date"].astype(str) >= f"{Y + k}-01-01"
    note([c for c in m.columns if c.startswith("listed_p")], "derived",
         "still filing 10-Ks k years after the base year (listed_p2 exists for every base year)")

    # ---------------------------------------------------------- 6. CRSP
    link = pd.read_csv(D / "wrds/ccm_link.csv.gz", dtype={"gvkey": str})
    link["gvkey"] = link["gvkey"].str.zfill(6)
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
    valid = link[(link["linkdt"] <= base_date) & (link["linkenddt"] >= pd.Timestamp(f"{Y - 1}-01-01"))]
    valid = valid.assign(prim=valid["linkprim"].eq("P")).sort_values(["gvkey", "prim", "linkenddt"])
    m = m.merge(valid.drop_duplicates("gvkey", keep="last")[["gvkey", "permno"]], on="gvkey", how="left")
    m["permno"] = m["permno"].astype("Int64")

    names = pd.read_csv(D / "wrds/msenames.csv.gz", parse_dates=["namedt", "nameendt"])
    nm = names[(names["namedt"] <= base_date) & (names["nameendt"] >= pd.Timestamp(f"{Y}-01-01"))]
    nm = nm.sort_values("nameendt").drop_duplicates("permno", keep="last")[["permno", "shrcd", "exchcd", "ticker"]]
    m = m.merge(nm.rename(columns={"shrcd": "crsp_shrcd", "exchcd": "crsp_exchcd", "ticker": "crsp_ticker"}),
                on="permno", how="left")
    m["crsp_common_stock"] = m["crsp_shrcd"].isin([10, 11])
    note(["permno", "crsp_shrcd", "crsp_exchcd", "crsp_ticker", "crsp_common_stock"], "CRSP",
         "stock id and listing in the base year; common stock = share code 10/11")

    msf = pd.read_csv(D / "wrds/msf.csv.gz", parse_dates=["date"])
    msf["ym"] = msf["date"].dt.to_period("M")
    if msf["ym"].min() > pd.Period(f"{Y - 2}-01", "M"):
        print(f"  WARNING: CRSP msf starts {msf['ym'].min()}; the pre-period placebo for {Y} will be short")
    dl = pd.read_csv(D / "wrds/msedelist.csv.gz", parse_dates=["dlstdt"])
    dl = dl.sort_values("dlstdt").drop_duplicates("permno", keep="last")
    dl["ym"] = dl["dlstdt"].dt.to_period("M")

    r = msf[["permno", "ym", "ret"]].copy()
    r["ret"] = pd.to_numeric(r["ret"], errors="coerce")
    dlr = dl[["permno", "ym", "dlret", "dlstcd"]].copy()
    c_ = dlr["dlstcd"]
    perf = c_.isin([500, 520, 574, 580, 584]) | c_.between(551, 573)   # Shumway (1997) performance delists
    dlr["dlret"] = dlr["dlret"].where(dlr["dlret"].notna() | ~perf, -0.30)
    r = r.merge(dlr[["permno", "ym", "dlret"]], on=["permno", "ym"], how="outer")
    r["ret_total"] = (1 + r["ret"].fillna(0)) * (1 + r["dlret"].fillna(0)) - 1
    r.loc[r["ret"].isna() & r["dlret"].isna(), "ret_total"] = np.nan
    mkt = r.groupby("ym")["ret_total"].mean().rename("mkt")

    def bhar(start, end, label):
        w = r[(r["ym"] >= pd.Period(start, "M")) & (r["ym"] <= pd.Period(end, "M"))].dropna(subset=["ret_total"])
        w = w.merge(mkt, left_on="ym", right_index=True)
        w["g1"], w["gm"] = 1 + w["ret_total"], 1 + w["mkt"]
        g = w.groupby("permno")
        return pd.DataFrame({f"bhr_{label}": g["g1"].prod() - 1,
                             f"bhar_{label}": g["g1"].prod() - g["gm"].prod(),
                             f"months_{label}": g.size()})

    windows = {"post": (f"{Y + 1}-01", f"{LAST}-12"), "p1": (f"{Y + 1}-01", f"{Y + 1}-12"),
               "p2": (f"{Y + 1}-01", f"{Y + 2}-12"), "pre": (f"{Y - 2}-01", f"{Y}-12")}
    for label, (a, b) in windows.items():
        if int(b[:4]) <= LAST:
            m = m.merge(bhar(a, b, label), left_on="permno", right_index=True, how="left")
    dec = msf[msf["ym"] == pd.Period(f"{Y}-12", "M")].drop_duplicates("permno")
    dec = dec.assign(mcap_base=dec["prc"].abs() * dec["shrout"] / 1000)[["permno", "mcap_base"]]
    m = m.merge(dec, on="permno", how="left")
    note([c for c in m.columns if c.startswith(("bhr_", "bhar_", "months_"))], "CRSP",
         "buy-and-hold return incl. delisting returns (bhar = minus equal-weighted market); "
         "post = Y+1..2024, p1 = Y+1, p2 = Y+1..Y+2, pre = Y-2..Y (placebo: before/at the logo year)")
    note("mcap_base", "CRSP", "market cap at the end of the base year, USD millions")

    dl2 = dl[(dl["dlstdt"] > base_date) & (dl["dlstcd"] != 100)][["permno", "dlstdt", "dlstcd"]]
    m = m.merge(dl2.rename(columns={"dlstdt": "crsp_delist_date", "dlstcd": "crsp_dlstcd"}), on="permno", how="left")
    code = m["crsp_dlstcd"]
    m = m.copy()
    m["crsp_exit"] = np.select(
        [m["permno"].isna(), code.isna(), code.between(200, 299), code.between(300, 399),
         code.between(400, 499), code.between(501, 505), code.eq(574), code.between(500, 599)],
        ["no_crsp", "still_listed", "merger", "share_exchange", "liquidated", "moved_exchange",
         "bankruptcy", "dropped_performance"], default="other")
    ed = m["outcome_2024"]
    m["outcome_reconciled"] = np.select(
        [ed.eq("bankrupt") | m["crsp_exit"].eq("bankruptcy"),
         ed.eq("acquired") | (ed.eq("survived") & m["crsp_exit"].eq("merger")),
         ed.eq("survived") & m["crsp_exit"].isin(["dropped_performance", "liquidated"])],
        ["bankrupt", "acquired", "distressed_delist"], default=ed)
    m["outcome_conflict"] = m["outcome_reconciled"] != ed
    note(["crsp_delist_date", "crsp_dlstcd", "crsp_exit"], "CRSP msedelist",
         "delisting after the base year: 2xx merger, 3xx share exchange, 4xx liquidation, 501-505 moved exchange, "
         "574 bankruptcy, other 5xx dropped for performance")
    note(["outcome_reconciled", "outcome_conflict"], "derived",
         "EDGAR outcome corrected with CRSP (bankruptcy or merger in either source wins); conflict = changed")

    # ---------------------------------------------------------- 7. flags, save, report
    m["has_compustat"] = m["gvkey"].notna()
    m["has_crsp_returns"] = m["months_p1"].notna() if "months_p1" in m else False
    m["in_main_sample"] = m["in_logo_sample"] & m["tobins_q_base"].notna()
    note(["has_compustat", "has_crsp_returns", "in_main_sample"], "derived",
         "coverage flags; in_main_sample = usable logo + base-year Tobin's Q")

    m = m.sort_values("cik").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / f"firms_master_{Y}.csv", index=False)

    blocks = {"logo (analysis sample)": m["in_logo_sample"], "Compustat match": m["has_compustat"],
              "website domain": m["domain"].notna(), "Tobin's Q base year": m["tobins_q_base"].notna(),
              "Q two years later": m["c_q_p2"].notna(), "advertising reported": m["ad_reported"],
              "years listed (IPO date)": m["years_listed_base"].notna(),
              "CRSP return, 1 year": m["has_crsp_returns"],
              "main sample (logo + Q)": m["in_main_sample"]}
    cov = pd.DataFrame({k: v.groupby(m["outcome_2024"]).mean() for k, v in blocks.items()}).T
    cov.insert(0, "all", [v.mean() for v in blocks.values()])
    cov.insert(0, "n", [int(v.sum()) for v in blocks.values()])
    lines = [f"firms_master_{Y}.csv: {len(m):,} firms x {m.shape[1]} columns", "",
             "Coverage (share of firms), overall and by EDGAR outcome:",
             cov.to_string(float_format=lambda x: f"{x:.0%}" if x <= 1 else f"{x:,.0f}"), "",
             "Usable logos with no financial data at all (left out of the main sample):",
             f"  {int((m['in_logo_sample'] & m['tobins_q_base'].isna() & m['log_assets_base'].isna()).sum())}", "",
             "EDGAR outcome vs CRSP delisting reason:", pd.crosstab(m["outcome_2024"], m["crsp_exit"]).to_string(), "",
             "Reconciled outcome:", pd.crosstab(m["outcome_2024"], m["outcome_reconciled"], margins=True).to_string(), "",
             "Main sample by sector:", m[m["in_main_sample"]]["sector"].value_counts().to_string()]
    report = "\n".join(lines)
    (OUT / f"merge_report_{Y}.txt").write_text(report)
    print(report)

    dd = pd.DataFrame(DICT, columns=["column", "source", "meaning"]).drop_duplicates("column")
    path = OUT / "data_dictionary.csv"
    if path.exists():
        dd = pd.concat([pd.read_csv(path), dd]).drop_duplicates("column", keep="last")
    dd.to_csv(path, index=False)


def pool():
    files = sorted(glob.glob(str(OUT / "firms_master_[0-9][0-9][0-9][0-9].csv")))
    if not files:
        raise SystemExit("No firms_master_<year>.csv yet.")
    p = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    p = p.copy()
    p["firm_year"] = p["cik"].astype(str) + "_" + p["base_year"].astype(str)
    n_years = p.groupby("cik")["base_year"].transform("nunique")
    p["n_base_years"] = n_years
    p.to_csv(OUT / "firms_pooled.csv", index=False)
    print(f"firms_pooled.csv: {len(p):,} firm-years, {p['cik'].nunique():,} firms, years {sorted(p['base_year'].unique())}")
    print(p.groupby("base_year")[["in_logo_sample", "in_main_sample"]].sum().to_string())
    s = p[p["in_logo_sample"]]
    print(f"firms with a usable logo in 2+ years: {(s.groupby('cik').size() >= 2).sum():,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--pool", action="store_true")
    args = ap.parse_args()
    if args.pool:
        pool()
    else:
        build(args.year)
