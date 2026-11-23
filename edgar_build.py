"""
Build the free public-firm dataset from SEC EDGAR (no WRDS, no paid sources)
for one base year (2019 by default; also used for the 2015 and 2022 groups).

Step 1  Firm universe: every company that filed a 10-K in the base calendar year
        (EDGAR quarterly master index), so firms that later failed or were
        acquired are included.
Step 2  Company details and outcomes from the bulk submissions.zip:
        SIC code, sector, name, tickers, exchanges, website, and from the
        full filing history from the base year on:
          survived     still filing 10-Ks in 2024
          acquired     merger / tender-offer filings or 8-K item 5.01
                       (change in control) before it stopped filing
          bankrupt     8-K item 1.03 (bankruptcy) before it stopped filing
          other_exit   stopped filing for another reason (Form 15 / 25 / none)
Step 3  Financials from the year before the base year to 2024 (bulk companyfacts.zip, XBRL):
        assets, liabilities, equity, revenue, net income, advertising expense,
        public float. Each value records which tag and method produced it.

Step 4  Keep publicly traded operating companies: positive public float in the base year
        or a listed ticker/exchange, and not an asset-backed securities issuer
        (SIC 6189). Compute Tobin's Q and SG&A intensity.

Outputs (data/edgar/ for 2019, data/edgar_<year>/ for other years):
  firms_<year>_all.csv  every base-year 10-K filer kept after SIC exclusions
  firms_<year>.csv      traded operating companies only (the analysis sample)
  financials_long.csv   cik, year, item, value, tag, method
  financials_wide.csv   one row per firm-year, one column per item, plus
                        tobins_q and sga_to_revenue
  build_report.txt      coverage summary for the analysis sample

Usage:
  python3 edgar_build.py --name "First Last" --email you@school.edu                 # 2019
  python3 edgar_build.py --name "First Last" --email you@school.edu --year 2022
Downloads about 3 GB once into data/edgar/raw/ and reuses it for every year.
SEC fair access: the User-Agent carries your name and email, and this script makes
only a handful of requests (4 index files and 2 bulk files per year).
"""
import argparse
import io
import json
import os
import time
import zipfile
from collections import Counter

import pandas as pd
import requests

LAST_YEAR = 2024
EXCLUDE_SIC = list(range(6700, 6800)) + [6189]   # holding cos, SPACs (6770), funds, trusts, ABS issuers
MERGER_FORMS = {"DEFM14A", "PREM14A", "DEFM14C", "PREM14C", "SC 14D9", "SC TO-T", "SC TO-T/A",
                "SC 14D9/A", "425"}
EXIT_FORMS = {"15-12B", "15-12G", "15-15D", "25", "25-NSE", "15F-12B", "15F-12G", "15F-15D"}

ITEMS = {
    # item: (taxonomy, [tags in order of preference], kind)
    "assets": ("us-gaap", ["Assets"], "instant"),
    "liabilities": ("us-gaap", ["Liabilities"], "instant"),
    "equity": ("us-gaap", ["StockholdersEquity",
                           "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "instant"),
    "revenue": ("us-gaap", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                            "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"], "duration"),
    "net_income": ("us-gaap", ["NetIncomeLoss", "ProfitLoss"], "duration"),
    "advertising": ("us-gaap", ["AdvertisingExpense", "MarketingAndAdvertisingExpense"], "duration"),
    "sga": ("us-gaap", ["SellingGeneralAndAdministrativeExpense",
                        "GeneralAndAdministrativeExpense"], "duration"),
    "public_float": ("dei", ["EntityPublicFloat"], "float"),
}


def sector_of(sic):
    """Map a 4-digit SIC code to ~17 sectors (3-digit detail where it matters)."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    d2, d3 = s // 100, s // 10
    if d3 == 283:
        return "Pharma & Biotech"
    if d3 in (384, 385) or d2 == 80:
        return "Healthcare devices & services"
    if d3 == 737:
        return "Software & IT services"
    if d3 in (357, 366, 367, 381, 382):
        return "Tech hardware & electronics"
    if d2 == 48:
        return "Telecom"
    if d2 == 49:
        return "Utilities"
    if 10 <= d2 <= 14 or d2 == 29:
        return "Energy & Mining"
    if d2 in (60, 61, 62):
        return "Banks & financial services"
    if d2 in (63, 64):
        return "Insurance"
    if d2 == 65:
        return "Real estate"
    if 52 <= d2 <= 59:
        return "Retail"
    if d2 in (50, 51):
        return "Wholesale"
    if d2 in (20, 21):
        return "Food, beverage & tobacco"
    if d2 in (22, 23, 25, 31, 39) or s == 2844:
        return "Consumer products"
    if d2 == 37 or 40 <= d2 <= 47:
        return "Autos & transportation"
    if 15 <= d2 <= 17:
        return "Construction"
    if 70 <= d2 <= 89:
        return "Services"
    if 24 <= d2 <= 39:
        return "Industrial manufacturing"
    return "Other"


def download(url, path, headers):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  have {os.path.basename(path)}")
        return
    print(f"  downloading {url}")
    tmp = path + ".part"
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done % (200 << 20) < (1 << 20):
                    print(f"    {done / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)
    os.replace(tmp, path)


def step1_universe(raw, headers, year):
    rows = []
    for q in range(1, 5):
        path = os.path.join(raw, f"master_{year}_Q{q}.idx")
        download(f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx", path, headers)
        time.sleep(0.5)
        with open(path, encoding="latin-1") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) == 5 and parts[2] == "10-K":
                    rows.append({"cik": int(parts[0]), f"name_{year}": parts[1], f"filed_10k_{year}": parts[3]})
    df = pd.DataFrame(rows).sort_values(f"filed_10k_{year}").drop_duplicates("cik")
    print(f"  companies that filed a 10-K in {year}: {len(df):,}")
    return df


def all_filings(zf, cik, sub):
    """Combine the 'recent' block with any overflow files for one company."""
    blocks = [sub.get("filings", {}).get("recent", {})]
    for extra in sub.get("filings", {}).get("files", []):
        try:
            blocks.append(json.loads(zf.read(extra["name"])))
        except KeyError:
            pass
    rows = []
    for b in blocks:
        n = len(b.get("form", []))
        items = b.get("items", [""] * n)
        for i in range(n):
            rows.append((b["form"][i], b["filingDate"][i], items[i] if i < len(items) else ""))
    return rows


def classify(filings, year):
    tenk = sorted(d for f, d, _ in filings if f in ("10-K", "10-K/A", "10-KT"))
    last_10k = tenk[-1] if tenk else ""
    if any(d >= f"{LAST_YEAR}-01-01" for d in tenk):
        return "survived", last_10k, ""
    after = [(f, d, it) for f, d, it in filings if f"{year}-01-01" <= d]
    window_end = (pd.Timestamp(last_10k) + pd.DateOffset(years=2)).strftime("%Y-%m-%d") if last_10k else "2099"
    win = [(f, d, it) for f, d, it in after if d <= window_end]
    if any(f == "8-K" and "1.03" in str(it) for f, d, it in win):
        return "bankrupt", last_10k, "8-K item 1.03"
    if any(f in MERGER_FORMS for f, d, it in win) or any(f == "8-K" and "5.01" in str(it) for f, d, it in win):
        why = next((f for f, d, it in win if f in MERGER_FORMS), "8-K item 5.01")
        return "acquired", last_10k, why
    exit_form = next((f for f, d, it in win if f in EXIT_FORMS), "")
    return "other_exit", last_10k, exit_form or "stopped filing"


def step2_details(raw, headers, universe, year):
    path = os.path.join(raw, "submissions.zip")
    download("https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip", path, headers)
    out = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for k, cik in enumerate(universe["cik"], 1):
            fn = f"CIK{cik:010d}.json"
            if fn not in names:
                out.append({"cik": cik, "in_submissions": False})
                continue
            sub = json.loads(zf.read(fn))
            filings = all_filings(zf, cik, sub)
            outcome, last_10k, evidence = classify(filings, year)
            out.append({
                "cik": cik, "in_submissions": True,
                "name": sub.get("name"), "entity_type": sub.get("entityType"),
                "sic": sub.get("sic"), "sic_description": sub.get("sicDescription"),
                "sector": sector_of(sub.get("sic")),
                "tickers": ";".join(sub.get("tickers") or []),
                "exchanges": ";".join(x for x in (sub.get("exchanges") or []) if x),
                "website_edgar": sub.get("website") or "",
                "state_of_incorporation": sub.get("stateOfIncorporation"),
                "fiscal_year_end": sub.get("fiscalYearEnd"),
                "outcome_2024": outcome, "last_10k_date": last_10k, "outcome_evidence": evidence,
            })
            if k % 1000 == 0:
                print(f"    {k:,}/{len(universe):,} companies read", flush=True)
    return universe.merge(pd.DataFrame(out), on="cik", how="left")


def pick_value(units, kind, year):
    """Choose one value for a calendar year. Prefer SEC 'frame' alignment."""
    if kind == "instant":
        want = f"CY{year}Q4I"
        hits = [u for u in units if u.get("frame") == want]
        if hits:
            return hits[-1]["val"], "frame"
        cands = [u for u in units if u.get("form", "").startswith("10-K") and str(u.get("end", "")).startswith(str(year))]
    elif kind == "duration":
        want = f"CY{year}"
        hits = [u for u in units if u.get("frame") == want]
        if hits:
            return hits[-1]["val"], "frame"
        cands = []
        for u in units:
            if not u.get("form", "").startswith("10-K") or not str(u.get("end", "")).startswith(str(year)):
                continue
            try:
                days = (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days
            except Exception:
                continue
            if 330 <= days <= 400:
                cands.append(u)
    else:  # public float: measured once a year, usually mid-year
        cands = [u for u in units if str(u.get("end", "")).startswith(str(year))]
    if not cands:
        return None, None
    best = sorted(cands, key=lambda u: (u.get("end", ""), u.get("filed", "")))[-1]
    return best["val"], "latest_10k"


def step3_financials(raw, headers, firms, years):
    path = os.path.join(raw, "companyfacts.zip")
    download("https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip", path, headers)
    rows = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for k, cik in enumerate(firms["cik"], 1):
            fn = f"CIK{cik:010d}.json"
            if fn not in names:
                continue
            facts = json.loads(zf.read(fn)).get("facts", {})
            for item, (tax, tags, kind) in ITEMS.items():
                for year in years:
                    for tag in tags:
                        units = facts.get(tax, {}).get(tag, {}).get("units", {}).get("USD", [])
                        if not units:
                            continue
                        val, method = pick_value(units, kind, year)
                        if val is not None:
                            rows.append({"cik": cik, "year": year, "item": item, "value": val,
                                         "tag": f"{tax}:{tag}", "method": method})
                            break
            if k % 1000 == 0:
                print(f"    {k:,}/{len(firms):,} companies read", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="your email, required by SEC in the User-Agent")
    ap.add_argument("--name", required=True, help='your full name, required by SEC in the User-Agent')
    ap.add_argument("--year", type=int, default=2019, help="base year (2019, 2015, 2022, ...)")
    ap.add_argument("--out", default=None, help="default data/edgar for 2019, data/edgar_<year> otherwise")
    ap.add_argument("--raw", default="data/edgar/raw", help="bulk SEC files, shared by all years")
    args = ap.parse_args()
    Y = args.year
    out = args.out or ("data/edgar" if Y == 2019 else f"data/edgar_{Y}")
    headers = {"User-Agent": f"Stony Brook CSE519 research {args.name} {args.email}",
               "Accept-Encoding": "gzip, deflate"}
    raw = args.raw
    os.makedirs(raw, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    print(f"Step 1: {Y} universe")
    uni = step1_universe(raw, headers, Y)

    print("Step 2: company details and outcomes")
    firms = step2_details(raw, headers, uni, Y)
    sic = pd.to_numeric(firms["sic"], errors="coerce")
    before = len(firms)
    firms = firms[firms["in_submissions"].fillna(False) & ~sic.isin(EXCLUDE_SIC) & sic.notna()]
    print(f"  kept {len(firms):,} of {before:,} (dropped SIC 6700-6799 and firms without SIC)")
    firms["base_year"] = Y
    firms.to_csv(os.path.join(out, f"firms_{Y}_all.csv"), index=False)

    print("Step 3: financials")
    fin = step3_financials(raw, headers, firms, range(Y - 1, LAST_YEAR + 1))
    fin.to_csv(os.path.join(out, "financials_long.csv"), index=False)
    wide = fin.pivot_table(index=["cik", "year"], columns="item", values="value", aggfunc="first").reset_index()
    for col in ITEMS:
        if col not in wide:
            wide[col] = float("nan")
    ok = (wide["assets"] > 0) & wide["equity"].notna() & (wide["public_float"] > 0)
    wide["tobins_q"] = ((wide["assets"] - wide["equity"] + wide["public_float"]) / wide["assets"]).where(ok)
    wide["sga_to_revenue"] = (wide["sga"] / wide["revenue"]).where(wide["revenue"] > 0)
    wide.to_csv(os.path.join(out, "financials_wide.csv"), index=False)

    print("Step 4: keep publicly traded operating companies")
    float_base = wide.loc[wide["year"] == Y].set_index("cik")["public_float"]
    firms[f"public_float_{Y}"] = firms["cik"].map(float_base)
    firms["traded"] = (firms[f"public_float_{Y}"] > 0) | (firms["tickers"].fillna("") != "") | \
                      (firms["exchanges"].fillna("") != "")
    all_n = len(firms)
    firms = firms[firms["traded"]].copy()
    print(f"  kept {len(firms):,} of {all_n:,} (positive {Y} public float, or a ticker/exchange)")
    firms.to_csv(os.path.join(out, f"firms_{Y}.csv"), index=False)
    wide = wide[wide["cik"].isin(firms["cik"])]

    lines = [f"Traded operating companies in the {Y} sample: {len(firms):,} "
             f"(of {all_n:,} 10-K filers after SIC exclusions)", "",
             "Outcome by 2024:"]
    lines += [f"  {k:12s} {v:6,}  ({v / len(firms):.0%})" for k, v in firms["outcome_2024"].value_counts().items()]
    lines += ["", "Firms by sector:"]
    lines += [f"  {k:32s} {v:6,}" for k, v in firms["sector"].value_counts().items()]
    lines += ["", f"Website listed in EDGAR: {(firms['website_edgar'] != '').mean():.0%}",
              f"Has a ticker: {(firms['tickers'] != '').mean():.0%}", "",
              f"Coverage of {Y} financials (share of firms with a value):"]
    w19 = wide[wide["year"] == Y].set_index("cik")
    for item in list(ITEMS) + ["tobins_q", "sga_to_revenue"]:
        share = w19[item].notna().sum() / len(firms) if item in w19 else 0
        lines.append(f"  {item:14s} {share:.0%}")
    q = w19["tobins_q"].dropna()
    if len(q):
        lines += ["", f"Tobin's Q {Y}: median {q.median():.2f}, "
                      f"10th-90th percentile {q.quantile(.1):.2f} to {q.quantile(.9):.2f}"]
    meth = Counter(fin.loc[fin["year"] == Y, "method"])
    lines += ["", f"{Y} values matched by SEC frame: {meth.get('frame', 0):,}, "
                  f"by latest 10-K fallback: {meth.get('latest_10k', 0):,}"]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(out, "build_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
