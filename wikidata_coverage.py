"""
Link the 2019 EDGAR firms to Wikidata by SEC CIK (property P5531) and collect:
  website (P856) -> domain, for logo lookups
  logo image (P154), with its start/end dates (P580/P582) when Wikidata has them
  inception / founding date (P571), dissolution date (P576)

No name matching: the join is on the CIK number only.

Outputs (data/wikidata/):
  raw_cik_batches.json    raw query results by batch (cached; a rerun resumes)
  firm_links.csv          one row per 2019 firm: wikidata id, domain, logo file,
                          logo status (dated_2019 / undated / none), logo URL
  coverage_report.txt     coverage overall and by sector

Usage:
  python3 wikidata_coverage.py --email you@example.com
"""
import argparse
import json
import os
import re
import time
from urllib.parse import quote, urlparse

import pandas as pd
import requests

ENDPOINT = "https://query.wikidata.org/sparql"

QUERY = """
SELECT ?item ?cik ?website ?logo ?logoStart ?logoEnd ?logoRank ?inception ?dissolved WHERE {
  VALUES ?cik { %s }
  ?item wdt:P5531 ?cik .
  OPTIONAL { ?item wdt:P856 ?website . }
  OPTIONAL {
    ?item p:P154 ?ls .
    ?ls ps:P154 ?logo ; wikibase:rank ?logoRank .
    FILTER(?logoRank != wikibase:DeprecatedRank)
    OPTIONAL { ?ls pq:P580 ?logoStart . }
    OPTIONAL { ?ls pq:P582 ?logoEnd . }
  }
  OPTIONAL { ?item wdt:P571 ?inception . }
  OPTIONAL { ?item wdt:P576 ?dissolved . }
}
"""

BASE_START = pd.Timestamp("2019-12-31")
BASE_END = pd.Timestamp("2019-01-01")


def run_query(email, cache, ciks, batch=200):
    """Query in small batches of CIKs (Wikidata stops queries after ~60 s).
    Each CIK is sent both zero-padded to 10 digits and unpadded, since both
    spellings occur on Wikidata. Finished batches are cached, so a rerun resumes."""
    done = json.load(open(cache)) if os.path.exists(cache) else {"batches": {}}
    headers = {"User-Agent": f"CSE519 logo research ({email})",
               "Accept": "application/sparql-results+json"}
    ciks = sorted(set(int(c) for c in ciks))
    chunks = [ciks[i:i + batch] for i in range(0, len(ciks), batch)]
    for k, chunk in enumerate(chunks):
        key = str(k)
        if key in done["batches"]:
            continue
        values = " ".join(f'"{c:010d}" "{c}"' for c in chunk)
        for attempt in range(5):
            try:
                r = requests.post(ENDPOINT, data={"query": QUERY % values}, headers=headers, timeout=120)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 30))
                    print(f"    rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                done["batches"][key] = r.json()["results"]["bindings"]
                break
            except (requests.RequestException, ValueError) as e:
                wait = 5 * (attempt + 1)
                print(f"    batch {k + 1}/{len(chunks)} failed ({type(e).__name__}), retrying in {wait}s")
                time.sleep(wait)
        else:
            raise SystemExit(f"Batch {k + 1} kept failing. Run the script again later; finished batches are saved.")
        json.dump(done, open(cache, "w"))
        print(f"  batch {k + 1}/{len(chunks)} done", flush=True)
        time.sleep(1)
    return {"results": {"bindings": [b for v in done["batches"].values() for b in v]}}


def val(b, key):
    return b[key]["value"] if key in b else None


def domain_of(url):
    if not url:
        return None
    try:
        host = urlparse(url if "://" in url else "http://" + url).netloc.lower()
    except ValueError:
        return None
    host = host.split(":")[0]
    return host[4:] if host.startswith("www.") else host or None


def logo_url(file_url, width=512):
    # Wikidata returns e.g. http://commons.wikimedia.org/wiki/Special:FilePath/Apple%20logo.svg
    name = file_url.rsplit("/", 1)[-1]
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{name}?width={width}"


def to_date(s):
    if not s:
        return pd.NaT
    return pd.to_datetime(s[:10], errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--firms", default="data/edgar/firms_2019.csv")
    ap.add_argument("--out", default="data/wikidata")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    firms = pd.read_csv(args.firms)
    data = run_query(args.email, os.path.join(args.out, "raw_cik_batches.json"), firms["cik"])
    rows = []
    for b in data["results"]["bindings"]:
        cik_raw = re.sub(r"\D", "", val(b, "cik") or "")
        if not cik_raw:
            continue
        rows.append({
            "cik": int(cik_raw), "qid": val(b, "item").rsplit("/", 1)[-1],
            "website": val(b, "website"), "logo": val(b, "logo"),
            "logo_start": to_date(val(b, "logoStart")), "logo_end": to_date(val(b, "logoEnd")),
            "logo_preferred": (val(b, "logoRank") or "").endswith("PreferredRank"),
            "inception": to_date(val(b, "inception")), "dissolved": to_date(val(b, "dissolved")),
        })
    wd = pd.DataFrame(rows)
    print(f"  Wikidata rows with a CIK: {len(wd):,}, distinct CIKs: {wd['cik'].nunique():,}")

    out = []
    for cik, g in wd.groupby("cik"):
        sites = [w for w in g["website"].dropna().unique()]
        domains = sorted({d for d in (domain_of(w) for w in sites) if d}, key=len)
        logos = g.dropna(subset=["logo"]).drop_duplicates("logo")
        status, chosen = "none", None
        if len(logos):
            dated = logos[logos["logo_start"].notna() | logos["logo_end"].notna()]
            in_2019 = dated[(dated["logo_start"].isna() | (dated["logo_start"] <= BASE_START)) &
                            (dated["logo_end"].isna() | (dated["logo_end"] >= BASE_END))]
            if len(in_2019):
                status, chosen = "dated_2019", in_2019.sort_values("logo_start").iloc[-1]
            else:
                undated = logos[logos["logo_start"].isna() & logos["logo_end"].isna()]
                pool = undated if len(undated) else logos
                pool = pool.sort_values("logo_preferred", ascending=False)
                status, chosen = ("undated" if len(undated) else "dated_other_years"), pool.iloc[0]
        out.append({
            "cik": cik, "qid": g["qid"].iloc[0],
            "n_qids": g["qid"].nunique(),
            "domain": domains[0] if domains else None,
            "all_domains": ";".join(domains),
            "logo_status": status,
            "logo_file": chosen["logo"].rsplit("/", 1)[-1] if chosen is not None else None,
            "logo_url": logo_url(chosen["logo"]) if chosen is not None else None,
            "n_logos": len(logos),
            "inception": g["inception"].min(),
            "dissolved": g["dissolved"].min(),
        })
    links = pd.DataFrame(out)

    m = firms[["cik", "name", "sector", "outcome_2024"]].merge(links, on="cik", how="left")
    m["logo_status"] = m["logo_status"].fillna("no_wikidata_item").where(m["qid"].notna(), "no_wikidata_item")
    m.to_csv(os.path.join(args.out, "firm_links.csv"), index=False)

    n = len(m)
    lines = [f"2019 traded firms: {n:,}",
             f"  with a Wikidata item (by CIK): {m['qid'].notna().mean():.0%}",
             f"  with a website/domain:         {m['domain'].notna().mean():.0%}",
             f"  with any logo:                 {m['logo_file'].notna().mean():.0%}",
             f"  with a logo dated to 2019:     {(m['logo_status'] == 'dated_2019').mean():.0%}",
             f"  with an inception date:        {m['inception'].notna().mean():.0%}",
             f"  CIKs linked to 2+ Wikidata items (check these): {(m['n_qids'] > 1).sum():,}", "",
             "Logo status:"]
    lines += [f"  {k:20s} {v:6,}" for k, v in m["logo_status"].value_counts().items()]
    lines += ["", "By outcome (share with domain / with logo):"]
    for k, g in m.groupby("outcome_2024"):
        lines.append(f"  {k:12s} n={len(g):5,}  domain {g['domain'].notna().mean():.0%}  "
                     f"logo {g['logo_file'].notna().mean():.0%}")
    lines += ["", "By sector (share with domain / with logo):"]
    for k, g in m.groupby("sector"):
        lines.append(f"  {k:32s} n={len(g):4,}  domain {g['domain'].notna().mean():.0%}  "
                     f"logo {g['logo_file'].notna().mean():.0%}")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(args.out, "coverage_report.txt"), "w").write(report + "\n")


if __name__ == "__main__":
    main()
