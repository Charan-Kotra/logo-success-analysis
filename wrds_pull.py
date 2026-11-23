"""
Pull the firm list, financials and stock returns from WRDS for the
logo distinctiveness study.

Saves (in data/wrds/):
  company.csv.gz     Compustat company header: gvkey, name, CIK, website, SIC
  funda.csv.gz       Compustat annual fundamentals 2012-2024: assets, book equity,
                     shares, price, advertising, sales, net income
  ccm_link.csv.gz    CRSP-Compustat link table (gvkey <-> permno)
  msf.csv.gz         CRSP monthly stock file 2012-2024: returns, price, shares
  msedelist.csv.gz   CRSP delisting returns
  msenames.csv.gz    CRSP names: share code, exchange code, ticker, SIC over time

Usage:
  pip3 install wrds
  python3 wrds_pull.py YOUR_WRDS_USERNAME
The first run asks for your WRDS password (and a Duo approval) and offers to
save it in ~/.pgpass so later runs connect without asking.

Covers base years 2015, 2019 and 2022. A table saved earlier with a shorter date
range is pulled again automatically (the old file is kept as *.old.csv.gz).

WRDS terms of use: these files are licensed to Stony Brook WRDS users. Keep them
out of GitHub (data/ is in .gitignore) and do not share them outside the team.
"""
import os
import sys

import wrds

user = sys.argv[1] if len(sys.argv) > 1 else None
os.makedirs("data/wrds", exist_ok=True)
db = wrds.Connection(wrds_username=user)

QUERIES = {
    "company": """
        select gvkey, conm, conml, cik, weburl, sic, naics, ipodate, loc, fic, state
        from comp.company
    """,
    "funda": """
        select gvkey, datadate, fyear, at, ceq, csho, prcc_f, xad, sale, ni, lt
        from comp.funda
        where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'
          and fyear between 2012 and 2024
    """,
    "ccm_link": """
        select gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt
        from crsp.ccmxpf_lnkhist
        where linktype in ('LU', 'LC') and linkprim in ('P', 'C')
    """,
    "msf": """
        select permno, date, ret, retx, prc, shrout
        from crsp.msf
        where date between '2012-01-01' and '2024-12-31'
    """,
    "msedelist": """
        select permno, dlstdt, dlstcd, dlret
        from crsp.msedelist
        where dlstdt between '2012-01-01' and '2024-12-31'
    """,
    "msenames": """
        select permno, namedt, nameendt, shrcd, exchcd, ticker, siccd, comnam
        from crsp.msenames
        where nameendt >= '2012-01-01'
    """,
}

# earliest date each saved table must reach; an older, shorter pull is replaced
NEEDS = {"funda": ("fyear", 2012), "msf": ("date", "2012-02"), "msedelist": ("dlstdt", "2012-12"),
         "msenames": ("nameendt", None)}


def covers(name, path):
    if name not in NEEDS:
        return True
    col, start = NEEDS[name]
    if start is None:        # msenames: check the earliest end date reached
        d = pd.read_csv(path, usecols=["nameendt"])
        return str(d["nameendt"].min()) <= "2012-12-31"
    d = pd.read_csv(path, usecols=[col])
    return str(d[col].min()) <= str(start)


import pandas as pd

failed = []
for name, sql in QUERIES.items():
    path = f"data/wrds/{name}.csv.gz"
    if os.path.exists(path):
        if covers(name, path):
            print(f"{name}: already saved, skipping")
            continue
        os.replace(path, path.replace(".csv.gz", ".old.csv.gz"))
        print(f"{name}: saved file starts too late for 2015, pulling again")
    try:
        df = db.raw_sql(sql)
        df.to_csv(path, index=False, compression="gzip")
        print(f"{name}: {len(df):,} rows, {df.shape[1]} columns -> {path}")
    except Exception as e:
        failed.append(name)
        print(f"{name}: FAILED - {str(e).splitlines()[0]}")

if failed:
    print("\nSome tables failed. If the CRSP ones failed, your subscription may use the newer")
    print("CRSP format. Run this to see which CRSP monthly tables you can access:")
    print("  python3 -c \"import wrds; db=wrds.Connection(); "
          "print([t for t in db.list_tables('crsp') if 'msf' in t or 'mth' in t])\"")

# Quick look at the 2019 base-year universe
import pandas as pd
if os.path.exists("data/wrds/company.csv.gz") and os.path.exists("data/wrds/funda.csv.gz"):
    comp = pd.read_csv("data/wrds/company.csv.gz", dtype={"gvkey": str, "cik": str, "sic": str})
    fa = pd.read_csv("data/wrds/funda.csv.gz", dtype={"gvkey": str})
    base = fa[fa["fyear"] == 2019].merge(comp, on="gvkey")
    base = base[base["loc"] == "USA"]
    sic = pd.to_numeric(base["sic"], errors="coerce")
    base = base[~sic.between(6700, 6799)]
    print(f"\n2019 base year, US firms, excluding SIC 67xx (shells, SPACs, funds): {len(base):,}")
    print(f"  with a website (weburl): {base['weburl'].notna().mean():.0%}")
    print(f"  with a CIK:              {base['cik'].notna().mean():.0%}")
    print(f"  with advertising (xad):  {base['xad'].notna().mean():.0%}")
db.close()
