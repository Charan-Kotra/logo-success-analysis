"""
Step A of logo extraction: collect candidate logo images from SEC filings for one base year.

For each firm (base year Y):
  pass 1    its filings dated in Y: up to 15 8-Ks, 2 DEF 14As, 2 10-Ks
            (only the first 3 images of each filing, in file-name order;
             stops early once one image has appeared in 3 separate filings)
  pass 2    only if fewer than 2 filings had images: 8-Ks and proxies from
            July of Y-1 to June of Y+1 (same logo era), up to 10 more filings
  fallback  if no image repeats in 2+ filings yet, fetch up to 12 more images
            of the filings already visited (press releases first)
Every image is downloaded once, fingerprinted (pHash) and measured.
Nothing is chosen here; logo_score.py does the choosing.

SEC fair access (https://www.sec.gov/os/accessing-edgar-data):
  - every request carries a User-Agent with your name and email
  - all worker threads share ONE rate limiter; the default is 6 requests/second,
    below SEC's 10/second limit. If two people run this on the same network
    (same campus Wi-Fi) at the same time, use --rate 4 each.
  - on a 403/429 from SEC every thread pauses, then slows down

Outputs (data/logos/ for 2019, data/logos_<year>/ otherwise):
  edgar_raw/<cik>/<accession>_<file>   downloaded images
  candidates.csv (candidates_v3.csv for 2019)   one row per image, or one empty row per firm

Usage:
  python3 edgar_logos_collect.py --year 2022 --name "First Last" --email you@school.edu --sample 30   # pilot
  python3 edgar_logos_collect.py --year 2022 --name "First Last" --email you@school.edu               # all
A rerun skips firms already collected, so it is safe to stop (Ctrl+C) and restart.
"""
import argparse
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import imagehash
import pandas as pd
import requests
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".gif", ".png")
PASS1 = {"8-K": 15, "DEF 14A": 2, "10-K": 2}
PASS2 = {"8-K": 8, "DEF 14A": 2}
MAX_IMG_PER_FILING = 3     # in the pilot the logo was among a filing's first 3 images 88% of the time
FORMS = set(PASS1) | set(PASS2)
COLS = ["cik", "form", "date", "accession", "window", "file", "path", "w", "h", "phash", "colors", "fallback"]


class Sec:
    """Shared, thread-safe SEC client: one global request schedule for all threads."""

    def __init__(self, name, email, rate):
        self.headers = {"User-Agent": f"Stony Brook CSE519 research {name} {email}",
                        "Accept-Encoding": "gzip, deflate"}
        self.gap = 1.0 / rate
        self.lock = threading.Lock()
        self.next_slot = 0.0
        self.local = threading.local()
        self.count = 0
        self.throttled = 0

    def _session(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
            self.local.s.headers.update(self.headers)
        return self.local.s

    def _wait_turn(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.gap
            self.count += 1
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _pause_all(self, seconds):
        with self.lock:
            self.next_slot = max(self.next_slot, time.monotonic() + seconds)
            self.gap = min(self.gap * 1.5, 1.0)       # slow down for the rest of the run
            self.throttled += 1

    def get(self, url):
        for attempt in range(4):
            self._wait_turn()
            try:
                r = self._session().get(url, timeout=30)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 503):
                self._pause_all(60 * (attempt + 1))
                continue
            return None
        return None


def firm_filings(zf, names, cik, lo, hi, sec=None, stats=None):
    """This firm's 8-K / DEF 14A / 10-K filings dated between lo and hi.
    The main JSON holds only a company's most recent ~1,000 filings; older ones are in overflow
    files. An overflow file covering the window that is missing from the zip is fetched from SEC."""
    fn = f"CIK{cik:010d}.json"
    if fn not in names:
        if stats is not None:
            stats["not_in_zip"] += 1
        return []
    sub = json.loads(zf.read(fn))
    blocks = [sub.get("filings", {}).get("recent", {})]
    for extra in sub.get("filings", {}).get("files", []):
        if extra.get("filingTo", "9999") < lo or extra.get("filingFrom", "0000") > hi:
            continue                                   # overflow file outside the window
        if extra["name"] in names:
            blocks.append(json.loads(zf.read(extra["name"])))
        elif sec is not None:
            r = sec.get(f"https://data.sec.gov/submissions/{extra['name']}")
            if stats is not None:
                stats["overflow_fetched" if r is not None else "overflow_failed"] += 1
            if r is not None:
                try:
                    blocks.append(r.json())
                except ValueError:
                    pass
    out = []
    for b in blocks:
        for i in range(len(b.get("form", []))):
            f, d = b["form"][i], b["filingDate"][i]
            if f in FORMS and lo <= d <= hi:
                out.append((f, d, b["accessionNumber"][i]))
    return out


def pick(filings, caps, lo, hi, exclude=()):
    chosen = []
    for form, cap in caps.items():
        pool = sorted(f for f in filings if f[0] == form and lo <= f[1] <= hi and f[2] not in exclude)
        if len(pool) > cap:                       # spread picks across the window
            step = len(pool) / cap
            pool = [pool[int(i * step)] for i in range(cap)]
        chosen += pool
    return chosen


def image_record(path):
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            if w * h > 4_000_000:
                return None
            rgb = im.convert("RGB")
            small = rgb.copy()
            small.thumbnail((64, 64))
            return {"w": w, "h": h, "phash": str(imagehash.phash(rgb)),
                    "colors": len(small.getcolors(64 * 64) or [])}
    except Exception:
        return None


def confirmed(rows, need=3, hamming=6):
    """True once one image (near-identical by pHash) appears in `need` separate filings."""
    groups = []
    for r in rows:
        h = imagehash.hex_to_hash(r["phash"])
        for g in groups:
            if h - g["hash"] <= hamming:
                g["acc"].add(r["accession"])
                if len(g["acc"]) >= need:
                    return True
                break
        else:
            groups.append({"hash": h, "acc": {r["accession"]}})
    return False


def collect_filing(sec, raw_dir, cik, form, date, acc, window, index_cache, start=0, limit=MAX_IMG_PER_FILING):
    """Download images number start .. start+limit-1 (file-name order) of one filing."""
    acc_nd = acc.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nd}"
    if acc not in index_cache:
        r = sec.get(f"{base}/index.json")
        try:
            items = r.json()["directory"]["item"] if r is not None else []
        except (ValueError, KeyError):
            items = []
        index_cache[acc] = sorted((it for it in items if it.get("name", "").lower().endswith(IMG_EXT)),
                                  key=lambda it: it["name"])
    rows, kept = [], 0
    for it in index_cache[acc][start:]:
        if kept >= limit:
            break
        name = it.get("name", "")
        size = str(it.get("size", ""))
        if size.isdigit() and int(size) > 800_000:
            continue
        kept += 1
        d = os.path.join(raw_dir, str(cik))
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{acc_nd}_{name}")
        if not os.path.exists(path):
            img = sec.get(f"{base}/{name}")
            if img is None:
                continue
            with open(path, "wb") as fh:
                fh.write(img.content)
        rec = image_record(path)
        if rec:
            rows.append({"cik": cik, "form": form, "date": date, "accession": acc,
                         "window": window, "file": name, "path": path, **rec})
    return rows


def collect_firm(sec, raw_dir, cik, fl, year):
    y_lo, y_hi = f"{year}-01-01", f"{year}-12-31"
    p2_lo, p2_hi = f"{year - 1}-07-01", f"{year + 1}-06-30"
    rows, cache, visited = [], {}, []
    first = pick(fl, PASS1, y_lo, y_hi)
    for form, date, acc in first:
        rows += collect_filing(sec, raw_dir, cik, form, date, acc, str(year), cache)
        visited.append((form, date, acc, str(year)))
        if confirmed(rows):                      # logo already repeats in 3 filings: stop early
            break
    if len({r["accession"] for r in rows}) < 2:
        more = pick(fl, PASS2, p2_lo, p2_hi, exclude={a for _, _, a in first})
        for form, date, acc in more:
            rows += collect_filing(sec, raw_dir, cik, form, date, acc, f"{year - 1}-{year + 1}", cache)
            visited.append((form, date, acc, f"{year - 1}-{year + 1}"))
    if not confirmed(rows, need=2):
        for form, date, acc, window in sorted(visited, key=lambda v: v[0] != "8-K"):
            if len(cache.get(acc, [])) <= MAX_IMG_PER_FILING:
                continue
            extra = collect_filing(sec, raw_dir, cik, form, date, acc, window, cache,
                                   start=MAX_IMG_PER_FILING, limit=12)
            for r in extra:
                r["fallback"] = True
            rows += extra
            if confirmed(rows, need=2):
                break
    for r in rows:
        r.setdefault("fallback", False)
    if not rows:
        rows = [{c: None for c in COLS} | {"cik": cik, "fallback": False}]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--name", required=True, help="your full name (SEC User-Agent)")
    ap.add_argument("--email", required=True, help="your email (SEC User-Agent)")
    ap.add_argument("--firms", default=None, help="default: the firm list edgar_build.py made for --year")
    ap.add_argument("--submissions", default="data/edgar/raw/submissions.zip")
    ap.add_argument("--out", default=None, help="default data/logos (2019) or data/logos_<year>")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rate", type=float, default=6.0, help="max SEC requests per second (SEC limit is 10)")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--redo_empty", action="store_true",
                    help="collect again the firms that so far have no image at all")
    ap.add_argument("--seed", type=int, default=519)
    args = ap.parse_args()
    Y = args.year
    if args.rate > 9:
        raise SystemExit("--rate must stay below SEC's 10 requests/second")

    firms_path = args.firms or ("data/edgar/firms_2019.csv" if Y == 2019 else f"data/edgar_{Y}/firms_{Y}.csv")
    out = args.out or ("data/logos" if Y == 2019 else f"data/logos_{Y}")
    raw_dir = os.path.join(out, "edgar_raw")
    os.makedirs(raw_dir, exist_ok=True)
    cand_path = os.path.join(out, "candidates_v3.csv" if Y == 2019 else "candidates.csv")

    firms = pd.read_csv(firms_path)
    if args.sample:
        firms = firms.sample(min(args.sample, len(firms)), random_state=args.seed)
    done = set()
    if os.path.exists(cand_path):
        prev = pd.read_csv(cand_path, usecols=["cik", "phash"])
        done = set(prev["cik"].unique())
        if args.redo_empty:
            has_img = set(prev.loc[prev["phash"].notna(), "cik"])
            done = has_img
            print(f"--redo_empty: {len(set(prev['cik']) - has_img):,} firms without images will be collected again")
    todo = [int(c) for c in firms["cik"] if int(c) not in done]
    print(f"Base year {Y}: firms {len(firms):,}  already collected {len(firms) - len(todo):,}  to do {len(todo):,}")
    print(f"Rate limit {args.rate}/s shared by {args.workers} workers")

    sec = Sec(args.name, args.email, args.rate)
    print("Reading filing lists from submissions.zip...")
    stats = {"not_in_zip": 0, "overflow_fetched": 0, "overflow_failed": 0}
    with zipfile.ZipFile(args.submissions) as zf:
        names = set(zf.namelist())
        fl = {cik: firm_filings(zf, names, cik, f"{Y - 1}-07-01", f"{Y + 1}-06-30", sec, stats) for cik in todo}
    no_filings = sum(1 for v in fl.values() if not v)
    print(f"  firms with no 8-K/DEF 14A/10-K in the window: {no_filings:,} of {len(todo):,}; "
          f"overflow files fetched from SEC: {stats['overflow_fetched']}, failed: {stats['overflow_failed']}, "
          f"firms missing from the zip: {stats['not_in_zip']}")
    t0 = time.time()
    header = not os.path.exists(cand_path)
    finished = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(collect_firm, sec, raw_dir, cik, fl[cik], Y): cik for cik in todo}
            for fut in as_completed(futs):
                try:
                    rows = fut.result()
                except Exception as e:                 # one firm failing never stops the run
                    print(f"  firm {futs[fut]} failed: {type(e).__name__}: {e}")
                    continue
                pd.DataFrame(rows, columns=COLS).to_csv(cand_path, mode="a", header=header, index=False)
                header = False
                finished += 1
                if finished % 25 == 0 or finished == len(todo):
                    el = time.time() - t0
                    print(f"  {finished:,}/{len(todo):,} firms  {sec.count / el:.1f} req/s  "
                          f"~{el / finished * (len(todo) - finished) / 60:.0f} min left"
                          + (f"  (SEC throttled {sec.throttled}x)" if sec.throttled else ""), flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Rerun the same command to continue where it left off.")
        os._exit(1)

    if not os.path.exists(cand_path):
        return
    c = pd.read_csv(cand_path)
    c = c[c["cik"].isin(firms["cik"])]
    per = c.dropna(subset=["phash"]).groupby("cik")["accession"].nunique()
    n = firms["cik"].nunique()
    print(f"\nFirms with any image: {len(per):,} of {n:,} ({len(per) / n:.0%})")
    print(f"Firms with images in 2+ filings: {(per >= 2).sum():,} ({(per >= 2).sum() / n:.0%})")
    print(f"Images collected: {c['phash'].notna().sum():,}")
    print(f"Next: python3 logo_score.py --year {Y}")


if __name__ == "__main__":
    main()
