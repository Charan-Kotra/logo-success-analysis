"""
Find each firm's 2019 logo in its own SEC filings.

Companies attach image files ("GRAPHIC" documents) to filings. Press releases
(8-K), proxy statements (DEF 14A) and some 10-Ks carry the company logo in their
header. A logo is reused from filing to filing, while charts and photos change,
so the image that recurs across a firm's 2019 filings is almost always its logo.

For each firm:
  1. take its 2019 filings: up to 6 8-Ks, 2 DEF 14As and 2 10-Ks
  2. list each filing's folder on EDGAR and download its images
  3. fingerprint every image (pHash) and group near-identical ones
  4. pick the group that appears in the most separate filings
     confidence: high = 3+ filings, medium = 2, low = 1, none = no images
  5. drop images that recur across many different firms (wire-service logos,
     icons): these are not the firm's own

Outputs (data/logos/):
  edgar_raw/<cik>/...          every downloaded image
  edgar/<cik>.png              the chosen logo
  edgar_candidates.csv         every image with size and fingerprint
  edgar_choice.csv             one row per firm: chosen image and confidence
  edgar_review.html            a page of chosen logos to check by eye

Usage:
  python3 edgar_logos.py --email you@example.com --sample 50      # pilot
  python3 edgar_logos.py --email you@example.com                  # all firms
A rerun skips firms already done.
"""
import argparse
import html
import json
import os
import time
import zipfile

import imagehash
import pandas as pd
import requests
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".gif", ".png")
PER_FORM = {"8-K": 6, "DEF 14A": 2, "10-K": 2}
HAMMING = 6                 # pHash distance that counts as "the same image"
GENERIC_FIRMS = 3           # an image shared by this many firms is not anyone's logo


class Sec:
    def __init__(self, email):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": f"CSE519 logo research {email}",
                               "Accept-Encoding": "gzip, deflate"})
        self.last = 0.0

    def get(self, url):
        for attempt in range(4):
            wait = 0.12 - (time.time() - self.last)     # stay under 10 requests/second
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.s.get(url, timeout=30)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 503):
                time.sleep(10 * (attempt + 1))
                continue
            return None
        return None


def filings_2019(zf, names, cik):
    fn = f"CIK{cik:010d}.json"
    if fn not in names:
        return []
    sub = json.loads(zf.read(fn))
    blocks = [sub.get("filings", {}).get("recent", {})]
    for extra in sub.get("filings", {}).get("files", []):
        if extra["name"] in names:
            blocks.append(json.loads(zf.read(extra["name"])))
    out = []
    for b in blocks:
        for i in range(len(b.get("form", []))):
            d = b["filingDate"][i]
            if d.startswith("2019") and b["form"][i] in PER_FORM:
                out.append((b["form"][i], d, b["accessionNumber"][i]))
    picked = []
    for form, cap in PER_FORM.items():
        picked += sorted(x for x in out if x[0] == form)[:cap]
    return picked


def image_record(path):
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            if min(w, h) < 24 or w * h > 2_000_000:
                return None
            rgb = im.convert("RGB")
            small = rgb.copy()
            small.thumbnail((64, 64))
            colors = len(small.getcolors(64 * 64) or [])
            return {"w": w, "h": h, "phash": str(imagehash.phash(rgb)), "colors": colors}
    except Exception:
        return None


def group(cands):
    """Greedy grouping of near-identical images by pHash distance."""
    groups = []
    for c in cands:
        h = imagehash.hex_to_hash(c["phash"])
        for g in groups:
            if h - g["hash"] <= HAMMING:
                g["items"].append(c)
                break
        else:
            groups.append({"hash": h, "items": [c]})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--firms", default="data/edgar/firms_2019.csv")
    ap.add_argument("--submissions", default="data/edgar/raw/submissions.zip")
    ap.add_argument("--out", default="data/logos")
    ap.add_argument("--sample", type=int, default=None, help="random firms for a pilot")
    ap.add_argument("--seed", type=int, default=519)
    args = ap.parse_args()

    raw_dir = os.path.join(args.out, "edgar_raw")
    pick_dir = os.path.join(args.out, "edgar")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(pick_dir, exist_ok=True)
    cand_path = os.path.join(args.out, "edgar_candidates.csv")

    firms = pd.read_csv(args.firms)
    if args.sample:
        firms = firms.sample(args.sample, random_state=args.seed)
    done = set()
    if os.path.exists(cand_path):
        done = set(pd.read_csv(cand_path)["cik"].unique())
    todo = [c for c in firms["cik"] if c not in done]
    print(f"Firms: {len(firms):,}  already done: {len(firms) - len(todo):,}  to do: {len(todo):,}")

    sec = Sec(args.email)
    header = not os.path.exists(cand_path)
    with zipfile.ZipFile(args.submissions) as zf:
        names = set(zf.namelist())
        for k, cik in enumerate(todo, 1):
            rows = []
            for form, date, acc in filings_2019(zf, names, int(cik)):
                acc_nd = acc.replace("-", "")
                base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nd}"
                r = sec.get(f"{base}/index.json")
                if r is None:
                    continue
                try:
                    items = r.json()["directory"]["item"]
                except (ValueError, KeyError):
                    continue
                for it in items:
                    name = it.get("name", "")
                    size = int(it.get("size") or 0) if str(it.get("size", "")).isdigit() else 0
                    if not name.lower().endswith(IMG_EXT) or size > 600_000:
                        continue
                    d = os.path.join(raw_dir, str(int(cik)))
                    os.makedirs(d, exist_ok=True)
                    path = os.path.join(d, f"{acc_nd}_{name}")
                    if not os.path.exists(path):
                        img = sec.get(f"{base}/{name}")
                        if img is None:
                            continue
                        open(path, "wb").write(img.content)
                    rec = image_record(path)
                    if rec:
                        rows.append({"cik": int(cik), "form": form, "date": date, "accession": acc,
                                     "file": name, "path": path, **rec})
            if not rows:
                rows = [{"cik": int(cik), "form": None, "date": None, "accession": None, "file": None,
                         "path": None, "w": None, "h": None, "phash": None, "colors": None}]
            pd.DataFrame(rows).to_csv(cand_path, mode="a", header=header, index=False)
            header = False
            if k % 10 == 0 or k == len(todo):
                print(f"  {k:,}/{len(todo):,} firms searched", flush=True)

    cands = pd.read_csv(cand_path)
    cands = cands[cands["cik"].isin(firms["cik"])]
    imgs = cands.dropna(subset=["phash"])

    # images that show up for several different firms are generic, not a firm's logo
    firm_groups = group(imgs.drop_duplicates(["cik", "phash"]).to_dict("records"))
    generic = set()
    for g in firm_groups:
        if len({c["cik"] for c in g["items"]}) >= GENERIC_FIRMS:
            generic.update(c["phash"] for c in g["items"])

    choices = []
    for cik, g in cands.groupby("cik"):
        ok = g.dropna(subset=["phash"])
        ok = ok[~ok["phash"].isin(generic)]
        if ok.empty:
            choices.append({"cik": cik, "confidence": "none", "n_images": 0})
            continue
        groups = group(ok.to_dict("records"))
        for gr in groups:
            gr["n_filings"] = len({c["accession"] for c in gr["items"]})
            aspect = max(c["w"] / c["h"] for c in gr["items"])
            gr["logo_shaped"] = 0.8 <= aspect <= 8
        groups.sort(key=lambda gr: (gr["n_filings"], gr["logo_shaped"],
                                    -min(c["w"] * c["h"] for c in gr["items"])), reverse=True)
        best = groups[0]
        rep = max(best["items"], key=lambda c: c["w"] * c["h"])
        conf = "high" if best["n_filings"] >= 3 else "medium" if best["n_filings"] == 2 else "low"
        out_png = os.path.join(pick_dir, f"{cik}.png")
        try:
            Image.open(rep["path"]).convert("RGBA").save(out_png)
        except Exception:
            out_png = None
        choices.append({"cik": cik, "confidence": conf, "n_images": len(ok),
                        "n_filings_in_group": best["n_filings"], "n_groups": len(groups),
                        "chosen_file": rep["file"], "chosen_form": rep["form"],
                        "w": rep["w"], "h": rep["h"], "colors": rep["colors"],
                        "phash": rep["phash"], "logo_path": out_png})
    ch = pd.DataFrame(choices).merge(firms[["cik", "name", "sector", "outcome_2024"]], on="cik", how="left")
    ch.to_csv(os.path.join(args.out, "edgar_choice.csv"), index=False)

    cells = []
    for r in ch.sort_values(["confidence", "sector"]).itertuples():
        img = (f'<img src="edgar/{r.cik}.png">' if isinstance(r.logo_path, str) else "<div class=none>no image</div>")
        cells.append(f'<div class="c {r.confidence}">{img}<b>{html.escape(str(r.name))}</b>'
                     f'<span>{r.confidence} · {html.escape(str(r.sector))} · {r.outcome_2024}</span></div>')
    page = ("<!doctype html><meta charset=utf-8><title>EDGAR logo review</title><style>"
            "body{font-family:sans-serif;margin:16px}.g{display:grid;grid-template-columns:repeat(auto-fill,180px);gap:12px}"
            ".c{border:1px solid #ddd;padding:8px;font-size:12px}.c img{width:160px;height:90px;object-fit:contain;background:#fff;display:block}"
            ".none{width:160px;height:90px;background:#f4f4f4;display:flex;align-items:center;justify-content:center;color:#999}"
            ".high{border-color:#3a3}.medium{border-color:#e90}.low{border-color:#d55}span{display:block;color:#666}</style>"
            "<h3>Chosen 2019 logos from EDGAR filings (green = high, orange = medium, red = low)</h3>"
            f"<div class=g>{''.join(cells)}</div>")
    open(os.path.join(args.out, "edgar_review.html"), "w").write(page)

    n = len(ch)
    print(f"\nFirms: {n}")
    for c in ["high", "medium", "low", "none"]:
        k = (ch["confidence"] == c).sum()
        print(f"  {c:7s} {k:5,} ({k / n:.0%})")
    print("\nBy outcome (share with a high or medium logo):")
    for o, g in ch.groupby("outcome_2024"):
        print(f"  {o:12s} n={len(g):4}  {g['confidence'].isin(['high', 'medium']).mean():.0%}")
    print(f"\nGeneric images dropped (shared by {GENERIC_FIRMS}+ firms): {len(generic)}")
    print(f"Review page: {os.path.join(args.out, 'edgar_review.html')}")


if __name__ == "__main__":
    main()
