"""
Make a compact copy of one year's chosen logos for the (private) GitHub repo.

The analysis always uses the full-resolution files in data/logos[_<year>]/final/.
This copy is for browsing and sharing inside the team: each logo is resized to at
most 400 px on its longer side and saved as an optimized PNG.

Output (logos/<year>/):
  <cik>.png      one logo per firm (high / medium / low confidence; "none" has no file)
  index.csv      cik, base-year name, sector, outcome, confidence, in_logo_sample
  review.html    the same review page as logo_score.py, pointing at these files
  README.md      source and terms of use

Usage:
  python3 export_logos.py              # 2019
  python3 export_logos.py --year 2022
"""
import argparse
import html
import os
from pathlib import Path

import pandas as pd
from PIL import Image

MAX_SIDE = 400

README = """# Company logos, base year {year}

Logos extracted from each firm's own SEC EDGAR filings (8-K, DEF 14A, 10-K) dated in or
around {year}, chosen automatically and checked by hand on a random sample.

- `index.csv`: one row per firm; `in_logo_sample` marks the logos used in the analysis
  (high or medium confidence and not part of a slide).
- Images are resized copies (at most {max_side} px); the analysis uses the full-resolution originals.

All logos are trademarks of their respective owners. They are included only for
non-commercial academic research (CSE 519, Stony Brook University) and are not licensed
for any other use. This repository must stay private.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()
    Y = args.year
    src = Path("data/logos" if Y == 2019 else f"data/logos_{Y}")
    out = Path("logos") / str(Y)
    out.mkdir(parents=True, exist_ok=True)

    ch = pd.read_csv(src / "logo_choice.csv")
    name_col = f"name_{Y}"
    slide = ch["logo_in_slide"].astype(str).str.lower().eq("true")
    ch["in_logo_sample"] = ch["confidence"].isin(["high", "medium"]) & ~slide

    n, size = 0, 0
    for r in ch.itertuples():
        if not isinstance(r.logo_path, str) or not os.path.exists(r.logo_path):
            continue
        dst = out / f"{r.cik}.png"
        im = Image.open(r.logo_path).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        im.save(dst, optimize=True)
        n += 1
        size += dst.stat().st_size

    idx = ch[["cik", name_col, "sector", "outcome_2024", "confidence", "logo_in_slide", "in_logo_sample"]]
    idx.rename(columns={name_col: "name_base"}).to_csv(out / "index.csv", index=False)

    order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    cells = []
    for r in ch.assign(o=ch["confidence"].map(order)).sort_values(["o", "sector"]).itertuples():
        has = (out / f"{r.cik}.png").exists()
        img = f'<img src="{r.cik}.png" loading="lazy">' if has else "<div class=none>no logo</div>"
        cells.append(f'<div class="c {r.confidence}">{img}<b>{html.escape(str(getattr(r, name_col)))}</b>'
                     f'<span>{r.confidence}{" · in sample" if r.in_logo_sample else ""}</span>'
                     f'<span>{html.escape(str(r.sector))} · {r.outcome_2024}</span></div>')
    page = ("<!doctype html><meta charset=utf-8><title>Logos " + str(Y) + "</title><style>"
            "body{font-family:sans-serif;margin:16px}.g{display:grid;grid-template-columns:repeat(auto-fill,190px);gap:12px}"
            ".c{border:2px solid #ddd;padding:8px;font-size:12px}.c img{width:170px;height:90px;object-fit:contain;background:#fff;display:block}"
            ".none{width:170px;height:90px;background:#f4f4f4;display:flex;align-items:center;justify-content:center;color:#999}"
            ".high{border-color:#3a3}.medium{border-color:#e90}.low{border-color:#d55}span{display:block;color:#666}</style>"
            f"<h3>{Y} logos (green = high, orange = medium, red = low)</h3><div class=g>{''.join(cells)}</div>")
    (out / "review.html").write_text(page)
    (out / "README.md").write_text(README.format(year=Y, max_side=MAX_SIDE))
    print(f"{n:,} logos -> {out}/  ({size / 1e6:.0f} MB)  in analysis sample: {int(ch['in_logo_sample'].sum()):,}")


if __name__ == "__main__":
    main()
