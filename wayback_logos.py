"""
Second logo source: each firm's own website as archived by the Wayback Machine in the base year.

Why: about a third of firms have no usable logo from their SEC filings (mostly small firms
that later exited, whose filings are plain text). Their 2019 homepage usually shows the logo.
The archive's timestamp makes it a true base-year logo, not today's.

Stage "collect" (network, archive.org):
  for each firm with a website domain
    1. find the archived homepage snapshot closest to July 1 of the base year
       (searching Jan 1 of the base year to Jun 30 of the next year)
    2. read the page and list logo candidates: <img> tags whose src / alt / class / id
       mention "logo" or the firm's name, header images, inline SVG logos, og:image,
       apple-touch-icon
    3. download the best 4 candidates from the archive
Stage "score" (CLIP + OCR, same checks as logo_score.py):
  pick each firm's logo by: logo hint in the page + firm name read in the image + CLIP logo
  probability; assign high / medium / low confidence.
Stage "combine":
  writes data/logos[_Y]/logo_choice_all.csv = EDGAR logos, plus Wayback logos for firms
  without a usable EDGAR logo (column logo_source). logo_embed.py uses this file if present.
  Also reports how often the two sources agree for firms that have both (a validation check).

Which firms (--which):
  missing   firms without a usable EDGAR logo (default)
  both      missing + a random sample of --check firms that DO have one (for the agreement check)

Archive.org etiquette: one request at a time, --rate requests/second (default 1), a
User-Agent with your name and email, and a pause on 429/503.

Usage:
  python3 wayback_logos.py --stage collect --which both --name "First Last" --email you@school.edu
  python3 wayback_logos.py --stage score
  python3 wayback_logos.py --stage combine
Needs data/analysis/firms_master_<year>.csv (merge_all.py) for the domains.
A rerun skips firms already collected.
"""
import argparse
import html as htmlmod
import io
import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin

import numpy as np
import pandas as pd
import requests
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
COLS = ["cik", "domain", "snapshot", "page_url", "cand_rank", "hint", "kind", "src", "path", "w", "h", "error"]


# ------------------------------------------------------------------ helpers
def tokens(name):
    stop = {"inc", "corp", "corporation", "co", "company", "ltd", "llc", "lp", "plc", "holdings",
            "holding", "group", "the", "and", "of", "trust", "bancorp", "international"}
    return [t for t in re.findall(r"[a-z0-9]+", str(name).lower()) if t not in stop and len(t) >= 3]


class Archive:
    def __init__(self, name, email, rate):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": f"Stony Brook CSE519 research {name} {email}"})
        self.gap = 1.0 / rate
        self.last = 0.0
        self.count = 0

    def get(self, url, timeout=40):
        for attempt in range(4):
            wait = self.gap - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.count += 1
            try:
                r = self.s.get(url, timeout=timeout)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503, 502, 504):
                time.sleep(30 * (attempt + 1))
                self.gap = min(self.gap * 1.5, 5.0)
                continue
            return None
        return None


def best_snapshot(arc, domain, year):
    target = int(f"{year}0701000000")
    for host in (domain, f"www.{domain}"):
        url = ("https://web.archive.org/cdx/search/cdx?url=" + quote(host) +
               f"&from={year}0101&to={year + 1}0630&output=json&filter=statuscode:200"
               "&filter=mimetype:text/html&collapse=timestamp:6&limit=40")
        r = arc.get(url)
        if r is None:
            continue
        try:
            rows = r.json()
        except ValueError:
            continue
        if len(rows) < 2:
            continue
        head, data = rows[0], rows[1:]
        ti, oi = head.index("timestamp"), head.index("original")
        best = min(data, key=lambda row: abs(int(row[ti]) - target))
        return best[ti], best[oi]
    return None, None


class LogoParser(HTMLParser):
    """Collects logo candidates from one HTML page."""

    def __init__(self, name_toks):
        super().__init__(convert_charrefs=True)
        self.name_toks = name_toks
        self.cands = []          # (hint score, kind, url or svg text)
        self.stack = []          # open tags with their "logo-ness"
        self.n_img = 0
        self.svg_depth = 0
        self.svg_buf = []
        self.svg_hint = 0

    def _attr_hint(self, attrs):
        text = " ".join(str(v) for k, v in attrs if k in ("src", "alt", "class", "id", "title", "href",
                                                          "data-src", "aria-label")).lower()
        h = 0.0
        if "logo" in text:
            h += 3
        if any(t in text for t in self.name_toks):
            h += 1.5
        if any(w in text for w in ("icon", "sprite", "social", "facebook", "twitter", "linkedin",
                                   "youtube", "instagram", "banner", "slide", "hero", "photo", "award")):
            h -= 2
        return h

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.svg_depth:
            self.svg_depth += 1
            self.svg_buf.append(self.get_starttag_text() or "")
            return
        ctx = sum(1 for _, hint in self.stack if hint > 0)
        in_header = any(t in ("header", "nav") for t, _ in self.stack)
        if tag == "svg":
            self.svg_depth = 1
            self.svg_buf = [self.get_starttag_text() or "<svg>"]
            self.svg_hint = self._attr_hint(attrs) + (2 if ctx else 0) + (1 if in_header else 0)
            return
        if tag == "img":
            src = a.get("src") or a.get("data-src") or ""
            if src and not src.startswith("data:"):
                self.n_img += 1
                h = self._attr_hint(attrs) + (2 if ctx else 0) + (1 if in_header else 0) + (1 if self.n_img <= 3 else 0)
                self.cands.append((h, "img", src))
        elif tag == "meta" and a.get("property", "").lower() == "og:image" and a.get("content"):
            self.cands.append((0.5, "og_image", a["content"]))
        elif tag == "link" and "apple-touch-icon" in a.get("rel", "").lower() and a.get("href"):
            self.cands.append((0.5, "touch_icon", a["href"]))
        if tag not in ("br", "hr", "input", "meta", "link", "img"):
            self.stack.append((tag, self._attr_hint(attrs) if tag in ("a", "div", "span", "header", "figure") else 0))

    def handle_endtag(self, tag):
        if self.svg_depth:
            self.svg_buf.append(f"</{tag}>")
            self.svg_depth -= 1
            if self.svg_depth == 0:
                svg = "".join(self.svg_buf)
                if self.svg_hint > 0 and 200 < len(svg) < 200_000:
                    self.cands.append((self.svg_hint, "inline_svg", svg))
            return
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_startendtag(self, tag, attrs):
        if self.svg_depth:
            self.svg_buf.append(self.get_starttag_text() or "")
            return
        self.handle_starttag(tag, attrs)
        if tag not in ("br", "hr", "input", "meta", "link", "img") and self.stack and self.stack[-1][0] == tag:
            self.stack.pop()

    def handle_data(self, data):
        if self.svg_depth:
            self.svg_buf.append(htmlmod.escape(data))


def svg_to_png(svg_bytes):
    try:
        import cairosvg
        return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_bytes, output_width=600))).convert("RGBA")
    except Exception:
        return None


def save_image(content, path, is_svg):
    im = svg_to_png(content) if is_svg else None
    if im is None and not is_svg:
        try:
            im = Image.open(io.BytesIO(content))
            im.load()
        except Exception:
            return None
    if im is None:
        return None
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    if im.width < 16 or im.height < 16 or im.width * im.height > 6_000_000:
        return None
    im.save(path)
    return im.size


# ------------------------------------------------------------------ stage: collect
def collect(args, firms, out):
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cand_path = out / "candidates.csv"
    done = set(pd.read_csv(cand_path, usecols=["cik"])["cik"]) if cand_path.exists() else set()
    todo = firms[~firms["cik"].isin(done)]
    print(f"collect: {len(firms):,} firms, {len(done):,} done, {len(todo):,} to do")
    arc = Archive(args.name, args.email, args.rate)
    t0 = time.time()
    header = not cand_path.exists()
    for n, f in enumerate(todo.itertuples(index=False), 1):
        rows = []
        base = {"cik": f.cik, "domain": f.domain, "snapshot": None, "page_url": None, "error": ""}
        ts, orig = best_snapshot(arc, f.domain, args.year)
        if ts is None:
            rows.append({**base, "error": "no_snapshot"})
        else:
            page = arc.get(f"https://web.archive.org/web/{ts}id_/{orig}")
            if page is None:
                rows.append({**base, "snapshot": ts, "error": "page_failed"})
            else:
                p = LogoParser(tokens(f.name_base))
                try:
                    p.feed(page.text[:2_000_000])
                except Exception:
                    pass
                cands = sorted(p.cands, key=lambda c: -c[0])
                seen, kept = set(), []
                for h, kind, val in cands:
                    key = val[:300]
                    if key in seen:
                        continue
                    seen.add(key)
                    kept.append((h, kind, val))
                    if len(kept) >= args.per_firm:
                        break
                if not kept:
                    rows.append({**base, "snapshot": ts, "page_url": orig, "error": "no_candidates"})
                d = raw / str(f.cik)
                d.mkdir(exist_ok=True)
                for rank, (h, kind, val) in enumerate(kept):
                    path = d / f"{rank}.png"
                    row = {**base, "snapshot": ts, "page_url": orig, "cand_rank": rank, "hint": h, "kind": kind,
                           "src": val[:300] if kind != "inline_svg" else "inline_svg", "path": None}
                    if kind == "inline_svg":
                        size = save_image(val.encode("utf-8", "ignore"), path, True)
                    else:
                        absu = urljoin(orig, val)
                        r = arc.get(f"https://web.archive.org/web/{ts}im_/{absu}")
                        size = save_image(r.content, path, absu.lower().split("?")[0].endswith(".svg")) if r else None
                    if size:
                        row.update(path=str(path), w=size[0], h=size[1])
                    else:
                        row["error"] = "image_failed"
                    rows.append(row)
        pd.DataFrame(rows, columns=COLS).to_csv(cand_path, mode="a", header=header, index=False)
        header = False
        if n % 20 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n:,}/{len(todo):,} firms  {arc.count / el:.2f} req/s  ~{el / n * (len(todo) - n) / 60:.0f} min left",
                  flush=True)


# ------------------------------------------------------------------ stage: score
def score(args, firms, out):
    import logo_score as LS          # same CLIP prompts, OCR name match and quality checks
    c = pd.read_csv(out / "candidates.csv")
    c = c[c["path"].notna()].copy()
    names = firms.set_index("cik")["name_base"].to_dict()
    print(f"score: {len(c):,} candidate images for {c['cik'].nunique():,} firms")
    ims, q, asp = [], [], []
    for p in c["path"]:
        try:
            im = LS.crop_whitespace(LS.on_white(p))
        except Exception:
            im = Image.new("RGB", (64, 64), "white")
        q.append(LS.quality(im))
        asp.append(im.width / max(1, im.height))
        im.thumbnail((448, 448))
        ims.append(im)
    c["logo_prob"] = LS.Clip().logo_prob(ims, batch=16)
    import easyocr
    import torch
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    texts, nm, ns = [], [], []
    for im, cik in zip(ims, c["cik"]):
        if max(im.size) < 300:
            im = im.resize((im.width * 2, im.height * 2))
        t = " ".join(reader.readtext(np.array(im), detail=0))
        ok, sim = LS.name_match(t, names.get(cik, ""))
        texts.append(t); nm.append(ok); ns.append(sim)
    c["ocr_text"], c["name_match"], c["name_sim"] = texts, nm, ns
    c["short_side"] = [a for a, _ in q]
    c["sharpness"] = [b for _, b in q]
    c["aspect"] = asp
    c["quality"] = [LS.quality_label(a, b, r) for (a, b), r in zip(q, asp)]
    c["score"] = c["hint"].clip(-2, 5) + 2 * c["name_match"].astype(int) + 2 * c["logo_prob"] \
        - 1.5 * (c["kind"] == "og_image") - 1.0 * (c["kind"] == "touch_icon") + 0.5 * (c["quality"] == "good")
    c.to_csv(out / "candidate_scores.csv", index=False)

    final = out / "final"
    final.mkdir(exist_ok=True)
    rows = []
    for cik, g in c.groupby("cik"):
        g = g[g["logo_prob"] >= 0.4]
        if g.empty:
            rows.append({"cik": cik, "confidence": "none"})
            continue
        b = g.sort_values("score", ascending=False).iloc[0]
        hint_logo = b["hint"] >= 3
        if b["name_match"] and (hint_logo or b["logo_prob"] >= 0.7):
            conf = "high"
        elif (hint_logo and b["logo_prob"] >= 0.6) or b["name_match"]:
            conf = "medium"
        else:
            conf = "low"
        path = final / f"{cik}.png"
        LS.crop_whitespace(LS.on_white(b["path"])).save(path)
        rows.append({"cik": cik, "confidence": conf, "snapshot": b["snapshot"], "kind": b["kind"],
                     "hint": b["hint"], "logo_prob": round(b["logo_prob"], 3), "name_match": bool(b["name_match"]),
                     "name_sim": b["name_sim"], "ocr_text": b["ocr_text"], "quality": b["quality"],
                     "short_side": b["short_side"], "sharpness": b["sharpness"], "logo_path": str(path)})
    ch = pd.DataFrame(rows).merge(firms[["cik", "name_base", "sector", "outcome_2024", "in_logo_sample"]],
                                  on="cik", how="left")
    ch.to_csv(out / "wayback_choice.csv", index=False)
    print(ch["confidence"].value_counts().to_string())

    order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    cells = []
    for r in ch.assign(o=ch["confidence"].map(order)).sort_values(["o", "sector"]).itertuples():
        img = f'<img src="final/{r.cik}.png">' if isinstance(r.logo_path, str) else "<div class=none>none</div>"
        cells.append(f'<div class="c {r.confidence}">{img}<b>{htmlmod.escape(str(r.name_base))}</b>'
                     f'<span>{r.confidence} · {r.kind} · p={r.logo_prob}{" · name ✓" if r.name_match is True else ""}</span>'
                     f'<span>{htmlmod.escape(str(r.sector))} · {r.outcome_2024}</span></div>')
    (out / "wayback_review.html").write_text(
        "<!doctype html><meta charset=utf-8><style>body{font-family:sans-serif}.g{display:grid;"
        "grid-template-columns:repeat(auto-fill,190px);gap:12px}.c{border:2px solid #ddd;padding:8px;font-size:12px}"
        ".c img{width:170px;height:90px;object-fit:contain;background:#fff;display:block}.high{border-color:#3a3}"
        ".medium{border-color:#e90}.low{border-color:#d55}span{display:block;color:#666}</style>"
        f"<h3>Wayback logos {args.year}</h3><div class=g>{''.join(cells)}</div>")
    print(f"review page: {out / 'wayback_review.html'}")


# ------------------------------------------------------------------ stage: combine
def combine(args, firms, out):
    edir = Path("data/logos" if args.year == 2019 else f"data/logos_{args.year}")
    ed = pd.read_csv(edir / "logo_choice.csv")
    slide = ed["logo_in_slide"].astype(str).str.lower().eq("true")
    ed["usable"] = ed["confidence"].isin(["high", "medium"]) & ~slide
    ed["logo_source"] = "edgar"
    wb = pd.read_csv(out / "wayback_choice.csv")
    wb_ok = wb[wb["confidence"].isin(["high", "medium"])]

    # agreement check on firms that have both
    both = ed[ed["usable"]].merge(wb_ok[["cik", "logo_path"]], on="cik", suffixes=("", "_wb"))
    if len(both):
        import logo_score as LS
        clip = LS.Clip()
        torch = clip.torch

        def emb(paths):
            vs = []
            for i in range(0, len(paths), 16):
                ims = [LS.on_white(p) for p in paths[i:i + 16]]
                with torch.no_grad():
                    px = clip.proc(images=ims, return_tensors="pt")["pixel_values"].to(clip.dev)
                    v = clip.model.visual_projection(clip.model.vision_model(pixel_values=px).pooler_output)
                    vs.append((v / v.norm(dim=-1, keepdim=True)).float().cpu().numpy())
            return np.vstack(vs)
        sim = (emb(both["logo_path"].tolist()) * emb(both["logo_path_wb"].tolist())).sum(axis=1)
        both["clip_sim"] = sim
        both[["cik", "clip_sim", "logo_path", "logo_path_wb"]].to_csv(out / "agreement.csv", index=False)
        msg = (f"Agreement (firms with both sources, n={len(both)}): median CLIP similarity {np.median(sim):.2f}; "
               f"share >= 0.85 (same logo): {(sim >= 0.85).mean():.0%}; >= 0.75: {(sim >= 0.75).mean():.0%}")
        print(msg)
        (out / "agreement.txt").write_text(msg + "\nHand-check pairs near the cut-off in agreement.csv.\n")

    add = wb_ok[~wb_ok["cik"].isin(ed.loc[ed["usable"], "cik"])].copy()
    add["logo_source"] = "wayback"
    add["logo_in_slide"] = False
    keep = [c for c in ed.columns if c not in ("usable",)]
    ed2 = ed.copy()
    ed2 = ed2[~ed2["cik"].isin(add["cik"])]
    add = add.rename(columns={"name_base": f"name_{args.year}"})
    for col in keep:
        if col not in add:
            add[col] = np.nan
    allc = pd.concat([ed2[keep], add[keep]], ignore_index=True).sort_values("cik")
    path = edir / "logo_choice_all.csv"
    allc.to_csv(path, index=False)
    usable = allc["confidence"].isin(["high", "medium"]) & ~allc["logo_in_slide"].astype(str).str.lower().eq("true")
    print(f"{path}: usable logos {int(usable.sum()):,} "
          f"(EDGAR {int((usable & allc['logo_source'].eq('edgar')).sum()):,} + "
          f"Wayback {int((usable & allc['logo_source'].eq('wayback')).sum()):,})")
    print("usable share by outcome:")
    fo = firms.set_index("cik")["outcome_2024"]
    u = allc[usable]["cik"]
    for o, g in fo.groupby(fo):
        print(f"  {o:12s} {g.index.isin(u).mean():.0%}")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["collect", "score", "combine"])
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--which", default="missing", choices=["missing", "both"])
    ap.add_argument("--check", type=int, default=300, help="firms WITH an EDGAR logo to fetch for the agreement check")
    ap.add_argument("--name", default=None)
    ap.add_argument("--email", default=None)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--per_firm", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    master = pd.read_csv(f"data/analysis/firms_master_{args.year}.csv", low_memory=False)
    master["cik"] = master["cik"].astype(int)
    firms = master[master["domain"].notna()][["cik", "name_base", "sector", "outcome_2024", "domain", "in_logo_sample"]]
    if args.stage == "collect":
        if not (args.name and args.email):
            raise SystemExit("--name and --email are required for collect (archive.org User-Agent)")
        miss = firms[~firms["in_logo_sample"].astype(bool)]
        pick = miss
        if args.which == "both":
            have = firms[firms["in_logo_sample"].astype(bool)].sample(min(args.check, len(firms)), random_state=519)
            pick = pd.concat([miss, have])
        if args.limit:
            pick = pick.head(args.limit)
        out = Path("data/logos_wayback" if args.year == 2019 else f"data/logos_wayback_{args.year}")
        collect(args, pick, out)
    else:
        out = Path("data/logos_wayback" if args.year == 2019 else f"data/logos_wayback_{args.year}")
        (score if args.stage == "score" else combine)(args, firms, out)
