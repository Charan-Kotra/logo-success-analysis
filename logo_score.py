"""
Step B of logo extraction: choose each firm's base-year logo from the collected images.

For every firm:
  1. drop tiny images (under 40 px on a side) and images shared by 3+ firms
     (wire-service logos, auditor logos, icons)
  2. group near-identical images (pHash) and count how many filings each
     group appears in
  3. for each group's best image:
       CLIP zero-shot: probability that it is a company logo, against a
         handwritten signature, a chart, a diagram, a photo, an icon, a page of text
       OCR + name match: does the text in the image match the firm's base-year name?
  4. keep groups that look like a logo; pick the best by
       filings it repeats in + name match + logo probability
  5. crop whitespace, save, and assign a confidence:
       high    logo-like and (3+ filings, or 2+ filings with a name match,
               or a name match with logo probability >= 0.8)
       medium  logo-like and (2 filings, or a name match)
       low     logo-like, seen once, no name match
       none    nothing logo-like

Reads data/logos/candidates_v3.csv for 2019, data/logos_<year>/candidates.csv otherwise.

Outputs (data/logos/ for 2019, data/logos_<year>/ otherwise):
  final/<cik>.png          chosen logo, whitespace cropped
  logo_choice.csv          one row per firm with scores and confidence
  group_scores.csv         every candidate group with its scores
  logo_review.html         chosen logos to check by eye

Usage:
  pip install torch transformers easyocr
  python3 logo_score.py                  # 2019
  python3 logo_score.py --year 2022
"""
import argparse
import difflib
import html
import os
import re

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageChops

MIN_SIDE = 40
GENERIC_FIRMS = 3
HAMMING = 6
LOGO_PROMPTS = ["a company logo", "a brand logo with the company name"]
OTHER_PROMPTS = ["a handwritten signature", "a chart or graph", "a diagram or flowchart",
                 "a photograph", "a small icon", "a page of printed text",
                 "a website address written in text",
                 "a presentation slide with a title", "a product photo",
                 "a letterhead with an address", "an official seal or stamp",
                 "a stock certificate"]

# Words that show an image is a slide, a document or another firm's letterhead,
# not a bare logo (found in a 60-logo audit of the first full run).
DOC_WORDS = re.compile(
    r"\b(earnings|presentation|conference call|annual meeting|stockholders|shareholders|investor|"
    r"quarter|fiscal year|q[1-4]\b|[1-4]q\d{2}|news release|press release|webcast|nasdaq:|nyse:|"
    r"dear|street|suite|p\.?o\.? box|telephone|tel\b|fax|accountants|chartered|cpa|llp|"
    r"certificate|seal of)", re.I)
STOP = {"inc", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "lp", "plc",
        "holdings", "holding", "group", "the", "de", "nv", "sa", "ag", "trust", "fund", "and", "of"}


def norm_tokens(s):
    s = re.sub(r"/[a-z]{2}/", " ", str(s).lower())
    toks = re.findall(r"[a-z0-9]+", s)
    return [t for t in toks if t not in STOP and len(t) >= 2]


def name_match(ocr_text, name):
    """True when a distinctive word of the firm's name appears in the image text."""
    want = [t for t in norm_tokens(name) if len(t) >= 3] or norm_tokens(name)
    got = norm_tokens(ocr_text)
    if not want or not got:
        return False, 0.0
    best = 0.0
    for w in want:
        for g in got:
            best = max(best, difflib.SequenceMatcher(None, w, g).ratio())
    joined_w, joined_g = "".join(want), "".join(got)
    best = max(best, difflib.SequenceMatcher(None, joined_w, joined_g).ratio())
    return best >= 0.8, round(best, 3)


def on_white(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def crop_whitespace(im, pad=6):
    bg = Image.new("RGB", im.size, im.getpixel((0, 0)))
    box = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 25 else 0).getbbox()
    if not box:
        return im
    l, t, r, b = box
    return im.crop((max(0, l - pad), max(0, t - pad), min(im.width, r + pad), min(im.height, b + pad)))


def quality(im):
    """Usable-detail measures on the cropped logo.
    short_side: pixels on the shorter side of the logo itself (after cropping)
    sharpness:  variance of the Laplacian on a 256-px-wide gray copy; low = blurry
    """
    gray = np.asarray(im.convert("L"), dtype=float)
    short_side = min(gray.shape)
    if gray.shape[1] > 256:
        h = max(1, round(gray.shape[0] * 256 / gray.shape[1]))
        gray = np.asarray(im.convert("L").resize((256, h)), dtype=float)
    if min(gray.shape) < 3:
        return short_side, 0.0
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    return short_side, float(lap.var())


def quality_label(short_side, sharpness, aspect):
    if short_side < 40 or sharpness < 50:
        return "poor"
    if short_side < 80 or sharpness < 200 or aspect > 6:
        return "fair"
    return "good"


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hash_bytes(hex_str):
    return np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)


def group(records):
    groups = []
    if not records:
        return groups
    nbytes = len(_hash_bytes(records[0]["phash"]))
    reps = np.empty((len(records), nbytes), dtype=np.uint8)
    n = 0
    show = len(records) > 5000
    for i, c in enumerate(records, 1):
        v = _hash_bytes(c["phash"])
        if n:
            dist = _POPCOUNT[np.bitwise_xor(reps[:n], v)].sum(axis=1, dtype=np.int32)
            hit = np.flatnonzero(dist <= HAMMING)
            if hit.size:
                groups[hit[0]]["items"].append(c)
                continue
        reps[n] = v
        n += 1
        groups.append({"hash": c["phash"], "items": [c]})
        if show and i % 5000 == 0:
            print(f"  grouped {i:,}/{len(records):,} images, {n:,} groups", flush=True)
    if show:
        print(f"  grouped {len(records):,} images into {n:,} groups", flush=True)
    return groups


class Clip:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        prompts = LOGO_PROMPTS + OTHER_PROMPTS
        with torch.no_grad():
            t = self.proc(text=prompts, return_tensors="pt", padding=True)
            out = self.model.text_model(input_ids=t["input_ids"].to(self.dev),
                                        attention_mask=t["attention_mask"].to(self.dev))
            tf = self.model.text_projection(out.pooler_output)
            self.text = tf / tf.norm(dim=-1, keepdim=True)

    def logo_prob(self, images, batch=32):
        torch = self.torch
        probs = []
        for i in range(0, len(images), batch):
            with torch.no_grad():
                px = self.proc(images=images[i:i + batch], return_tensors="pt")["pixel_values"].to(self.dev)
                vf = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
                vf = vf / vf.norm(dim=-1, keepdim=True)
                p = (100 * vf @ self.text.T).softmax(dim=-1)[:, :len(LOGO_PROMPTS)].sum(dim=-1)
            probs += p.float().cpu().tolist()
        return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--firms", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min_logo_prob", type=float, default=0.5)
    ap.add_argument("--no_ocr", action="store_true", help="skip OCR (faster, no name match)")
    args = ap.parse_args()
    Y = args.year
    args.out = args.out or ("data/logos" if Y == 2019 else f"data/logos_{Y}")
    args.candidates = args.candidates or os.path.join(args.out, "candidates_v3.csv" if Y == 2019 else "candidates.csv")
    args.firms = args.firms or ("data/edgar/firms_2019.csv" if Y == 2019 else f"data/edgar_{Y}/firms_{Y}.csv")
    name_col = f"name_{Y}"

    firms = pd.read_csv(args.firms)
    name19 = firms.set_index("cik")[name_col].to_dict()
    c = pd.read_csv(args.candidates)
    all_ciks = c["cik"].unique()
    imgs = c.dropna(subset=["phash"])
    imgs = imgs[(imgs["w"] >= MIN_SIDE) & (imgs["h"] >= MIN_SIDE)].copy()

    generic = set()
    for g in group(imgs.drop_duplicates(["cik", "phash"]).to_dict("records")):
        if len({r["cik"] for r in g["items"]}) >= GENERIC_FIRMS:
            generic.update(r["phash"] for r in g["items"])
    imgs = imgs[~imgs["phash"].isin(generic)]
    print(f"Firms: {len(all_ciks):,}  usable images: {len(imgs):,}  generic dropped: {len(generic):,}")

    groups = []
    for cik, g in imgs.groupby("cik"):
        for gr in group(g.to_dict("records")):
            rep = max(gr["items"], key=lambda r: r["w"] * r["h"])
            groups.append({"cik": cik, "n_filings": len({r["accession"] for r in gr["items"]}),
                           "rep_path": rep["path"], "rep_file": rep["file"], "rep_form": rep["form"],
                           "window": rep["window"], "w": rep["w"], "h": rep["h"]})
    gs = pd.DataFrame(groups)
    print(f"Candidate groups to score: {len(gs):,}")

    # Load each image once, measure it at full size, then keep only a small copy
    # (keeping every full-size image in memory crashed an 8 GB laptop).
    print("Preparing images...")
    ims, q, aspects = [], [], []
    for p in gs["rep_path"]:
        try:
            im = crop_whitespace(on_white(p))
        except Exception:
            im = Image.new("RGB", (64, 64), "white")
        q.append(quality(im))
        aspects.append(im.width / max(1, im.height))
        im.thumbnail((448, 448))
        ims.append(im)
    print("Scoring with CLIP...")
    clip = Clip()
    gs["logo_prob"] = clip.logo_prob(ims, batch=16)

    gs["ocr_text"], gs["name_match"], gs["name_sim"] = "", False, 0.0
    # reuse OCR text from an earlier run (same image file -> same text)
    prev_path = os.path.join(args.out, "group_scores.csv")
    ocr_cache = {}
    if os.path.exists(prev_path):
        prev = pd.read_csv(prev_path, usecols=["rep_path", "ocr_text"])
        prev = prev[prev["ocr_text"].notna()]
        ocr_cache = dict(zip(prev["rep_path"], prev["ocr_text"].astype(str)))
        print(f"Reusing OCR text for {len(ocr_cache):,} images from the previous run")
    if not args.no_ocr:
        import easyocr
        import torch
        reader = None
        todo_ocr = gs.index[gs["logo_prob"] >= 0.3]
        print(f"Reading text with OCR ({len(todo_ocr):,} logo-like groups)...")
        for n, i in enumerate(todo_ocr, 1):
            if gs.at[i, "rep_path"] in ocr_cache:
                text = ocr_cache[gs.at[i, "rep_path"]]
            else:
                if reader is None:
                    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
                im = ims[i]
                if max(im.size) < 300:
                    im = im.resize((im.width * 2, im.height * 2))
                text = " ".join(reader.readtext(np.array(im), detail=0))
            if n % 2000 == 0:
                print(f"  OCR {n:,}/{len(todo_ocr):,}", flush=True)
            ok, sim = name_match(text, name19.get(gs.at[i, "cik"], ""))
            gs.at[i, "ocr_text"], gs.at[i, "name_match"], gs.at[i, "name_sim"] = text, ok, sim
    gs["short_side"] = [a for a, _ in q]
    gs["sharpness"] = [round(b, 1) for _, b in q]
    gs["aspect"] = aspects
    gs["slide_shape"] = [(1.2 <= a <= 1.9) and (w >= 600) for a, w in zip(aspects, gs["w"])]
    gs["doc_words"] = gs["ocr_text"].fillna("").map(lambda t: bool(DOC_WORDS.search(t)))
    gs["quality"] = [quality_label(a, b, c) for (a, b), c in zip(q, gs["aspect"])]
    gs["banner"] = gs["aspect"] > 6
    # a slide or document is ineligible unless the firm's own name is in it
    gs["eligible"] = (gs["logo_prob"] >= args.min_logo_prob) & ~(gs["doc_words"] & ~gs["name_match"])
    # a large slide-shaped image that does carry the name: keep, but flag
    gs["logo_in_slide"] = gs["doc_words"] | gs["slide_shape"]
    gs["score"] = (gs["n_filings"] + 2 * gs["name_match"].astype(int) + gs["logo_prob"]
                   - gs["banner"].astype(int) + 0.5 * (gs["quality"] == "good").astype(int)
                   - 1.5 * gs["logo_in_slide"].astype(int))
    gs.drop(columns=[]).to_csv(os.path.join(args.out, "group_scores.csv"), index=False)

    final_dir = os.path.join(args.out, "final")
    os.makedirs(final_dir, exist_ok=True)
    rows = []
    for cik in all_ciks:
        g = gs[(gs["cik"] == cik) & gs["eligible"]] if len(gs) else gs
        if g.empty:
            rows.append({"cik": cik, "confidence": "none"})
            continue
        b = g.sort_values("score", ascending=False).iloc[0]
        nf, nm, lp = b["n_filings"], bool(b["name_match"]), b["logo_prob"]
        if b["logo_in_slide"]:
            conf = "medium" if nm else "low"     # logo sits inside a slide or document page
        elif nf >= 3 or (nf >= 2 and nm) or (nm and lp >= 0.8):
            conf = "high"
        elif nf == 2 or nm:
            conf = "medium"
        else:
            conf = "low"
        path = os.path.join(final_dir, f"{cik}.png")
        crop_whitespace(on_white(b["rep_path"])).save(path)      # full-resolution copy
        rows.append({"cik": cik, "confidence": conf, "n_filings": nf, "name_match": nm,
                     "name_sim": b["name_sim"], "logo_prob": round(lp, 3), "banner": bool(b["banner"]),
                     "window": b["window"], "form": b["rep_form"], "ocr_text": b["ocr_text"],
                     "quality": b["quality"], "short_side": b["short_side"], "sharpness": b["sharpness"],
                     "logo_in_slide": bool(b["logo_in_slide"]),
                     "logo_path": path})
    ch = pd.DataFrame(rows).merge(firms[["cik", name_col, "name", "sector", "outcome_2024"]],
                                  on="cik", how="left")
    ch["base_year"] = Y
    ch.to_csv(os.path.join(args.out, "logo_choice.csv"), index=False)

    order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    cells = []
    for r in ch.assign(o=ch["confidence"].map(order)).sort_values(["o", "sector"]).itertuples():
        img = (f'<img src="final/{r.cik}.png">' if isinstance(r.logo_path, str)
               else "<div class=none>no logo</div>")
        extra = "" if r.confidence == "none" else f" · {int(r.n_filings)} filings · p={r.logo_prob}" + \
            (" · name ✓" if r.name_match else "") + f" · {r.quality} quality"
        cells.append(f'<div class="c {r.confidence}">{img}<b>{html.escape(str(getattr(r, name_col)))}</b>'
                     f'<span>{r.confidence}{extra}</span><span>{html.escape(str(r.sector))} · {r.outcome_2024}</span></div>')
    page = ("<!doctype html><meta charset=utf-8><title>Logo review v2</title><style>"
            "body{font-family:sans-serif;margin:16px}.g{display:grid;grid-template-columns:repeat(auto-fill,190px);gap:12px}"
            ".c{border:2px solid #ddd;padding:8px;font-size:12px}.c img{width:170px;height:90px;object-fit:contain;background:#fff;display:block}"
            ".none{width:170px;height:90px;background:#f4f4f4;display:flex;align-items:center;justify-content:center;color:#999}"
            ".high{border-color:#3a3}.medium{border-color:#e90}.low{border-color:#d55}span{display:block;color:#666}</style>"
            f"<h3>Chosen {Y} logos (green = high, orange = medium, red = low; name is the {Y} name)</h3>"
            f"<div class=g>{''.join(cells)}</div>")
    open(os.path.join(args.out, "logo_review.html"), "w").write(page)

    n = len(ch)
    print(f"\nFirms: {n}")
    for k in ["high", "medium", "low", "none"]:
        m = (ch["confidence"] == k).sum()
        print(f"  {k:7s} {m:5,} ({m / n:.0%})")
    hm = ch["confidence"].isin(["high", "medium"])
    print(f"  high or medium: {hm.mean():.0%}")
    ok = ch[ch["confidence"] != "none"]
    if len(ok):
        print("\nImage quality of chosen logos:")
        for k in ["good", "fair", "poor"]:
            print(f"  {k:5s} {(ok['quality'] == k).mean():.0%}")
        print("  poor-quality share by outcome:",
              ", ".join(f"{o} {(g['quality'] == 'poor').mean():.0%}" for o, g in ok.groupby("outcome_2024")))
    print("\nBy outcome (high or medium):")
    for o, g in ch.groupby("outcome_2024"):
        print(f"  {o:12s} n={len(g):5,}  {g['confidence'].isin(['high', 'medium']).mean():.0%}")
    print(f"\nReview page: {os.path.join(args.out, 'logo_review.html')}")


if __name__ == "__main__":
    main()
