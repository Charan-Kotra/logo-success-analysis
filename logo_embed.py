"""
Text masking, image embeddings and the sector validation check for one base year's logo set.

Stages (run in order, each one resumes or skips work that is already done):
  mask      EasyOCR finds text boxes in each logo, paints them with the background colour,
            and labels each logo as text_only / symbol_and_text / symbol_only
  review    HTML page of 80 random originals next to their masked versions, for a hand check
  embed     CLIP ViT-B/32 and DINOv2-small embeddings of the original and masked logos
  validate  do logos sit near logos from the same sector? kNN purity vs chance,
            label-permutation test, per-sector lift, and a linear probe

Inputs : data/logos/logo_choice.csv (2019) or data/logos_<year>/logo_choice.csv, and the final/ logos
Outputs: data/embed/ (2019) or data/embed_<year>/  (index.csv, *.npy, masked/, ocr_boxes.jsonl, validation.txt)

Examples:
  python3 logo_embed.py --stage mask --limit 20      # quick test on 20 logos (2019)
  python3 logo_embed.py --stage review
  python3 logo_embed.py --stage all                  # full run, 2019
  python3 logo_embed.py --stage all --year 2022
"""
import argparse
import html
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

YEAR = 2019
NAME_COL = "name_2019"
CHOICE = Path("data/logos/logo_choice.csv")
OUT = Path("data/embed")
MASK_DIR = OUT / "masked"
OCR_LOG = OUT / "ocr_boxes.jsonl"
INDEX = OUT / "index.csv"


def set_year(y):
    global YEAR, NAME_COL, CHOICE, OUT, MASK_DIR, OCR_LOG, INDEX
    YEAR, NAME_COL = y, f"name_{y}"
    CHOICE = Path("data/logos/logo_choice.csv" if y == 2019 else f"data/logos_{y}/logo_choice.csv")
    OUT = Path("data/embed" if y == 2019 else f"data/embed_{y}")
    MASK_DIR, OCR_LOG, INDEX = OUT / "masked", OUT / "ocr_boxes.jsonl", OUT / "index.csv"
SIZE = 224
SYMBOL_MIN_INK = 0.15   # share of logo "ink" that must remain after masking to count as a symbol

MODELS = {
    "clip": ("openai/clip-vit-base-patch32",
             [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    "dino": ("facebook/dinov2-small",
             [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


# ---------------------------------------------------------------- helpers
def analysis_set():
    d = pd.read_csv(CHOICE)
    slide = d["logo_in_slide"].astype(str).str.lower().eq("true")
    a = d[d["confidence"].isin(["high", "medium"]) & ~slide].copy()
    a = a[a["logo_path"].map(lambda p: Path(str(p)).exists())]
    a["cik"] = a["cik"].astype(int)
    return a.sort_values("cik").reset_index(drop=True)


def load_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def ring_color(arr):
    """Median colour of the 2-pixel outer frame = background estimate."""
    ring = np.concatenate([arr[:2].reshape(-1, 3), arr[-2:].reshape(-1, 3),
                           arr[:, :2].reshape(-1, 3), arr[:, -2:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(ring, axis=0))


def ink_mask(arr, bg, tol=40):
    """Pixels that differ clearly from the background = the drawn part of the logo."""
    return np.abs(arr.astype(np.int16) - np.array(bg, dtype=np.int16)).sum(axis=2) > tol


def square(im, size=SIZE):
    """Pad to a square on white (keeps the aspect ratio), then resize."""
    w, h = im.size
    m = max(w, h)
    canvas = Image.new("RGB", (m, m), (255, 255, 255))
    canvas.paste(im, ((m - w) // 2, (m - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_ocr_log():
    if not OCR_LOG.exists():
        return pd.DataFrame(columns=["cik"])
    with open(OCR_LOG) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows).drop_duplicates("cik", keep="last")


# ---------------------------------------------------------------- stage: mask
def run_mask(a, limit=None, max_side=900):
    import easyocr
    import torch

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    done = set(read_ocr_log()["cik"].astype(int)) if OCR_LOG.exists() else set()
    todo = a[~a["cik"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"mask: {len(done)} already done, {len(todo)} to process")
    if todo.empty:
        return

    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    t0 = time.time()
    with open(OCR_LOG, "a") as log:
        for n, row in enumerate(todo.itertuples(index=False), 1):
            rec = {"cik": int(row.cik), "masked_path": "", "text": "", "n_boxes": 0,
                   "text_share": 0.0, "ink_left": 1.0, "error": ""}
            try:
                im = load_rgb(row.logo_path)
                arr = np.asarray(im)
                # small logos are upscaled for OCR, huge ones downscaled for speed
                s = 2.0 if max(im.size) < 300 else min(1.0, max_side / max(im.size))
                work = im if s == 1.0 else im.resize(
                    (max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)

                boxes, words = [], []
                for pts, text, conf in reader.readtext(np.asarray(work)):
                    if conf < 0.3 or not any(ch.isalnum() for ch in text):
                        continue
                    xs = [p[0] / s for p in pts]
                    ys = [p[1] / s for p in pts]
                    pad = 0.08 * (max(ys) - min(ys)) + 2
                    boxes.append((max(0, min(xs) - pad), max(0, min(ys) - pad),
                                  min(im.width - 1, max(xs) + pad), min(im.height - 1, max(ys) + pad)))
                    words.append(text)

                bg = ring_color(arr)
                ink = ink_mask(arr, bg)
                in_box = np.zeros(ink.shape, dtype=bool)
                for x0, y0, x1, y1 in boxes:
                    in_box[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True

                masked = im.copy()
                draw = ImageDraw.Draw(masked)
                for b in boxes:
                    draw.rectangle(b, fill=bg)
                out = MASK_DIR / f"{row.cik}.png"
                masked.save(out)

                total = int(ink.sum())
                rec.update(masked_path=str(out), text=" ".join(words), n_boxes=len(boxes),
                           text_share=round(float(in_box.mean()), 4),
                           ink_left=round(float((ink & ~in_box).sum() / total), 4) if total else 0.0)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:200]
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if n % 50 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(todo)}  {rate:.2f}s/logo  ~{rate * (len(todo) - n) / 60:.0f} min left",
                      flush=True)


def build_index(a):
    ocr = read_ocr_log()
    if ocr.empty or "masked_path" not in ocr:
        raise SystemExit("Run --stage mask first.")
    idx = a.merge(ocr, on="cik", how="left")
    missing = idx["masked_path"].isna() | (idx["masked_path"] == "")
    idx["mask_failed"] = missing
    idx.loc[missing, "masked_path"] = idx.loc[missing, "logo_path"]
    idx["n_boxes"] = idx["n_boxes"].fillna(0).astype(int)
    idx["ink_left"] = idx["ink_left"].fillna(1.0)
    idx["symbol_type"] = np.where(idx["n_boxes"] == 0, "symbol_only",
                          np.where(idx["ink_left"] < SYMBOL_MIN_INK, "text_only", "symbol_and_text"))
    idx["base_year"] = YEAR
    idx = idx.rename(columns={NAME_COL: "name_base"})
    keep = ["cik", "base_year", "name_base", "sector", "outcome_2024", "confidence", "logo_path", "masked_path",
            "mask_failed", "text", "n_boxes", "text_share", "ink_left", "symbol_type"]
    idx = idx[keep].sort_values("cik").reset_index(drop=True)
    idx.to_csv(INDEX, index=False)
    return idx


# ---------------------------------------------------------------- stage: review
def run_review(n=80, seed=7):
    ocr = read_ocr_log()
    ocr = ocr[ocr["error"] == ""]
    a = analysis_set()[["cik", NAME_COL, "sector", "logo_path"]].rename(columns={NAME_COL: "name_base"})
    d = a.merge(ocr, on="cik").sample(min(n, len(ocr)), random_state=seed)
    d["symbol_type"] = np.where(d["n_boxes"] == 0, "symbol_only",
                        np.where(d["ink_left"] < SYMBOL_MIN_INK, "text_only", "symbol_and_text"))
    rows = []
    for r in d.itertuples(index=False):
        orig = os.path.relpath(r.logo_path, OUT)
        mask = os.path.relpath(r.masked_path, OUT)
        rows.append(
            f"<tr><td>{r.cik}<br><b>{html.escape(str(r.name_base))}</b><br>{html.escape(str(r.sector))}</td>"
            f"<td><img src='{orig}'></td><td><img src='{mask}'></td>"
            f"<td>{r.symbol_type}<br>ink left {r.ink_left:.2f}<br>text: {html.escape(str(r.text))[:80]}</td></tr>")
    page = ("<html><head><style>body{font-family:sans-serif}td{border-bottom:1px solid #ccc;padding:6px;"
            "vertical-align:middle}img{max-width:260px;max-height:110px;background:#eee}</style></head><body>"
            "<h3>Masking check: original vs text-masked</h3><table>"
            "<tr><th>firm</th><th>original</th><th>masked</th><th>label</th></tr>"
            + "".join(rows) + "</table></body></html>")
    out = OUT / "mask_review.html"
    out.write_text(page)
    print("review page:", out)
    print(d["symbol_type"].value_counts().to_string())


# ---------------------------------------------------------------- stage: embed
def embed_paths(model_key, paths, device, batch=16):
    import torch
    from transformers import AutoModel, CLIPModel

    name, mean, std = MODELS[model_key]
    if model_key == "clip":
        m = CLIPModel.from_pretrained(name).to(device).eval()
        fwd = lambda x: m.visual_projection(m.vision_model(pixel_values=x).pooler_output)
    else:
        m = AutoModel.from_pretrained(name).to(device).eval()
        fwd = lambda x: m(pixel_values=x).last_hidden_state[:, 0]   # CLS token
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)

    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            arr = np.stack([np.asarray(square(load_rgb(p)), dtype=np.float32) / 255.0
                            for p in paths[i:i + batch]])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2)
            x = ((x - mean_t) / std_t).to(device)
            e = fwd(x).float()
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy())
            if (i // batch) % 40 == 0:
                print(f"    {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    del m
    if device == "mps":
        torch.mps.empty_cache()
    return np.vstack(out).astype(np.float32)


def run_embed(a):
    idx = build_index(a)
    device = pick_device()
    print(f"embed: {len(idx)} logos on {device}")
    for model_key in MODELS:
        for version, col in [("orig", "logo_path"), ("masked", "masked_path")]:
            f = OUT / f"{model_key}_{version}.npy"
            if f.exists() and np.load(f, mmap_mode="r").shape[0] == len(idx):
                print(f"  {f.name} already done")
                continue
            print(f"  {model_key} / {version}")
            X = embed_paths(model_key, idx[col].tolist(), device)
            np.save(f, X)
            print(f"  saved {f.name} {X.shape}")


# ---------------------------------------------------------------- stage: validate
def knn(X, k):
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def sector_check(X, labels, k, perms, rng):
    lab = pd.factorize(labels)[0]
    n = len(lab)
    counts = np.bincount(lab)
    nn = knn(X, k)

    per_firm = (lab[nn] == lab[:, None]).mean(axis=1)
    obs = per_firm.mean()
    chance = ((counts[lab] - 1) / (n - 1)).mean()

    sectors = np.arange(len(counts))
    obs_sec = np.array([per_firm[lab == s].mean() for s in sectors])
    null, null_sec = np.empty(perms), np.empty((perms, len(counts)))
    for p in range(perms):
        pl = rng.permutation(lab)
        pf = (pl[nn] == pl[:, None]).mean(axis=1)
        null[p] = pf.mean()
        null_sec[p] = [pf[pl == s].mean() for s in sectors]
    pval = (1 + (null >= obs).sum()) / (perms + 1)

    names = pd.factorize(labels)[1]
    by_sec = pd.DataFrame({
        "sector": names, "n": counts, "purity": obs_sec,
        "chance": (counts - 1) / (n - 1),
        "lift": obs_sec / ((counts - 1) / (n - 1)),
        "z": (obs_sec - null_sec.mean(0)) / null_sec.std(0),
        "p_perm": (1 + (null_sec >= obs_sec).sum(0)) / (perms + 1),
    }).sort_values("lift", ascending=False)
    return obs, chance, pval, null, by_sec


def linear_probe(X, labels):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        return None
    y = pd.Series(labels)
    keep = y.map(y.value_counts()) >= 10
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(clf, X[keep.values], y[keep].values, cv=cv, scoring="balanced_accuracy")
    return acc.mean(), 1 / y[keep].nunique()


def run_validate(k=10, perms=1000):
    idx = pd.read_csv(INDEX)
    rng = np.random.default_rng(519)
    has_symbol = (idx["symbol_type"] != "text_only").values
    lines = [f"Sector validation: do logos sit near same-sector logos?  (k={k}, {perms} permutations)",
             f"Firms: {len(idx)}   symbol types: " + idx["symbol_type"].value_counts().to_dict().__str__(), ""]
    tables = []
    runs = [("orig", "all logos", np.ones(len(idx), bool)),
            ("orig", "logos with a symbol", has_symbol),
            ("masked", "logos with a symbol", has_symbol)]
    for model_key in MODELS:
        for version, subset, keep in runs:
            f = OUT / f"{model_key}_{version}.npy"
            if not f.exists():
                continue
            X = np.load(f)[keep]
            labels = idx.loc[keep, "sector"].values
            obs, chance, p, null, by_sec = sector_check(X, labels, k, perms, rng)
            probe = linear_probe(X, labels)
            tag = f"{model_key:4s} {version:6s} {subset}"
            lines.append(f"{tag:42s} n={keep.sum():5d}  purity {obs:.3f} vs chance {chance:.3f}  "
                         f"lift {obs / chance:.2f}x  perm p={p:.4f}  (null max {null.max():.3f})"
                         + (f"  probe bal.acc {probe[0]:.3f} vs {probe[1]:.3f}" if probe else ""))
            by_sec.insert(0, "run", tag.strip())
            tables.append(by_sec)
    lines += ["", "Reading it:",
              " - lift > 1 with p < 0.05 means sectors have a visual 'norm' that distinctiveness can be measured against.",
              " - if 'orig' beats 'masked' by a lot, part of the sector signal comes from words in the logo",
              "   (e.g. 'Bank', 'Energy'), which is why the main distinctiveness scores should use masked logos too.",
              " - see validation_by_sector.csv for which sectors look most alike."]
    report = "\n".join(lines)
    (OUT / "validation.txt").write_text(report)
    pd.concat(tables).to_csv(OUT / "validation_by_sector.csv", index=False)
    print(report)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["mask", "review", "embed", "validate", "all"])
    ap.add_argument("--limit", type=int, default=None, help="mask stage only: process at most N logos")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()
    set_year(args.year)

    OUT.mkdir(parents=True, exist_ok=True)
    a = analysis_set()
    print(f"base year {YEAR}: analysis set {len(a)} firms -> {OUT}")
    if args.stage in ("mask", "all"):
        run_mask(a, args.limit)
    if args.stage in ("review", "all"):
        run_review()
    if args.stage in ("embed", "all"):
        run_embed(a)
    if args.stage in ("validate", "all"):
        run_validate(args.k, args.perms)
