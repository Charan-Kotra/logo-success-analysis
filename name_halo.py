"""
Name-halo experiment: does a vision-language model judge a startup logo by its
design or by the company name attached to it?

Each sampled company is shown to the model in five conditions:
  A  original logo, no name in the prompt (name visible only if it is in the logo)
  B  text-masked logo, no name
  C  text-masked logo + the company's real name in the prompt
  D  text-masked logo + a famous startup name in the prompt
  E  text-masked logo + a made-up neutral name in the prompt
  F  text-masked logo + the name of a famous startup FAILURE in the prompt
  G  text-masked logo + a look-alike of the famous name used in D (e.g. "Strype")

F and G separate a fame halo from factual recall: if the model only recalls
facts, famous failures should score low and look-alike names should score like
made-up ones.

The model rates success likelihood on a 0-9 scale. Instead of sampling one digit,
we read the probabilities of the ten digit tokens and keep the expected score,
which is continuous and less noisy.

Usage:
  python name_halo.py --n 400 --out data/name_halo_qwen.csv
  python name_halo.py --limit 5            # quick test on 5 companies
"""
import argparse
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

FAMOUS = ["Stripe", "Airbnb", "Coinbase", "DoorDash", "Dropbox", "Instacart"]
FAILED = ["Theranos", "Quibi", "Juicero", "Webvan", "Pets.com", "Jawbone"]
LOOKALIKE = {"Stripe": "Strype", "Airbnb": "Airbnd", "Coinbase": "Coinbace",
             "DoorDash": "DoorDosh", "Dropbox": "Dropbocks", "Instacart": "Instakart"}
MADE_UP = ["Velora", "Quintly", "Brimo", "Taskel", "Novari", "Plendo",
           "Zorvia", "Kelvo", "Truvo", "Mirabe", "Daxon", "Lumera"]

PROMPT = (
    "You are an experienced venture capital investor. Below is the logo of an "
    "early-stage startup.{name_line} Based only on what you see, how likely is "
    "this startup to become a big success (acquired or IPO)? Answer with a single "
    "digit from 0 (very unlikely) to 9 (very likely)."
)


def load_image(path, size=448):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def pick_sample(df, n, seed):
    df = df.copy()
    df["year"] = df["batch"].astype(str).str.extract(r"(\d{4})").astype(float)
    df = df[(df["year"] >= 2015) & (df["year"] <= 2024)]
    df = df[df["status"].isin(["Acquired", "Public", "Inactive"])]
    df["success"] = df["status"].isin(["Acquired", "Public"]).astype(int)
    famous_lower = {f.lower() for f in FAMOUS + FAILED + list(LOOKALIKE.values())}
    df = df[~df["name"].astype(str).str.lower().isin(famous_lower)]
    df = df[df["masked_path"].map(os.path.exists) & df["logo_path"].map(os.path.exists)]
    half = n // 2
    win = df[df["success"] == 1]
    lose = df[df["success"] == 0]
    half = min(half, len(win), len(lose))
    return pd.concat([win.sample(half, random_state=seed),
                      lose.sample(half, random_state=seed)]).sample(frac=1, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/yc/clean.csv")
    ap.add_argument("--logos", default="data/yc/logos")
    ap.add_argument("--masked", default="data/yc/masked")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=519)
    ap.add_argument("--out", default="data/name_halo_qwen.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"id", "name", "status", "batch"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"clean.csv is missing columns {missing}. Columns found: {list(df.columns)}")
    df["logo_path"] = df["id"].map(lambda i: os.path.join(args.logos, f"{i}.png"))
    df["masked_path"] = df["id"].map(lambda i: os.path.join(args.masked, f"{i}.png"))

    sample = pick_sample(df, args.n, args.seed)
    if args.limit:
        sample = sample.head(args.limit)
    print(f"Companies in sample: {len(sample)} (successes: {int(sample['success'].sum())})")

    rng = random.Random(args.seed)
    sample["famous_name"] = [rng.choice(FAMOUS) for _ in range(len(sample))]
    sample["madeup_name"] = [rng.choice(MADE_UP) for _ in range(len(sample))]
    sample["failed_name"] = [rng.choice(FAILED) for _ in range(len(sample))]
    sample["lookalike_name"] = sample["famous_name"].map(LOOKALIKE)

    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev["id"], prev["condition"]))
        print(f"Resuming: {len(done)} ratings already saved")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(device).eval()

    # Each digit can appear as more than one token depending on the tokenizer
    # (e.g. "3" in Qwen, "▁3" in SentencePiece-based models such as Mistral).
    tok = processor.tokenizer
    unk = tok.unk_token_id
    digit_ids = []
    for d in range(10):
        cands = set()
        for s in (str(d), "▁" + str(d), "Ġ" + str(d)):
            i = tok.convert_tokens_to_ids(s)
            if i is not None and i != unk:
                cands.add(i)
        enc = tok.encode(str(d), add_special_tokens=False)
        if enc:
            cands.add(enc[-1])
        digit_ids.append(sorted(cands))
    flat = [i for ids in digit_ids for i in ids]
    assert len(flat) == len(set(flat)), "digit token ids overlap"

    def digit_probs(logits):
        p = torch.softmax(logits, dim=0)
        per_digit = torch.stack([p[ids].sum() for ids in digit_ids])
        return per_digit, per_digit.sum().item()

    def rate(img, name):
        name_line = f" The company is called {name}." if name else ""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT.format(name_line=name_line)},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
            per_digit, mass = digit_probs(logits)
            # Some models emit a space or similar token before the digit.
            # If most probability is elsewhere, take that one token and look again.
            for _ in range(2):
                if mass >= 0.5:
                    break
                nxt = logits.argmax().view(1, 1)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], nxt], dim=1)
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones_like(nxt)], dim=1)
                logits = model(**inputs).logits[0, -1].float()
                per_digit, mass = digit_probs(logits)
        probs = (per_digit / per_digit.sum()).cpu().numpy()
        return float((probs * np.arange(10)).sum()), int(probs.argmax()), mass

    header = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = len(sample)
    for k, row in enumerate(sample.itertuples(index=False), 1):
        original = load_image(row.logo_path)
        masked = load_image(row.masked_path)
        conditions = [
            ("A_original", original, None),
            ("B_masked", masked, None),
            ("C_masked_real", masked, row.name),
            ("D_masked_famous", masked, row.famous_name),
            ("E_masked_madeup", masked, row.madeup_name),
            ("F_masked_failed", masked, row.failed_name),
            ("G_masked_lookalike", masked, row.lookalike_name),
        ]
        rows = []
        for cond, img, name in conditions:
            if (row.id, cond) in done:
                continue
            score, top, mass = rate(img, name)
            rows.append({
                "id": row.id, "name": row.name, "status": row.status,
                "success": row.success, "year": row.year,
                "industry": getattr(row, "industry", None),
                "condition": cond, "name_shown": name or "",
                "score": score, "top_digit": top, "digit_mass": mass,
                "model": args.model,
            })
        if rows:
            pd.DataFrame(rows).to_csv(args.out, mode="a", header=header, index=False)
            header = False
        if k % 20 == 0 or k == total:
            print(f"{k}/{total} companies done", flush=True)

    res = pd.read_csv(args.out)
    print("\nMean score by condition:")
    print(res.groupby("condition")["score"].agg(["mean", "std", "count"]).round(3))
    print("\nMean probability mass on digit tokens (should be close to 1):",
          round(res["digit_mass"].mean(), 3))


if __name__ == "__main__":
    main()
