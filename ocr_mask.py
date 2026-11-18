import json
import sys
from pathlib import Path

import easyocr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

name, csv, path_col, id_col = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None

out_dir = Path(f"data/{name}/masked")
out_dir.mkdir(parents=True, exist_ok=True)
log_path = Path(f"data/{name}_ocr.jsonl")

df = pd.read_csv(csv)
if limit:
    df = df.head(limit)

done = set()
if log_path.exists():
    with open(log_path) as f:
        done = {json.loads(line)["id"] for line in f if line.strip()}
todo = df[~df[id_col].astype(str).isin({str(d) for d in done})]
print(f"{len(done)} already done, {len(todo)} to process")

reader = easyocr.Reader(["en"], gpu=False, verbose=False)

def load(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")

def background_color(arr):
    corners = np.concatenate([arr[:4, :4].reshape(-1, 3), arr[:4, -4:].reshape(-1, 3),
                              arr[-4:, :4].reshape(-1, 3), arr[-4:, -4:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(corners, axis=0))

with open(log_path, "a") as log:
    for n, row in enumerate(todo.itertuples(index=False), 1):
        rid = str(getattr(row, id_col))
        src = getattr(row, path_col)
        record = {"id": rid, "masked_path": "", "text": "", "n_boxes": 0, "text_share": 0.0, "error": ""}
        try:
            im = load(src)
            scale = 2 if max(im.size) < 300 else 1
            big = im.resize((im.width * scale, im.height * scale))
            results = reader.readtext(np.asarray(big))
            boxes, words = [], []
            for pts, text, conf in results:
                if conf < 0.3 or not any(ch.isalnum() for ch in text):
                    continue
                xs = [p[0] / scale for p in pts]
                ys = [p[1] / scale for p in pts]
                boxes.append((min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2))
                words.append(text)

            masked = im.copy()
            draw = ImageDraw.Draw(masked)
            fill = background_color(np.asarray(im))
            mask = np.zeros((im.height, im.width), dtype=bool)
            for x0, y0, x1, y1 in boxes:
                draw.rectangle([x0, y0, x1, y1], fill=fill)
                mask[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = True

            out = out_dir / f"{rid}.png"
            masked.save(out)
            record.update(masked_path=str(out), text=" ".join(words),
                          n_boxes=len(boxes), text_share=float(mask.mean()))
        except Exception as e:
            record["error"] = type(e).__name__
        log.write(json.dumps(record) + "\n")
        log.flush()
        if n % 100 == 0:
            print(f"{n}/{len(todo)}")

with open(log_path) as f:
    ocr = pd.DataFrame([json.loads(line) for line in f if line.strip()])
ocr = ocr.rename(columns={"error": "ocr_error"})
df[id_col] = df[id_col].astype(str)
merged = df.merge(ocr, left_on=id_col, right_on="id", how="left")
merged.to_csv(f"data/{name}_ocr.csv", index=False)

ok = merged[merged["ocr_error"] == ""]
print(f"\nProcessed: {len(merged)}, errors: {int((merged['ocr_error'] != '').sum())}")
print(f"Logos with text: {(ok['n_boxes'] > 0).mean():.0%}")
print(f"Median text share (logos with text): {ok.loc[ok['n_boxes'] > 0, 'text_share'].median():.2f}")
print("Examples:", ok.loc[ok["n_boxes"] > 0, "text"].head(10).tolist())
