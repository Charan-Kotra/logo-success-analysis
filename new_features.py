import colorsys
import io

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

df = pd.read_csv("data/yc_simple_features.csv")
df["row"] = np.arange(len(df))
df["id"] = df["id"].astype(str)
ocr = pd.read_csv("data/yc_ocr.csv", dtype={"id": str})[["id", "n_boxes", "text_share", "masked_path"]]
df = df.merge(ocr, on="id", how="left")
df["n_boxes"] = df["n_boxes"].fillna(0)
df["text_share"] = df["text_share"].fillna(0)

def hue_family(rgb):
    h, s, v = colorsys.rgb_to_hsv(*(np.asarray(rgb) / 255))
    if s < 0.2 or v < 0.15:
        return "neutral"
    deg = h * 360
    for limit, name in [(15, "red"), (45, "orange"), (70, "yellow"), (170, "green"),
                        (260, "blue"), (345, "purple")]:
        if deg < limit:
            return name
    return "red"

def measures(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB").resize((128, 128))
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    compress_ratio = len(buf.getvalue()) / (128 * 128 * 3)

    arr = np.asarray(im).astype(float)
    corners = np.concatenate([arr[:4, :4].reshape(-1, 3), arr[:4, -4:].reshape(-1, 3),
                              arr[-4:, :4].reshape(-1, 3), arr[-4:, -4:].reshape(-1, 3)])
    bg_color = np.median(corners, axis=0)
    content = arr[np.abs(arr - bg_color).sum(axis=2) > 40]
    if len(content) < 20:
        return compress_ratio, 0, "neutral"
    rng = np.random.default_rng(519)
    if len(content) > 2000:
        content = content[rng.choice(len(content), 2000, replace=False)]
    k = min(5, len(np.unique(content, axis=0)))
    km = KMeans(n_clusters=k, n_init=1, random_state=519).fit(content)
    weights = np.bincount(km.labels_, minlength=k) / len(km.labels_)
    n_dominant = int((weights >= 0.1).sum())
    order = np.argsort(-weights)
    main = "neutral"
    for i in order:
        family = hue_family(km.cluster_centers_[i])
        if family != "neutral":
            main = family
            break
    return compress_ratio, n_dominant, main

vals = []
for n, p in enumerate(df["logo_path"], 1):
    try:
        vals.append(measures(p))
    except Exception:
        vals.append((np.nan, np.nan, "neutral"))
    if n % 1000 == 0:
        print(f"{n}/{len(df)}")
df[["compress_ratio", "n_dominant", "main_hue"]] = pd.DataFrame(vals, index=df.index)

df["has_text"] = (df["n_boxes"] > 0).astype(int)
df["logo_type"] = np.where(df["n_boxes"] == 0, "symbol",
                           np.where(df["text_share"] >= 0.5, "wordmark", "both"))
df = df.sort_values("row").reset_index(drop=True)
df.to_csv("data/yc_all_features.csv", index=False)
print("Saved data/yc_all_features.csv")

lab = pd.read_csv("data/ocr_labels.csv", dtype={"id": str})
check = lab.merge(df[["id", "logo_type"]].rename(columns={"logo_type": "rule"}), on="id")
check["hand"] = check["logo_type"].str.strip().str.lower()
print(f"\nLogo-type rule agrees with hand labels: {(check.hand == check.rule).mean():.0%} of {len(check)}")
print(pd.crosstab(check.hand, check.rule, rownames=["hand"], colnames=["rule"]).to_string())
print("\nMain hue:")
print(df["main_hue"].value_counts().to_string())
