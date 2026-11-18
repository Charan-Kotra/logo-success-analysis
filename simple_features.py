import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

def features(path):
    try:
        im = Image.open(path)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB").resize((128, 128))
        rgb = np.asarray(im).astype(float) / 255
        hsv = np.asarray(im.convert("HSV")).astype(float) / 255
        gray = np.asarray(im.convert("L")).astype(float) / 255

        corners = np.concatenate([rgb[:4, :4].reshape(-1, 3), rgb[:4, -4:].reshape(-1, 3),
                                  rgb[-4:, :4].reshape(-1, 3), rgb[-4:, -4:].reshape(-1, 3)])
        bg_color = np.median(corners, axis=0)
        is_bg = np.abs(rgb - bg_color).sum(axis=2) < 0.15
        content = ~is_bg

        rows, cols = np.where(content)
        if len(rows):
            h = rows.max() - rows.min() + 1
            w = cols.max() - cols.min() + 1
            aspect = w / h
        else:
            aspect = 1.0

        quant = (rgb[content] * 7).round().astype(int) if content.any() else np.zeros((1, 3), int)
        n_colors = len({tuple(c) for c in quant})

        edges = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES)).astype(float) / 255
        symmetry = 1 - np.abs(gray - gray[:, ::-1]).mean()

        return {
            "n_colors": n_colors,
            "saturation": hsv[..., 1][content].mean() if content.any() else 0,
            "brightness": hsv[..., 2].mean(),
            "background_frac": is_bg.mean(),
            "edge_density": (edges > 0.2).mean(),
            "aspect": aspect,
            "symmetry": symmetry,
            "feature_error": "",
        }
    except Exception as e:
        return {"feature_error": type(e).__name__}

for name, csv, col in [("yc", "data/yc/clean.csv", "logo_path"),
                       ("logodet", "data/logodet/sample.csv", "crop_path")]:
    df = pd.read_csv(csv)
    feats = pd.DataFrame([features(p) for p in df[col]], index=df.index)
    out = pd.concat([df, feats], axis=1)
    out.to_csv(f"data/{name}_simple_features.csv", index=False)
    print(f"{name}: {len(out)} logos, {int((out['feature_error'] != '').sum())} errors")
    print(out[["n_colors", "saturation", "brightness", "background_frac",
               "edge_density", "aspect", "symmetry"]].describe().loc[["mean", "50%"]].round(3).to_string())
