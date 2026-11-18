import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

name, csv, col = sys.argv[1], sys.argv[2], sys.argv[3]
out_path = Path(f"data/{name}_clip.npy")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")

paths = pd.read_csv(csv)[col].tolist()
partial = Path(f"data/{name}_clip_partial.npy")
done = np.load(partial) if partial.exists() else np.zeros((0, 512), dtype=np.float32)
print(f"{len(done)} already done, {len(paths)} total")

batch = 32
with torch.no_grad():
    for start in range(len(done), len(paths), batch):
        images = [load(p) for p in paths[start:start + batch]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        vision = model.vision_model(pixel_values=inputs["pixel_values"])
        emb = model.visual_projection(vision.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        done = np.vstack([done, emb.cpu().numpy().astype(np.float32)])
        if (start // batch) % 20 == 0:
            np.save(partial, done)
            print(f"{len(done)}/{len(paths)}")

np.save(out_path, done)
partial.unlink(missing_ok=True)
print("Saved", out_path, done.shape)
