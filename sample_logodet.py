from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

PER_BRAND = 8
PER_INDUSTRY = 4000
out_dir = Path("data/logodet/crops")
out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("axonstan/LogoDet-3K", split="train", streaming=True)
ds = ds.shuffle(seed=519, buffer_size=5000)

brand_count, industry_count, rows = Counter(), Counter(), []
for i, row in enumerate(ds):
    brand, industry = row["company_name"], row["industry_name"]
    if brand_count[brand] >= PER_BRAND or industry_count[industry] >= PER_INDUSTRY:
        continue
    try:
        x0, y0, x1, y1 = row["bbox"]
        crop = row["image_path"].crop((x0, y0, x1, y1)).convert("RGB")
        if crop.width < 16 or crop.height < 16:
            continue
        crop.thumbnail((224, 224))
        path = out_dir / f"{len(rows):06d}.jpg"
        crop.save(path, quality=90)
    except Exception:
        continue
    brand_count[brand] += 1
    industry_count[industry] += 1
    rows.append({"crop_path": str(path), "brand_id": brand, "industry": industry})
    if len(rows) % 1000 == 0:
        print(f"{len(rows)} crops saved, {i + 1} rows read")
        pd.DataFrame(rows).to_csv("data/logodet/sample.csv", index=False)

pd.DataFrame(rows).to_csv("data/logodet/sample.csv", index=False)
print("Total crops:", len(rows))
print(pd.DataFrame(rows)["industry"].value_counts())
