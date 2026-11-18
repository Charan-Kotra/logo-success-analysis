import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

out_dir = Path("data/yc/logos")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_json("data/yc/all.json")
keep = ["id", "name", "slug", "website", "industry", "subindustry", "batch",
        "status", "team_size", "top_company", "small_logo_thumb_url"]
df = df[keep]

def fetch(row):
    path = out_dir / f"{row.id}.png"
    if path.exists() and path.stat().st_size > 0:
        return row.id, str(path), "ok"
    url = row.small_logo_thumb_url
    if not isinstance(url, str) or not url.startswith("http"):
        return row.id, "", "no_url"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cse519-student-project"})
            with urllib.request.urlopen(req, timeout=20) as r:
                path.write_bytes(r.read())
            time.sleep(0.1)
            return row.id, str(path), "ok"
        except Exception as e:
            err = type(e).__name__
            time.sleep(1 + attempt)
    return row.id, "", err

results = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for i, res in enumerate(pool.map(fetch, df.itertuples(index=False)), 1):
        results.append(res)
        if i % 500 == 0:
            print(f"{i}/{len(df)} done")

res = pd.DataFrame(results, columns=["id", "logo_path", "download_status"])
df = df.merge(res, on="id")
df.to_csv("data/yc/companies_with_logos.csv", index=False)
print(df["download_status"].value_counts())
