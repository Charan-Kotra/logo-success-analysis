import hashlib

import pandas as pd
from PIL import Image

df = pd.read_csv("data/yc/companies_with_logos.csv")
df = df[df["download_status"] == "ok"].copy()

def inspect(path):
    try:
        data = open(path, "rb").read()
        with Image.open(path) as im:
            im.load()
            return hashlib.md5(data).hexdigest(), im.width, im.height, im.mode, ""
    except Exception as e:
        return "", 0, 0, "", type(e).__name__

info = df["logo_path"].apply(inspect)
df[["md5", "width", "height", "mode", "error"]] = pd.DataFrame(info.tolist(), index=df.index)

shared = df["md5"].value_counts()
df["problem"] = ""
df.loc[df["error"] != "", "problem"] = "broken"
df.loc[(df["problem"] == "") & ((df["width"] < 32) | (df["height"] < 32)), "problem"] = "tiny"
df.loc[(df["problem"] == "") & (df["md5"].map(shared) >= 3), "problem"] = "placeholder"

print("Problems:")
print(df["problem"].replace("", "ok").value_counts())
print("\nImage sizes (ok logos):")
print(df.loc[df["problem"] == "", ["width", "height"]].describe().loc[["min", "50%", "max"]])

clean = df[df["problem"] == ""].copy()
clean.to_csv("data/yc/clean.csv", index=False)
print(f"\nClean logos: {len(clean)}")
print("\nStatus:")
print(clean["status"].value_counts())
print("\nIndustry:")
print(clean["industry"].value_counts())
