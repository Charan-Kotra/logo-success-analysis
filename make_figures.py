from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

Path("figures").mkdir(exist_ok=True)
df = pd.read_csv("data/yc_all_features.csv")
df["year"] = df["batch"].str.extract(r"(\d{4})").astype(float)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

trend = df[df["year"].between(2010, 2026)].groupby("year")
counts = trend.size()
years = counts[counts >= 20].index
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
for ax, (col, label, how) in zip(axes.ravel(), [
        ("saturation", "Median saturation", "median"),
        ("n_dominant", "Mean number of main colors", "mean"),
        ("has_text", "Share of logos with text", "mean"),
        ("compress_ratio", "Median complexity (compressed size)", "median")]):
    s = getattr(trend[col], how)().loc[years]
    ax.plot(s.index, s.values, marker="o", color="#4a6fa5")
    ax.set_title(label)
for ax in axes[1]:
    ax.set_xlabel("YC batch year")
fig.suptitle("YC startup logo style by batch year")
fig.tight_layout()
fig.savefig("figures/style_trends.png", dpi=200)

known = df[df["status"].isin(["Acquired", "Public", "Inactive"])].copy()
known["success"] = known["status"].isin(["Acquired", "Public"])
rate = known[known["year"].between(2010, 2026)].groupby("year")["success"].agg(["mean", "size"])
rate = rate[rate["size"] >= 10]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(rate.index, rate["mean"], color="#9aa7b8")
ax.set_ylabel("Acquired or public share")
ax.set_xlabel("YC batch year")
ax.set_title("Older batches have had more time to succeed")
fig.tight_layout()
fig.savefig("figures/success_by_year.png", dpi=200)

labels = ["Trained,\nall logos", "Trained,\nlogos with text", "Zero-shot,\nall logos", "Zero-shot,\nlogos with text"]
shown = [0.320, 0.307, 0.287, 0.301]
hidden = [0.295, 0.254, 0.255, 0.222]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - 0.2, shown, 0.4, label="Text shown", color="#4a6fa5")
ax.bar(x + 0.2, hidden, 0.4, label="Text hidden", color="#c9d3e0")
ax.axhline(0.20, color="gray", linestyle="--", linewidth=1)
ax.text(len(labels) - 0.5, 0.205, "chance", color="gray", ha="right")
ax.set_xticks(x, labels)
ax.set_ylabel("Balanced accuracy, 5 industries")
ax.set_title("Hiding the company name cuts industry prediction toward chance")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/name_vs_design.png", dpi=200)

sample = known[known["year"].between(2015, 2024)]
win = sample[sample["success"]].sample(8, random_state=519)
lose = sample[~sample["success"]].sample(8, random_state=519)
fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for row_axes, group, title in [(axes[0], win, "Acquired or public"), (axes[1], lose, "Inactive")]:
    for ax, path in zip(row_axes, group["logo_path"]):
        ax.imshow(Image.open(path).convert("RGBA"))
        ax.axis("off")
    row_axes[0].set_title(title, loc="left", fontsize=11)
fig.tight_layout()
fig.savefig("figures/example_logos.png", dpi=200)
print("Saved 4 figures in figures/")
