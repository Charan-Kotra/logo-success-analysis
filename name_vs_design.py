import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPTokenizer

df = pd.read_csv("data/yc_ocr.csv")
orig = np.load("data/yc_clip.npy")
masked = np.load("data/yc_masked_clip.npy")
assert len(df) == len(orig) == len(masked)

classes = ["B2B", "Consumer", "Healthcare", "Fintech", "Industrials"]
keep = df["industry"].isin(classes).to_numpy()
df, orig, masked = df[keep].reset_index(drop=True), orig[keep], masked[keep]
y = df["industry"].to_numpy()
has_text = (df["n_boxes"].fillna(0) > 0).to_numpy()
print(f"Logos: {len(df)}, with text found by OCR: {has_text.sum()}\n")

def paired_ci(pred_a, pred_b, mask):
    rng = np.random.default_rng(519)
    idx_all = np.where(mask)[0]
    diffs = []
    for _ in range(1000):
        i = rng.choice(idx_all, len(idx_all))
        diffs.append(balanced_accuracy_score(y[i], pred_a[i]) - balanced_accuracy_score(y[i], pred_b[i]))
    return np.percentile(diffs, [2.5, 97.5])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=519)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))
trained = {name: cross_val_predict(model, X, y, cv=cv) for name, X in [("original", orig), ("masked", masked)]}

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
prompts = ["the logo of a B2B software company", "the logo of a consumer products company",
           "the logo of a healthcare company", "the logo of a financial technology company",
           "the logo of an industrial or hardware company"]
with torch.no_grad():
    t = tok(prompts, padding=True, return_tensors="pt")
    text = clip.text_projection(clip.text_model(**t).pooler_output)
    text = (text / text.norm(dim=-1, keepdim=True)).numpy()
zero = {name: np.array(classes)[(X @ text.T).argmax(axis=1)] for name, X in [("original", orig), ("masked", masked)]}

print("Balanced accuracy (chance = 0.20):")
print(f"{'Model':12s} {'Logos':16s} {'Original':>9s} {'Masked':>8s}  Drop (95% CI)")
for model_name, preds in [("Trained", trained), ("Zero-shot", zero)]:
    for subset_name, mask in [("All", np.ones(len(y), bool)), ("With text", has_text)]:
        a = balanced_accuracy_score(y[mask], preds["original"][mask])
        b = balanced_accuracy_score(y[mask], preds["masked"][mask])
        lo, hi = paired_ci(preds["original"], preds["masked"], mask)
        print(f"{model_name:12s} {subset_name:16s} {a:9.3f} {b:8.3f}  {a - b:+.3f} ({lo:+.3f} to {hi:+.3f})")

print("\nPer-industry recall, trained model, logos with text:")
for c in classes:
    m = has_text & (y == c)
    print(f"  {c:12s} original {np.mean(trained['original'][m] == c):.2f}   masked {np.mean(trained['masked'][m] == c):.2f}   (n={m.sum()})")
