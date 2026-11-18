import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import CLIPModel, CLIPTokenizer

all_df = pd.read_csv("data/yc/clean.csv")
emb = np.load("data/yc_clip.npy")
assert len(all_df) == len(emb), "Embeddings and table do not line up"
all_df["year"] = all_df["batch"].str.extract(r"(\d{4})").astype(float)

# Idea 1: distinctiveness = 1 - mean cosine similarity to same-year peers
distinct = np.full(len(all_df), np.nan)
for year, idx in all_df.groupby("year").indices.items():
    if len(idx) < 5:
        continue
    sims = emb[idx] @ emb[idx].T
    np.fill_diagonal(sims, np.nan)
    distinct[idx] = 1 - np.nanmean(sims, axis=1)
all_df["distinctiveness"] = distinct

# Idea 2: professionalism from CLIP text prompts
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
prompts = ["a professionally designed company logo",
           "an amateur, poorly designed company logo"]
with torch.no_grad():
    t = tok(prompts, padding=True, return_tensors="pt")
    text = model.text_projection(model.text_model(**t).pooler_output)
    text = (text / text.norm(dim=-1, keepdim=True)).numpy()
sims = emb @ text.T
all_df["professionalism"] = sims[:, 0] - sims[:, 1]
all_df.to_csv("data/yc_clip_scores.csv", index=False)

mask = (all_df["status"].isin(["Acquired", "Public", "Inactive"]) & all_df["year"].between(2015, 2024)).to_numpy()
df = all_df[mask].copy().reset_index(drop=True)
df["success"] = df["status"].isin(["Acquired", "Public"]).astype(int)
y = df["success"].to_numpy()
print(f"Companies: {len(df)}, success rate: {y.mean():.2f}\n")

clip_cols = [f"c{i}" for i in range(emb.shape[1])]
full = pd.concat([df[["year", "industry"]], pd.DataFrame(emb[mask], columns=clip_cols)], axis=1)

def boot_ci(prob):
    rng = np.random.default_rng(519)
    vals = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], prob[i]))
    return np.percentile(vals, [2.5, 97.5])

def run(use_base, use_clip):
    parts = []
    if use_base:
        parts.append(("year", StandardScaler(), ["year"]))
        parts.append(("ind", OneHotEncoder(handle_unknown="ignore"), ["industry"]))
    if use_clip:
        parts.append(("clip", make_pipeline(StandardScaler(), PCA(n_components=30, random_state=519)), clip_cols))
    model = make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=3000, C=0.5))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=519)
    prob = cross_val_predict(model, full, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, prob), boot_ci(prob)

print("Cross-validated AUC (0.5 = chance):")
for name, base, clip in [("Year + industry", True, False),
                         ("Year + industry + CLIP", True, True),
                         ("CLIP only", False, True)]:
    auc, ci = run(base, clip)
    print(f"  {name:25s} {auc:.3f}  (95% CI {ci[0]:.3f}-{ci[1]:.3f})")

for score in ["distinctiveness", "professionalism"]:
    df[f"{score}_z"] = (df[score] - df[score].mean()) / df[score].std()
    m = smf.logit(f"success ~ {score}_z + year + C(industry)", data=df).fit(disp=False)
    row = m.summary2().tables[1].loc[f"{score}_z"]
    print(f"\n{score}: coef {row['Coef.']:.3f}, odds ratio {np.exp(row['Coef.']):.2f} per SD, p = {row['P>|z|']:.3f}")
    print("  median success vs failure:",
          round(df.loc[df.success == 1, score].median(), 4),
          round(df.loc[df.success == 0, score].median(), 4))
