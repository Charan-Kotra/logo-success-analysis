import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

all_df = pd.read_csv("data/yc_all_features.csv")
orig = np.load("data/yc_clip.npy")
masked = np.load("data/yc_masked_clip.npy")
all_df["year"] = all_df["batch"].str.extract(r"(\d{4})").astype(float)

mask = (all_df["status"].isin(["Acquired", "Public", "Inactive"]) & all_df["year"].between(2015, 2024)).to_numpy()
df = all_df[mask].reset_index(drop=True)
rows = df["row"].to_numpy()
y = df["status"].isin(["Acquired", "Public"]).astype(int).to_numpy()
print(f"Companies: {len(df)}, success rate: {y.mean():.2f}\n")


def z(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x)


base = pd.concat([pd.Series(z(df["year"]), name="year"),
                  pd.get_dummies(df["industry"], prefix="ind", drop_first=True).astype(float)], axis=1)


def numeric(cols):
    return pd.DataFrame({c: z(df[c].fillna(df[c].median())) for c in cols})


def dummies(col):
    return pd.get_dummies(df[col], prefix=col, drop_first=True).astype(float)


def pca(X):
    comps = PCA(n_components=10, random_state=519).fit_transform(StandardScaler().fit_transform(X))
    return pd.DataFrame(comps, columns=[f"pc{i}" for i in range(10)])


groups = {
    "Simple features": numeric(["n_colors", "saturation", "brightness", "background_frac",
                                "edge_density", "aspect", "symmetry"]),
    "Color palette": pd.concat([numeric(["saturation", "n_dominant"]), dummies("main_hue")], axis=1),
    "Text and logo type": pd.concat([numeric(["text_share", "has_text"]), dummies("logo_type")], axis=1),
    "Complexity": numeric(["compress_ratio", "edge_density"]),
    "CLIP (original)": pca(orig[rows]),
    "CLIP (text hidden)": pca(masked[rows]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=519)


def cv_auc(X):
    model = LogisticRegression(max_iter=3000, C=1.0)
    p = cross_val_predict(model, StandardScaler().fit_transform(X), y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


base_auc = cv_auc(base.to_numpy())
m0 = sm.Logit(y, sm.add_constant(base.to_numpy())).fit(disp=0, maxiter=500)
rng = np.random.default_rng(519)
results = []
for name, G in groups.items():
    X1 = np.hstack([base.to_numpy(), G.to_numpy()])
    m1 = sm.Logit(y, sm.add_constant(X1)).fit(disp=0, maxiter=500)
    lr = 2 * (m1.llf - m0.llf)
    p_lr = chi2.sf(lr, G.shape[1])
    gain = cv_auc(X1) - base_auc
    perm = []
    for _ in range(200):
        Gp = G.to_numpy()[rng.permutation(len(G))]
        perm.append(cv_auc(np.hstack([base.to_numpy(), Gp])) - base_auc)
    p_perm = (1 + np.sum(np.array(perm) >= gain)) / 201
    results.append([name, G.shape[1], lr, p_lr, gain, p_perm])
    print(f"done: {name}")

res = pd.DataFrame(results, columns=["group", "n_features", "LR_stat", "p_LR", "AUC_gain", "p_perm"])
res["p_LR_holm"] = multipletests(res["p_LR"], method="holm")[1]
res["p_perm_holm"] = multipletests(res["p_perm"], method="holm")[1]
print(f"\nBaseline AUC (year + industry): {base_auc:.3f}\n")
print(res.round(3).to_string(index=False))
res.to_csv("data/formal_tests.csv", index=False)

raw = pd.read_json("data/yc/all.json")
raw["year"] = raw["batch"].str.extract(r"(\d{4})").astype(float)
raw = raw[raw["status"].isin(["Acquired", "Public", "Inactive"]) & raw["year"].between(2015, 2024)].copy()
raw["success"] = raw["status"].isin(["Acquired", "Public"]).astype(int)
raw["has_logo"] = raw["small_logo_thumb_url"].astype(str).str.startswith("http").astype(int)
m = smf.logit("success ~ has_logo + year + C(industry)", data=raw).fit(disp=False)
r = m.summary2().tables[1].loc["has_logo"]
print(f"\nMissing-logo test (n={len(raw)}): success rate with logo "
      f"{raw.loc[raw.has_logo == 1, 'success'].mean():.2f}, "
      f"without {raw.loc[raw.has_logo == 0, 'success'].mean():.2f}")
print(f"  Has-logo odds ratio {np.exp(r['Coef.']):.2f}, p = {r['P>|z|']:.4f}")
