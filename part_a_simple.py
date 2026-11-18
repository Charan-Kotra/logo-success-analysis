import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["n_colors", "saturation", "brightness", "background_frac",
            "edge_density", "aspect", "symmetry"]

df = pd.read_csv("data/yc_simple_features.csv")
df["year"] = df["batch"].str.extract(r"(\d{4})").astype(float)
df = df[df["status"].isin(["Acquired", "Public", "Inactive"]) & df["year"].between(2015, 2024)].copy()
df["success"] = df["status"].isin(["Acquired", "Public"]).astype(int)
print(f"Companies: {len(df)}, success rate: {df['success'].mean():.2f}\n")

print("Feature medians (success vs failure):")
for f in FEATURES:
    a, b = df.loc[df.success == 1, f], df.loc[df.success == 0, f]
    p = mannwhitneyu(a, b).pvalue
    print(f"  {f:16s} {a.median():8.3f} {b.median():8.3f}   p = {p:.3f}")

def cv_auc(numeric, categorical):
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=519)
    prob = cross_val_predict(model, df[numeric + categorical], df["success"], cv=cv, method="predict_proba")[:, 1]
    rng = np.random.default_rng(519)
    y = df["success"].to_numpy()
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc_score(y[idx], prob[idx]))
    return roc_auc_score(y, prob), np.percentile(boots, [2.5, 97.5])

print("\nCross-validated AUC (0.5 = chance):")
for name, num, cat in [
    ("Year + industry", ["year"], ["industry"]),
    ("Year + industry + logo", ["year"] + FEATURES, ["industry"]),
    ("Logo features only", FEATURES, []),
]:
    auc, ci = cv_auc(num, cat)
    print(f"  {name:25s} {auc:.3f}  (95% CI {ci[0]:.3f}-{ci[1]:.3f})")
