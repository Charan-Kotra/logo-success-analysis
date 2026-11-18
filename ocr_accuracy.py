import pandas as pd

lab = pd.read_csv("data/ocr_labels.csv", dtype={"id": str})
ocr = pd.read_csv("data/ocr_check.csv", dtype={"id": str})
df = lab.merge(ocr[["id", "n_boxes"]], on="id")
df["ocr_found"] = df["n_boxes"] > 0
df["really"] = df["has_text_really"].str.strip().str.lower() == "yes"

tp = (df.ocr_found & df.really).sum()
fp = (df.ocr_found & ~df.really).sum()
fn = (~df.ocr_found & df.really).sum()
tn = (~df.ocr_found & ~df.really).sum()
print(f"Logos with text (hand labels): {df.really.sum()} of {len(df)}")
print(f"Recall: {tp}/{tp + fn} = {tp / max(tp + fn, 1):.0%}")
print(f"Precision: {tp}/{tp + fp} = {tp / max(tp + fp, 1):.0%}")
print(f"Agreement: {(tp + tn) / len(df):.0%}")
covered = df.loc[df.really, "text_fully_covered"].str.strip().str.lower() == "yes"
print(f"Text fully hidden after masking: {covered.sum()}/{len(covered)} = {covered.mean():.0%}")
print("\nLogo types (hand labels):")
print(df["logo_type"].str.strip().str.lower().value_counts().to_string())
