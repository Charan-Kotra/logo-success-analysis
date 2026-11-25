"""
Stage the project's code and shareable data for the private GitHub repo.

Included
  code and docs        every .py, .md, requirements.txt, .gitignore in the project folder
  SEC-derived data     data/edgar*/  firm lists, financials, build reports      (public SEC data)
  logo selection       data/logos*/  logo_choice, candidates, group_scores CSVs
  embeddings / scores  data/embed*/  index, *.npy, distinctiveness, validation, OCR boxes
  Wikidata links       data/wikidata/firm_links.csv                             (CC0)
  results              data/results_*/  tables, figures, summaries (aggregate results)
  compressed logos     logos/<year>/  (made by export_logos.py)

Left out on purpose
  data/wrds/           WRDS licence: Compustat / CRSP data may not be posted to GitHub, even privately
  data/analysis/       firms_master files contain firm-level Compustat and CRSP values (same licence);
                       share them with the team through a private Google Drive folder instead
  firm-level result files with WRDS values (e.g. rebrand_pairs.csv)
  raw downloads        SEC bulk zips, edgar_raw image folders, masked images, full-size logos (rebuildable)
  any *.zip and any file over 95 MB (GitHub rejects files over 100 MB)

Usage:
  python3 share_to_github.py            # shows what would be added
  python3 share_to_github.py --add      # runs git add for those files
"""
import argparse
import fnmatch
import subprocess
from pathlib import Path

MAX_MB = 95
INCLUDE = [
    "*.py", "*.md", "requirements.txt", ".gitignore",
    "data/edgar*/firms_*.csv", "data/edgar*/financials_wide.csv", "data/edgar*/financials_long.csv",
    "data/edgar*/build_report.txt",
    "data/logos*/logo_choice*.csv", "data/logos*/candidates*.csv", "data/logos*/group_scores.csv",
    "data/embed*/index.csv", "data/embed*/*.npy", "data/embed*/distinctiveness.csv",
    "data/embed*/distinctiveness_summary.txt", "data/embed*/validation*", "data/embed*/ocr_boxes.jsonl",
    "data/wikidata*/firm_links.csv",
    "data/results_*/*.csv", "data/results_*/*.txt", "data/results_*/*.png",
    "logos/*/*",
]
EXCLUDE = [
    "data/wrds/*", "data/analysis/*", "*.zip", "*/raw/*", "*/edgar_raw/*", "*/masked/*", "*/final/*",
    "data/results_pooled/rebrand_pairs.csv",      # firm-level Tobin's Q from Compustat
    "*.old.csv.gz", "*/__pycache__/*",
]


def wanted(rel):
    if any(fnmatch.fnmatch(rel, p) for p in EXCLUDE):
        return False
    if "/" not in rel:
        return any(fnmatch.fnmatch(rel, p) for p in INCLUDE[:4])
    return any(fnmatch.fnmatch(rel, p) for p in INCLUDE[4:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="store_true")
    args = ap.parse_args()
    root = Path(".")
    picked, too_big = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = p.as_posix()
        if not wanted(rel):
            continue
        mb = p.stat().st_size / 1e6
        (too_big if mb > MAX_MB else picked).append((rel, mb))
    total = sum(mb for _, mb in picked)
    print(f"{len(picked)} files, {total:.0f} MB in total")
    by_dir = {}
    for rel, mb in picked:
        top = "/".join(rel.split("/")[:2]) if rel.count("/") >= 1 else "(project folder)"
        n, s = by_dir.get(top, (0, 0.0))
        by_dir[top] = (n + 1, s + mb)
    for k, (n, s) in sorted(by_dir.items()):
        print(f"  {k:40s} {n:6d} files  {s:8.1f} MB")
    if too_big:
        print("\nSkipped, over 95 MB (GitHub limit 100 MB):")
        for rel, mb in too_big:
            print(f"  {rel}  ({mb:.0f} MB)")
    print("\nNot included on purpose: data/wrds/ and data/analysis/ (WRDS licence), raw downloads, zips.")
    if args.add:
        files = [rel for rel, _ in picked]
        for i in range(0, len(files), 200):
            subprocess.run(["git", "add", "-f", "--"] + files[i:i + 200], check=True)
        print("\nStaged. Next:  git status   then   git commit   and   git push")
    else:
        print("\nNothing staged yet. Run again with --add to stage these files.")


if __name__ == "__main__":
    main()
