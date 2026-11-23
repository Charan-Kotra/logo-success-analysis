# Running one base year (logos + firm data)

This builds the firm list and the logo set for one base year. The 2019 year is already done;
these steps are for **2015** and **2022**. Each run is independent, so two people can do one year each.

Everything here uses **public SEC EDGAR data only**. You do not need WRDS for these steps.

---

## 0. One-time setup (about 15 minutes)

You need Python 3.11 or 3.12. Python 3.13+ may fail to install torch.

**Mac**
```bash
cd ~/Desktop/logo-project
python3.12 -m venv ~/venvs/logo
source ~/venvs/logo/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd $HOME\Desktop\logo-project
py -3.12 -m venv $HOME\venvs\logo
$HOME\venvs\logo\Scripts\Activate.ps1
pip install -r requirements.txt
```

Each time you open a new terminal, activate the environment again (the `source ...` or
`Activate.ps1` line) and `cd` into the project folder.

In every command below, replace `YEAR` with your year (2015 or 2022), and put in
**your real name and email**. SEC requires them in every request.

---

## 1. Firm list, financials and outcomes (about 30 minutes, downloads ~3 GB once)

```bash
python3 edgar_build.py --year YEAR --name "Your Name" --email you@stonybrook.edu
```
Output: `data/edgar_YEAR/` with `firms_YEAR.csv`, `financials_wide.csv` and `build_report.txt`.
Look at `build_report.txt`; expect roughly 4,500-5,000 traded firms.

## 2. Pilot: logos for 30 firms (about 2 minutes)

```bash
python3 edgar_logos_collect.py --year YEAR --name "Your Name" --email you@stonybrook.edu --sample 30
```
It should print `Firms with any image:` at around 70-90%. If it shows `SEC throttled`, stop
(Ctrl+C) and tell the team.

## 3. Full logo collection (overnight)

The pilot firms are kept; the run continues with the rest. Safe to stop and restart.

**Mac** (keeps running with the screen off):
```bash
nohup caffeinate -i python3 -u edgar_logos_collect.py --year YEAR --name "Your Name" --email you@stonybrook.edu > collect_YEAR.log 2>&1 &
tail -3 collect_YEAR.log          # check progress any time
```
**Windows**: set Sleep to "Never" while plugged in (Settings > System > Power), then:
```powershell
python -u edgar_logos_collect.py --year YEAR --name "Your Name" --email you@stonybrook.edu *> collect_YEAR.log
```

**If both of us are on the same network (e.g. campus Wi-Fi) at the same time, add `--rate 4`**
so the two runs together stay under SEC's 10 requests/second limit.

## 4. Choose each firm's logo (1-3 hours; first run downloads the CLIP and OCR models)

```bash
python3 -u logo_score.py --year YEAR > score_YEAR.log 2>&1
tail -25 score_YEAR.log
```
Then open `data/logos_YEAR/logo_review.html` in a browser and scroll through it: green =
high confidence, orange = medium, red = low. Note anything systematically wrong.

## 5. Send the results back

Zip these (NOT `edgar_raw/`, which is large and not needed):
```bash
zip -r results_YEAR.zip data/edgar_YEAR \
    data/logos_YEAR/logo_choice.csv data/logos_YEAR/group_scores.csv \
    data/logos_YEAR/candidates.csv data/logos_YEAR/logo_review.html data/logos_YEAR/final \
    collect_YEAR.log score_YEAR.log
```
(Windows: right-click those files and folders > Send to > Compressed folder.)
Share the zip through Google Drive or OneDrive, **not GitHub**.

---

## Rules we follow

- **SEC EDGAR**: real name and email in the User-Agent (the scripts require them); never more
  than 10 requests/second (the default is 6, one shared limit for all threads). If SEC blocks
  us, the script pauses and slows down. Do not run two copies on the same machine.
- **WRDS (Compustat/CRSP)**: licensed to Stony Brook users. It is only used in `merge_all.py`
  on the machine of the WRDS account holder. Never upload WRDS files to GitHub or share them
  outside the team.
- **Logos are company trademarks**: we use them only for research analysis. Keep image folders
  and data out of GitHub (`data/` is in `.gitignore`); the report shows only a few examples.
- **Data never goes on GitHub**: only code. Check `git status` before every commit.
