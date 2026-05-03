"""
Step 1 — Data Acquisition
Uses the Kaggle CLI via subprocess (avoids the Python API import chain entirely,
which is broken due to a kagglesdk conflict with TF 2.18).
"""
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent       # scripts/
REPO_ROOT    = SCRIPT_DIR.parent                     # airsight/
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Repo root  : {REPO_ROOT}")
print(f"Raw data   : {RAW_DATA_DIR}")
print()

# ── Download via Kaggle CLI ───────────────────────────────────────────────────
DATASET_SLUG = "adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal"

print(f"Downloading '{DATASET_SLUG}' via Kaggle CLI ...")
print(f"Destination : {RAW_DATA_DIR}")
print("This may take several minutes.\n")

result = subprocess.run(
    [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(RAW_DATA_DIR),
        "--unzip",
    ],
    capture_output=False,  # stream output live
    text=True,
)

if result.returncode != 0:
    raise RuntimeError(f"Kaggle CLI failed with exit code {result.returncode}")

print("\n[OK] Download and extraction complete.\n")

# ── Directory tree ────────────────────────────────────────────────────────────
MAX_DEPTH = 3

def print_tree(root: Path, prefix: str = "", depth: int = 0):
    if depth > MAX_DEPTH:
        return
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    for i, entry in enumerate(entries):
        connector = "L-- " if i == len(entries) - 1 else "+-- "
        print(prefix + connector + entry.name)
        if entry.is_dir() and depth < MAX_DEPTH:
            extension = "    " if i == len(entries) - 1 else "|   "
            print_tree(entry, prefix + extension, depth + 1)

print("data/raw/")
print_tree(RAW_DATA_DIR)
print()

# ── File count by extension ───────────────────────────────────────────────────
ext_counter = Counter(
    p.suffix.lower()
    for p in RAW_DATA_DIR.rglob("*")
    if p.is_file()
)

print("File counts by extension:")
for ext, count in ext_counter.most_common():
    label = ext if ext else "(no extension)"
    print(f"  {label:15s}  {count:>6,}")

image_exts   = {".jpg", ".jpeg", ".png", ".webp"}
total_images = sum(ext_counter[e] for e in image_exts)
print(f"\nTotal image files : {total_images:,}")
if total_images == 0:
    print("[WARNING] No image files detected -- verify the dataset slug and retry.")
else:
    print("[OK] Image files found.\n")

# ── CSV preview (cell 1.7 equivalent) ────────────────────────────────────────
csv_files = list(RAW_DATA_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV files found under data/raw/.\n"
        "Check the dataset contents in the tree printed above."
    )

print(f"Found {len(csv_files)} CSV file(s):\n")
for csv_path in csv_files:
    df_full    = pd.read_csv(csv_path)
    df_preview = df_full.head(5)
    print("-" * 60)
    print(f"  File    : {csv_path.relative_to(REPO_ROOT)}")
    print(f"  Shape   : {df_full.shape}")
    print(f"  Columns : {list(df_preview.columns)}")
    print()
    print(df_preview.to_string(index=False))
    print()

# ── Validate required columns (cell 1.8 equivalent) ──────────────────────────
REQUIRED_TARGETS = ["PM2.5", "PM10", "O3", "CO", "SO2", "NO2", "AQI"]

valid_csvs = []
for csv_path in csv_files:
    df      = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_TARGETS if col not in df.columns]
    if not missing:
        valid_csvs.append(csv_path)
        print(f"[OK] {csv_path.name}  -- all 7 target columns present.")
    else:
        print(f"[WARN] {csv_path.name}  -- missing columns: {missing}")

if not valid_csvs:
    raise ValueError(
        "None of the CSVs contain all required target columns.\n"
        f"Required: {REQUIRED_TARGETS}\n"
    )

print("\n[OK] Step 1 complete. Proceed to 02_build_dataset.")
