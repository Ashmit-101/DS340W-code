import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_copy"

# ── Load behavioral data ──────────────────────────────────────────────────────
data = pd.read_pickle(DATA_DIR / "behavioral_summary.pkl")
print("behavioral_summary shape:", data.shape)
print("NaN per task:")
print(data.isna().sum())
print()

# ── Remap P* → BFM_AMT_* ─────────────────────────────────────────────────────
key_df = pd.read_csv(DATA_DIR / "raw_formatted" / "id_filename_key.csv", index_col=0)
p_to_bfm = {f"P{i}": row["filenames"] for i, row in key_df.iterrows()}
data.index = [p_to_bfm[p] for p in data.index]
data.index.name = "participant_id"

# Drop NaN tasks/participants (same logic as testing.py)
data = data.dropna(axis=1).dropna(axis=0)
print(f"After NaN drop: {data.shape}  tasks: {list(data.columns)}")
print()

# ── Build dataset on train split ──────────────────────────────────────────────
print("Loading fitness data (reading_data.py)...")
from reading_data import train_final

from temporal_dataset import FitnessSequenceDataset

dataset = FitnessSequenceDataset(
    fitness_df=train_final,
    behavioral_df=data,
    task_cols=list(data.columns),
    window_days=90,
    lag_days=14,
)

# ── Diagnostic: unique labels per participant ─────────────────────────────────
# Each participant gets one fixed cognitive score repeated across all their windows.
# If unique_labels == 1, the dataset is working correctly (same y for all windows).
print("\n── Label uniqueness per participant ──────────────────────────────────")
print(f"{'participant_id':<20} {'windows':>8} {'unique_labels':>14}  status")
print("-" * 60)

pid_windows = defaultdict(list)
for window, y, pid in dataset.samples:
    pid_windows[pid].append(tuple(y.tolist()))

all_ok = True
for pid, labels in sorted(pid_windows.items()):
    n_windows = len(labels)
    n_unique = len(set(labels))
    status = "OK" if n_unique == 1 else "MISMATCH"
    if n_unique != 1:
        all_ok = False
    print(f"{pid:<20} {n_windows:>8} {n_unique:>14}  {status}")

print("-" * 60)
if all_ok:
    print("All participants: 1 unique label per participant (correct)")
else:
    print("WARNING: some participants have mismatched labels across windows")
