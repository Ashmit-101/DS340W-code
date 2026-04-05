"""
testing.py — Full experimental pipeline with real data.

Pipeline:
  1. reading_data.py        → daily fitness DataFrames (train / test / val)
  2. behavioral_summary.pkl → cognitive task scores per participant
  3. id_filename_key.csv    → maps row index i to BFM_AMT_* (so Pi = BFM_AMT row i)
  4. temporal_dataset.py    → FitnessSequenceDataset / make_dataloaders
  5. temporal_model.py      → LSTM + Trainer + run_lag_analysis
"""

from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_copy"

# ─── Step 1: Fitness preprocessing ───────────────────────────────────────────
print("=" * 70)
print("STEP 1: Fitness preprocessing (reading_data.py)")
print("=" * 70)
from reading_data import train_final, test_final, val_final

print(f"\nTrain: {train_final.shape}  participants: {train_final['participant_id'].nunique()}")
print(f"Test:  {test_final.shape}  participants: {test_final['participant_id'].nunique()}")
print(f"Val:   {val_final.shape}  participants: {val_final['participant_id'].nunique()}")

# ─── Step 2: Load behavioral scores ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Loading behavioral scores")
print("=" * 70)

behavioral_df = pd.read_pickle(DATA_DIR / "behavioral_summary.pkl")
print(f"Shape: {behavioral_df.shape}")
print(f"Tasks: {list(behavioral_df.columns)}")
print(f"Index sample (before remap): {list(behavioral_df.index[:5])}")

# ─── Step 3: Map participant IDs (Pi → BFM_AMT_*) ────────────────────────────
# id_filename_key.csv row i has filenames=BFM_AMT_XXXX, matching Pi in the pkls
print("\n" + "=" * 70)
print("STEP 3: Mapping P* → BFM_AMT_* participant IDs")
print("=" * 70)

key_df = pd.read_csv(DATA_DIR / "raw_formatted" / "id_filename_key.csv", index_col=0)
p_to_bfm = {f"P{i}": row["filenames"] for i, row in key_df.iterrows()}

missing = [p for p in behavioral_df.index if p not in p_to_bfm]
if missing:
    raise ValueError(f"No BFM mapping for P-IDs: {missing}")

behavioral_df.index = [p_to_bfm[p] for p in behavioral_df.index]
behavioral_df.index.name = "participant_id"
print(f"Remapped index sample: {list(behavioral_df.index[:5])}")

# Verify overlap with fitness data
all_fitness_pids = set(train_final["participant_id"]) | set(test_final["participant_id"]) | set(val_final["participant_id"])
overlap = all_fitness_pids & set(behavioral_df.index)
print(f"\nFitness participants: {len(all_fitness_pids)}")
print(f"Behavioral participants: {len(behavioral_df)}")
print(f"Overlap (will be used): {len(overlap)}")

# ─── Step 4: Build DataLoaders ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Building DataLoaders")
print("=" * 70)

from temporal_dataset import make_dataloaders

# Drop tasks with any NaN across participants — NaN targets cause NaN loss/gradients
before = list(behavioral_df.columns)
behavioral_df = behavioral_df.dropna(axis=1)  # drop columns with any NaN
dropped_tasks = [c for c in before if c not in behavioral_df.columns]
if dropped_tasks:
    print(f"Dropped tasks with NaN values: {dropped_tasks}")

# Drop participants that still have NaN in any remaining task column
n_before = len(behavioral_df)
behavioral_df = behavioral_df.dropna(axis=0)
if len(behavioral_df) < n_before:
    print(f"Dropped {n_before - len(behavioral_df)} participants with NaN behavioral scores")

print(f"Using {len(behavioral_df)} participants, {len(behavioral_df.columns)} tasks: {list(behavioral_df.columns)}")

TASK_COLS = list(behavioral_df.columns)
WINDOW_DAYS = 90
LAG_DAYS = 14
BATCH_SIZE = 32

# Detect device early so DataLoaders can set pin_memory correctly
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
pin = device == "cuda"  # pin_memory only works on CUDA, not MPS

train_loader, test_loader, val_loader = make_dataloaders(
    train_final, test_final, val_final, behavioral_df,
    task_cols=TASK_COLS,
    window_days=WINDOW_DAYS,
    lag_days=LAG_DAYS,
    batch_size=BATCH_SIZE,
    pin_memory=pin,
)

# ─── Step 5: Single-model training (verification) ─────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Training a single LSTM (baseline verification)")
print("=" * 70)

from temporal_model import FitnessLSTM, Trainer

print(f"Device: {device}\n")

model = FitnessLSTM(n_tasks=len(TASK_COLS))
trainer = Trainer(model, device=device, max_epochs=50, patience=8, verbose=True)
trainer.fit(train_loader, val_loader)
metrics = trainer.evaluate(test_loader)

print(f"\nLSTM Test Results:")
print(f"  MSE:  {metrics['mse']:.4f}")
print(f"  R²:   {metrics['r2']:.4f}")
for col, r2 in zip(TASK_COLS, metrics["r2_per_task"]):
    print(f"    {col:<45s}  R²={r2:.4f}")

# ─── Step 6: Lag analysis ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Lag analysis — fitness window timing vs cognition")
print("=" * 70)

from temporal_model import run_lag_analysis

lag_results = run_lag_analysis(
    train_df=train_final,
    test_df=test_final,
    val_df=val_final,
    behavioral_df=behavioral_df,
    task_cols=TASK_COLS,
    lag_values=[0, 7, 14, 21, 30, 45, 60, 90],
    window_days=WINDOW_DAYS,
    model_type="lstm",
    n_epochs=50,
    batch_size=BATCH_SIZE,
    device=device,
    save_plot=True,
)

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print("\nLag analysis summary:")
for lag, m in sorted(lag_results.items()):
    print(f"  lag={lag:3d}d  R²={m['r2']:+.4f}  MSE={m['mse']:.4f}")

best_lag = max(lag_results, key=lambda l: lag_results[l]["r2"])
print(f"\nBest lag: {best_lag} days  (R²={lag_results[best_lag]['r2']:.4f})")
print(f"Plots saved to: {PROJECT_ROOT / 'results'}/")
