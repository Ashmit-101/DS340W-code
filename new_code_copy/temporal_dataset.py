"""
temporal_dataset.py — Sequence dataset for temporal modeling of fitness → cognition

Consumes the output of reading_data.py (train_final, test_final, val_final) and
behavioral_summary.pkl to produce sliding-window PyTorch Datasets.

Data layout produced by reading_data.py:
    columns: participant_id | datetime | bmi | cal | cal_bmr | distance |
             fair_act_mins | food_cal_log | light_act_mins | sed_mins |
             steps | very_act_mins | water_log | weight |
             steps_rel_change | distance_rel_change | cal_rel_change
    rows: one row per (participant, day), sorted by participant then date

behavioral_summary.pkl layout (produced by loader.py pipeline):
    pandas DataFrame, index = participant_id, columns = behavioral score names
    e.g.: free_recall_accuracy | temporal_clustering | semantic_clustering |
          pfr_primacy | pfr_recency | lag_crp_forward | lag_crp_backward |
          nat_recall_immediate | nat_recall_delayed | ...
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── project paths (mirror reading_data.py) ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_copy"

# ── feature columns (must match reading_data.py output after bodyfat removal) ─
FITNESS_FEATURE_COLS = [
    "bmi", "cal", "cal_bmr", "distance",
    "fair_act_mins", "food_cal_log", "light_act_mins", "sed_mins",
    "steps", "very_act_mins", "water_log", "weight",
    "steps_rel_change", "distance_rel_change", "cal_rel_change",
]
N_FEATURES = len(FITNESS_FEATURE_COLS)  # 15


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FitnessSequenceDataset(Dataset):
    """
    One sample = (X, y) where:
        X : float32 tensor  (window_days, N_FEATURES)  — standardised daily fitness
        y : float32 tensor  (n_tasks,)                 — behavioural scores

    Sliding-window construction per participant:
        For each valid end-index `t` in [window_days, len(seq) - lag_days):
            X = fitness[t - window_days : t]     # history window
            y = behavioral score(s) for this participant

    With lag_days > 0 the fitness window ends `lag_days` before the score,
    so the model can only see fitness *prior* to assessment — the key setup
    for the lag analysis described in the paper extension.

    Args:
        fitness_df   : preprocessed DataFrame from reading_data.py
        behavioral_df: DataFrame indexed by participant_id, columns = task scores
        task_cols    : list of column names from behavioral_df to predict;
                       None = use all columns
        window_days  : number of consecutive days in each input window (default 90)
        lag_days     : gap (days) between end of window and score assessment (default 0)
        min_seq_len  : skip participants with fewer usable days than this
        stride       : step size between consecutive windows (default 1)
    """

    def __init__(
        self,
        fitness_df: pd.DataFrame,
        behavioral_df: pd.DataFrame,
        task_cols: list[str] | None = None,
        window_days: int = 90,
        lag_days: int = 0,
        min_seq_len: int | None = None,
        stride: int = 1,
    ):
        if task_cols is None:
            task_cols = [c for c in behavioral_df.columns
                         if c != "participant_id"]

        self.task_cols = task_cols
        self.window_days = window_days
        self.lag_days = lag_days
        self.n_tasks = len(task_cols)

        if min_seq_len is None:
            min_seq_len = window_days + lag_days + 1

        # make sure behavioral_df is indexed by participant_id
        if behavioral_df.index.name != "participant_id":
            if "participant_id" in behavioral_df.columns:
                behavioral_df = behavioral_df.set_index("participant_id")

        # shared participants only
        fitness_pids = set(fitness_df["participant_id"].unique())
        behav_pids = set(behavioral_df.index)
        shared = fitness_pids & behav_pids
        if len(shared) == 0:
            raise ValueError(
                "No participant IDs overlap between fitness_df and behavioral_df. "
                "Check that participant_id values match (e.g. 'BFM_AMT_001' in both)."
            )

        missing_tasks = [c for c in task_cols if c not in behavioral_df.columns]
        if missing_tasks:
            raise ValueError(f"task_cols not found in behavioral_df: {missing_tasks}")

        # build (X_window, y_vector, participant_id) triples
        self.samples: list[tuple[np.ndarray, np.ndarray, str]] = []
        skipped = 0

        for pid, group in fitness_df.groupby("participant_id"):
            if pid not in shared:
                continue

            group = group.sort_values("datetime")
            seq = group[FITNESS_FEATURE_COLS].values.astype(np.float32)  # (T, 15)

            if len(seq) < min_seq_len:
                skipped += 1
                continue

            y = behavioral_df.loc[pid, task_cols].values.astype(np.float32)  # (n_tasks,)

            # sliding windows: end index runs from window_days to T - lag_days
            end_max = len(seq) - lag_days
            for end in range(window_days, end_max, stride):
                window = seq[end - window_days : end]  # (window_days, 15)
                self.samples.append((window, y, pid))

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples created. Skipped {skipped} participants (too short). "
                f"Try reducing window_days ({window_days}) or lag_days ({lag_days})."
            )

        print(
            f"FitnessSequenceDataset: {len(self.samples)} windows from "
            f"{len(shared)} participants | window={window_days}d lag={lag_days}d "
            f"tasks={task_cols} | skipped {skipped} (too short)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window, y, _ = self.samples[idx]
        return torch.from_numpy(window), torch.from_numpy(y)

    def participant_ids(self) -> list[str]:
        return [pid for _, _, pid in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_behavioral_summary(path: Path | None = None) -> pd.DataFrame:
    """
    Load behavioral_summary.pkl produced by the paper's analysis notebooks.
    Expected index: participant_id
    Expected columns: free_recall_accuracy, temporal_clustering,
                      semantic_clustering, pfr_primacy, pfr_recency,
                      lag_crp_forward, lag_crp_backward,
                      nat_recall_immediate, nat_recall_delayed, ...
    """
    if path is None:
        path = DATA_DIR / "behavioral_summary.pkl"
    with open(path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"behavioral_summary.pkl should be a DataFrame, got {type(df)}")
    print(f"Loaded behavioral summary: {df.shape}  index={df.index.name}  cols={list(df.columns)}")
    return df


def make_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    behavioral_df: pd.DataFrame,
    task_cols: list[str] | None = None,
    window_days: int = 90,
    lag_days: int = 0,
    stride: int = 1,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function: build train/test/val DataLoaders in one call.

    Returns:
        (train_loader, test_loader, val_loader)

    Usage:
        from reading_data import train_final, test_final, val_final
        behavioral_df = load_behavioral_summary()
        train_loader, test_loader, val_loader = make_dataloaders(
            train_final, test_final, val_final, behavioral_df,
            task_cols=["free_recall_accuracy", "nat_recall_immediate"],
            window_days=90, lag_days=14,
        )
    """
    train_ds = FitnessSequenceDataset(
        train_df, behavioral_df, task_cols, window_days, lag_days, stride=stride
    )
    test_ds = FitnessSequenceDataset(
        test_df, behavioral_df, task_cols, window_days, lag_days, stride=stride
    )
    val_ds = FitnessSequenceDataset(
        val_df, behavioral_df, task_cols, window_days, lag_days, stride=stride
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Build a tiny synthetic dataset to verify shapes without needing real pkl
    print("Running shape verification with synthetic data...")

    n_participants = 20
    n_days = 200
    n_tasks = 3
    window_days = 90
    lag_days = 14

    rng = np.random.default_rng(0)

    rows = []
    for i in range(n_participants):
        pid = f"BFM_AMT_{i:03d}"
        dates = pd.date_range("2020-01-01", periods=n_days)
        for d in dates:
            row = {"participant_id": pid, "datetime": d}
            row.update({col: rng.random() for col in FITNESS_FEATURE_COLS})
            rows.append(row)
    fitness_df = pd.DataFrame(rows)

    task_names = ["free_recall_accuracy", "nat_recall_immediate", "temporal_clustering"]
    behav_rows = {
        f"BFM_AMT_{i:03d}": rng.random(n_tasks)
        for i in range(n_participants)
    }
    behavioral_df = pd.DataFrame.from_dict(
        behav_rows, orient="index", columns=task_names
    )
    behavioral_df.index.name = "participant_id"

    ds = FitnessSequenceDataset(
        fitness_df, behavioral_df,
        task_cols=task_names,
        window_days=window_days,
        lag_days=lag_days,
    )

    X, y = ds[0]
    print(f"\nSample shapes — X: {tuple(X.shape)}  y: {tuple(y.shape)}")
    assert X.shape == (window_days, N_FEATURES), f"X shape mismatch: {X.shape}"
    assert y.shape == (n_tasks,), f"y shape mismatch: {y.shape}"
    print("Shape verification passed.")
