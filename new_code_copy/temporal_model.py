"""
temporal_model.py — LSTM and Transformer models for fitness → cognition prediction

Extends Manning et al. (2022) from static bootstrap correlations to temporal
sequence modelling.  Two architectures are provided:

    FitnessLSTM        — bidirectional LSTM, good baseline
    FitnessTransformer — causal Transformer, captures long-range interactions,
                         produces interpretable attention weights

Training utilities:
    Trainer            — training loop with early stopping
    run_lag_analysis   — sweeps lag_days values, plots R² vs lag

Input tensor shape:  (batch, window_days, N_FEATURES)   — e.g. (32, 90, 15)
Output tensor shape: (batch, n_tasks)                   — e.g. (32, 3)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from temporal_dataset import (
    FITNESS_FEATURE_COLS,
    FitnessSequenceDataset,
    N_FEATURES,
    make_dataloaders,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 1 — LSTM
# ─────────────────────────────────────────────────────────────────────────────

class FitnessLSTM(nn.Module):
    """
    Bidirectional LSTM over daily fitness sequences.

    Architecture:
        Input projection  (N_FEATURES → d_model)
        BiLSTM stack      (d_model → 2 * hidden_size per layer)
        Final hidden      (2 * hidden_size of last layer)
        Prediction head   (→ n_tasks)

    Why bidirectional? The sequence represents historical fitness leading up to
    an assessment — we know the full window at inference time, so non-causal
    modelling is appropriate here.  If you want a strictly causal variant
    (e.g. for forecasting), set bidirectional=False.

    Args:
        n_tasks     : number of behavioural scores to predict
        d_model     : projection dimension before LSTM (default 32)
        hidden_size : LSTM hidden state size per direction (default 64)
        n_layers    : number of stacked LSTM layers (default 2)
        dropout     : dropout between LSTM layers (default 0.3)
    """

    def __init__(
        self,
        n_tasks: int,
        d_model: int = 32,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(N_FEATURES, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        lstm_out_dim = hidden_size * 2  # bidirectional
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, n_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, window_days, N_FEATURES)
        Returns:
            (batch, n_tasks)
        """
        x = self.input_proj(x)              # (B, T, d_model)
        out, (h_n, _) = self.lstm(x)        # h_n: (2*n_layers, B, hidden)
        # concatenate the last forward and backward hidden states
        h_fwd = h_n[-2]                     # (B, hidden)
        h_bwd = h_n[-1]                     # (B, hidden)
        h = torch.cat([h_fwd, h_bwd], dim=-1)   # (B, 2*hidden)
        h = self.norm(h)
        return self.head(h)                 # (B, n_tasks)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture 2 — Temporal Transformer
# ─────────────────────────────────────────────────────────────────────────────

class FitnessTransformer(nn.Module):
    """
    Transformer encoder over daily fitness sequences with learnable positional
    embeddings.  Each day is one token; the model attends over the full window.

    Attention weights are preserved for interpretability — you can inspect
    which days the model attends to when predicting each cognitive score.

    Architecture:
        Input projection    (N_FEATURES → d_model)
        Positional embedding (window_days → d_model)
        Transformer encoder (n_layers × multi-head attention)
        Mean pooling        over time dimension
        Prediction head     (→ n_tasks)

    Args:
        n_tasks    : number of behavioural scores to predict
        window_days: sequence length (must match dataset window_days)
        d_model    : token embedding dimension (default 64)
        nhead      : number of attention heads (default 4; must divide d_model)
        n_layers   : number of Transformer encoder layers (default 3)
        d_ff       : feedforward inner dimension (default 256)
        dropout    : dropout in attention and feedforward (default 0.1)
    """

    def __init__(
        self,
        n_tasks: int,
        window_days: int = 90,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.window_days = window_days
        self.d_model = d_model

        self.input_proj = nn.Linear(N_FEATURES, d_model)
        self.pos_embedding = nn.Embedding(window_days, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # pre-norm: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_tasks),
        )

        # Store attention weights for the last forward pass (set by hook)
        self._attn_weights: list[torch.Tensor] = []
        self._register_attn_hooks()

    def _register_attn_hooks(self):
        """Register forward hooks to capture attention weights per layer."""
        self._attn_weights = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # TransformerEncoderLayer returns (output,) — attention weights
                # are NOT returned by default.  We re-run attn manually here.
                pass
            return hook

        # Note: PyTorch's TransformerEncoderLayer does not expose attention
        # weights directly.  To get them, call get_attention_weights() below,
        # which runs a separate forward pass with need_weights=True.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, window_days, N_FEATURES)
        Returns:
            (batch, n_tasks)
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)       # (T,)
        tokens = self.input_proj(x) + self.pos_embedding(positions)  # (B, T, d_model)
        encoded = self.encoder(tokens)                      # (B, T, d_model)
        pooled = self.norm(encoded.mean(dim=1))             # (B, d_model)
        return self.head(pooled)                            # (B, n_tasks)

    @torch.no_grad()
    def get_attention_weights(
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Return per-layer attention weight matrices for a batch.

        Returns:
            list of (batch, nhead, window_days, window_days) tensors,
            one per Transformer layer.

        Usage:
            attn = model.get_attention_weights(X_batch)   # list of 3 tensors
            # Average over heads and layers:
            avg_attn = torch.stack(attn).mean(dim=[0, 2])  # (B, T, T)
            # Row-slice at last token to see what day-90 attended to:
            attn_profile = avg_attn[0, -1, :]              # (T,)
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)
        tokens = self.input_proj(x) + self.pos_embedding(positions)

        attn_per_layer = []
        h = tokens
        for layer in self.encoder.layers:
            # Use the layer's self-attention module directly
            attn_out, attn_w = layer.self_attn(
                h, h, h, need_weights=True, average_attn_weights=False
            )
            attn_per_layer.append(attn_w)   # (B, nhead, T, T)
            # Complete the rest of the layer manually (post-attn path)
            h = layer(h)

        return attn_per_layer


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Coefficient of determination averaged across tasks."""
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum(dim=0)
    r2_per_task = 1 - ss_res / (ss_tot + 1e-8)
    return r2_per_task.mean().item()


class Trainer:
    """
    Train a temporal model with MSE loss and early stopping.

    Args:
        model       : FitnessLSTM or FitnessTransformer
        device      : 'cuda', 'mps', or 'cpu'
        lr          : initial learning rate (default 3e-4)
        weight_decay: L2 regularisation (default 1e-4)
        patience    : early stopping patience in epochs (default 10)
        max_epochs  : maximum training epochs (default 100)

    Usage:
        trainer = Trainer(model, device='cpu')
        history = trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        patience: int = 10,
        max_epochs: int = 100,
        verbose: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose

        self.criterion = nn.MSELoss()
        self.optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimiser, T_max=max_epochs)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[float, float]:
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        pred = self.model(X)
        loss = self.criterion(pred, y)
        r2 = r2_score(pred.detach(), y.detach())
        return loss, r2

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool | None = None,
    ) -> dict[str, list[float]]:
        """
        Run training loop.

        Returns:
            history dict with keys: train_loss, val_loss, train_r2, val_r2
        """
        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []
        }
        if verbose is None:
            verbose = self.verbose

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.max_epochs + 1):
            # ── train ──
            self.model.train()
            train_losses, train_r2s = [], []
            for batch in train_loader:
                self.optimiser.zero_grad()
                loss, r2 = self._step(batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimiser.step()
                train_losses.append(loss.item())
                train_r2s.append(r2)
            self.scheduler.step()

            # ── validate ──
            self.model.eval()
            val_losses, val_r2s = [], []
            with torch.no_grad():
                for batch in val_loader:
                    loss, r2 = self._step(batch)
                    val_losses.append(loss.item())
                    val_r2s.append(r2)

            t_loss = np.mean(train_losses)
            v_loss = np.mean(val_losses)
            t_r2   = np.mean(train_r2s)
            v_r2   = np.mean(val_r2s)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_r2"].append(t_r2)
            history["val_r2"].append(v_r2)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"train loss {t_loss:.4f}  R²={t_r2:.3f} | "
                    f"val loss {v_loss:.4f}  R²={v_r2:.3f}"
                )

            # early stopping
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """
        Compute MSE and R² on a DataLoader split.

        Returns:
            {"mse": float, "r2": float, "r2_per_task": list[float]}
        """
        self.model.eval()
        all_pred, all_true = [], []
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            all_pred.append(pred)
            all_true.append(y)

        pred = torch.cat(all_pred, dim=0)
        true = torch.cat(all_true, dim=0)

        mse = nn.functional.mse_loss(pred, true).item()
        ss_res = ((true - pred) ** 2).sum(dim=0)
        ss_tot = ((true - true.mean(dim=0)) ** 2).sum(dim=0)
        r2_per_task = (1 - ss_res / (ss_tot + 1e-8)).tolist()
        r2_mean = float(np.mean(r2_per_task))

        return {"mse": mse, "r2": r2_mean, "r2_per_task": r2_per_task}


# ─────────────────────────────────────────────────────────────────────────────
# Lag analysis — the core scientific extension
# ─────────────────────────────────────────────────────────────────────────────

def run_lag_analysis(
    train_df,
    test_df,
    val_df,
    behavioral_df,
    task_cols: list[str],
    lag_values: list[int] | None = None,
    window_days: int = 90,
    model_type: str = "lstm",  # "lstm" or "transformer"
    n_epochs: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
    save_plot: bool = True,
) -> dict[int, dict]:
    """
    Train a separate model for each lag value and record test R².

    This is the key analysis extending the paper: if R² peaks at lag > 0,
    it means fitness N days *before* assessment predicts cognition — supporting
    a predictive rather than merely concurrent relationship.

    Args:
        lag_values  : list of lag_days values to sweep (default [0,7,14,21,30,45,60,90])
        model_type  : "lstm" or "transformer"

    Returns:
        results dict: {lag_days: {"r2": float, "r2_per_task": list, "mse": float}}

    Saves:
        results/lag_analysis.png — R² vs lag plot
        results/lag_analysis_per_task.png — per-task R² vs lag
    """
    if lag_values is None:
        lag_values = [0, 7, 14, 21, 30, 45, 60, 90]

    n_tasks = len(task_cols)
    results: dict[int, dict] = {}

    for lag in lag_values:
        print(f"\n{'─'*60}")
        print(f"Lag = {lag} days")
        print(f"{'─'*60}")

        try:
            train_loader, test_loader, val_loader = make_dataloaders(
                train_df, test_df, val_df, behavioral_df,
                task_cols=task_cols,
                window_days=window_days,
                lag_days=lag,
                batch_size=batch_size,
            )
        except ValueError as e:
            print(f"  Skipping lag={lag}: {e}")
            continue

        if model_type == "lstm":
            model = FitnessLSTM(n_tasks=n_tasks)
        elif model_type == "transformer":
            model = FitnessTransformer(n_tasks=n_tasks, window_days=window_days)
        else:
            raise ValueError(f"model_type must be 'lstm' or 'transformer', got '{model_type}'")

        trainer = Trainer(model, device=device, max_epochs=n_epochs, patience=8, verbose=False)
        trainer.fit(train_loader, val_loader, verbose=False)
        metrics = trainer.evaluate(test_loader)
        results[lag] = metrics
        print(f"  Test R² = {metrics['r2']:.4f}  MSE = {metrics['mse']:.4f}")
        for tc, r2t in zip(task_cols, metrics["r2_per_task"]):
            print(f"    {tc}: R²={r2t:.4f}")

    # ── plot: mean R² vs lag ──
    lags = sorted(results.keys())
    r2s  = [results[l]["r2"] for l in lags]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, r2s, marker="o", linewidth=2, color="steelblue")
    ax.axvline(lags[int(np.argmax(r2s))], linestyle="--", color="salmon",
               label=f"peak lag = {lags[int(np.argmax(r2s))]}d")
    ax.set_xlabel("Lag (days between fitness window end and cognitive assessment)")
    ax.set_ylabel("Test R²")
    ax.set_title(f"Fitness → Cognition: predictive lag analysis ({model_type.upper()})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot:
        path = RESULTS_DIR / "lag_analysis.png"
        fig.savefig(path, dpi=150)
        print(f"\nSaved lag analysis plot → {path}")
    plt.close(fig)

    # ── plot: per-task R² vs lag ──
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for i, tc in enumerate(task_cols):
        task_r2s = [results[l]["r2_per_task"][i] for l in lags]
        ax2.plot(lags, task_r2s, marker="o", linewidth=2, label=tc)
    ax2.set_xlabel("Lag (days)")
    ax2.set_ylabel("Test R² per task")
    ax2.set_title("Task-specific predictive lag profiles")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot:
        path2 = RESULTS_DIR / "lag_analysis_per_task.png"
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"Saved per-task lag plot → {path2}")
    plt.close(fig2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    print("Running model shape verification with synthetic data...\n")

    rng = np.random.default_rng(42)
    n_participants = 30
    n_days = 200
    n_tasks = 3
    window_days = 90
    lag_days = 14

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
    behavioral_df = pd.DataFrame(
        rng.random((n_participants, n_tasks)),
        index=[f"BFM_AMT_{i:03d}" for i in range(n_participants)],
        columns=task_names,
    )
    behavioral_df.index.name = "participant_id"

    # Split naively: first 20 train, next 5 test, last 5 val
    pids = [f"BFM_AMT_{i:03d}" for i in range(n_participants)]
    train_df = fitness_df[fitness_df["participant_id"].isin(pids[:20])]
    test_df  = fitness_df[fitness_df["participant_id"].isin(pids[20:25])]
    val_df   = fitness_df[fitness_df["participant_id"].isin(pids[25:])]

    train_loader, test_loader, val_loader = make_dataloaders(
        train_df, test_df, val_df, behavioral_df,
        task_cols=task_names, window_days=window_days, lag_days=lag_days, batch_size=16,
    )

    for arch, Model in [("LSTM", FitnessLSTM), ("Transformer", FitnessTransformer)]:
        print(f"\n{'='*50}")
        print(f"Architecture: {arch}")
        if arch == "LSTM":
            model = Model(n_tasks=n_tasks)
        else:
            model = Model(n_tasks=n_tasks, window_days=window_days)

        X_batch, y_batch = next(iter(train_loader))
        out = model(X_batch)
        print(f"  Input:  {tuple(X_batch.shape)}")
        print(f"  Output: {tuple(out.shape)}")
        assert out.shape == (X_batch.shape[0], n_tasks)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")

        trainer = Trainer(model, device="cpu", max_epochs=3, verbose=True)
        trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(test_loader)
        print(f"  Test metrics: {metrics}")

    print("\nAll shape checks passed.")

    # ── Lag analysis ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Running lag analysis (LSTM, synthetic data) ...")
    print("="*60)
    lag_results = run_lag_analysis(
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        behavioral_df=behavioral_df,
        task_cols=task_names,
        lag_values=[0, 7, 14, 21, 30],   # reduced set for quick demo
        window_days=window_days,
        model_type="lstm",
        n_epochs=20,
        batch_size=16,
        device="cpu",
        save_plot=True,
    )
    print("\nLag analysis complete.")
    print(f"PNGs written to: {RESULTS_DIR}")