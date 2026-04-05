"""
ridge_regression.py — Baseline Ridge regression: fitness summary → cognition

Collapses each participant's daily fitness time-series to one row
(mean + std + linear trend per feature), then fits RidgeCV per task.
Serves as a static baseline to compare against the temporal LSTM/Transformer.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_copy"

# ── Step 1: Load fitness data ─────────────────────────────────────────────────
print("=" * 60)
print("Loading fitness data (reading_data.py)...")
print("=" * 60)
from reading_data import train_final, test_final, val_final

feature_cols = [c for c in train_final.columns
                if c not in ("datetime", "participant_id")]
print(f"Features: {len(feature_cols)}  {feature_cols}")

# ── Step 2: Load + clean behavioral scores ────────────────────────────────────
print("\n" + "=" * 60)
print("Loading behavioral scores...")
print("=" * 60)

key_df = pd.read_csv(DATA_DIR / "raw_formatted" / "id_filename_key.csv", index_col=0)
p_to_bfm = {f"P{i}": row["filenames"] for i, row in key_df.iterrows()}

behavioral_summary = pd.read_pickle(DATA_DIR / "behavioral_summary.pkl")
behavioral_summary.index = [p_to_bfm[p] for p in behavioral_summary.index]
behavioral_summary.index.name = "participant_id"
behavioral_summary = behavioral_summary.dropna(axis=1).dropna(axis=0)

print(f"Behavioral shape: {behavioral_summary.shape}")
print(f"Tasks: {list(behavioral_summary.columns)}")

# ── Step 3: Aggregate each participant to one row ─────────────────────────────
print("\n" + "=" * 60)
print("Aggregating participants (mean + std + trend per feature)...")
print("=" * 60)

def aggregate_participant(df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: mean, std, and linear trend per feature."""
    rows = []
    for pid, grp in df.groupby("participant_id"):
        grp = grp.sort_values("datetime")
        X = grp[feature_cols].values.astype(float)
        n = len(X)
        t = np.arange(n) / n  # normalised time 0→1

        feats: dict = {}
        for i, col in enumerate(feature_cols):
            feats[f"{col}_mean"]  = X[:, i].mean()
            feats[f"{col}_std"]   = X[:, i].std()
            feats[f"{col}_trend"] = np.polyfit(t, X[:, i], 1)[0]
        feats["participant_id"] = pid
        rows.append(feats)

    return pd.DataFrame(rows).set_index("participant_id")

train_agg = aggregate_participant(train_final)
test_agg  = aggregate_participant(test_final)
val_agg   = aggregate_participant(val_final)

print(f"Train agg: {train_agg.shape}")
print(f"Test  agg: {test_agg.shape}")
print(f"Val   agg: {val_agg.shape}")

# ── Step 4: Align labels, drop participants missing from behavioral_summary ────
task_cols = list(behavioral_summary.columns)

train_idx = train_agg.index.intersection(behavioral_summary.index)
test_idx  = test_agg.index.intersection(behavioral_summary.index)
val_idx   = val_agg.index.intersection(behavioral_summary.index)

train_agg = train_agg.loc[train_idx]
test_agg  = test_agg.loc[test_idx]
val_agg   = val_agg.loc[val_idx]

y_train = behavioral_summary.loc[train_idx, task_cols].values.astype(float)
y_test  = behavioral_summary.loc[test_idx,  task_cols].values.astype(float)
y_val   = behavioral_summary.loc[val_idx,   task_cols].values.astype(float)

print(f"\nAligned — train: {len(train_idx)}  test: {len(test_idx)}  val: {len(val_idx)} participants")

# ── Step 5: Fit RidgeCV ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Fitting RidgeCV (alphas=[0.01, 0.1, 1, 10, 100], cv=5)...")
print("=" * 60)

ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge.fit(train_agg.values, y_train)
print(f"Best alpha (per task): {ridge.alpha_}")

# ── Step 6: Evaluate on test set ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Test set R² per task")
print("=" * 60)

y_pred = ridge.predict(test_agg.values)
r2s = []
for i, task in enumerate(task_cols):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    r2s.append(r2)
    print(f"  {task:<45s}  R²={r2:+.4f}")

print(f"\n  Mean R²: {np.mean(r2s):+.4f}")

# ── Step 7: Evaluate on val set ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Validation set R² per task")
print("=" * 60)

y_pred_val = ridge.predict(val_agg.values)
r2s_val = []
for i, task in enumerate(task_cols):
    r2 = r2_score(y_val[:, i], y_pred_val[:, i])
    r2s_val.append(r2)
    print(f"  {task:<45s}  R²={r2:+.4f}")

print(f"\n  Mean R²: {np.mean(r2s_val):+.4f}")


# ── Step 8: Permutation test ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Permutation test (n=1000) — is real R² above chance?")
print("=" * 60)

N_PERMS = 1000
rng = np.random.default_rng(42)

# null_r2s[i] = list of 1000 R² values for task i under shuffled labels
null_r2s = [[] for _ in task_cols]

for perm in range(N_PERMS):
    shuffled_idx = rng.permutation(len(y_train))
    y_train_perm = y_train[shuffled_idx]

    r_perm = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
    r_perm.fit(train_agg.values, y_train_perm)
    y_pred_perm = r_perm.predict(test_agg.values)

    for i in range(len(task_cols)):
        null_r2s[i].append(r2_score(y_test[:, i], y_pred_perm[:, i]))

    if (perm + 1) % 100 == 0:
        print(f"  {perm + 1}/{N_PERMS} permutations done...")

print()
print(f"  {'Task':<45}  {'Real R²':>8}  {'Null mean':>10}  {'Null std':>9}  {'Percentile':>11}  Verdict")
print("  " + "-" * 100)

perm_results = []
for i, task in enumerate(task_cols):
    null = np.array(null_r2s[i])
    real = r2s[i]
    percentile = float(np.mean(null < real)) * 100
    null_mean  = null.mean()
    null_std   = null.std()

    if percentile >= 95:
        verdict = "significant (p<0.05)"
    elif percentile >= 90:
        verdict = "marginal (p<0.10)"
    elif percentile >= 75:
        verdict = "weak signal"
    else:
        verdict = "chance / noise"

    perm_results.append((task, real, null_mean, null_std, percentile, verdict))
    print(f"  {task:<45}  {real:>+8.4f}  {null_mean:>+10.4f}  {null_std:>9.4f}  "
          f"{percentile:>10.1f}%  {verdict}")

print()
sig_tasks = [r for r in perm_results if r[4] >= 90]
if sig_tasks:
    print(f"  Tasks above 90th percentile of null: {[r[0] for r in sig_tasks]}")
else:
    print("  No tasks exceeded 90th percentile — results are consistent with chance.")
    print("  Interpretation: N is likely underpowered for this prediction task.")


# ── Step 9: Coefficient analysis ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Ridge coefficient analysis — which features drive predictions?")
print("=" * 60)

agg_feature_names = train_agg.columns.tolist()
coef_df = pd.DataFrame(
    ridge.coef_,           # shape (n_tasks, n_agg_features)
    index=task_cols,
    columns=agg_feature_names,
)

# Top-5 features by mean absolute coefficient across all tasks
mean_abs_coef = coef_df.abs().mean(axis=0).sort_values(ascending=False)

print(f"\n  Top 15 features by mean |coefficient| across all tasks:")
print(f"  {'Feature':<35}  {'Mean |coef|':>12}  {'Sign pattern'}")
print("  " + "-" * 75)
for feat, mean_abs in mean_abs_coef.head(15).items():
    signs = "".join(
        "+" if coef_df.loc[task, feat] > 0 else "-"
        for task in task_cols
    )
    print(f"  {feat:<35}  {mean_abs:>12.4f}  [{signs}]  ({task_cols[0][0]}..)")

# Per-task top-3 features
print(f"\n  Top 3 features per task:")
print("  " + "-" * 75)
for task in task_cols:
    top3 = coef_df.loc[task].abs().nlargest(3)
    parts = [f"{f} ({coef_df.loc[task, f]:+.3f})" for f in top3.index]
    print(f"  {task:<45}  {',  '.join(parts)}")

# Manning-aligned features: check if activity features have expected sign
print(f"\n  Manning-alignment check (activity ↑ → recall ↑, expected positive):")
print("  " + "-" * 75)
manning_features = [f for f in agg_feature_names
                    if any(k in f for k in ("steps", "very_act_mins", "distance", "sed_mins"))]
for feat in manning_features:
    row = coef_df[feat]
    signs = {task: ("+") if row[task] > 0 else "-" for task in task_cols}
    sign_str = "  ".join(f"{t[:20]}: {s}" for t, s in signs.items())
    print(f"  {feat:<35}  {sign_str}")

print("\n  Interpretation guide:")
print("  • High |coef| + sign consistent with Manning → feature-level replication")
print("  • Low percentile in permutation test → effect too small to generalise at N=", len(train_idx))


# ── Step 10: Visualisations ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
short_tasks = [t.replace("(", "").replace(")", "").strip() for t in task_cols]

# ── Plot 1: Permutation test — null distributions with real R² marked ─────────
n_tasks = len(task_cols)
ncols = min(n_tasks, 4)
nrows = (n_tasks + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for i, (task, real, null_mean, null_std, pct, verdict) in enumerate(perm_results):
    ax = axes[i]
    null = np.array(null_r2s[i])

    ax.hist(null, bins=40, color="steelblue", alpha=0.75, edgecolor="white", label="Null (shuffled)")
    ax.axvline(real, color="crimson", linewidth=2.2, label=f"Real R²={real:+.3f}")
    ax.axvline(np.percentile(null, 95), color="orange", linewidth=1.5,
               linestyle="--", label="p=0.05 threshold")

    colour = "crimson" if pct >= 95 else ("darkorange" if pct >= 90 else "gray")
    ax.set_title(f"{short_tasks[i]}", fontsize=11, fontweight="bold")
    ax.set_xlabel("R² under shuffled labels", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8)
    ax.annotate(
        f"{pct:.0f}th pct\n{verdict}",
        xy=(real, ax.get_ylim()[1] * 0.85),
        fontsize=8, color=colour, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour, alpha=0.8),
    )

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Permutation Test: Real R² vs Null Distribution (1000 shuffles)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
path1 = RESULTS_DIR / "ridge_permutation_test.png"
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path1}")

# ── Plot 2: Coefficient heatmap (top 20 features × tasks) ─────────────────────
top20_feats = mean_abs_coef.head(20).index.tolist()
heatmap_data = coef_df[top20_feats].T          # (20 features, n_tasks)
heatmap_data.index = [f.replace("_", " ") for f in heatmap_data.index]
heatmap_data.columns = short_tasks

vmax = heatmap_data.abs().values.max()

fig, ax = plt.subplots(figsize=(max(8, n_tasks * 1.8), 9))
sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    vmin=-vmax, vmax=vmax,
    annot=True, fmt=".2f", annot_kws={"size": 8},
    linewidths=0.4, linecolor="lightgray",
    cbar_kws={"label": "Ridge coefficient", "shrink": 0.8},
)
ax.set_title("Ridge Coefficients: Top 20 Features × Cognitive Tasks",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Cognitive task", fontsize=11)
ax.set_ylabel("Fitness feature", fontsize=11)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(fontsize=8)
plt.tight_layout()
path2 = RESULTS_DIR / "ridge_coefficient_heatmap.png"
fig.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path2}")

# ── Plot 3: Mean |coefficient| bar chart — overall feature importance ──────────
top15 = mean_abs_coef.head(15)
colours = ["#2196F3" if any(k in f for k in ("steps", "very_act_mins", "distance", "sed_mins"))
           else "#90A4AE" for f in top15.index]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(
    [f.replace("_", " ") for f in top15.index[::-1]],
    top15.values[::-1],
    color=colours[::-1], edgecolor="white",
)
ax.set_xlabel("Mean |Ridge coefficient| across tasks", fontsize=11)
ax.set_title("Feature Importance: Mean |Coefficient| (blue = Manning activity features)",
             fontsize=12, fontweight="bold")
ax.axvline(top15.values.mean(), color="crimson", linestyle="--",
           linewidth=1.5, label=f"Mean = {top15.values.mean():.3f}")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
path3 = RESULTS_DIR / "ridge_feature_importance.png"
fig.savefig(path3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path3}")

# ── Plot 4: Manning activity features — coefficient per task ───────────────────
manning_feats = [f for f in agg_feature_names
                 if any(k in f for k in ("steps", "very_act_mins", "distance", "sed_mins"))]
manning_coef = coef_df[manning_feats].T
manning_coef.index = [f.replace("_", " ") for f in manning_coef.index]
manning_coef.columns = short_tasks

fig, ax = plt.subplots(figsize=(max(8, n_tasks * 1.8), max(4, len(manning_feats) * 0.55)))
sns.heatmap(
    manning_coef, ax=ax,
    cmap="RdBu_r", center=0,
    vmin=-manning_coef.abs().values.max(),
    vmax=manning_coef.abs().values.max(),
    annot=True, fmt=".3f", annot_kws={"size": 9},
    linewidths=0.5, linecolor="lightgray",
    cbar_kws={"label": "Coefficient (+ = activity ↑ → cognition ↑)", "shrink": 0.8},
)
ax.set_title(
    "Manning-Alignment Check: Activity Feature Coefficients\n"
    "(blue = positive, consistent with Manning; red = negative, contradicts)",
    fontsize=11, fontweight="bold", pad=12,
)
ax.set_xlabel("Cognitive task", fontsize=10)
ax.set_ylabel("Activity feature", fontsize=10)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
path4 = RESULTS_DIR / "ridge_manning_alignment.png"
fig.savefig(path4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path4}")

# ── Plot 5: Real R² vs permutation percentile summary bar chart ───────────────
fig, ax = plt.subplots(figsize=(max(8, n_tasks * 1.5), 5))
percentiles = [r[4] for r in perm_results]
bar_colours = ["#4CAF50" if p >= 95 else "#FF9800" if p >= 90 else "#90A4AE"
               for p in percentiles]
bars = ax.bar(short_tasks, percentiles, color=bar_colours, edgecolor="white", width=0.6)
ax.axhline(95, color="green",  linestyle="--", linewidth=1.5, label="p<0.05 (95th pct)")
ax.axhline(90, color="orange", linestyle="--", linewidth=1.5, label="p<0.10 (90th pct)")
ax.set_ylim(0, 105)
ax.set_ylabel("Percentile of real R² in null distribution", fontsize=11)
ax.set_title("Permutation Test Summary: How unlikely is each real R²?",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.xticks(rotation=25, ha="right", fontsize=9)

for bar, pct in zip(bars, percentiles):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{pct:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
path5 = RESULTS_DIR / "ridge_permutation_summary.png"
fig.savefig(path5, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path5}")

print("\n" + "=" * 60)
print("All plots saved to results/")
print("=" * 60)
