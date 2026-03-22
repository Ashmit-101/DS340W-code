from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_copy"

# Define the 13 core features used in the paper's Fitness Matrix pipeline
FEATURE_COLS = [
    "bmi",
    "bodyfat",
    "cal",
    "cal_bmr",
    "distance",
    "fair_act_mins",
    "food_cal_log",
    "light_act_mins",
    "sed_mins",
    "steps",
    "very_act_mins",
    "water_log",
    "weight",
]
CORE_FEATURES = FEATURE_COLS.copy()

SPLIT_NAME_MAP = {
    "train": "Train",
    "test": "Test",
    "val": "Validation",
    "validation": "Validation",
}

# Feature categories for paper-aligned imputation and filling
ACTIVITY_METRICS = [
    "steps",
    "distance",
    "cal",
    "cal_bmr",
    "fair_act_mins",
    "light_act_mins",
    "very_act_mins",
    "sed_mins",
]
PHYSICAL_STATE_METRICS = ["weight", "bmi", "bodyfat"]
NUTRITION_METRICS = ["food_cal_log", "water_log"]
REL_CHANGE_BASE_FEATURES = ["steps", "distance", "cal"]


def resolve_split_dir(split_name: str) -> Path:
    folder_name = SPLIT_NAME_MAP.get(split_name.lower(), split_name)
    split_dir = DATA_DIR / folder_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    return split_dir


def load_participant_wide(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    df = df[df["variable"].isin(FEATURE_COLS)].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    participant_wide = (
        df.pivot_table(
            index="datetime",
            columns="variable",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    participant_wide.columns.name = None
    participant_wide.insert(0, "participant_id", file_path.stem)
    return participant_wide


def load_split(split_name: str) -> pd.DataFrame:
    split_dir = resolve_split_dir(split_name)
    participant_files = sorted(split_dir.glob("BFM_AMT_*.csv"))
    split_frames = [load_participant_wide(file_path) for file_path in participant_files]
    split_df = pd.concat(split_frames, ignore_index=True, sort=False)

    for col in FEATURE_COLS:
        if col not in split_df.columns:
            split_df[col] = np.nan

    split_df = split_df[["participant_id", "datetime"] + FEATURE_COLS]
    split_df = split_df.sort_values(["participant_id", "datetime"]).reset_index(drop=True)
    return split_df


def filter_participants_by_data_quality(
    df: pd.DataFrame,
    min_days: int = 100,
    min_non_null_core_features: int = 5,
) -> pd.DataFrame:
    """
    Keep only participants with >= min_days where at least
    min_non_null_core_features are present among core features.
    """
    print(
        "Step 0: Filtering participants by data quality "
        f"(>= {min_days} days with >= {min_non_null_core_features} core features)..."
    )

    core_features_present = [c for c in CORE_FEATURES if c in df.columns]
    quality_flag = df[core_features_present].notna().sum(axis=1) >= min_non_null_core_features
    quality_df = df.assign(_quality_day=quality_flag)

    participant_day_counts = (
        quality_df.groupby("participant_id", as_index=False)["_quality_day"].sum()
        .rename(columns={"_quality_day": "quality_days"})
    )
    participant_day_counts["keep"] = participant_day_counts["quality_days"] >= min_days

    keep_ids = set(participant_day_counts.loc[participant_day_counts["keep"], "participant_id"])
    dropped_ids = sorted(
        participant_day_counts.loc[~participant_day_counts["keep"], "participant_id"].tolist()
    )

    if dropped_ids:
        print(f"   Dropped participants ({len(dropped_ids)}): {', '.join(dropped_ids)}")
    else:
        print("   Dropped participants: none")

    filtered_df = df[df["participant_id"].isin(keep_ids)].copy()
    filtered_df = filtered_df.sort_values(["participant_id", "datetime"]).reset_index(drop=True)
    print(f"   Shape after quality filtering: {filtered_df.shape}")
    return filtered_df


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure continuous daily sequences for each participant.
    Missing dates are inserted as explicit NaN rows.
    """
    print("Step 1: Resampling to daily frequency...")
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    resampled_frames = []
    for participant_id, group in df.groupby("participant_id"):
        group = group.sort_values("datetime").set_index("datetime")
        group_resampled = group.resample("D").asfreq()
        group_resampled["participant_id"] = participant_id
        group_resampled = group_resampled.reset_index()
        resampled_frames.append(group_resampled)

    result = pd.concat(resampled_frames, ignore_index=True)
    print(f"   Shape after resampling: {result.shape}")
    return result


def convert_physical_zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert physical zeros (0.0) to NaN for bmi, weight, and bodyfat.
    """
    print("Step 2: Converting physical zeros to NaNs...")
    df = df.copy()
    for col in PHYSICAL_STATE_METRICS:
        if col in df.columns:
            zeros_before = (df[col] == 0.0).sum()
            df.loc[df[col] == 0.0, col] = np.nan
            print(f"   {col}: converted {zeros_before} zeros")
    return df


def apply_within_participant_physical_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ffill + bfill ONLY to physical metrics per participant.
    Activity and nutrition metrics are intentionally left as NaN here.
    """
    print("Step 3: Within-participant fill for physical metrics only (ffill -> bfill)...")
    df = df.copy()
    filled_frames = []

    for participant_id, group in df.groupby("participant_id"):
        group = group.sort_values("datetime").copy()
        for col in PHYSICAL_STATE_METRICS:
            if col in group.columns:
                group[col] = group[col].ffill().bfill()
        filled_frames.append(group)

    result = pd.concat(filled_frames, ignore_index=True)
    remaining_physical_nans = result[PHYSICAL_STATE_METRICS].isna().sum().to_dict()
    print(f"   Remaining NaNs in physical metrics after within-participant fill: {remaining_physical_nans}")
    return result


def apply_7day_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply 7-day rolling mean to partially imputed data.
    min_periods=3 allows stable estimates even with missing logs.
    """
    print("Step 4: Applying 7-day rolling mean smoothing (min_periods=3)...")
    df = df.copy()
    smoothed_frames = []

    for participant_id, group in df.groupby("participant_id"):
        group = group.sort_values("datetime").copy()
        for col in FEATURE_COLS:
            if col in group.columns:
                group[col] = group[col].rolling(window=7, min_periods=3, center=False).mean()
        smoothed_frames.append(group)

    result = pd.concat(smoothed_frames, ignore_index=True)
    print(f"   Shape after smoothing: {result.shape}")
    return result


def compute_global_means_from_train(train_df_partial: pd.DataFrame) -> dict:
    """
    Compute global means for physical metrics from TRAIN split only.
    """
    global_means = {}
    for col in PHYSICAL_STATE_METRICS:
        if col in train_df_partial.columns:
            global_means[col] = train_df_partial[col].mean(skipna=True)
    return global_means


def fill_remaining_gaps_after_smoothing(df: pd.DataFrame, global_means: dict) -> pd.DataFrame:
    """
    Post-smoothing filling order:
    1) Activity/Nutrition remaining NaNs -> 0
    2) Physical remaining NaNs -> train-only global means
    """
    print("Step 5: Filling remaining gaps AFTER smoothing...")
    df = df.copy()

    for col in ACTIVITY_METRICS + NUTRITION_METRICS:
        if col in df.columns:
            nan_count = int(df[col].isna().sum())
            if nan_count > 0:
                df[col] = df[col].fillna(0.0)
                print(f"   {col}: filled {nan_count} NaNs with 0")

    for col in PHYSICAL_STATE_METRICS:
        if col in df.columns:
            nan_count = int(df[col].isna().sum())
            if nan_count > 0:
                fill_value = global_means.get(col, np.nan)
                df[col] = df[col].fillna(fill_value)
                print(f"   {col}: filled {nan_count} NaNs with train global mean {fill_value:.4f}")

    total_nans = int(df[FEATURE_COLS].isna().sum().sum())
    print(f"   Remaining NaNs after post-smoothing fill: {total_nans}")
    return df


def create_relative_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create relative change features as 7-day / 30-day ratios.
    """
    print("Step 6: Creating relative change features (7-day / 30-day ratios)...")
    df = df.copy()
    enhanced_frames = []

    for participant_id, group in df.groupby("participant_id"):
        group = group.sort_values("datetime").copy()
        for col in REL_CHANGE_BASE_FEATURES:
            if col in group.columns:
                rolling_7day = group[col].rolling(window=7, min_periods=1).mean()
                rolling_30day = group[col].rolling(window=30, min_periods=1).mean()
                group[f"{col}_rel_change"] = rolling_7day / (rolling_30day + 1e-8)
        enhanced_frames.append(group)

    result = pd.concat(enhanced_frames, ignore_index=True)
    new_cols = [f"{col}_rel_change" for col in REL_CHANGE_BASE_FEATURES]
    print(f"   Added features: {new_cols}")
    return result


def preprocess_split(df: pd.DataFrame, global_means: dict) -> pd.DataFrame:
    """
    Complete split preprocessing in strict paper-aligned order.
    """
    df = resample_to_daily(df)
    df = convert_physical_zeros_to_nan(df)
    df = apply_within_participant_physical_fill(df)
    df = apply_7day_smoothing(df)
    df = fill_remaining_gaps_after_smoothing(df, global_means=global_means)
    df = create_relative_change_features(df)
    return df


def apply_leakage_free_standardization(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit StandardScaler on train only, transform train/test/val.
    """
    print("Step 7: Applying leakage-free standardization (fit train, transform all)...")
    feature_cols = [col for col in train_df.columns if col not in ["participant_id", "datetime"]]

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    val_scaled = val_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])

    print(f"   Standardized {len(feature_cols)} features")
    return train_scaled, test_scaled, val_scaled


def clean_final_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup: only drop rows that STILL contain NaN after all processing.
    """
    initial_shape = df.shape
    df_clean = df.dropna().reset_index(drop=True)
    dropped = initial_shape[0] - df_clean.shape[0]
    print(f"   Dropped {dropped} rows that still had NaNs")
    return df_clean


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("LOADING RAW DATA")
print("=" * 80)
train_df = load_split("train")
test_df = load_split("test")
val_df = load_split("val")

print(f"\nRaw train shape: {train_df.shape}")
print(f"Raw test shape: {test_df.shape}")
print(f"Raw val shape: {val_df.shape}")

print("\n" + "=" * 80)
print("QUALITY CONTROL FILTERING")
print("=" * 80)
print("Train split:")
train_df = filter_participants_by_data_quality(train_df)
print("Test split:")
test_df = filter_participants_by_data_quality(test_df)
print("Validation split:")
val_df = filter_participants_by_data_quality(val_df)

print("\n" + "=" * 80)
print("COMPUTING TRAIN-ONLY GLOBAL MEANS FOR PHYSICAL METRICS")
print("=" * 80)
train_partial_for_means = resample_to_daily(train_df)
train_partial_for_means = convert_physical_zeros_to_nan(train_partial_for_means)
train_partial_for_means = apply_within_participant_physical_fill(train_partial_for_means)
global_means = compute_global_means_from_train(train_partial_for_means)
for col, mean in global_means.items():
    print(f"   {col}: {mean:.4f}")

print("\n" + "=" * 80)
print("PREPROCESSING TRAIN/TEST/VAL SPLITS")
print("=" * 80)
print("Train split:")
train_processed = preprocess_split(train_df, global_means=global_means)
print("Test split:")
test_processed = preprocess_split(test_df, global_means=global_means)
print("Validation split:")
val_processed = preprocess_split(val_df, global_means=global_means)

print("\n" + "=" * 80)
print("STANDARDIZATION")
print("=" * 80)
train_scaled, test_scaled, val_scaled = apply_leakage_free_standardization(
    train_processed, test_processed, val_processed
)

print("\n" + "=" * 80)
print("FINAL CLEANUP (MAXIMUM ROW RETENTION)")
print("=" * 80)
print("Cleaning train split...")
train_final = clean_final_nans(train_scaled)
print("Cleaning test split...")
test_final = clean_final_nans(test_scaled)
print("Cleaning val split...")
val_final = clean_final_nans(val_scaled)

# Remove bodyfat from all final output dataframes
for final_df in (train_final, test_final, val_final):
    final_df.drop(columns=["bodyfat"], inplace=True, errors="ignore")

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

train_retention = (train_final.shape[0] / train_df.shape[0]) * 100 if len(train_df) else 0.0
test_retention = (test_final.shape[0] / test_df.shape[0]) * 100 if len(test_df) else 0.0
val_retention = (val_final.shape[0] / val_df.shape[0]) * 100 if len(val_df) else 0.0

print(f"\nTrain: {train_final.shape} (retained {train_retention:.1f}% after QC)")
print(f"Test:  {test_final.shape} (retained {test_retention:.1f}% after QC)")
print(f"Val:   {val_final.shape} (retained {val_retention:.1f}% after QC)")

print("\nParticipants in final datasets:")
print(f"   Train: {train_final['participant_id'].nunique()} participants")
print(f"   Test:  {test_final['participant_id'].nunique()} participants")
print(f"   Val:   {val_final['participant_id'].nunique()} participants")

print(f"\nTrain columns ({len(train_final.columns)}): {list(train_final.columns)}")

print("\n" + "-" * 80)
print("SAMPLE DATA (First 5 rows of train)")
print("-" * 80)
print(train_final.head())

print("\n" + "-" * 80)
print("NaN VERIFICATION")
print("-" * 80)
print(f"Train NaNs: {int(train_final.isna().sum().sum())}")
print(f"Test NaNs: {int(test_final.isna().sum().sum())}")
print(f"Val NaNs: {int(val_final.isna().sum().sum())}")

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)
print("Model-ready dataframes: train_final, test_final, val_final")
