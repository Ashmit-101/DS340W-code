# BrainFit Temporal Modeling Pipeline

This repository contains a Python pipeline for:

- preprocessing daily fitness CSV data,
- aligning behavioral/cognitive targets,
- training temporal models (LSTM),
- running lag analysis,
- running a static Ridge baseline,
- generating analysis plots in the results folder.

## 1. First-Time Setup

Run all commands from the repository root.

```bash
cd /Users/ashmit/Downloads/testing

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Verify Required Data Files

The pipeline expects these paths to exist:

- `data_copy/Train/`
- `data_copy/Test/`
- `data_copy/Validation/`
- `data_copy/raw_formatted/id_filename_key.csv`
- `data_copy/behavioral_summary.pkl`

Quick check:

```bash
ls data_copy
ls data_copy/raw_formatted | head
```

## 3. Run the Main End-to-End Pipeline

This is the recommended first run:

```bash
python new_code_copy/testing.py
```

What it does:

1. Loads and preprocesses train/test/validation fitness splits.
2. Loads behavioral targets and remaps participant IDs.
3. Builds temporal dataloaders.
4. Trains an LSTM model.
5. Runs lag analysis across multiple lag values.
6. Saves plots to `results/`.

## 4. Optional Scripts

Run these after the main pipeline if needed.

### Ridge baseline

```bash
python new_code_copy/ridge_regression.py
```

### Bootstrap correlation analysis

```bash
python new_code_copy/pkl_reading.py
```

## 5. Expected Outputs

You should see output artifacts in `results/`, including files such as:

- `lag_analysis.png`
- `lag_analysis_per_task.png`
- `ridge_feature_importance.png`
- `ridge_coefficient_heatmap.png`
- `correlation_heatmap.png`

## 6. Common Issues

### `ModuleNotFoundError` for local files

Run scripts from the repository root, not from inside another folder:

```bash
cd /Users/ashmit/Downloads/testing
python new_code_copy/testing.py
```

### `FileNotFoundError` under `data_copy`

Confirm all required folders/files listed above exist and match exact names.

### Slow training on CPU

This pipeline can take a while on first run. If Apple Metal (MPS) is available, PyTorch will use it automatically.

## 7. Re-running Later

```bash
cd /Users/ashmit/Downloads/testing
source .venv/bin/activate
python new_code_copy/testing.py
```
