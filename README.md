# BrainFit Temporal Modeling Pipeline

A Python pipeline for analyzing fitness-cognition relationships through temporal modeling. This repository replicates and extends [Manning et al. (2022)](https://www.nature.com/articles/s41598-022-17781-0) using LSTM and Transformer models to predict cognitive task performance from daily fitness tracking data.

**What it does:**
- Preprocesses daily fitness CSV data with quality control filtering
- Aligns fitness metrics with behavioral/cognitive target scores
- Builds temporal sequence datasets with configurable lag windows
- Trains deep learning models (LSTM, Transformer)
- Runs lag analysis to identify predictive time windows
- Compares against a Ridge regression baseline
- Generates publication-quality plots and analysis reports

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Data Setup](#data-setup)
5. [Running the Pipeline](#running-the-pipeline)
6. [Expected Outputs](#expected-outputs)
7. [Understanding the Pipeline](#understanding-the-pipeline)
8. [Troubleshooting](#troubleshooting)
9. [Project Structure](#project-structure)

---

## Quick Start

PASTE ALL THE COMMANDS IN TERMINAL OF YOUR COMPUTER 
MACOS: CMD+ space -> TERMINAL 
Windows: pressing Win + X -> "Terminal" or "Windows Terminal"


**TL;DR** — if you have all data files and Python 3.9+:

```bash
git clone https://github.com/Ashmit-101/DS340W-code.git
cd DS340W-code
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python new_code_copy/testing.py
```

Expected runtime: **10–30 minutes on CPU** (faster with GPU/Apple Silicon).

Results will appear in `results/` folder.

---

## System Requirements

### Operating System
- **macOS** (Intel or Apple Silicon)
- **Linux** (Ubuntu 18.04+, Debian, etc.)
- **Windows** 10 or 11 (Command Prompt, PowerShell, or WSL)

### Python Version
- **Python 3.9, 3.10, 3.11, or 3.12**
- Download from https://www.python.org/downloads/
- **Tested on:** Python 3.9.13, 3.10.12, 3.11.7

### Dependencies
All Python packages are listed in `requirements.txt` and will be installed automatically. Key packages:
- `torch` (deep learning, includes CPU and GPU support)
- `pandas` (data manipulation)
- `scikit-learn` (machine learning baselines)
- `matplotlib`, `seaborn` (plotting)
- `statsmodels`, `scipy` (statistical tests)

### Hardware
- **Minimum:** 4 GB RAM, 2 CPU cores
- **Recommended:** 8+ GB RAM, 4+ cores
- **GPU support:** NVIDIA CUDA 11.8+ (automatic), or Apple Silicon (MPS, automatic)
- **Estimated storage:** ~2 GB for data + results

---
# Installation

### Step 1: Install Git

Download and install Git from https://git-scm.com/downloads

**Verify:**
```bash
git --version
```

### Step 2: Install Python 3.9+

Download from https://www.python.org/downloads/

**On Windows:** During installation, **CHECK** the box "Add Python to PATH"

**Verify:**
```bash
python --version
```

> **Note:** On macOS/Linux, you may need to use `python3` instead of `python`. Commands below assume `python`; substitute `python3` if needed.

### Step 3: Clone the Repository

```bash
git clone https://github.com/Ashmit-101/DS340W-code.git
cd DS340W-code
```

**Verify:** You should see folders like `data_copy/`, `new_code_copy/`, `results/`, etc.

### Step 4: Create and Activate Virtual Environment

A virtual environment isolates this project's dependencies from your system Python.

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Verify activation:** Your terminal prompt should now show `(.venv)` at the start.

### Step 5: Install Python Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:** About 20–30 packages will install. This takes 2–5 minutes.

**Verify:** No error messages ending with `ERROR:`.

---

## Running the Pipeline

### ✅ Pre-Flight Checklist

Before running, verify:

- [ ] Virtual environment is activated (you see `(.venv)` in terminal)
- [ ] You're in the repository root directory (`DS340W-code/`)
- [ ] All data files exist in `data_copy/` (see [Data Setup](#data-setup))
- [ ] `requirements.txt` and `new_code_copy/testing.py` are present

### Main Pipeline

```bash
python new_code_copy/testing.py
```

## Data Setup

### Required Data Files

The pipeline expects a `data_copy/` folder containing:

```
data_copy/
├── Train/
│   ├── BFM_AMT_001.csv
│   ├── BFM_AMT_002.csv
│   └── ... (participant CSV files)
├── Test/
│   ├── BFM_AMT_XXX.csv
│   └── ...
├── Validation/
│   ├── BFM_AMT_XXX.csv
│   └── ...
├── raw_formatted/
│   └── id_filename_key.csv
├── behavioral_summary.pkl
└── fitness_summary.pkl
```

### Data File Descriptions

| File/Folder | Purpose | Format |
|---|---|---|
| `Train/`, `Test/`, `Validation/` | Daily fitness data split by train/test/val | CSV files, one per participant |
| `raw_formatted/id_filename_key.csv` | Maps participant IDs (P1, P2, ...) to filenames | CSV with index and `filenames` column |
| `behavioral_summary.pkl` | Cognitive task scores per participant | Pickle file (pandas DataFrame) |
| `fitness_summary.pkl` | Aggregated fitness features per participant | Pickle file (pandas DataFrame) |

### Where to Get Data

If you are part of the Manning et al. (2022) replication project, data files should already be in `data_copy/`. 

**If data is missing:**
1. Check if `data_copy/` folder exists and is not empty
2. Verify all subfolders (`Train/`, `Test/`, `Validation/`, `raw_formatted/`) are present
3. Ensure CSV files in `Train/`, `Test/`, `Validation/` follow the naming pattern `BFM_AMT_XXX.csv`
4. Confirm `.pkl` files (`behavioral_summary.pkl`, `fitness_summary.pkl`) exist

If any files are missing, contact your project administrator or refer to the data preparation documentation.

---

**Expected behavior:**
- **First 2 minutes:** Loads and preprocesses fitness data
- **Next 5 minutes:** Loads behavioral scores and builds dataloaders
- **Next 10–20 minutes:** Trains LSTM model and runs lag analysis
- **Final output:** Plots and summary printed to console

**Success indicators:**
- No error messages (warnings are OK)
- Console output shows progress bars (with `tqdm`)
- Files appear in `results/` folder
- Final line says "PIPELINE COMPLETE"

### Optional: Ridge Baseline

After the main pipeline, run the static Ridge regression baseline (faster):

```bash
python new_code_copy/ridge_regression.py
```

**Runtime:** 1–2 minutes

### Optional: Bootstrap Correlation Analysis

Run detailed correlation analysis with permutation testing:

```bash
python new_code_copy/pkl_reading.py
```

**Runtime:** 10–15 minutes

---

## Expected Outputs

After running `testing.py`, the `results/` folder will contain:

### Plots

| File | Purpose |
|---|---|
| `lag_analysis.png` | **Main result:** Test R² vs fitness window lag (days). Peak indicates when fitness best predicts cognition. |
| `lag_analysis_per_task.png` | Same as above, but separate line per cognitive task. Shows task-specific lag profiles. |

### Additional Plots (Ridge Baseline)

After running `ridge_regression.py`:

| File | Purpose |
|---|---|
| `ridge_feature_importance.png` | Bar chart of feature importance (mean coefficient magnitude). |
| `ridge_coefficient_heatmap.png` | Heatmap of Ridge coefficients: features × cognitive tasks. |
| `ridge_permutation_test.png` | Null distribution of R² values under shuffled labels (permutation test). |
| `ridge_manning_alignment.png` | Checks if activity features have expected sign (Manning replication). |

### Data & Reports (Bootstrap Analysis)

After running `pkl_reading.py`:

| File | Purpose |
|---|---|
| `bootstrap_correlations_fdr.csv` | Full correlation results (FDR-corrected) |
| `significant_correlations_only.csv` | Subset of statistically significant correlations |
| `correlation_heatmap.png` | Heatmap of significant fitness-behavior correlations |
| `top_correlations_scatter.png` | Scatter plots of top 12 correlations with regression lines |
| `task_specificity.png` | Task-specific association analysis |
| `analysis_report.txt` | Comprehensive text report of all findings |

### Console Output

The main script prints:
- Data shapes and participant counts
- Task names and number of features
- Training progress (one line per 10 epochs)
- Test R² for each cognitive task
- Lag analysis summary table

---

## Understanding the Pipeline

### Data Flow

```
reading_data.py
    ↓
(train_final, test_final, val_final)
    ↓
behavioral_summary.pkl
    ↓
temporal_dataset.py
    ↓
(train_loader, test_loader, val_loader)
    ↓
temporal_model.py
    ↓
LSTM training + Lag analysis
    ↓
results/ (plots, metrics)
```

### What Each Script Does

| Script | Purpose | Input | Output |
|---|---|---|---|
| `reading_data.py` | Preprocesses raw fitness CSVs | `data_copy/Train/`, etc. | `train_final`, `test_final`, `val_final` DataFrames |
| `temporal_dataset.py` | Builds sliding-window sequences | Fitness + behavioral DataFrames | PyTorch DataLoaders |
| `temporal_model.py` | LSTM, Transformer, and Trainer | DataLoaders | Trained model, lag analysis plots |
| `testing.py` | Main orchestration | All above | `results/lag_analysis.png`, metrics |
| `ridge_regression.py` | Static Ridge baseline | `train_final` + behavioral | Ridge plots + permutation test |
| `pkl_reading.py` | Bootstrap correlation analysis | `.pkl` files | Correlation heatmaps + report |

### Key Parameters

You can modify these in `testing.py` (lines 45–49) to customize the analysis:

```python
TASK_COLS = list(behavioral_df.columns)   # Which cognitive tasks to predict
WINDOW_DAYS = 90                           # Length of fitness history window (days)
LAG_DAYS = 14                              # Gap between window end and assessment
BATCH_SIZE = 32                            # Batch size during training
```

For the lag analysis, change line 72:

```python
lag_values=[0, 7, 14, 21, 30, 45, 60, 90]  # Which lag values to sweep
```

---

## Troubleshooting

### `python` not found

**Error message:**
```
'python' is not recognized as an internal or external command...
```

**Solution:**
Try `python3` instead:
```bash
python3 -m venv .venv
python3 -m pip install -r requirements.txt
python3 new_code_copy/testing.py
```

If neither works, Python is not installed or not on PATH. Reinstall from https://www.python.org/downloads/ and check "Add Python to PATH" on Windows.

---

### Virtual environment not activated

**Error message:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
You forgot to activate the virtual environment. Run:

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```cmd
.venv\Scripts\activate
```

Verify the prompt shows `(.venv)` before running scripts.

---

### `FileNotFoundError: data_copy/...`

**Error message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data_copy/Train/...'
```

**Solutions:**
1. **Check you're in the correct directory:**
   ```bash
   ls                    # macOS/Linux
   dir                   # Windows
   ```
   You should see `data_copy/`, `new_code_copy/`, `results/`, etc.

2. **Verify data files exist:**
   ```bash
   ls data_copy/Train/   # macOS/Linux
   dir data_copy\Train   # Windows
   ```
   You should see `BFM_AMT_*.csv` files.

3. **If files are missing:**
   Download or restore data files to `data_copy/` and re-run.

---

### `ModuleNotFoundError` for a specific package

**Error message:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
Packages were not installed. Run:
```bash
pip install -r requirements.txt
```

If this fails with a network error, check your internet connection and retry.

---

### Slow training on CPU

**Expected behavior:**
Training takes **10–20 minutes on CPU**, **1–3 minutes on GPU**.

**To check if GPU is being used:**
- **NVIDIA GPU:** Should say `device: cuda`
- **Apple Silicon Mac:** Should say `device: mps`
- **CPU:** Should say `device: cpu`

Look for the line `Device: [device_type]` at the start of training output.

---

### Out of memory (`RuntimeError: CUDA out of memory`)

**Solution:**
Reduce batch size in `testing.py`:
```python
BATCH_SIZE = 16  # Instead of 32
```

Or reduce `WINDOW_DAYS`:
```python
WINDOW_DAYS = 60  # Instead of 90
```

Then re-run.

---

### Script runs but produces no output plots

**Symptoms:**
- Script says "PIPELINE COMPLETE" but `results/` folder is empty or has no PNGs

**Solutions:**
1. **Check for errors in console output** — scroll up for `ValueError` or `RuntimeError`

2. **Verify `results/` folder exists:**
   ```bash
   ls results/       # macOS/Linux
   dir results       # Windows
   ```

3. **Check file permissions** — `results/` folder should be writable

4. **Re-run with verbose output** — add `print()` statements in `testing.py` to trace execution

---

### Issues on macOS with Apple Silicon

If you see `AssertionError: Unexpected arch` or similar MPS errors:

1. **Upgrade PyTorch:**
   ```bash
   pip install --upgrade torch
   ```

2. **Fallback to CPU:**
   Edit `testing.py` line 62:
   ```python
   device = "cpu"  # Force CPU instead of MPS
   ```

---

## Project Structure

```
DS340W-code/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── package.json                       # (ignore — artifact)
├── data_copy/                         # Input data
│   ├── Train/                         # Train split fitness CSVs
│   ├── Test/                          # Test split fitness CSVs
│   ├── Validation/                    # Validation split fitness CSVs
│   ├── raw_formatted/
│   │   └── id_filename_key.csv        # Participant ID mapping
│   ├── behavioral_summary.pkl         # Cognitive task scores
│   └── fitness_summary.pkl            # (optional, aggregated features)
│
├── new_code_copy/                     # Main analysis code
│   ├── testing.py                     # ← RUN THIS (main pipeline)
│   ├── reading_data.py                # Fitness preprocessing
│   ├── temporal_dataset.py            # Sequence dataset builder
│   ├── temporal_model.py              # LSTM/Transformer + lag analysis
│   ├── ridge_regression.py            # Ridge baseline (optional)
│   ├── pkl_reading.py                 # Bootstrap analysis (optional)
│   └── readomg_temp.py                # (internal diagnostic)
│
├── results/                           # Output folder (created on first run)
│   ├── lag_analysis.png               # Main result plot
│   ├── lag_analysis_per_task.png
│   ├── ridge_feature_importance.png
│   ├── ridge_coefficient_heatmap.png
│   ├── correlation_heatmap.png
│   └── ... (other plots & CSVs)
│
└── node_modules/                      # (ignore — artifact)
```

---

## FAQ

**Q: Can I run this on Windows?**
A: Yes! Use Command Prompt or PowerShell. See installation step 4 for platform-specific activation commands.

**Q: How long does the pipeline take?**
A: Typically 10–30 minutes on CPU, 1–5 minutes on GPU (depends on dataset size and hardware).

**Q: Can I modify the cognitive tasks or features?**
A: Yes. See [Understanding the Pipeline → Key Parameters](#key-parameters). Modify `TASK_COLS` or `WINDOW_DAYS` in `testing.py`.

**Q: What's the difference between lag analysis and the Ridge baseline?**
A: Lag analysis trains deep temporal models (LSTM) and sweeps the time gap between fitness and cognition assessment. Ridge is a static baseline that aggregates fitness to one row per participant. Both are useful — lag analysis is the main scientific extension, Ridge is a sanity check.

**Q: Can I use this with my own data?**
A: Yes, but you'll need to reformat it to match the expected CSV and pickle structures. See [Data Setup](#data-setup).

**Q: What citation should I use?**
A: This code replicates Manning et al. (2022):
```
Manning, J. R., et al. (2022). Fitness tracking reveals task-specific 
associations between memory, mental health, and physical activity. 
Scientific Reports, 12(1), 13822.
https://doi.org/10.1038/s41598-022-17781-0
```

---

## License

This project replicates research from Manning et al. (2022). Code is provided for educational and research purposes.

---

## Support

For issues, questions, or contributions:
1. Check [Troubleshooting](#troubleshooting) section above
2. Review inline code comments in `new_code_copy/*.py`
3. See the original paper: https://www.nature.com/articles/s41598-022-17781-0

---

**Last updated:** 2025-04-20  
**Repository:** https://github.com/Ashmit-101/DS340W-code  
**Paper:** Manning et al. (2022), Sci Rep 12:13822
