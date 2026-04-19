# BrainFit Temporal Modeling Pipeline

This repository contains a Python pipeline for:

- preprocessing daily fitness CSV data,
- aligning behavioral/cognitive targets,
- training temporal models (LSTM),
- running lag analysis,
- running a static Ridge baseline,
- generating analysis plots in the `results/` folder.

---

## 1. Prerequisites

### Install Git and GitHub CLI

- **Git**: https://git-scm.com/downloads
- **GitHub CLI** (`gh`): https://cli.github.com/

After installing `gh`, authenticate once:

```bash
gh auth login
```

### Install Python 3.9+

- Download from https://www.python.org/downloads/
- During installation on Windows, check **"Add Python to PATH"**
- Verify it works:

```bash
python --version
```

> On macOS/Linux you may need to use `python3` instead of `python` in all commands below.

---

## 2. Clone the Repository

```bash
gh repo clone Ashmit-101/DS340W-code
cd DS340W-code
```

---

## 3. Create a Virtual Environment and Install Dependencies

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> You must activate the virtual environment every time you open a new terminal before running the pipeline.

---

## 4. Run the Pipeline

From the repository root (the `DS340W-code` folder), run:

```bash
python new_code_copy/testing.py
```

> On macOS/Linux, use `python3` if `python` is not recognized.

What it does:

1. Loads and preprocesses train/test/validation fitness splits.
2. Loads behavioral targets and remaps participant IDs.
3. Builds temporal dataloaders.
4. Trains an LSTM model.
5. Runs lag analysis across multiple lag values.
6. Saves plots to `results/`.

---

## 5. Optional Scripts

Run these from the repository root after the main pipeline if needed.

### Ridge baseline

```bash
python new_code_copy/ridge_regression.py
```

### Bootstrap correlation analysis

```bash
python new_code_copy/pkl_reading.py
```

---

## 6. Expected Outputs

After running, check the `results/` folder for:

- `lag_analysis.png`
- `lag_analysis_per_task.png`
- `ridge_feature_importance.png`
- `ridge_coefficient_heatmap.png`
- `correlation_heatmap.png`

---

## 7. Common Issues

### `python` not found

Try `python3` instead. If neither works, Python is not installed or not on your PATH — revisit step 1.

### `ModuleNotFoundError`

Make sure you:
1. Activated the virtual environment (step 3).
2. Are running commands from the repository root (`DS340W-code/`), not from inside a subfolder.

### `FileNotFoundError` under `data_copy/`

The pipeline expects these folders/files to exist inside the repo:

- `data_copy/Train/`
- `data_copy/Test/`
- `data_copy/Validation/`
- `data_copy/behavioral_summary.pkl`

If they are missing, confirm the clone completed successfully.

### Slow training

Training can take several minutes on CPU. On Apple Silicon Macs, PyTorch will use Metal (MPS) automatically for faster training.
