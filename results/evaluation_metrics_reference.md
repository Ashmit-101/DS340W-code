# Evaluation Metrics Reference — DS340W Research Paper
**All metrics extracted from: `testing.py` (LSTM + Lag Analysis) and `ridge_regression.py` (Ridge Regression)**

---

## 1. R² — Coefficient of Determination

**What it is:**
R² measures how much of the variance in cognitive scores your model explains compared to simply predicting the mean score for everyone. A value of 1.0 means perfect prediction. A value of 0.0 means the model does no better than predicting the mean. A **negative** value means the model is actively *worse* than predicting the mean — it is making predictions that are more wrong than just guessing the average.

**Where it comes from:**
Computed in both `temporal_model.py` (the `r2_score()` function and `Trainer.evaluate()`) and `ridge_regression.py` (via scikit-learn's `r2_score`). Applied to the test set after training is complete.

**Formula:** R² = 1 − (SS_residual / SS_total), where SS_residual = Σ(y_true − y_pred)² and SS_total = Σ(y_true − mean(y_true))²

---

### 1a. LSTM — Single Model (testing.py, Step 5)
*Window = 90 days, Lag = 14 days, trained on MPS device, early stopped at epoch 9*

| Task | R² | Interpretation |
|---|---|---|
| Free recall (immediate) | −0.2006 | Slightly worse than mean prediction |
| Free recall (delayed) | −0.5367 | Moderately worse than mean prediction |
| Spatial learning (intercept) | −1.3483 | Substantially worse than mean — high-variance task |
| Spatial learning (slope) | −1.5346 | Substantially worse than mean — high-variance task |
| **Mean R²** | **−0.9050** | **Overall: model fails to generalize cross-subject** |

**Good or bad:** Bad as a predictive result. However, this result is interpretable — it establishes that a deep sequence model cannot generalize at N=104 with 1 cognitive label per participant.

---

### 1b. Ridge Regression — Test Set (ridge_regression.py)
*Features: mean + std + linear trend per feature = 45 features, N_train = 74, N_test = 20*

| Task | R² | Interpretation |
|---|---|---|
| Free recall (immediate) | **+0.0110** | Tiny positive signal — the only positive R² across all models |
| Free recall (delayed) | −0.0951 | Near-zero negative — close to chance |
| Spatial learning (intercept) | −0.2155 | Below mean predictor |
| Spatial learning (slope) | −0.0705 | Near-zero negative |
| **Mean R²** | **−0.0925** | Marginally better than LSTM but still predominantly negative |

**Good or bad:** Still negative on average, but much better than LSTM (−0.09 vs −0.91). The pattern of improvement from LSTM to Ridge tells you the LSTM was not the right tool, not that the data is without signal.

---

### 1c. Ridge Regression — Validation Set (ridge_regression.py)

| Task | R² | Interpretation |
|---|---|---|
| Free recall (immediate) | −0.1266 | Flips negative from test — confirms +0.011 was noise |
| Free recall (delayed) | −0.2015 | Consistent with test pattern |
| Spatial learning (intercept) | −0.0877 | Slightly better than test for this task |
| Spatial learning (slope) | −0.3053 | Worse on validation |
| **Mean R²** | **−0.1803** | Worse than test set — reflects small N instability |

**Good or bad:** The gap between test (−0.09) and validation (−0.18) is expected with only 10 validation participants. Do not over-interpret the difference; both point to the same conclusion.

---

### 1d. LSTM Lag Analysis — Mean R² per Lag (testing.py, Step 6)
*Each lag trains a separate LSTM from scratch. Best lag = 0 days (least negative).*

| Lag (days) | Mean R² | MSE | Interpretation |
|---|---|---|---|
| 0 | −0.7162 | 0.0328 | Concurrent fitness window — best performing lag |
| 7 | −0.8166 | 0.0348 | Slight degradation |
| 14 | −0.8277 | 0.0354 | Slight degradation (this was the single-model lag used) |
| 21 | −1.1959 | 0.0409 | Notable drop — fewer windows, less training data |
| 30 | −0.7185 | 0.0318 | Anomalous slight recovery — likely noise |
| 45 | −1.9268 | 0.0523 | Sharp degradation — data starvation as windows shrink |
| 60 | (pipeline cut off in output) | — | — |

**Good or bad:** All lags are negative. The lack of a clear peak is consistent with the fundamental limitation (1 label per participant). There is no meaningful lag effect detectable at this sample size.

---

## 2. MSE — Mean Squared Error

**What it is:**
The average squared difference between predicted and actual cognitive scores. Unlike R², MSE is in the same units as the squared cognitive scores (which are standardized in this pipeline, so units are approximately "standard deviations squared"). Lower is better. MSE does not tell you if the model is better than the mean — R² does that.

**Where it comes from:**
Computed in `temporal_model.py` `Trainer.evaluate()` via `nn.functional.mse_loss`. Also stored in lag analysis results dict.

| Model / Setting | MSE | Notes |
|---|---|---|
| LSTM single model (lag=14d) | 0.0353 | All splits standardized; roughly 0.19 SD average error |
| Lag=0d | 0.0328 | Lowest MSE in lag sweep |
| Lag=30d | 0.0318 | Numerically lowest but R² is also bad — predicts near-mean |
| Lag=45d | 0.0523 | Highest MSE — data starvation |
| Ridge (not directly computed) | — | Ridge uses R² as primary metric; MSE not printed |

**Good or bad:** The MSE values are low (0.03–0.05) because the targets are standardized and the model defaults toward predicting the mean. Low MSE here does not mean good predictions — it means the model is conservative. Always pair MSE interpretation with R².

---

## 3. Early Stopping Epoch

**What it is:**
The epoch at which training was halted because validation loss stopped improving. Patience was set to 8 (stop if no improvement for 8 consecutive epochs). This tells you how quickly the model overfitted to the training participants.

**Where it comes from:**
`Trainer.fit()` in `temporal_model.py`, printed during training with `verbose=True`.

| Model | Stopped at Epoch | Max Epochs | Patience |
|---|---|---|---|
| LSTM single model | **Epoch 9** | 50 | 8 |

**Good or bad:** Stopping at epoch 9 out of 50 is a red flag. It means the model overfit the training set almost immediately. With N=74 training participants and an LSTM with thousands of parameters, the model memorized the training labels rather than learning generalizable patterns.

---

## 4. Permutation Test — p-value / Percentile

**What it is:**
A non-parametric significance test that asks: "Is our model's R² higher than what you'd get by randomly shuffling the cognitive labels?" We ran 1,000 permutations — each time randomly reassigning cognitive scores to different participants, retraining Ridge, and recording R². The percentile tells us where our real R² falls in that null distribution. A percentile ≥ 95 corresponds to p < 0.05 (significant). This is the most important statistical test in the paper because standard p-values assume large N; the permutation test is valid for N=74.

**Where it comes from:**
`ridge_regression.py` Step 8. Uses `np.random.default_rng(42)` for reproducibility. N=1000 permutations. Each permutation refits a full RidgeCV.

| Task | Real R² | Null Mean | Null Std | Percentile | Verdict |
|---|---|---|---|---|---|
| Free recall (immediate) | +0.0110 | −0.1896 | 0.1081 | **98.5%** | **Significant (p < 0.05)** |
| Free recall (delayed) | −0.0951 | −0.3417 | 0.1422 | **97.3%** | **Significant (p < 0.05)** |
| Spatial learning (intercept) | −0.2155 | −0.2396 | 0.0860 | 55.4% | Chance / noise |
| Spatial learning (slope) | −0.0705 | −0.1230 | 0.0585 | 84.7% | Weak signal |

**Good or bad:** This is the key positive result. Even though absolute R² values look bad, the two free recall tasks are statistically above chance — the model extracts real signal for episodic memory. Spatial learning tasks do not show significant signal. This task dissociation is a scientifically meaningful finding.

**Critical point for your paper:** Report permutation percentiles / p-values, not just R². With N=20 test participants, raw R² is highly unstable. The permutation test accounts for this instability and provides valid inference.

---

## 5. Null Distribution (Permutation Test)

**What it is:**
The distribution of R² values produced by 1000 random label shuffles. This tells you what "chance performance" actually looks like for your specific dataset and sample size — and it is not R²=0. With only N=20 test participants, random models score around R²=−0.19 on average due to sampling noise alone.

**Where it comes from:**
`ridge_regression.py` Step 8, stored in `null_r2s` list.

| Task | Null Mean R² | Null Std |
|---|---|---|
| Free recall (immediate) | −0.1896 | ±0.1081 |
| Free recall (delayed) | −0.3417 | ±0.1422 |
| Spatial learning (intercept) | −0.2396 | ±0.0860 |
| Spatial learning (slope) | −0.1230 | ±0.0585 |

**Good or bad:** The null means being negative (not zero) is expected and important to report. It shows your test set is small enough that even random models produce noise. The wide standard deviations confirm that individual R² values are unreliable without permutation testing.

---

## 6. Regularization Strength (Ridge Alpha)

**What it is:**
The regularization hyperparameter selected by 5-fold cross-validation. Higher alpha = stronger shrinkage of coefficients toward zero. RidgeCV tested alphas [0.01, 0.1, 1, 10, 100].

**Where it comes from:**
`ridge_regression.py` Step 5, `RidgeCV.alpha_` attribute.

| Selected Alpha | Implication |
|---|---|
| **100.0** (maximum tested) | The model needed maximum regularization — coefficients were shrunk heavily toward zero. The cross-validation kept selecting higher alpha, suggesting even 100 may not be enough. |

**Good or bad:** Alpha=100 (the ceiling of your search grid) suggests the true optimal alpha may be even higher. It means the 45-feature matrix is too high-dimensional relative to N=74 training participants, and the model can only find signal by heavily constraining the coefficients. Consider reporting this as evidence of underpowering.

---

## 7. Ridge Coefficients — Feature Importance

**What it is:**
The learned weights for each of the 45 aggregated features (15 fitness features × 3 summary statistics: mean, std, trend). Larger absolute values indicate stronger influence on prediction. Sign (+/−) indicates direction: positive means higher feature value predicts higher cognitive score.

**Where it comes from:**
`ridge_regression.py` Step 9, `ridge.coef_` attribute, shape (4 tasks × 45 features).

### Top 5 Features by Mean |Coefficient| Across All Tasks

| Rank | Feature | Mean |coef| | Direction | Manning-aligned? |
|---|---|---|---|---|
| 1 | bmi_mean | 0.0122 | Mixed (+++−) | N/A (not a Manning activity feature) |
| 2 | fair_act_mins_trend | 0.0095 | Mixed (−−+−) | Partial |
| 3 | distance_rel_change_mean | 0.0083 | Mixed (++−+) | Partial |
| 4 | distance_mean | 0.0080 | Positive all (++++) | **Yes — fully aligned** |
| 5 | cal_bmr_mean | 0.0077 | Mixed (++-+) | Partial |

### Top 3 Predictors Per Task

| Task | Top Features |
|---|---|
| Free recall (immediate) | fair_act_mins_trend (−0.018), very_act_mins_trend (−0.017), distance_mean (+0.014) |
| Free recall (delayed) | distance_rel_change_mean (+0.017), distance_mean (+0.015), steps_rel_change_mean (+0.013) |
| Spatial learning (intercept) | bmi_mean (+0.021), cal_rel_change_mean (+0.010), sed_mins_mean (−0.010) |
| Spatial learning (slope) | bmi_mean (−0.021), cal_bmr_mean (+0.012), food_cal_log_mean (+0.012) |

**Good or bad:** The feature-level pattern is scientifically coherent. `distance_mean` being positive across all tasks replicates Manning's finding that higher activity correlates with better cognition. The trend features (fair_act_mins_trend, very_act_mins_trend) being top predictors for free recall is the novel temporal contribution — it suggests *trajectory* of activity matters, not just its average level. Spatial learning being dominated by BMI rather than activity features suggests a different mechanistic pathway.

---

## 8. Dataset / Model Configuration (Report These in Methods)

These are not metrics per se but must be reported alongside the metrics for reproducibility.

| Parameter | Value | Where set |
|---|---|---|
| Train participants | 74 | reading_data.py QC filter |
| Test participants | 20 | reading_data.py QC filter |
| Val participants | 10 | reading_data.py QC filter |
| LSTM window size | 90 days | testing.py `WINDOW_DAYS` |
| LSTM single-model lag | 14 days | testing.py `LAG_DAYS` |
| LSTM max epochs | 50 | testing.py `Trainer(max_epochs=50)` |
| LSTM early stopping patience | 8 | testing.py `Trainer(patience=8)` |
| LSTM architecture | Bidirectional, 2 layers, hidden=64, d_model=32 | temporal_model.py defaults |
| Ridge features | 45 (15 features × mean/std/trend) | ridge_regression.py aggregate_participant() |
| Ridge alphas tested | [0.01, 0.1, 1, 10, 100] | ridge_regression.py |
| Ridge CV folds | 5 | ridge_regression.py |
| Permutation N | 1000 | ridge_regression.py |
| Random seed | 42 | ridge_regression.py `np.random.default_rng(42)` |
| Lag values tested | 0, 7, 14, 21, 30, 45, 60 days | testing.py `run_lag_analysis` |
| Device | MPS (Apple Silicon) | testing.py auto-detect |

---

## Summary Table — All Key Metrics at a Glance

| Model | Task | R² | MSE | Perm. Percentile | Conclusion |
|---|---|---|---|---|---|
| LSTM (lag=14d) | Free recall (imm.) | −0.2006 | 0.0353 | — | Fails to generalize |
| LSTM (lag=14d) | Free recall (del.) | −0.5367 | 0.0353 | — | Fails to generalize |
| LSTM (lag=14d) | Spatial (intercept) | −1.3483 | 0.0353 | — | Fails to generalize |
| LSTM (lag=14d) | Spatial (slope) | −1.5346 | 0.0353 | — | Fails to generalize |
| **LSTM Mean** | — | **−0.9050** | **0.0353** | — | **Deep learning not viable at N=104** |
| Ridge | Free recall (imm.) | +0.0110 | — | **98.5% (p<0.05)** | **Significant — episodic memory signal** |
| Ridge | Free recall (del.) | −0.0951 | — | **97.3% (p<0.05)** | **Significant — episodic memory signal** |
| Ridge | Spatial (intercept) | −0.2155 | — | 55.4% (chance) | No signal |
| Ridge | Spatial (slope) | −0.0705 | — | 84.7% (weak) | No signal |
| **Ridge Mean** | — | **−0.0925** | — | — | **Weak but real signal for recall tasks** |

---

## How to Use These Metrics in Your Paper

**Methods section:** Report window size (90d), lag values swept (0–60d), Ridge feature construction (mean/std/trend), alpha search grid, CV folds, and permutation N=1000 with seed=42.

**Results section:** Lead with permutation test results — they are the statistically valid claim. Report "Free recall tasks showed above-chance prediction (98.5th and 97.3rd percentile of null distribution, p<0.05)." Then report raw R² values with the caveat that absolute R² is unreliable at N=20.

**Discussion section:** Use the LSTM vs Ridge comparison to argue that the fitness→cognition signal is real (Ridge finds it) but weak (requires permutation testing to detect) and insufficient for sequence modeling at this sample size (LSTM fails). Use the null distribution means to explain why raw R² appears negative.
