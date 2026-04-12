# Abstract Outline

**Title:** Temporal Patterns of Physical Activity Predict Episodic Memory but Not Spatial Cognition: A Longitudinal Analysis with Wearable Fitness Data

---

## Section 1: Background & Motivation (2–3 sentences)
- Manning et al. (2022) demonstrated cross-sectional correlations between fitness metrics and cognitive performance
- Prior work relied on static aggregate metrics, missing potential temporal dynamics
- Questions remain: Do fitness *trends* (not just averages) predict cognition? Can these relationships support individual-level prediction?

---

## Section 2: Methods (3–4 sentences)
- **Participants:** 104 participants (74 train, 20 test, 10 val) with continuous wearable fitness data and cognitive assessments
- **Features:** Aggregated daily wearable data into temporal summaries (mean, standard deviation, linear trend) for 15 fitness metrics
- **Approach:** Ridge regression with RidgeCV hyperparameter tuning; LSTM sequence model for comparison
- **Statistical validation:** Permutation testing (n=1000) to assess significance at small N

---

## Section 3: Key Results (4–5 sentences)
- **Episodic memory (free recall tasks):** Ridge regression achieved statistically significant prediction (free recall immediate: 98.5th percentile of null distribution, p<0.05; free recall delayed: 97.3th percentile, p<0.05)
  - *Despite* raw R² values of +0.01 and −0.10, both exceeded chance due to permutation test controlling for small-N sampling noise
  - Trend features (7-day and 30-day activity trajectory slopes) were among the top predictors, indicating temporal dynamics matter
- **Spatial learning tasks:** No significant association with fitness features (spatial intercept: 55th percentile; spatial slope: 85th percentile)
- **Deep learning (LSTM):** Sequence models failed to generalize cross-subject (mean R² = −0.91), suggesting the signal is too weak or inconsistent for deep learning at this sample size
- **Lag analysis:** Concurrent fitness windows (lag=0) performed better than lagged windows, suggesting immediate rather than delayed fitness-cognition relationships

---

## Section 4: Interpretation & Implications (2–3 sentences)
- Replicates Manning's feature-level findings (activity level predicts cognition) while extending them temporally: activity *trends* matter beyond static averages
- The task-specific pattern (memory yes, spatial no) aligns with neuroscience: aerobic exercise promotes hippocampal function relevant to episodic memory
- Effect sizes are too small for clinical prediction (mean R² = −0.09) but statistically robust via permutation testing; larger samples or richer feature sets may improve predictive power

---

## Section 5: Conclusion (1–2 sentences)
- Temporal feature engineering extends Manning et al.'s static correlations and reveals that activity *trajectories* predict episodic memory at above-chance levels
- Deep learning is not viable at current sample sizes; linear models with permutation testing provide more honest inference for small-N longitudinal studies

---

## Word Count Target
Typically 200–250 words for a conference/journal abstract. This outline sits at ~190 words unfilled; expand each section with specific values and brief explanations to reach target length.

---

## Key Metrics to Include in Each Section
- **Methods:** N=104, 15 fitness features, 4 cognitive tasks, permutation N=1000, alpha range [0.01–100]
- **Results:** Real R² values, permutation percentiles, null distribution means, top features (distance_mean, fair_act_mins_trend, very_act_mins_trend)
- **Conclusion:** Replication status, task dissociation, sample size limitations
