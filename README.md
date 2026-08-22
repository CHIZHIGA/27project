# Biomedical Signal & Clinical Data Analytics

This portfolio project applies unsupervised learning, time-series modelling and association-rule mining to ECG and cardiovascular datasets. The analysis was rebuilt from coursework feedback into a reproducible, evaluation-aware notebook with explicit assumptions, diagnostics and limitations.

## What the project demonstrates

- Audited and standardised 1,800 ECG feature records across 83 engineered features and three balanced classes.
- Implemented one-to-one cluster-label alignment with the Hungarian algorithm, keeping ground-truth labels strictly post-hoc for external evaluation.
- Benchmarked K-Means, full-covariance GMM and agglomerative clustering across original and PCA-reduced spaces.
- Retained 90.2% of variance with 20 principal components; GMM + PCA achieved the best clustering result (accuracy **0.897**, weighted F1 **0.893**).
- Diagnosed single-linkage chaining: 99.9% of observations collapsed into one cluster in the full feature space.
- Tested ECG transformations with ADF, compared AR/MA/ARMA/ARIMA forecasts on a 1,200-sample hold-out segment, and visualised why low RSS does not imply waveform fidelity.
- Mined Apriori rules from 270 cardiovascular records and interpreted support, confidence, lift and conviction without treating association as causation.

## Files

- `CW4.ipynb` — fully executed analysis with figures, tables and interpretation.
- `src/rebuild_cw4_notebook.py` — deterministic notebook source/rebuilder.
- `src/CW4/Q1.py` … `Q4.py` — four standalone question-level analysis scripts.
- `results/CW4/Q1` … `Q4` — generated metrics, assignments, forecasts, tables and figures.
- `requirements.txt` — tested Python dependencies.
- `feedback.md` — assessment feedback addressed by the revised analysis.

## Reproduce

From this directory:

```bash
python src/rebuild_cw4_notebook.py
jupyter nbconvert --to notebook --execute --inplace CW4.ipynb --ExecutePreprocessor.timeout=180
```

Run the four standalone analyses:

```bash
python src/CW4/Q1.py
python src/CW4/Q2.py
python src/CW4/Q3.py
python src/CW4/Q4.py
```
