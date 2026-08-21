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

## CV-ready project entry

**Biomedical Data Analytics — ECG Clustering, Forecasting & Clinical Pattern Mining**  
Python · pandas · scikit-learn · SciPy · statsmodels · mlxtend

- Built a reproducible biomedical analytics pipeline spanning PCA, K-Means, Gaussian mixtures, hierarchical clustering, ARIMA-family forecasting and Apriori association rules.
- Reduced 83 engineered ECG features to 20 principal components while retaining 90.2% variance; achieved 89.7% Hungarian-aligned clustering accuracy and 0.893 weighted F1 with GMM.
- Benchmarked four hierarchical linkages and diagnosed single-linkage chaining through cluster-balance and confusion-matrix analysis; improved Ward accuracy from 77.8% to 87.7% with PCA.
- Evaluated ECG forecasts on a 1,200-sample temporal hold-out using RSS, RMSE and MAE, and communicated model limitations through forecast diagnostics.
- Translated cardiovascular association rules into auditable support/lift/conviction evidence while documenting confounding, redundancy and non-causal interpretation.

### 中文简历精简版

**生物医学数据分析｜ECG 聚类、时序预测与临床关联规则**

- 搭建覆盖 PCA、K-Means、GMM、层次聚类、ARIMA 族模型与 Apriori 的可复现分析流程；使用 Hungarian algorithm 对无监督簇标签进行一对一事后对齐。
- 将 83 维 ECG 特征压缩至 20 个主成分并保留 90.2% 方差，GMM + PCA 达到 89.7% 聚类准确率与 0.893 weighted F1。
- 系统比较 4 种 linkage，识别 single linkage 的 chaining effect；PCA 将 Ward 聚类准确率由 77.8% 提升至 87.7%。
- 在 1,200 点时间留出集上用 RSS/RMSE/MAE 评估 ECG 预测，并通过预测图揭示“数值误差较低但无法重现尖峰”的模型局限。

## Interview framing

The strongest discussion point is not just the top score. It is the evaluation discipline: cluster IDs are arbitrary, so Hungarian alignment is required; a high weighted precision can be misleading when one cluster absorbs nearly all samples; and the best aggregate forecast error can still hide clinically important peak failures. Those checks turn a set of algorithms into a defensible analysis.
