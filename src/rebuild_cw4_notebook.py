"""Rebuild CW4.ipynb as a reproducible, feedback-driven portfolio notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "CW4.ipynb"


def clean(text: str) -> str:
    return dedent(text).strip() + "\n"


def md(text: str):
    return nbf.v4.new_markdown_cell(clean(text))


def code(text: str):
    return nbf.v4.new_code_cell(clean(text))


cells = [
    md(
        r"""
        # ECG Analytics: Clustering, Forecasting and Association Rules

        This notebook studies two biomedical datasets through five connected questions:

        1. K-Means clustering of engineered ECG features;
        2. probabilistic clustering with Gaussian Mixture Models (GMM);
        3. hierarchical clustering under four linkage criteria;
        4. stationarity analysis and forecasting of a single ECG trace; and
        5. interpretable association-rule mining on cardiovascular risk factors.

        The analysis is designed to be reproducible and evaluation-aware. Cluster IDs are aligned to the known classes **only after fitting**, using the Hungarian assignment algorithm. The labels therefore play no role in training; they are used solely to make external evaluation meaningful.

        **Headline findings.** PCA retains 90.2% of variance with 20 of 83 components. GMM + PCA gives the strongest clustering result (accuracy 0.897; weighted F1 0.893). Ward is the only competitive hierarchical linkage, while single linkage exhibits severe chaining. For the univariate ECG, the original signal is already stationary; MA(2) has the lowest test RSS among the four requested models but smooths away sharp ECG peaks. Association rules identify co-occurring risk-factor patterns, but they are descriptive associations rather than causal or diagnostic claims.
        """
    ),
    md(
        r"""
        ## Reproducible setup and evaluation protocol

        The ECG feature table contains three balanced classes and no missing values. Three exact duplicate rows are retained because no patient/beat identifier is available to establish that they are data-entry errors rather than legitimate repeated observations. All 83 features are standardised because K-Means, PCA and agglomerative clustering depend on distances or variances. PCA is fitted to the standardised data and the smallest number of components explaining at least 90% of total variance is retained.

        Cluster labels are arbitrary permutations. A naive majority-vote mapping can assign several clusters to the same class and inflate external scores. Instead, the contingency matrix is treated as a linear assignment problem: the Hungarian algorithm finds the one-to-one cluster-to-class permutation that maximises the diagonal total. This post-hoc step does not change the clusters.
        """
    ),
    code(
        """
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from scipy.optimize import linear_sum_assignment
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            recall_score,
        )
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        sns.set_theme(style="whitegrid", context="notebook")
        RANDOM_STATE = 4


        def hungarian_match(y_true, cluster_ids):
            # Return one-to-one aligned predictions, mapping, and contingency table.
            y_true = np.asarray(y_true)
            cluster_ids = np.asarray(cluster_ids)
            true_labels = np.unique(y_true)
            cluster_labels = np.unique(cluster_ids)

            contingency = pd.crosstab(
                pd.Series(y_true, name="true class"),
                pd.Series(cluster_ids, name="cluster"),
            ).reindex(index=true_labels, columns=cluster_labels, fill_value=0)

            row_ind, col_ind = linear_sum_assignment(-contingency.to_numpy())
            mapping = {
                cluster_labels[col]: true_labels[row]
                for row, col in zip(row_ind, col_ind)
            }
            aligned = np.array([mapping[cluster] for cluster in cluster_ids])
            return aligned, mapping, contingency


        def clustering_metrics(y_true, y_pred):
            # External clustering scores after label alignment.
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_weighted": precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
                "f1_weighted": f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
            }


        def plot_confusion(y_true, y_pred, title, cmap="Blues", ax=None):
            if ax is None:
                _, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap=cmap,
                cbar=False, square=True, ax=ax
            )
            ax.set(title=title, xlabel="Aligned cluster label", ylabel="True class")
            return ax


        ecg = pd.read_csv("ecg_signals_preprocessed.csv")
        X = ecg.drop(columns="classes")
        y = ecg["classes"]

        audit = pd.DataFrame(
            {
                "samples": [len(ecg)],
                "features": [X.shape[1]],
                "missing_values": [int(ecg.isna().sum().sum())],
                "duplicate_rows": [int(ecg.duplicated().sum())],
            }
        )
        display(audit)
        display(y.value_counts().sort_index().rename("samples per class").to_frame().T)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        """
    ),
    md(
        r"""
        # Question 1 — K-Means and PCA

        K-Means partitions observations by minimising within-cluster squared Euclidean distance. Standardisation prevents high-variance features from dominating this objective. `n_init=10` makes the optimisation more robust to centroid initialisation and explicit across scikit-learn versions.
        """
    ),
    code(
        """
        kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
        kmeans_clusters = kmeans.fit_predict(X_scaled)
        kmeans_aligned, kmeans_mapping, kmeans_contingency = hungarian_match(
            y, kmeans_clusters
        )
        kmeans_metrics = clustering_metrics(y, kmeans_aligned)

        print("Hungarian cluster-to-class mapping:", kmeans_mapping)
        display(pd.DataFrame([kmeans_metrics], index=["K-Means (83 features)"]).round(3))
        plot_confusion(y, kmeans_aligned, "K-Means — 83 standardised features")
        plt.show()
        """
    ),
    code(
        """
        pca_full = PCA().fit(X_scaled)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative_variance, 0.90) + 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, lw=2)
        ax.axhline(0.90, color="tab:red", ls="--", label="90% threshold")
        ax.axvline(n_components, color="tab:green", ls=":", label=f"{n_components} components")
        ax.set(
            xlabel="Number of principal components",
            ylabel="Cumulative explained variance",
            ylim=(0, 1.02),
            title="PCA dimensionality selection",
        )
        ax.legend()
        plt.show()

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        print(
            f"Selected {n_components} of {X.shape[1]} components; "
            f"retained variance = {pca.explained_variance_ratio_.sum():.3%}."
        )
        """
    ),
    code(
        """
        kmeans_pca = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
        kmeans_pca_clusters = kmeans_pca.fit_predict(X_pca)
        kmeans_pca_aligned, kmeans_pca_mapping, _ = hungarian_match(
            y, kmeans_pca_clusters
        )
        kmeans_pca_metrics = clustering_metrics(y, kmeans_pca_aligned)

        q1_comparison = pd.DataFrame(
            [kmeans_metrics, kmeans_pca_metrics],
            index=["K-Means (83 features)", f"K-Means (PCA, {n_components} PCs)"],
        )
        display(q1_comparison.round(3))
        plot_confusion(y, kmeans_pca_aligned, "K-Means — PCA feature space", "Greens")
        plt.show()
        """
    ),
    md(
        r"""
        ### Q1 interpretation

        K-Means reaches **0.864 accuracy and 0.858 weighted F1**. Class 0 is almost perfectly recovered (597/600), and class 2 is also well separated (562/600). Class 1 is the main source of error: 120 samples are assigned to class 0 and 83 to class 2. This pattern suggests that the engineered representation does not form an equally compact, spherical group for class 1.

        Reducing 83 features to 20 principal components (90.2% retained variance) leaves the assignments and scores unchanged. This is useful rather than trivial: approximately 76% of the dimensions can be removed without losing K-Means' external performance, revealing substantial linear redundancy. It does **not** prove that all discarded information is clinically irrelevant; PCA optimises variance preservation, not class separation.
        """
    ),
    md(
        r"""
        # Question 2 — Gaussian Mixture Model with EM

        A three-component GMM is fitted by expectation-maximisation (EM). The E-step estimates component responsibilities and the M-step updates means, covariances and mixture weights. Unlike K-Means' hard spherical partition, the default full-covariance GMM can represent elliptical, overlapping clusters and gives probabilistic memberships.
        """
    ),
    code(
        """
        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=9)
        gmm_clusters = gmm.fit_predict(X_scaled)
        gmm_aligned, gmm_mapping, _ = hungarian_match(y, gmm_clusters)
        gmm_metrics = clustering_metrics(y, gmm_aligned)

        gmm_pca = GaussianMixture(n_components=3, covariance_type="full", random_state=9)
        gmm_pca_clusters = gmm_pca.fit_predict(X_pca)
        gmm_pca_aligned, gmm_pca_mapping, _ = hungarian_match(y, gmm_pca_clusters)
        gmm_pca_metrics = clustering_metrics(y, gmm_pca_aligned)

        q2_comparison = pd.DataFrame(
            [gmm_metrics, gmm_pca_metrics],
            index=["GMM (83 features)", f"GMM (PCA, {n_components} PCs)"],
        )
        display(q2_comparison.round(3))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_confusion(y, gmm_aligned, "GMM — 83 features", "Blues", axes[0])
        plot_confusion(y, gmm_pca_aligned, "GMM — PCA", "Greens", axes[1])
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        ### Q2 interpretation

        Full-feature GMM improves on K-Means from 0.864 to **0.892 accuracy**, consistent with non-spherical or overlapping structure. PCA gives a further modest gain to **0.897 accuracy and 0.893 weighted F1**. In the PCA solution, classes 0 and 2 are recovered extremely well (594/600 and 595/600), while class 1 remains difficult and contributes 175 of the 186 errors.

        PCA may help because estimating three full 83×83 covariance matrices is parameter-intensive, whereas the 20-dimensional representation suppresses low-variance directions and makes covariance estimation more stable. The improvement is small (0.5 percentage points), so it should be described as evidence on this dataset—not a general guarantee that PCA improves GMMs. Soft responsibilities could also be used to flag ambiguous beats, although that uncertainty analysis is outside the coursework scope.
        """
    ),
    md(
        r"""
        # Question 3 — Agglomerative clustering and linkage sensitivity

        Agglomerative clustering begins with one cluster per sample and repeatedly merges clusters. Four linkage definitions are compared:

        - **Ward:** merge causing the smallest increase in within-cluster variance;
        - **complete:** maximum inter-cluster pairwise distance;
        - **average:** mean inter-cluster pairwise distance;
        - **single:** minimum inter-cluster pairwise distance.

        All methods are evaluated in both the 83-feature standardised space and the 20-component PCA space. Including single linkage directly addresses its characteristic failure mode: a chain of nearby samples can collapse most observations into one cluster.
        """
    ),
    code(
        """
        linkage_rows = []
        linkage_predictions = {}

        for space_name, feature_matrix in {
            "83 features": X_scaled,
            f"PCA ({n_components} PCs)": X_pca,
        }.items():
            for linkage in ["ward", "complete", "average", "single"]:
                model = AgglomerativeClustering(n_clusters=3, linkage=linkage)
                cluster_ids = model.fit_predict(feature_matrix)
                aligned, mapping, _ = hungarian_match(y, cluster_ids)
                sizes = np.bincount(cluster_ids, minlength=3)
                linkage_rows.append(
                    {
                        "space": space_name,
                        "linkage": linkage,
                        **clustering_metrics(y, aligned),
                        "largest_cluster_share": sizes.max() / len(cluster_ids),
                    }
                )
                linkage_predictions[(space_name, linkage)] = aligned

        linkage_results = pd.DataFrame(linkage_rows)
        display(linkage_results.round(3))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        sns.barplot(data=linkage_results, x="linkage", y="accuracy", hue="space", ax=axes[0])
        sns.barplot(
            data=linkage_results,
            x="linkage",
            y="largest_cluster_share",
            hue="space",
            ax=axes[1],
        )
        axes[0].set(title="External accuracy", ylim=(0, 1))
        axes[1].set(title="Largest-cluster share", ylabel="share of samples", ylim=(0, 1))
        axes[1].get_legend().remove()
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        ward_full = linkage_predictions[("83 features", "ward")]
        ward_pca = linkage_predictions[(f"PCA ({n_components} PCs)", "ward")]
        ward_metrics = clustering_metrics(y, ward_full)
        ward_pca_metrics = clustering_metrics(y, ward_pca)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_confusion(y, ward_full, "Ward — 83 features", "Blues", axes[0])
        plot_confusion(y, ward_pca, "Ward — PCA", "Greens", axes[1])
        plt.tight_layout()
        plt.show()

        clustering_summary = pd.DataFrame(
            [
                {"model": "K-Means", "space": "83 features", **kmeans_metrics},
                {"model": "K-Means", "space": f"PCA ({n_components} PCs)", **kmeans_pca_metrics},
                {"model": "GMM", "space": "83 features", **gmm_metrics},
                {"model": "GMM", "space": f"PCA ({n_components} PCs)", **gmm_pca_metrics},
                {"model": "Agglomerative (Ward)", "space": "83 features", **ward_metrics},
                {"model": "Agglomerative (Ward)", "space": f"PCA ({n_components} PCs)", **ward_pca_metrics},
            ]
        ).sort_values("accuracy", ascending=False)
        display(clustering_summary.round(3).reset_index(drop=True))
        """
    ),
    md(
        r"""
        ### Q3 interpretation and comparison with Q1–Q2

        Ward is decisively the strongest hierarchical criterion: its accuracy rises from **0.778 to 0.877** after PCA, and weighted F1 rises from 0.762 to 0.875. PCA is especially helpful here because Ward's variance objective is sensitive to noisy high-dimensional distances.

        Complete, average and single linkage all collapse almost the entire dataset into one cluster. Single linkage is the clearest case: its largest cluster contains 99.9% of samples, producing near-chance accuracy (0.334). This is the expected **chaining effect** and explains why single linkage is unsuitable for these ECG features. The superficially high weighted precision for some collapsed solutions is misleading; accuracy, F1, confusion matrices and cluster-size balance must be read together.

        Across the requested methods, **GMM + PCA ranks first** (accuracy 0.897; weighted F1 0.893), followed by Ward + PCA (0.877; 0.875) and K-Means (0.864; 0.858). The ranking supports a geometric interpretation: GMM benefits from flexible covariance, Ward becomes competitive after denoising, and K-Means is a stable but less flexible baseline. Because the same labelled dataset is used for comparison, these are descriptive external scores rather than estimates of generalisation to unseen patients.
        """
    ),
    md(
        r"""
        # Question 4 — Stationarity, decomposition and ECG forecasting

        The first CSV entry is zero and is removed as a documented acquisition artefact. The remaining 3,000 samples are treated as an equally spaced univariate series. The Augmented Dickey–Fuller (ADF) test has the null hypothesis of a unit root; a p-value below 0.05 is evidence against that null. Square-root and log transforms mainly compress amplitude/variance, while differencing targets changes in the level.
        """
    ),
    code(
        """
        signal_raw = pd.read_csv("single_ecg_signal.csv", header=None)[0]
        removed_sample = signal_raw.iloc[0]
        signal = signal_raw.iloc[1:].reset_index(drop=True).astype(float).rename("ECG")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(signal, lw=0.9)
        ax.set(
            title=f"Original ECG signal (first artefact value {removed_sample:g} removed)",
            xlabel="Sample index",
            ylabel="Amplitude",
        )
        plt.show()
        print(signal.describe().round(2))
        """
    ),
    code(
        """
        from statsmodels.tsa.stattools import adfuller

        transformed_signals = {
            "Original": signal,
            "First difference": signal.diff().dropna(),
            "Square root": np.sqrt(signal - signal.min() + 1),
            "Log": np.log(signal - signal.min() + 1),
        }

        stationarity_rows = []
        for name, series in transformed_signals.items():
            statistic, p_value, used_lag, n_obs, critical, _ = adfuller(series)
            stationarity_rows.append(
                {
                    "transform": name,
                    "ADF statistic": statistic,
                    "p-value": p_value,
                    "5% critical value": critical["5%"],
                    "lags used": used_lag,
                    "stationary at 5%": p_value < 0.05,
                }
            )

        stationarity_results = pd.DataFrame(stationarity_rows).set_index("transform")
        display(stationarity_results)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for ax, name in zip(axes, ["First difference", "Square root", "Log"]):
            ax.plot(transformed_signals[name], lw=0.8)
            ax.set(title=name, ylabel="Value")
        axes[-1].set_xlabel("Sample index")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        ### Transformation decision

        Every ADF p-value is far below 0.05, including the original series (ADF = −11.07, p ≈ 4.47×10⁻²⁰). The original ECG is therefore already stationary under this test. A more negative statistic after differencing is **not**, by itself, a reason to difference an already stationary series; unnecessary differencing can amplify noise and makes forecasts harder to interpret on the original amplitude scale.

        The untransformed signal is consequently retained for modelling. This choice preserves ECG amplitudes and avoids double differencing when the requested ARIMA model uses `d=1`. The ADF result only addresses a unit root—not all forms of non-stationarity or variance change—so visual diagnostics remain important.
        """
    ),
    code(
        """
        final_signal = signal
        split_index = 1800
        train = final_signal.iloc[:split_index]
        test = final_signal.iloc[split_index:]

        print(f"Training samples: {len(train)}; test samples: {len(test)}")
        print(f"Train mean/std: {train.mean():.2f}/{train.std():.2f}")
        print(f"Test mean/std:  {test.mean():.2f}/{test.std():.2f}")
        """
    ),
    code(
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        assumed_period = 200
        decomposition = seasonal_decompose(
            train, model="additive", period=assumed_period, extrapolate_trend="freq"
        )
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle(f"Additive decomposition (assumed period = {assumed_period} samples)", y=1.01)
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        plot_acf(train, lags=250, zero=False, ax=axes[0])
        plot_pacf(train, lags=40, zero=False, method="ywm", ax=axes[1])
        axes[0].set_title("ACF: short-range dependence and recurring peaks")
        axes[1].set_title("PACF: strongest direct effects at early lags")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        The decomposition uses 200 samples as an exploratory beat-scale period, not as a clinically estimated heart rate because the sampling frequency is unavailable. It shows a fairly stable level with recurring sharp excursions and residual variation. The ACF remains significant across many lags and displays recurring structure, while the PACF is strongest at the first few lags. Following the requested comparison, this motivates low-order candidates AR(3), MA(2), ARMA(3,2) and ARIMA(3,1,2). ACF/PACF are heuristics; out-of-sample errors decide the final ranking.
        """
    ),
    code(
        """
        from statsmodels.tsa.arima.model import ARIMA

        candidate_orders = {
            "AR(3)": (3, 0, 0),
            "MA(2)": (0, 0, 2),
            "ARMA(3,2)": (3, 0, 2),
            "ARIMA(3,1,2)": (3, 1, 2),
        }

        model_fits = {}
        forecasts = {}
        forecast_rows = []
        train_values = train.to_numpy()
        test_values = test.to_numpy()

        for model_name, order in candidate_orders.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(train_values, order=order).fit()
            prediction = np.asarray(fitted.forecast(steps=len(test_values)))
            errors = test_values - prediction
            model_fits[model_name] = fitted
            forecasts[model_name] = prediction
            forecast_rows.append(
                {
                    "model": model_name,
                    "RSS": np.sum(errors ** 2),
                    "RMSE": np.sqrt(mean_squared_error(test_values, prediction)),
                    "MAE": mean_absolute_error(test_values, prediction),
                    "train AIC": fitted.aic,
                }
            )

        mean_forecast = np.repeat(train.mean(), len(test_values))
        mean_errors = test_values - mean_forecast
        forecast_rows.append(
            {
                "model": "Training-mean baseline",
                "RSS": np.sum(mean_errors ** 2),
                "RMSE": np.sqrt(mean_squared_error(test_values, mean_forecast)),
                "MAE": mean_absolute_error(test_values, mean_forecast),
                "train AIC": np.nan,
            }
        )

        forecast_results = pd.DataFrame(forecast_rows).sort_values("RSS")
        display(forecast_results.round(2).reset_index(drop=True))
        """
    ),
    code(
        """
        best_requested_model = (
            forecast_results[forecast_results["model"] != "Training-mean baseline"]
            .iloc[0]["model"]
        )

        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        test_index = np.arange(split_index, split_index + len(test_values))
        axes[0].plot(test_index, test_values, color="black", lw=0.8, label="Observed")
        for name, prediction in forecasts.items():
            axes[0].plot(test_index, prediction, lw=1.2, alpha=0.8, label=name)
        axes[0].set(title="Multi-step forecasts over the complete hold-out segment", ylabel="Amplitude")
        axes[0].legend(ncol=3, fontsize=9)

        zoom = 300
        axes[1].plot(test_index[:zoom], test_values[:zoom], color="black", lw=1, label="Observed")
        axes[1].plot(
            test_index[:zoom], forecasts[best_requested_model][:zoom],
            color="tab:red", lw=1.5, label=f"Best requested model: {best_requested_model}"
        )
        axes[1].set(
            title="First 300 hold-out samples: local forecast behaviour",
            xlabel="Sample index",
            ylabel="Amplitude",
        )
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        ### Q4 forecast interpretation

        MA(2) gives the lowest requested-model test RSS (**4.70×10⁶**) and RMSE (**62.55**), narrowly improving on the training-mean baseline. AR(3) and ARMA(3,2) are close, while ARIMA(3,1,2) is markedly worse because its integration term is unnecessary for an ADF-stationary level series.

        Qualitatively, every long-horizon forecast rapidly reverts to a smooth level near the training mean. The models capture the central amplitude but fail to reproduce the timing and magnitude of sharp ECG peaks; this is visible in both the full hold-out plot and the 300-sample zoom. MA(2)'s lowest RSS therefore should not be mistaken for clinically faithful waveform prediction. Longer seasonal/state-space models, explicit beat alignment, or rolling one-step evaluation would be more appropriate follow-up work. The train/test mean also shifts (about 1026 versus 998), showing that an ADF rejection does not eliminate distribution shift.
        """
    ),
    md(
        r"""
        # Question 5 — Apriori association-rule mining

        The Statlog heart data are binarised using the coursework thresholds. `slope_not_1` deliberately reverses the reference encoding so that flat/downsloping ST segments are represented as the positive item. Both class states are one-hot encoded to allow separate disease-present and disease-absent consequents.

        Apriori first finds itemsets with support ≥ 0.25. Rules are then required to have lift ≥ 1.15, and the strongest set uses conviction > 1.5. For a rule A → B:

        - **support** is the fraction of all records containing A and B;
        - **confidence** is the fraction of records with A that also contain B;
        - **lift** compares confidence with B's baseline prevalence (values >1 indicate positive association);
        - **conviction** compares observed implication failures with those expected under independence (values >1 support directional association).
        """
    ),
    code(
        """
        from mlxtend.frequent_patterns import apriori, association_rules

        heart = pd.read_csv("heart-statlog.csv")
        binary = pd.DataFrame(index=heart.index)
        binary["age_gt_50"] = heart["age"] > 50
        binary["sex_1"] = heart["sex"] == 1
        binary["chest_pain_gt_2_5"] = heart["chest"] > 2.5
        binary["resting_bp_gt_125"] = heart["resting_blood_pressure"] > 125
        binary["cholesterol_gt_250"] = heart["serum_cholestoral"] > 250
        binary["fasting_bs_1"] = heart["fasting_blood_sugar"] == 1
        binary["restecg_not_0"] = heart["resting_electrocardiographic_results"] != 0
        binary["max_hr_gt_140"] = heart["maximum_heart_rate_achieved"] > 140
        binary["ex_angina_1"] = heart["exercise_induced_angina"] == 1
        binary["oldpeak_not_0"] = heart["oldpeak"] != 0
        binary["slope_not_1"] = heart["slope"] != 1
        binary["vessels_not_0"] = heart["number_of_major_vessels"] != 0
        binary["thal_not_3"] = heart["thal"] != 3
        binary["class_present"] = heart["class"].eq("present")
        binary["class_absent"] = heart["class"].eq("absent")

        binarisation_audit = pd.DataFrame(
            {
                "records": [len(binary)],
                "binary_items": [binary.shape[1]],
                "missing_values": [int(binary.isna().sum().sum())],
                "disease_prevalence": [binary["class_present"].mean()],
            }
        )
        display(binarisation_audit.round(3))
        display(binary.head())
        """
    ),
    code(
        """
        frequent_itemsets = apriori(binary, min_support=0.25, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.15)
        significant_rules = rules[rules["conviction"] > 1.5].copy()

        print(f"Frequent itemsets (support >= 0.25): {len(frequent_itemsets)}")
        print(f"Rules (lift >= 1.15): {len(rules)}")
        print(f"Strong rules (also conviction > 1.5): {len(significant_rules)}")


        def item_text(itemset):
            return " & ".join(sorted(itemset))


        def rule_table(frame, n=10):
            columns = ["antecedents", "consequents", "support", "confidence", "lift", "conviction"]
            result = frame.loc[:, columns].head(n).copy()
            result["antecedents"] = result["antecedents"].map(item_text)
            result["consequents"] = result["consequents"].map(item_text)
            return result.reset_index(drop=True)


        top_overall = significant_rules.sort_values(
            ["lift", "conviction"], ascending=False
        )
        display(rule_table(top_overall, n=10).round(3))
        """
    ),
    code(
        """
        disease_rules = rules[
            rules["consequents"].eq(frozenset({"class_present"}))
            & rules["conviction"].gt(1.5)
        ].sort_values(["lift", "conviction"], ascending=False)

        health_rules = rules[
            rules["consequents"].eq(frozenset({"class_absent"}))
        ].sort_values(["lift", "conviction"], ascending=False)

        print("Strong singleton-consequent rules for disease presence")
        display(rule_table(disease_rules, n=10).round(3))
        print("Singleton-consequent rules for disease absence (none reaches conviction > 1.5)")
        display(rule_table(health_rules, n=10).round(3))
        """
    ),
    md(
        r"""
        ### Q5 rule-by-rule interpretation

        The table makes the requested metrics explicit. Four representative rules are:

        1. **`vessels_not_0 & sex_1 → class_present`** — support 0.252 means the pattern occurs in 25.2% of all 270 records; confidence 0.829 means 82.9% of records with that antecedent have disease present. Lift 1.866 indicates 86.6% higher co-occurrence than the disease baseline, and conviction 3.254 is the strongest directional evidence among the singleton disease consequents.
        2. **`oldpeak_not_0 & thal_not_3 & chest_pain_gt_2_5 → class_present`** — support 0.267, confidence 0.809, lift 1.820 and conviction 2.908. The combination of ST depression, non-normal thallium coding and higher chest-pain category is both common enough to pass the support threshold and strongly associated with disease presence.
        3. **`thal_not_3 & chest_pain_gt_2_5 → class_present`** — support 0.296, confidence 0.792, lift 1.782 and conviction 2.672. Removing `oldpeak_not_0` increases coverage but slightly weakens confidence and conviction, illustrating the coverage–specificity trade-off.
        4. **`max_hr_gt_140 → class_absent`** — support 0.467, confidence 0.670, lift 1.206 and conviction 1.348. This is the strongest simple health-associated rule, but its conviction remains below 1.5, so evidence for disease absence is weaker under the chosen thresholds.

        There are 31 strong rules with exactly `class_present` as consequent and no `class_absent` rule above conviction 1.5. This asymmetry may reflect the available binary thresholds, target prevalence, correlated predictors and minimum-support pruning; it does not establish that “health is a default state.” Apriori rules are exploratory associations, can be redundant, and are not adjusted for confounding. They require validation before any clinical or diagnostic use.
        """
    ),
    md(
        r"""
        # Overall conclusions and limitations

        - A consistent preprocessing and Hungarian-aligned evaluation pipeline makes the clustering comparison valid and reproducible.
        - GMM + PCA is the strongest clustering configuration (0.897 accuracy; 0.893 weighted F1), while 20 PCs retain 90.2% of the original variance.
        - Ward + PCA is competitive; complete, average and especially single linkage produce severely imbalanced clusters.
        - The ECG level series is ADF-stationary. MA(2) has the lowest requested-model RSS, but long-horizon forecasts revert to the mean and miss sharp peaks.
        - Apriori surfaces interpretable risk-factor combinations with explicit support, confidence, lift and conviction, but association is not causation.

        Important limitations include evaluation on a single labelled feature dataset, post-hoc in-sample cluster scoring, no uncertainty intervals for metric differences, an unknown ECG sampling frequency, and no external validation of association rules. A production-grade extension would add patient-level hold-outs, clustering stability across seeds/resamples, uncertainty-aware ECG forecasting and validation on a second clinical cohort.
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
)

nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT} with {len(cells)} cells")
