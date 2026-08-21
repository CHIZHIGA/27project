"""Question 2: Gaussian Mixture clustering before and after PCA."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ecg_signals_preprocessed.csv"
RESULTS = ROOT / "results" / "CW4" / "Q2"
RESULTS.mkdir(parents=True, exist_ok=True)


def hungarian_match(y_true, cluster_ids):
    y_true = np.asarray(y_true)
    cluster_ids = np.asarray(cluster_ids)
    true_labels = np.unique(y_true)
    cluster_labels = np.unique(cluster_ids)
    table = pd.crosstab(
        pd.Series(y_true, name="true_class"),
        pd.Series(cluster_ids, name="cluster"),
    ).reindex(index=true_labels, columns=cluster_labels, fill_value=0)
    rows, cols = linear_sum_assignment(-table.to_numpy())
    mapping = {int(cluster_labels[c]): int(true_labels[r]) for r, c in zip(rows, cols)}
    aligned = np.array([mapping[int(cluster)] for cluster in cluster_ids])
    return aligned, mapping


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main():
    sns.set_theme(style="whitegrid")
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns="classes")
    y = data["classes"]
    X_scaled = StandardScaler().fit_transform(X)

    pca_full = PCA().fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, 0.90) + 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    configurations = {
        "GMM (83 features)": X_scaled,
        f"GMM (PCA, {n_components} PCs)": X_pca,
    }
    rows = []
    assignments = pd.DataFrame({"true_class": y})
    mappings = {}
    predictions = {}

    for name, feature_matrix in configurations.items():
        model = GaussianMixture(
            n_components=3, covariance_type="full", random_state=9
        )
        cluster_ids = model.fit_predict(feature_matrix)
        aligned, mapping = hungarian_match(y, cluster_ids)
        key = "full" if "83 features" in name else "pca"
        # AIC/BIC values are not compared across spaces of different dimensions;
        # retain the common external metrics for a like-for-like evaluation.
        rows.append({"model": name, **metrics(y, aligned)})
        assignments[f"cluster_{key}"] = cluster_ids
        assignments[f"aligned_class_{key}"] = aligned
        mappings[key] = mapping
        predictions[key] = aligned

    metric_table = pd.DataFrame(rows)
    metric_table.to_csv(RESULTS / "metrics.csv", index=False)
    assignments.to_csv(RESULTS / "cluster_assignments.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, key, title, cmap in [
        (axes[0], "full", "GMM — 83 features", "Blues"),
        (axes[1], "pca", "GMM — PCA", "Greens"),
    ]:
        sns.heatmap(
            confusion_matrix(y, predictions[key]),
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            square=True,
            ax=ax,
        )
        ax.set(title=title, xlabel="Aligned cluster label", ylabel="True class")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "samples": len(data),
        "original_features": X.shape[1],
        "selected_components": n_components,
        "retained_variance": float(pca.explained_variance_ratio_.sum()),
        "hungarian_mappings": mappings,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(metric_table.to_string(index=False))
    print(f"Saved Q2 outputs to {RESULTS}")


if __name__ == "__main__":
    main()
