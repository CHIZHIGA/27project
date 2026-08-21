"""Question 1: K-Means clustering before and after PCA."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ecg_signals_preprocessed.csv"
RESULTS = ROOT / "results" / "CW4" / "Q1"
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


def save_confusion(y_true, y_pred, title, filename, cmap):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        square=True,
        ax=ax,
    )
    ax.set(title=title, xlabel="Aligned cluster label", ylabel="True class")
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
        "K-Means (83 features)": X_scaled,
        f"K-Means (PCA, {n_components} PCs)": X_pca,
    }
    rows = []
    assignments = pd.DataFrame({"true_class": y})
    mappings = {}

    for name, feature_matrix in configurations.items():
        model = KMeans(n_clusters=3, random_state=4, n_init=10)
        cluster_ids = model.fit_predict(feature_matrix)
        aligned, mapping = hungarian_match(y, cluster_ids)
        rows.append({"model": name, **metrics(y, aligned), "inertia": model.inertia_})
        key = "full" if "83 features" in name else "pca"
        assignments[f"cluster_{key}"] = cluster_ids
        assignments[f"aligned_class_{key}"] = aligned
        mappings[key] = mapping
        save_confusion(
            y,
            aligned,
            name,
            f"confusion_matrix_{key}.png",
            "Blues" if key == "full" else "Greens",
        )

    metric_table = pd.DataFrame(rows)
    metric_table.to_csv(RESULTS / "metrics.csv", index=False)
    assignments.to_csv(RESULTS / "cluster_assignments.csv", index=False)
    pd.DataFrame(
        {
            "component": np.arange(1, len(cumulative) + 1),
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
        }
    ).to_csv(RESULTS / "pca_explained_variance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, lw=2)
    ax.axhline(0.90, color="tab:red", ls="--", label="90% threshold")
    ax.axvline(n_components, color="tab:green", ls=":", label=f"{n_components} PCs")
    ax.set(
        title="PCA dimensionality selection",
        xlabel="Number of principal components",
        ylabel="Cumulative explained variance",
        ylim=(0, 1.02),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "pca_cumulative_variance.png", dpi=180, bbox_inches="tight")
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
    print(f"Saved Q1 outputs to {RESULTS}")


if __name__ == "__main__":
    main()
