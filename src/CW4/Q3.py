"""Question 3: agglomerative-linkage study and cross-model comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
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
RESULTS = ROOT / "results" / "CW4" / "Q3"
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
    n_components = int(
        np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.90) + 1
    )
    X_pca = PCA(n_components=n_components).fit_transform(X_scaled)
    spaces = {"83 features": X_scaled, f"PCA ({n_components} PCs)": X_pca}

    rows = []
    predictions = {}
    assignments = pd.DataFrame({"true_class": y})
    mappings = {}

    for space_name, feature_matrix in spaces.items():
        for linkage in ["ward", "complete", "average", "single"]:
            model = AgglomerativeClustering(n_clusters=3, linkage=linkage)
            cluster_ids = model.fit_predict(feature_matrix)
            aligned, mapping = hungarian_match(y, cluster_ids)
            sizes = np.bincount(cluster_ids, minlength=3)
            rows.append(
                {
                    "space": space_name,
                    "linkage": linkage,
                    **metrics(y, aligned),
                    "cluster_0_size": int(sizes[0]),
                    "cluster_1_size": int(sizes[1]),
                    "cluster_2_size": int(sizes[2]),
                    "largest_cluster_share": sizes.max() / len(cluster_ids),
                }
            )
            key = f"{'full' if space_name == '83 features' else 'pca'}_{linkage}"
            assignments[f"cluster_{key}"] = cluster_ids
            assignments[f"aligned_class_{key}"] = aligned
            predictions[(space_name, linkage)] = aligned
            mappings[key] = mapping

    linkage_results = pd.DataFrame(rows)
    linkage_results.to_csv(RESULTS / "linkage_metrics.csv", index=False)
    assignments.to_csv(RESULTS / "cluster_assignments.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    sns.barplot(
        data=linkage_results, x="linkage", y="accuracy", hue="space", ax=axes[0]
    )
    sns.barplot(
        data=linkage_results,
        x="linkage",
        y="largest_cluster_share",
        hue="space",
        ax=axes[1],
    )
    axes[0].set(title="External accuracy", ylim=(0, 1))
    axes[1].set(
        title="Largest-cluster share", ylabel="Share of samples", ylim=(0, 1)
    )
    axes[1].get_legend().remove()
    fig.tight_layout()
    fig.savefig(RESULTS / "linkage_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, space_name, title, cmap in [
        (axes[0], "83 features", "Ward — 83 features", "Blues"),
        (axes[1], f"PCA ({n_components} PCs)", "Ward — PCA", "Greens"),
    ]:
        sns.heatmap(
            confusion_matrix(y, predictions[(space_name, "ward")]),
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            square=True,
            ax=ax,
        )
        ax.set(title=title, xlabel="Aligned cluster label", ylabel="True class")
    fig.tight_layout()
    fig.savefig(RESULTS / "ward_confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Refit the Q1/Q2 models so this script remains independently reproducible.
    comparison_rows = []
    for space_name, feature_matrix in spaces.items():
        models = {
            "K-Means": KMeans(n_clusters=3, random_state=4, n_init=10),
            "GMM": GaussianMixture(
                n_components=3, covariance_type="full", random_state=9
            ),
        }
        for model_name, model in models.items():
            aligned, _ = hungarian_match(y, model.fit_predict(feature_matrix))
            comparison_rows.append(
                {"model": model_name, "space": space_name, **metrics(y, aligned)}
            )
        ward = predictions[(space_name, "ward")]
        comparison_rows.append(
            {
                "model": "Agglomerative (Ward)",
                "space": space_name,
                **metrics(y, ward),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "accuracy", ascending=False
    )
    comparison.to_csv(RESULTS / "cross_model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=comparison, x="model", y="accuracy", hue="space", ax=ax)
    ax.set(title="Q1–Q3 clustering comparison", ylim=(0.7, 0.92))
    fig.tight_layout()
    fig.savefig(RESULTS / "cross_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    best = linkage_results.loc[linkage_results["accuracy"].idxmax()]
    summary = {
        "samples": len(data),
        "selected_components": n_components,
        "best_hierarchical_configuration": {
            "space": best["space"],
            "linkage": best["linkage"],
            "accuracy": float(best["accuracy"]),
            "f1_weighted": float(best["f1_weighted"]),
        },
        "hungarian_mappings": mappings,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(linkage_results.to_string(index=False))
    print("\nCross-model comparison:\n", comparison.to_string(index=False))
    print(f"Saved Q3 outputs to {RESULTS}")


if __name__ == "__main__":
    main()
