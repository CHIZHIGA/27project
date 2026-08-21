#!/usr/bin/env python3
# coding: utf-8

"""Run the ECG 1D-CNN architecture on ``ecg_signals_preprocessed.csv``.

The preprocessed feature table is padded to 260 features so it can pass through
the same ``ECGRecognition`` architecture used for the MIT-BIH heartbeat windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ECG import ECGRecognition, evaluate, plot_training_curves, seed_everything


def load_table(path: Path, label_col: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))

    header = rows[0]
    label_idx = header.index(label_col)
    features = []
    raw_labels = []

    for row in rows[1:]:
        raw_labels.append(row[label_idx])
        features.append([float(value) for idx, value in enumerate(row) if idx != label_idx])

    label_values = sorted(set(raw_labels))
    label_to_idx = {value: idx for idx, value in enumerate(label_values)}
    labels = np.array([label_to_idx[value] for value in raw_labels], dtype=np.int64)
    return np.array(features, dtype=np.float32), labels, label_to_idx


def zscore_and_pad_to_260(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = (features - mean) / np.maximum(std, 1e-8)

    if features.shape[1] < 260:
        features = np.pad(features, ((0, 0), (0, 260 - features.shape[1])), mode="constant")
    else:
        features = features[:, :260]
    return features.astype(np.float32)


def make_split_loaders(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[tuple[DataLoader, DataLoader, DataLoader], dict[str, int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(features))
    rng.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    train_end = int(len(features) * 0.8)
    valid_end = int(len(features) * 0.9)
    split_specs = [
        (0, train_end, True),
        (train_end, valid_end, False),
        (valid_end, len(features), False),
    ]

    loaders = []
    for start, end, shuffle in split_specs:
        x = torch.tensor(features[start:end], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(labels[start:end], dtype=torch.long)
        loaders.append(DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle))

    sizes = {"train": train_end, "valid": valid_end - train_end, "test": len(features) - valid_end}
    return (loaders[0], loaders[1], loaders[2]), sizes


def predict_labels(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_true.append(labels.numpy())
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[true_label, pred_label] += 1

    supports = confusion.sum(axis=1)
    precisions = []
    recalls = []
    f1s = []
    for class_id in range(num_classes):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    weights = supports / supports.sum()
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "precision_weighted": float(np.sum(np.array(precisions) * weights)),
        "recall_weighted": float(np.sum(np.array(recalls) * weights)),
        "f1_weighted": float(np.sum(np.array(f1s) * weights)),
        "confusion_matrix": confusion.tolist(),
    }


def train_tabular_with_same_cnn(
    name: str,
    path: Path,
    label_col: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    raw_features, labels, label_to_idx = load_table(path, label_col)
    original_feature_count = raw_features.shape[1]
    features = zscore_and_pad_to_260(raw_features)
    (train_loader, valid_loader, test_loader), split_sizes = make_split_loaders(
        features, labels, batch_size, seed
    )

    model = ECGRecognition(num_classes=len(label_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.7)
    losses = []
    accs = []

    for _ in range(epochs):
        model.train()
        for inputs, batch_labels in train_loader:
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accs.append((logits.argmax(dim=1) == batch_labels).float().mean().item())

    valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    y_true, y_pred = predict_labels(model, test_loader, device)
    test_metrics = classification_metrics(y_true, y_pred, len(label_to_idx))

    torch.save(model.state_dict(), output_dir / f"{name}_same_cnn.pt")
    plot_training_curves(losses, accs, output_dir / f"{name}_training_curves.png")
    np.savetxt(
        output_dir / f"{name}_confusion_matrix.csv",
        np.array(test_metrics["confusion_matrix"], dtype=int),
        fmt="%d",
        delimiter=",",
    )

    return {
        "name": name,
        "source": str(path),
        "original_feature_count": int(original_feature_count),
        "model_input_length": 260,
        "label_to_index": label_to_idx,
        "split_sizes": split_sizes,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision_weighted": test_metrics["precision_weighted"],
        "test_recall_weighted": test_metrics["recall_weighted"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run ecg_signals_preprocessed.csv experiment.")
    parser.add_argument("--output-dir", type=Path, default=project_root / "results" / "ECG" / "root_datasets")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_root / "ecg_signals_preprocessed.csv",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    results = [
        train_tabular_with_same_cnn(
            "ecg_signals_preprocessed",
            args.data_path,
            "classes",
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.seed,
            device,
        )
    ]

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
