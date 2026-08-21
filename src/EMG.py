#!/usr/bin/env python3
# coding: utf-8

"""EMG gesture classification with PyTorch.

The script reads the EMG Pattern Database text files under
``Data/EMG_data_for_gestures-master``. It removes unmarked samples by default,
creates fixed-length windows from contiguous gesture segments, and trains a
1D-CNN over the eight EMG channels. Optional Haar-wavelet denoising, an FFT
feature branch, and group-aware splitting support leakage-resistant ablations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GESTURE_NAMES = {
    0: "unmarked data",
    1: "hand at rest",
    2: "hand clenched in a fist",
    3: "wrist flexion",
    4: "wrist extension",
    5: "radial deviations",
    6: "ulnar deviations",
    7: "extended palm",
}


def load_emg_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.genfromtxt(
            path,
            delimiter="\t",
            skip_header=1,
            usecols=range(1, 10),
            dtype=np.float32,
            invalid_raise=False,
        )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    data = data[~np.isnan(data).any(axis=1)]
    channels = data[:, :8]
    labels = data[:, 8].astype(np.int64)
    return channels, labels


def window_file(
    path: Path,
    window_size: int,
    stride: int,
    include_unmarked: bool,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    channels, labels = load_emg_file(path)
    windows = []
    window_labels = []
    window_groups = []
    start = 0
    segment_id = 0

    while start < len(labels):
        label = int(labels[start])
        end = start + 1
        while end < len(labels) and labels[end] == label:
            end += 1

        if (include_unmarked or label != 0) and end - start >= window_size:
            group_id = f"{path.parent.name}/{path.name}:segment-{segment_id}"
            for idx in range(start, end - window_size + 1, stride):
                windows.append(channels[idx : idx + window_size].T)
                window_labels.append(label)
                window_groups.append(group_id)

        start = end
        segment_id += 1

    return windows, window_labels, window_groups


def load_emg_dataset(
    data_dir: Path,
    window_size: int,
    stride: int,
    include_unmarked: bool,
    max_windows_per_class: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], np.ndarray]:
    files = sorted(data_dir.glob("[0-9][0-9]/*raw_data*.txt"))
    if not files:
        raise FileNotFoundError(f"No raw EMG txt files found under {data_dir}")

    all_windows = []
    all_labels = []
    all_groups = []
    for path in files:
        windows, labels, groups = window_file(path, window_size, stride, include_unmarked)
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_groups.extend(groups)

    raw_labels = np.asarray(all_labels, dtype=np.int64)
    features = np.asarray(all_windows, dtype=np.float32)
    label_values = sorted(int(value) for value in np.unique(raw_labels))
    label_to_index = {label: idx for idx, label in enumerate(label_values)}
    labels = np.asarray([label_to_index[int(value)] for value in raw_labels], dtype=np.int64)
    groups = np.asarray(all_groups)

    if max_windows_per_class is not None:
        features, labels, groups = limit_windows_per_class(
            features, labels, groups, max_windows_per_class, seed
        )

    return features, labels, label_to_index, groups


def limit_windows_per_class(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    max_windows_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    kept_indices = []
    for class_id in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        kept_indices.extend(class_indices[:max_windows_per_class])

    kept_indices = np.asarray(kept_indices, dtype=np.int64)
    rng.shuffle(kept_indices)
    return features[kept_indices], labels[kept_indices], groups[kept_indices]


def stratified_split(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    train_indices = []
    valid_indices = []
    test_indices = []

    for class_id in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        train_end = int(len(class_indices) * 0.8)
        valid_end = int(len(class_indices) * 0.9)
        train_indices.extend(class_indices[:train_end])
        valid_indices.extend(class_indices[train_end:valid_end])
        test_indices.extend(class_indices[valid_end:])

    splits = []
    for indices in (train_indices, valid_indices, test_indices):
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        splits.append((features[indices], labels[indices]))

    return splits[0], splits[1], splits[2]


def group_stratified_split(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    dict[str, object],
]:
    """Split whole gesture segments, stratified by their single class label."""
    rng = np.random.default_rng(seed)
    split_indices: list[list[int]] = [[], [], []]
    split_groups: list[set[str]] = [set(), set(), set()]

    for class_id in sorted(np.unique(labels)):
        class_groups = np.unique(groups[labels == class_id])
        rng.shuffle(class_groups)
        if len(class_groups) < 3:
            raise ValueError(
                f"Class {class_id} has only {len(class_groups)} groups; at least 3 are required."
            )

        group_count = len(class_groups)
        if group_count == 3:
            train_end, valid_end = 1, 2
        else:
            train_end = max(1, int(group_count * 0.8))
            valid_end = max(train_end + 1, int(group_count * 0.9))
            valid_end = min(valid_end, group_count - 1)
        group_parts = (
            class_groups[:train_end],
            class_groups[train_end:valid_end],
            class_groups[valid_end:],
        )

        for split_id, group_part in enumerate(group_parts):
            mask = np.isin(groups, group_part) & (labels == class_id)
            split_indices[split_id].extend(np.flatnonzero(mask).tolist())
            split_groups[split_id].update(str(value) for value in group_part)

    splits = []
    for indices in split_indices:
        index_array = np.asarray(indices, dtype=np.int64)
        rng.shuffle(index_array)
        splits.append((features[index_array], labels[index_array]))

    overlaps = {
        "train_valid": len(split_groups[0] & split_groups[1]),
        "train_test": len(split_groups[0] & split_groups[2]),
        "valid_test": len(split_groups[1] & split_groups[2]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Group leakage detected: {overlaps}")

    metadata = {
        "group_counts": {
            "train": len(split_groups[0]),
            "valid": len(split_groups[1]),
            "test": len(split_groups[2]),
        },
        "group_overlaps": overlaps,
    }
    return splits[0], splits[1], splits[2], metadata


def haar_wavelet_denoise(
    features: np.ndarray,
    levels: int = 3,
    threshold_scale: float = 0.5,
) -> np.ndarray:
    """Vectorised multi-level Haar DWT soft-threshold denoising."""
    if levels < 1:
        raise ValueError("Wavelet levels must be at least 1.")
    divisor = 2**levels
    if features.shape[-1] % divisor != 0:
        raise ValueError(
            f"Window length {features.shape[-1]} must be divisible by 2**levels={divisor}."
        )
    if threshold_scale < 0:
        raise ValueError("Wavelet threshold scale must be non-negative.")

    sqrt_two = np.float32(np.sqrt(2.0))
    approximation = features.astype(np.float32, copy=True)
    details = []
    for _ in range(levels):
        even = approximation[..., 0::2]
        odd = approximation[..., 1::2]
        details.append((even - odd) / sqrt_two)
        approximation = (even + odd) / sqrt_two

    # Robust noise estimate from the finest-scale detail coefficients.
    sigma = np.median(np.abs(details[0]), axis=-1, keepdims=True) / np.float32(0.6745)
    threshold = (
        np.float32(threshold_scale)
        * sigma
        * np.float32(np.sqrt(2.0 * np.log(features.shape[-1])))
    )
    thresholded_details = [
        np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        for detail in details
    ]

    reconstructed = approximation
    for detail in reversed(thresholded_details):
        even = (reconstructed + detail) / sqrt_two
        odd = (reconstructed - detail) / sqrt_two
        merged = np.empty((*even.shape[:-1], even.shape[-1] * 2), dtype=np.float32)
        merged[..., 0::2] = even
        merged[..., 1::2] = odd
        reconstructed = merged
    return reconstructed.astype(np.float32, copy=False)


def standardize_splits(
    train: tuple[np.ndarray, np.ndarray],
    valid: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    train_x, train_y = train
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-8)

    return (
        (((train_x - mean) / std).astype(np.float32), train_y),
        (((valid[0] - mean) / std).astype(np.float32), valid[1]),
        (((test[0] - mean) / std).astype(np.float32), test[1]),
    )


def make_loader(
    split: tuple[np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    features, labels = split
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


class EMGRecognition(nn.Module):
    def __init__(self, num_classes: int, use_fft_branch: bool = False) -> None:
        super().__init__()
        self.use_fft_branch = use_fft_branch
        self.features = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        if use_fft_branch:
            self.fft_features = nn.Sequential(
                nn.Conv1d(8, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
        classifier_features = 128 + (64 if use_fft_branch else 0)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        time_features = self.features(inputs)
        if not self.use_fft_branch:
            return self.classifier(time_features)

        # Exclude the DC bin; log magnitude compresses the spectral dynamic range.
        spectrum = torch.log1p(torch.abs(torch.fft.rfft(inputs, dim=-1)))[..., 1:]
        frequency_features = self.fft_features(spectrum)
        return self.classifier(torch.cat([time_features, frequency_features], dim=1))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str, force_cpu: bool) -> torch.device:
    if force_cpu or device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    device: torch.device,
) -> tuple[list[float], list[float], dict[str, object]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    accs = []
    valid_losses = []
    valid_accs = []
    best_valid_loss = float("inf")
    best_valid_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    print(f"start training on {device} ...")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_id, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)

            if batch_id % 20 == 0:
                print(
                    f"epoch: {epoch}, batch: {batch_id}, "
                    f"loss: {loss.item():.6f}, acc: {acc:.4f}"
                )

        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        print(f"epoch: {epoch} validation, loss: {valid_loss:.6f}, acc: {valid_acc:.4f}")

        if valid_loss < best_valid_loss - early_stopping_min_delta:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0:
                print(
                    "early stopping counter: "
                    f"{epochs_without_improvement}/{early_stopping_patience}"
                )

        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(f"early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    history = {
        "epochs_run": len(valid_losses),
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "best_valid_acc": best_valid_acc,
        "valid_losses": valid_losses,
        "valid_accs": valid_accs,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "stopped_early": len(valid_losses) < epochs,
    }
    return losses, accs, history


def predict_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device, non_blocking=True))
            true_labels.append(labels.numpy())
            pred_labels.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(true_labels), np.concatenate(pred_labels)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
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
        "precision_weighted": float(np.sum(np.asarray(precisions) * weights)),
        "recall_weighted": float(np.sum(np.asarray(recalls) * weights)),
        "f1_weighted": float(np.sum(np.asarray(f1s) * weights)),
        "confusion_matrix": confusion.tolist(),
    }


def plot_training_curves(losses: list[float], accs: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(losses))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, losses, color="red")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("batch")
    axes[0].set_ylabel("loss")

    axes[1].plot(steps, accs, color="blue")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("batch")
    axes[1].set_ylabel("acc")
    axes[1].set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results" / "EMG"
    parser = argparse.ArgumentParser(description="Train a PyTorch EMG gesture classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "Data" / "EMG_data_for_gestures-master",
    )
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--split-mode",
        choices=("window", "group"),
        default="window",
        help=(
            "window reproduces the original stratified window split; group keeps all "
            "overlapping windows from a gesture segment in one split."
        ),
    )
    parser.add_argument(
        "--wavelet-denoise",
        action="store_true",
        help="Apply multi-level Haar-wavelet soft-threshold denoising before standardisation.",
    )
    parser.add_argument("--wavelet-levels", type=int, default=3)
    parser.add_argument("--wavelet-threshold-scale", type=float, default=0.5)
    parser.add_argument(
        "--fft-branch",
        action="store_true",
        help="Fuse a log-magnitude rFFT feature branch with the time-domain CNN.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop if validation loss does not improve for this many epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation loss improvement required to reset early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=results_dir)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--confusion-path", type=Path, default=None)
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=None,
        help="Optional cap for quick experiments or class balancing.",
    )
    parser.add_argument(
        "--include-unmarked",
        action="store_true",
        help="Include class 0 windows. By default, unmarked samples are removed.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device. auto uses CUDA when available.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path or args.results_dir / "EMG_Recognition.pt"
    args.plot_path = args.plot_path or args.results_dir / "training_curves.png"
    args.summary_path = args.summary_path or args.results_dir / "summary.json"
    args.confusion_path = args.confusion_path or args.results_dir / "confusion_matrix.csv"
    seed_everything(args.seed)
    device = resolve_device(args.device, args.cpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"using GPU: {torch.cuda.get_device_name()}")
    else:
        print("using CPU")

    features, labels, label_to_index, groups = load_emg_dataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        include_unmarked=args.include_unmarked,
        max_windows_per_class=args.max_windows_per_class,
        seed=args.seed,
    )
    print(f"loaded windows: {len(features)}, input shape: {features.shape[1:]}")
    print(f"label mapping: {label_to_index}")

    if args.split_mode == "group":
        train_split, valid_split, test_split, split_metadata = group_stratified_split(
            features, labels, groups, args.seed
        )
    else:
        train_split, valid_split, test_split = stratified_split(features, labels, args.seed)
        split_metadata = {
            "group_counts": None,
            "group_overlaps": "not controlled in window split",
        }

    if args.wavelet_denoise:
        print(
            "applying Haar-wavelet denoising: "
            f"levels={args.wavelet_levels}, threshold_scale={args.wavelet_threshold_scale}"
        )
        train_split = (
            haar_wavelet_denoise(
                train_split[0], args.wavelet_levels, args.wavelet_threshold_scale
            ),
            train_split[1],
        )
        valid_split = (
            haar_wavelet_denoise(
                valid_split[0], args.wavelet_levels, args.wavelet_threshold_scale
            ),
            valid_split[1],
        )
        test_split = (
            haar_wavelet_denoise(
                test_split[0], args.wavelet_levels, args.wavelet_threshold_scale
            ),
            test_split[1],
        )

    train_split, valid_split, test_split = standardize_splits(
        train_split, valid_split, test_split
    )
    split_sizes = {
        "train": int(len(train_split[0])),
        "valid": int(len(valid_split[0])),
        "test": int(len(test_split[0])),
    }
    print(f"split sizes: {split_sizes}")

    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_split, args.batch_size, True, pin_memory)
    valid_loader = make_loader(valid_split, args.batch_size, False, pin_memory)
    test_loader = make_loader(test_split, args.batch_size, False, pin_memory)

    model = EMGRecognition(
        num_classes=len(label_to_index), use_fft_branch=args.fft_branch
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    losses, accs, train_history = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        device=device,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(losses, accs, args.plot_path)
    torch.save(model.state_dict(), args.model_path)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    y_true, y_pred = predict_labels(model, test_loader, device)
    metrics = classification_metrics(y_true, y_pred, len(label_to_index))
    np.savetxt(args.confusion_path, np.asarray(metrics["confusion_matrix"]), fmt="%d", delimiter=",")

    index_to_label = {index: label for label, index in label_to_index.items()}
    summary = {
        "source": str(args.data_dir),
        "window_size": args.window_size,
        "stride": args.stride,
        "include_unmarked": args.include_unmarked,
        "split_mode": args.split_mode,
        "split_metadata": split_metadata,
        "wavelet_denoise": args.wavelet_denoise,
        "wavelet": {
            "name": "Haar",
            "levels": args.wavelet_levels,
            "threshold": "soft universal threshold from finest detail MAD",
            "threshold_scale": args.wavelet_threshold_scale,
        },
        "fft_branch": args.fft_branch,
        "fft_representation": "log1p(abs(rfft)), DC bin excluded" if args.fft_branch else None,
        "requested_epochs": args.epochs,
        "training": train_history,
        "label_to_index": {
            str(label): {
                "index": int(index),
                "name": GESTURE_NAMES.get(label, f"class {label}"),
            }
            for label, index in label_to_index.items()
        },
        "index_to_label": {str(index): int(label) for index, label in index_to_label.items()},
        "split_sizes": split_sizes,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision_weighted": metrics["precision_weighted"],
        "test_recall_weighted": metrics["recall_weighted"],
        "test_f1_weighted": metrics["f1_weighted"],
        "test_confusion_matrix": metrics["confusion_matrix"],
    }
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved model parameters to {args.model_path}")
    print(f"saved training curves to {args.plot_path}")
    print(f"saved confusion matrix to {args.confusion_path}")
    print(f"saved summary to {args.summary_path}")
    print("\nTest set")
    print(f"avg_loss={test_loss:.6f}, avg_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
