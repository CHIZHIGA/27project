#!/usr/bin/env python3
# coding: utf-8

"""ECG identity classification with PyTorch.

The script uses the MIT-BIH ECG files in ``Data/ECG``. It keeps the original
PaddlePaddle notebook's preprocessing and 1D-CNN architecture: five patients,
1000 R-peak centred windows per patient, window length 260, and an 8:1:1
train/validation/test split.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATIENT_LABELS = {"100": 0, "101": 1, "103": 2, "105": 3, "112": 4}
WINDOW_RADIUS = 130
WINDOW_LENGTH = WINDOW_RADIUS * 2


def load_patient_data(rr_file_path: Path, csv_file_path: Path, label: int) -> np.ndarray:
    """Load one patient's ECG windows and append the class label as last column."""
    rr_tokens = rr_file_path.read_text(encoding="utf-8").split()
    rr_data = np.array(rr_tokens).reshape([-1, 5])
    r_peak_times = rr_data[2:1002, 0]

    with csv_file_path.open(newline="", encoding="utf-8") as csv_file:
        rows = np.array(list(csv.reader(csv_file))[2:])

    ecg_times = rows[:, 0]
    ecg_values = rows[:, 1].astype(np.float32)
    time_to_index = {time_value: idx for idx, time_value in enumerate(ecg_times)}

    windows = []
    for r_time in r_peak_times:
        r_index = time_to_index.get(f"'{r_time}'")
        if r_index is None:
            continue

        start = r_index - WINDOW_RADIUS
        end = r_index + WINDOW_RADIUS
        if start < 0 or end > len(ecg_values):
            continue

        window = ecg_values[start:end]
        if len(window) == WINDOW_LENGTH:
            windows.append(window)

    ecg_windows = np.asarray(windows, dtype=np.float32)
    min_values = ecg_windows.min(axis=1, keepdims=True)
    max_values = ecg_windows.max(axis=1, keepdims=True)
    ranges = np.maximum(max_values - min_values, 1e-8)
    normalized = ((ecg_windows - min_values) / ranges) * 2.0 - 1.0

    labels = np.full((normalized.shape[0], 1), label, dtype=np.float32)
    return np.concatenate([normalized, labels], axis=1)


def load_ecg_dataset(data_dir: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_arrays = []
    for patient_id, label in PATIENT_LABELS.items():
        rr_path = data_dir / f"{patient_id}_RR.txt"
        samples_path = data_dir / f"{patient_id}_samples.csv"
        patient_data = load_patient_data(rr_path, samples_path, label)
        patient_arrays.append(patient_data)
        print(f"loaded patient {patient_id}: {len(patient_data)} samples")

    ecg_data = np.concatenate(patient_arrays, axis=0)
    rng = np.random.default_rng(seed)
    rng.shuffle(ecg_data)

    train_end = int(len(ecg_data) * 0.8)
    valid_end = int(len(ecg_data) * 0.9)
    train_set = ecg_data[:train_end]
    valid_set = ecg_data[train_end:valid_end]
    test_set = ecg_data[valid_end:]
    return train_set, valid_set, test_set


def make_loader(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool = False,
) -> DataLoader:
    ecgs = torch.tensor(data[:, :-1], dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(data[:, -1], dtype=torch.long)
    dataset = TensorDataset(ecgs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


class ECGRecognition(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=600, out_features=30),
            nn.ReLU(),
            nn.Linear(in_features=30, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean().item()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for ecgs, labels in data_loader:
            ecgs = ecgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(ecgs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    train_accs = []

    print(f"start training on {device} ...")
    for epoch in range(epochs):
        model.train()
        for batch_id, (ecgs, labels) in enumerate(train_loader):
            ecgs = ecgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(ecgs)
            loss = criterion(logits, labels)
            acc = accuracy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(acc)

            if batch_id % 10 == 0:
                print(
                    f"epoch: {epoch}, batch: {batch_id}, "
                    f"loss: {loss.item():.6f}, acc: {acc:.4f}"
                )

        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        print(
            f"epoch: {epoch} validation, "
            f"loss: {valid_loss:.6f}, acc: {valid_acc:.4f}"
        )

    return train_losses, train_accs


def plot_training_curves(
    losses: list[float],
    accs: list[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    steps = np.arange(len(losses))

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
    results_dir = project_root / "results" / "ECG"
    parser = argparse.ArgumentParser(description="Train a PyTorch ECG classifier.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "Data" / "ECG")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--momentum", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=results_dir)
    parser.add_argument("--model-path", type=Path, default=results_dir / "ECG_Recognition.pt")
    parser.add_argument("--plot-path", type=Path, default=results_dir / "training_curves.png")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device. auto uses CUDA when available.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str, force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"using GPU: {torch.cuda.get_device_name(device_index)}")
    else:
        print("using CPU")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device, args.cpu)
    describe_device(device)
    train_set, valid_set, test_set = load_ecg_dataset(args.data_dir, args.seed)
    print(f"split sizes: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}")

    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, pin_memory=pin_memory)
    valid_loader = make_loader(valid_set, args.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False, pin_memory=pin_memory)

    model = ECGRecognition().to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        device=device,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(train_losses, train_accs, args.plot_path)
    print(f"saved training curves to {args.plot_path}")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    print(f"saved model parameters to {args.model_path}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("\nTest set")
    print(f"avg_loss={test_loss:.6f}, avg_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
