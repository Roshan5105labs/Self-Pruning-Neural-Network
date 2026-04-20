#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_float_list(raw: str) -> List[float]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Lambda list cannot be empty.")
    return [float(v) for v in values]


def parse_int_list(raw: str) -> List[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Hidden layer list cannot be empty.")
    return [int(v) for v in values]


class PrunableLinear(nn.Module):
    """Linear layer with learnable gates per weight."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)


class SelfPruningMLP(nn.Module):
    """Feed-forward CIFAR-10 classifier with prunable layers."""

    def __init__(self, hidden_dims: Iterable[int], num_classes: int = 10) -> None:
        super().__init__()
        dims = [32 * 32 * 3, *hidden_dims, num_classes]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(PrunableLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.network(x)


def prunable_layers(model: nn.Module) -> List[PrunableLinear]:
    return [m for m in model.modules() if isinstance(m, PrunableLinear)]


def sparsity_l1_loss(model: nn.Module) -> torch.Tensor:
    penalties = [layer.gate_values().abs().sum() for layer in prunable_layers(model)]
    if not penalties:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(penalties).sum()


def get_all_gate_values(model: nn.Module) -> torch.Tensor:
    all_gates = [layer.gate_values().reshape(-1) for layer in prunable_layers(model)]
    return torch.cat(all_gates) if all_gates else torch.empty(0)


def compute_sparsity(model: nn.Module, threshold: float = 1e-2) -> float:
    gates = get_all_gate_values(model)
    if gates.numel() == 0:
        return 0.0
    sparse = (gates < threshold).float().mean().item()
    return sparse * 100.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return avg_loss, accuracy


@dataclass
class RunResult:
    lambda_value: float
    best_epoch: int
    test_accuracy: float
    test_loss: float
    sparsity_percent: float
    checkpoint_path: str
    run_dir: str


def get_data_loaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tfms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_tfms)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_tfms)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_for_lambda(
    lambda_value: float,
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
) -> RunResult:
    run_dir = output_dir / f"lambda_{lambda_value:.0e}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = SelfPruningMLP(hidden_dims=args.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_acc = -1.0
    best_epoch = -1

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_ce = 0.0
        running_sparse = 0.0
        seen = 0
        steps = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            ce_loss = criterion(logits, labels)
            sparse_loss = sparsity_l1_loss(model)
            total_loss = ce_loss + lambda_value * sparse_loss

            total_loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            seen += batch_size
            steps += 1
            running_total += total_loss.item() * batch_size
            running_ce += ce_loss.item() * batch_size
            running_sparse += sparse_loss.item()

        test_loss, test_acc = evaluate(model, test_loader, device=device, max_batches=args.max_test_batches)
        current_sparsity = compute_sparsity(model, threshold=args.gate_threshold)

        epoch_log = {
            "epoch": epoch,
            "train_total_loss": running_total / max(seen, 1),
            "train_ce_loss": running_ce / max(seen, 1),
            "train_sparse_loss": running_sparse / max(steps, 1),
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "sparsity_percent": current_sparsity,
        }
        history.append(epoch_log)

        print(
            f"[lambda={lambda_value:.0e}] "
            f"epoch={epoch:03d} "
            f"train_total={epoch_log['train_total_loss']:.4f} "
            f"test_acc={test_acc * 100:.2f}% "
            f"sparsity={current_sparsity:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no checkpoints.")

    ckpt_path = run_dir / "best_model.pt"
    torch.save(best_state, ckpt_path)

    model.load_state_dict(best_state)
    final_test_loss, final_test_acc = evaluate(model, test_loader, device=device, max_batches=args.max_test_batches)
    final_sparsity = compute_sparsity(model, threshold=args.gate_threshold)

    gate_values = get_all_gate_values(model).detach().cpu().numpy()
    torch.save(torch.tensor(gate_values), run_dir / "gate_values.pt")

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    result = RunResult(
        lambda_value=lambda_value,
        best_epoch=best_epoch,
        test_accuracy=final_test_acc * 100.0,
        test_loss=final_test_loss,
        sparsity_percent=final_sparsity,
        checkpoint_path=str(ckpt_path),
        run_dir=str(run_dir),
    )
    with (run_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    return result


def save_summary(results: list[RunResult], output_dir: Path) -> Path:
    summary_path = output_dir / "results_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda_value",
                "best_epoch",
                "test_accuracy",
                "test_loss",
                "sparsity_percent",
                "checkpoint_path",
                "run_dir",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return summary_path


def save_markdown_table(results: list[RunResult], output_dir: Path) -> Path:
    md_path = output_dir / "results_table.md"
    lines = [
        "| Lambda | Test Accuracy (%) | Sparsity Level (%) | Best Epoch |",
        "|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda x: x.lambda_value):
        lines.append(
            f"| {result.lambda_value:.0e} | {result.test_accuracy:.2f} | "
            f"{result.sparsity_percent:.2f} | {result.best_epoch} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def plot_gate_distribution(gates: torch.Tensor, save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(gates.cpu().numpy(), bins=100, color="#2C7FB8", edgecolor="black", alpha=0.85)
    plt.title("Distribution of Final Gate Values (Best Model)")
    plt.xlabel("Gate Value (sigmoid(gate_scores))")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-pruning neural network on CIFAR-10.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambdas", type=str, default="1e-6,1e-5,1e-4")
    parser.add_argument("--hidden-dims", type=str, default="1024,512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gate-threshold", type=float, default=1e-2)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()
    args.lambdas = parse_float_list(args.lambdas)
    args.hidden_dims = parse_int_list(args.hidden_dims)

    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Lambda values: {args.lambdas}")

    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size, args.num_workers)

    all_results: list[RunResult] = []
    for lambda_value in args.lambdas:
        result = train_for_lambda(
            lambda_value=lambda_value,
            args=args,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            output_dir=output_dir,
        )
        all_results.append(result)

    summary_path = save_summary(all_results, output_dir)
    table_path = save_markdown_table(all_results, output_dir)

    best = max(all_results, key=lambda x: x.test_accuracy)
    best_model = SelfPruningMLP(hidden_dims=args.hidden_dims).to(device)
    best_state = torch.load(best.checkpoint_path, map_location=device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    best_gates = get_all_gate_values(best_model).detach().cpu()
    plot_path = output_dir / "best_gate_distribution.png"
    plot_gate_distribution(best_gates, plot_path)

    print("\n=== Final Summary ===")
    for result in sorted(all_results, key=lambda x: x.lambda_value):
        print(
            f"lambda={result.lambda_value:.0e} | "
            f"test_acc={result.test_accuracy:.2f}% | "
            f"sparsity={result.sparsity_percent:.2f}% | "
            f"best_epoch={result.best_epoch}"
        )
    print(f"\nBest model (by test accuracy): lambda={best.lambda_value:.0e}")
    print(f"Summary CSV: {summary_path}")
    print(f"Markdown table: {table_path}")
    print(f"Gate distribution plot: {plot_path}")


if __name__ == "__main__":
    main()
