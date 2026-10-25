"""Full-batch logistic-regression convergence experiments with resumable traces.

Only corrupted training data enter this module. Candidate models are independent;
their logits are evaluated together to amortize matrix multiplication overhead.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _candidates(lambda_grid):
    values = sorted(set(float(value) for value in lambda_grid))
    if not values or values[0] != 0 or not np.isfinite(values).all():
        raise ValueError("lambda_grid must contain zero and finite nonnegative values.")
    return [("Baseline", 0.0)] + [
        (name, value) for name in ("L1", "L2") for value in values if value > 0
    ]


class _CandidateModels(nn.Module):
    def __init__(self, n_features, n_classes, candidates, seed):
        super().__init__()
        # Same nn.Linear initialization as the notebook, shared by all candidates.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            initial = nn.Linear(n_features, n_classes)
        self.weight = nn.Parameter(initial.weight.detach().repeat(len(candidates), 1, 1))
        self.bias = nn.Parameter(initial.bias.detach().repeat(len(candidates), 1))
        self.n_classes = n_classes
        self.n_candidates = len(candidates)

    def forward(self, X):
        logits = F.linear(X, self.weight.reshape(-1, X.shape[1]), self.bias.flatten())
        return logits.reshape(len(X), self.n_candidates, self.n_classes)


def _evaluate(models, X_train, y_train, X_val, y_val, candidates):
    """Metrics AND gradients at the current parameter state, after its updates."""
    models.zero_grad(set_to_none=True)
    logits = models(X_train)
    targets = y_train[:, None].expand(-1, len(candidates)).reshape(-1)
    train_ce = F.cross_entropy(logits.reshape(-1, models.n_classes), targets,
                               reduction="none").reshape(len(X_train), -1).mean(dim=0)
    l1 = models.weight.abs().sum(dim=(1, 2))
    l2_squared = models.weight.square().sum(dim=(1, 2))
    penalties = torch.stack([
        l1[i] if name == "L1" else l2_squared[i] if name == "L2" else l1[i] * 0
        for i, (name, _) in enumerate(candidates)
    ])
    strengths = train_ce.new_tensor([value for _, value in candidates])
    regularization_term = strengths * penalties
    objectives = train_ce + regularization_term
    # SUM, not mean: each parameter block receives its own independent gradient.
    objectives.sum().backward()
    with torch.no_grad():
        val_logits = models(X_val)
        val_targets = y_val[:, None].expand(-1, len(candidates)).reshape(-1)
        val_ce = F.cross_entropy(val_logits.reshape(-1, models.n_classes), val_targets,
                                 reduction="none").reshape(len(X_val), -1).mean(dim=0)
        metrics = {
            "train_ce": train_ce,
            "train_objective": objectives,
            "penalty": penalties,
            "regularization_term": regularization_term,
            "val_ce": val_ce,
            "train_accuracy": (logits.argmax(dim=-1) == y_train[:, None]).float().mean(0),
            "val_accuracy": (val_logits.argmax(dim=-1) == y_val[:, None]).float().mean(0),
            "weight_l1": l1,
            "weight_l2": l2_squared.sqrt(),
            "near_zero_fraction": (models.weight.abs() < 1e-3).float().mean((1, 2)),
            "exact_zero_fraction": (models.weight == 0).float().mean((1, 2)),
            "gradient_norm": (models.weight.grad.square().sum((1, 2))
                              + models.bias.grad.square().sum(1)).sqrt(),
        }
        return {key: value.detach().cpu().numpy() for key, value in metrics.items()}


def run_experiment(raw_datasets, output_dir, *, max_iter=2000, lambda_grid=(0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
                   lr=0.1, seed=42, n_splits=3, checkpoint_every=50, num_threads=2):
    """Run or extend the same experiment. Reuse completed folds without training.

    Iteration 0 is initialization. Row t measures parameters AFTER t SGD updates.
    Checkpoints commit every ``checkpoint_every`` iterations (and at completion).
    On interruption, discard CSV rows beyond the last committed checkpoint.
    Changing data, implementation, versions, folds or hyperparameters starts a
    separate experiment directory. Increasing max_iter extends existing traces.
    """
    if max_iter < 1 or checkpoint_every < 1 or lr <= 0:
        raise ValueError("max_iter, checkpoint_every and lr must be positive.")
    candidates = _candidates(lambda_grid)
    torch.set_num_threads(num_threads)
    config = {
        "schema_version": 1,
        "seed": int(seed), "n_splits": int(n_splits), "lr": float(lr),
        "lambda_grid": sorted(set(float(x) for x in lambda_grid)),
        "candidates": candidates, "optimizer": "SGD; momentum=0; weight_decay=0",
        "preprocessing": "float32; flatten; divide by 255 (no fitted statistics)",
        "objective": "mean cross-entropy + lambda * sum(abs(W) or W**2); bias unpenalized",
        "metric_timing": "after iteration updates; iteration 0 = initialization",
        "validation": "unpenalized cross-entropy and accuracy; no test data",
        "implementation_sha256": hashlib.sha256("\n".join(
            inspect.getsource(item) for item in (_candidates, _CandidateModels, _evaluate, run_experiment)
        ).encode()).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__,
        "device": "cpu", "dtype": "float32", "num_threads": num_threads,
        "datasets": {name: {"X_sha256": _array_hash(X), "y_sha256": _array_hash(y),
                              "shape": list(X.shape), "n_samples": len(y)}
                     for name, (X, y) in raw_datasets.items()},
    }
    experiment_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    run_dir = Path(output_dir) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError("Configuration mismatch; refusing to mix cached experiments.")
    _atomic_json(config_path, config)
    _atomic_json(run_dir / "request.json", {
        "target_iteration": max_iter, "checkpoint_every": checkpoint_every,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json(run_dir / "status.json", {
        "status": "incomplete", "target_iteration": max_iter, "experiment_id": experiment_id,
    })
    print(f"Experiment {experiment_id}: target={max_iter}; directory={run_dir}", flush=True)
    splits = {
        regime: list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
        for regime, (X, y) in raw_datasets.items()
    }
    split_rows = [{"regime": regime, "sample_index": int(i), "validation_fold": fold}
                  for regime, folds in splits.items()
                  for fold, (_, indices) in enumerate(folds, start=1) for i in indices]
    _atomic_csv(run_dir / "splits.csv", pd.DataFrame(split_rows))
    for regime, (X_raw, y_raw) in raw_datasets.items():
        for fold, (train_idx, val_idx) in enumerate(splits[regime], start=1):
            fold_dir = run_dir / regime / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            csv_path, state_path = fold_dir / "iterations.csv", fold_dir / "checkpoint.pt"
            models = _CandidateModels(int(np.prod(X_raw.shape[1:])), 10, candidates, seed + fold - 1)
            optimizer = torch.optim.SGD(models.parameters(), lr=lr)
            iteration, elapsed, rows = 0, 0.0, []
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                if state["experiment_id"] != experiment_id:
                    raise ValueError("Checkpoint belongs to a different experiment.")
                iteration, elapsed = state["iteration"], state["elapsed_seconds"]
                models.load_state_dict(state["model_state_dict"])
                optimizer.load_state_dict(state["optimizer_state_dict"])
                history = pd.read_csv(csv_path, float_precision="round_trip")
                history = history[history.iteration <= iteration]
                expected = (iteration + 1) * len(candidates)
                if len(history) != expected or history.duplicated(["iteration", "regularizer", "lambda"]).any():
                    raise ValueError(f"Missing/duplicate committed rows in {csv_path}")
                rows = history.to_dict("records")
                if iteration >= max_iter:
                    print(f"{regime} fold {fold}: cached through {iteration}", flush=True)
                    continue
            started = time.perf_counter()
            X_train = torch.from_numpy(np.asarray(X_raw[train_idx], dtype=np.float32).reshape(len(train_idx), -1) / 255)
            X_val = torch.from_numpy(np.asarray(X_raw[val_idx], dtype=np.float32).reshape(len(val_idx), -1) / 255)
            y_train = torch.as_tensor(y_raw[train_idx], dtype=torch.long)
            y_val = torch.as_tensor(y_raw[val_idx], dtype=torch.long)
            metrics = _evaluate(models, X_train, y_train, X_val, y_val, candidates)

            def record(update_norm, relative_update_norm):
                seconds = elapsed + time.perf_counter() - started
                for i, (name, strength) in enumerate(candidates):
                    rows.append({
                        "experiment_id": experiment_id, "regime": regime, "fold": fold,
                        "initialization_seed": seed + fold - 1, "regularizer": name,
                        "lambda": strength, "iteration": iteration, "lr": lr,
                        "n_train": len(train_idx), "n_val": len(val_idx),
                        **{key: float(values[i]) for key, values in metrics.items()},
                        "update_norm": float(update_norm[i]),
                        "relative_update_norm": float(relative_update_norm[i]),
                        "elapsed_seconds": seconds,
                    })

            def commit():
                # CSV first. If interrupted before checkpoint replacement, resume
                # trims the extra rows back to the previous checkpoint iteration.
                _atomic_csv(csv_path, pd.DataFrame(rows))
                temporary = state_path.with_suffix(".pt.tmp")
                torch.save({"experiment_id": experiment_id, "iteration": iteration,
                            "elapsed_seconds": rows[-1]["elapsed_seconds"],
                            "model_state_dict": models.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()}, temporary)
                os.replace(temporary, state_path)

            if not rows:
                record(np.zeros(len(candidates)), np.zeros(len(candidates)))
                commit()
            while iteration < max_iter:
                before_w, before_b = models.weight.detach().clone(), models.bias.detach().clone()
                optimizer.step()
                iteration += 1
                with torch.no_grad():
                    update = ((models.weight - before_w).square().sum((1, 2))
                              + (models.bias - before_b).square().sum(1)).sqrt()
                    parameter_norm = (before_w.square().sum((1, 2)) + before_b.square().sum(1)).sqrt()
                    relative_update = update / parameter_norm.clamp_min(1e-12)
                metrics = _evaluate(models, X_train, y_train, X_val, y_val, candidates)
                record(update.numpy(), relative_update.numpy())
                if iteration % checkpoint_every == 0 or iteration == max_iter:
                    commit()
                if iteration % 200 == 0 or iteration == max_iter:
                    baseline = float(metrics["val_accuracy"][0])
                    print(f"{regime} fold {fold}: {iteration}/{max_iter}; baseline val={baseline:.5f}", flush=True)
            del X_train, X_val, y_train, y_val, models, optimizer
    _atomic_csv(run_dir / "splits.csv", pd.DataFrame(split_rows))
    history = load_history(run_dir)
    _atomic_csv(run_dir / "iterations.csv", history)
    _atomic_json(run_dir / "status.json", {
        "status": "complete", "target_iteration": max_iter,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(history), "experiment_id": experiment_id,
    })
    return run_dir, history


def load_history(run_dir):
    """Read committed per-fold CSVs, including partially finished experiments."""
    frames = []
    for path in sorted(Path(run_dir).glob("*/fold_*/iterations.csv")):
        state_path = path.with_name("checkpoint.pt")
        if not state_path.exists():
            continue
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        frame = pd.read_csv(path, float_precision="round_trip")
        frames.append(frame[frame.iteration <= state["iteration"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(history, run_dir, *, budgets=(200, 500, 1000, 2000), window=100):
    """Aggregate folds at fixed budgets; diagnostic window is not early stopping."""
    baseline = history[history.regularizer == "Baseline"]
    expanded = pd.concat([
        history[history.regularizer != "Baseline"],
        baseline.assign(regularizer="L1"), baseline.assign(regularizer="L2"),
    ], ignore_index=True)
    metrics = ["train_ce", "train_objective", "val_ce", "train_accuracy", "val_accuracy",
               "gradient_norm", "relative_update_norm", "weight_l2"]
    aggregate = expanded.groupby(["regime", "regularizer", "lambda", "iteration"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        val_accuracy_std=("val_accuracy", "std"), val_ce_std=("val_ce", "std"),
        folds=("fold", "nunique"),
    )
    n_splits = json.loads((Path(run_dir) / "config.json").read_text())["n_splits"]
    aggregate = aggregate[aggregate.folds == n_splits].copy()
    aggregate["train_val_gap"] = aggregate.train_accuracy_mean - aggregate.val_accuracy_mean
    selection = aggregate[aggregate.iteration.isin(budgets)].sort_values(
        ["val_accuracy_mean", "lambda"], ascending=[False, True]
    ).drop_duplicates(["regime", "regularizer", "iteration"])
    tail_rows = []
    for keys, rows in expanded.groupby(["regime", "regularizer", "lambda", "fold"]):
        rows = rows.sort_values("iteration")
        last = rows.iloc[-1]
        earlier = rows[rows.iteration == last.iteration - window]
        if earlier.empty:
            continue
        first = earlier.iloc[0]
        tail = rows[rows.iteration > first.iteration]
        tail_rows.append(dict(zip(["regime", "regularizer", "lambda", "fold"], keys)) | {
            "last_iteration": int(last.iteration), "window": window,
            "objective_change": float(last.train_objective - first.train_objective),
            "objective_relative_change": float((last.train_objective - first.train_objective)
                                                / max(abs(first.train_objective), 1e-12)),
            "objective_tail_range": float(tail.train_objective.max() - tail.train_objective.min()),
            "val_ce_change": float(last.val_ce - first.val_ce),
            "val_accuracy_change": float(last.val_accuracy - first.val_accuracy),
            "gradient_norm_last": float(last.gradient_norm),
            "relative_update_norm_last": float(last.relative_update_norm),
        })
    diagnostics = pd.DataFrame(tail_rows)
    _atomic_csv(Path(run_dir) / "cv_by_iteration.csv", aggregate)
    _atomic_csv(Path(run_dir) / "selected_by_budget.csv", selection)
    _atomic_csv(Path(run_dir) / "convergence_diagnostics.csv", diagnostics)
    return aggregate, selection, diagnostics


def plot_convergence(aggregate, run_dir):
    """Mean curves across folds, fixed color per lambda and reference at 200."""
    import matplotlib.pyplot as plt

    figure_dir = Path(run_dir) / "figures"
    figure_dir.mkdir(exist_ok=True)
    figures = []
    for regime in aggregate.regime.unique():
        fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, constrained_layout=True)
        for col, name in enumerate(("L1", "L2")):
            subset = aggregate[(aggregate.regime == regime) & (aggregate.regularizer == name)]
            for strength, rows in subset.groupby("lambda"):
                rows = rows.sort_values("iteration")
                label = f"lambda={strength:g}"
                line, = axes[0, col].plot(rows.iteration, rows.train_objective_mean, label=label)
                axes[1, col].plot(rows.iteration, rows.train_ce_mean, color=line.get_color())
                axes[1, col].plot(rows.iteration, rows.val_ce_mean, "--", color=line.get_color())
                axes[2, col].plot(rows.iteration, rows.val_accuracy_mean, color=line.get_color())
            axes[0, col].set_title(name)
            axes[0, col].set_ylabel("Training objective (CE + penalty)")
            axes[1, col].set_ylabel("CE: train solid; validation dashed")
            axes[0, col].set_yscale("log")
            axes[1, col].set_yscale("log")
            axes[2, col].set_ylabel("Validation accuracy")
            axes[2, col].set_xlabel("SGD updates")
            axes[0, col].legend(fontsize=8, ncol=2)
            for ax in axes[:, col]:
                ax.axvline(200, color="gray", linestyle=":", linewidth=1)
                ax.grid(alpha=0.2)
        fig.suptitle(f"{regime}: convergence on corrupted training-data folds (means)")
        fig.savefig(figure_dir / f"convergence_{regime.lower()}.png", dpi=130, bbox_inches="tight")
        figures.append(fig)
    return figures


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    X = np.load(root / "data_2s2026/train/train_images.npy")
    y = np.load(root / "data_2s2026/train/train_labels.npy")
    # Exact sampling order from notebook cells 6/7; isolated RNG avoids dependence
    # on exploratory plots that subsequently consume NumPy's global RNG.
    rng = np.random.RandomState(42)
    partial_idx = rng.choice(len(y), size=10_000, replace=False)
    scarce_idx = rng.choice(len(y), size=1_000, replace=False)
    raw_datasets = {"Scarce": (X[scarce_idx], y[scarce_idx]),
                    "Partial": (X[partial_idx], y[partial_idx]), "Full": (X, y)}
    run_dir, history = run_experiment(raw_datasets, root / "resultados_convergencia",
                                      max_iter=args.max_iter)
    aggregate, selected, _ = summarize(history, run_dir,
                                       budgets=sorted({200, 500, 1000, 2000, args.max_iter}))
    figures = plot_convergence(aggregate, run_dir)
    import matplotlib.pyplot as plt
    for fig in figures:
        plt.close(fig)
    print(selected[["regime", "regularizer", "iteration", "lambda",
                    "val_accuracy_mean", "val_accuracy_std"]].sort_values(
                        ["regime", "regularizer", "iteration"]).to_string(index=False), flush=True)
    print("Completed:", run_dir, flush=True)


if __name__ == "__main__":
    main()
