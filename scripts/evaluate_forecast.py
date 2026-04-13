"""Evaluate a ConvLSTM checkpoint and save forecast skill plots.

Usage:
    python scripts/evaluate_forecast.py --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from pipeline.models.convlstm import ConvLSTMForecast
from pipeline.training.data_setup import build_datasets
from pipeline.training.metrics import ForecastMetricAccumulator, build_metric_context


def make_bar_plot(variable_names: list[str], values: list[float], title: str, ylabel: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(variable_names, values, color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_sample_plot(variable_name: str, target: torch.Tensor, pred: torch.Tensor, path: str) -> None:
    error = pred - target
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, field, title in zip(
        axes,
        [target, pred, error],
        [f"{variable_name} truth", f"{variable_name} forecast", f"{variable_name} error"],
    ):
        im = ax.imshow(field.cpu().numpy(), cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="viz/evaluation")
    parser.add_argument("--data_mode", choices=["synthetic", "real"], default="real")
    parser.add_argument("--zarr_path", type=str, default="data/era5_subset.zarr")
    parser.add_argument("--variables", nargs="+", default=None)
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--synthetic_vars", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    _, val_dataset, variable_names = build_datasets(
        data_mode=args.data_mode,
        seq_len=args.seq_len,
        synthetic_vars=args.synthetic_vars,
        n_train=args.n_train,
        n_val=args.n_val,
        zarr_path=args.zarr_path,
        variables=args.variables,
        val_fraction=args.val_fraction,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metric_context = build_metric_context(val_loader, device, variable_names)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ConvLSTMForecast(
        in_channels=len(variable_names),
        hidden_channels=64,
        out_channels=len(variable_names),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accumulator = ForecastMetricAccumulator(variable_names, device=device)
    sample_target = None
    sample_pred = None

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            pred_denorm = metric_context.denormalize(pred)
            y_denorm = metric_context.denormalize(y)
            accumulator.update(
                pred_denorm,
                y_denorm,
                lat_weights=metric_context.lat_weights,
                climatology=metric_context.climatology,
            )
            if sample_target is None:
                sample_target = y_denorm[0]
                sample_pred = pred_denorm[0]

    metrics = accumulator.compute()
    metrics["variables"] = variable_names
    metrics["checkpoint"] = args.checkpoint

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    make_bar_plot(
        variable_names,
        metrics["rmse"],
        "Validation RMSE by Variable",
        "RMSE",
        os.path.join(args.output_dir, "rmse_by_variable.png"),
    )
    make_bar_plot(
        variable_names,
        metrics["anomaly_correlation"],
        "Validation Anomaly Correlation by Variable",
        "ACC",
        os.path.join(args.output_dir, "acc_by_variable.png"),
    )

    if sample_target is not None and sample_pred is not None:
        make_sample_plot(
            variable_names[0],
            sample_target[0],
            sample_pred[0],
            os.path.join(args.output_dir, "sample_forecast_comparison.png"),
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
