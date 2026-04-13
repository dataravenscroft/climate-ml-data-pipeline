from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset


def unwrap_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


@dataclass
class ForecastMetricContext:
    variable_names: list[str]
    means: torch.Tensor | None = None
    stds: torch.Tensor | None = None
    lat_weights: torch.Tensor | None = None
    climatology: torch.Tensor | None = None

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.means is None or self.stds is None:
            return tensor
        return tensor * self.stds + self.means


def build_metric_context(loader, device: torch.device, variable_names: list[str]) -> ForecastMetricContext:
    base_dataset = unwrap_dataset(loader.dataset)
    means = None
    stds = None
    lat_weights = None
    climatology = None

    if hasattr(base_dataset, "variables") and hasattr(base_dataset, "means") and hasattr(base_dataset, "stds"):
        means = torch.tensor(
            [base_dataset.means[v] for v in base_dataset.variables],
            dtype=torch.float32,
            device=device,
        ).view(1, -1, 1, 1)
        stds = torch.tensor(
            [base_dataset.stds[v] for v in base_dataset.variables],
            dtype=torch.float32,
            device=device,
        ).view(1, -1, 1, 1)

    if hasattr(base_dataset, "ds") and "lat" in base_dataset.ds.coords:
        lat = np.asarray(base_dataset.ds["lat"].values, dtype=np.float32)
        weights = np.cos(np.deg2rad(lat))
        weights = weights / weights.mean()
        lat_weights = torch.tensor(weights, dtype=torch.float32, device=device).view(1, 1, -1, 1)

    if hasattr(base_dataset, "ds") and hasattr(base_dataset, "variables"):
        climatology_fields = []
        for variable in base_dataset.variables:
            field = base_dataset.ds[variable].mean("time").compute().values.astype(np.float32)
            climatology_fields.append(field)
        climatology = torch.tensor(
            np.stack(climatology_fields, axis=0),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

    return ForecastMetricContext(
        variable_names=variable_names,
        means=means,
        stds=stds,
        lat_weights=lat_weights,
        climatology=climatology,
    )


class ForecastMetricAccumulator:
    def __init__(self, variable_names: list[str], device: torch.device) -> None:
        n_vars = len(variable_names)
        self.variable_names = variable_names
        self.device = device
        self.sse = torch.zeros(n_vars, device=device)
        self.count = torch.zeros(1, device=device)
        self.weighted_sse = torch.zeros(n_vars, device=device)
        self.weighted_count = torch.zeros(1, device=device)
        self.acc_num = torch.zeros(n_vars, device=device)
        self.acc_pred = torch.zeros(n_vars, device=device)
        self.acc_true = torch.zeros(n_vars, device=device)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lat_weights: torch.Tensor | None = None,
        climatology: torch.Tensor | None = None,
    ) -> None:
        err = pred - target
        self.sse += (err ** 2).sum(dim=(0, 2, 3))
        self.count += target.shape[0] * target.shape[2] * target.shape[3]

        if lat_weights is not None:
            weighted_err = (err ** 2) * lat_weights
            self.weighted_sse += weighted_err.sum(dim=(0, 2, 3))
            self.weighted_count += lat_weights.sum() * target.shape[0] * target.shape[3]
        else:
            self.weighted_sse += (err ** 2).sum(dim=(0, 2, 3))
            self.weighted_count += target.shape[0] * target.shape[2] * target.shape[3]

        if climatology is None:
            clim = target.mean(dim=0, keepdim=True)
        else:
            clim = climatology

        pred_anom = pred - clim
        true_anom = target - clim

        self.acc_num += (pred_anom * true_anom).sum(dim=(0, 2, 3))
        self.acc_pred += (pred_anom ** 2).sum(dim=(0, 2, 3))
        self.acc_true += (true_anom ** 2).sum(dim=(0, 2, 3))

    def reduce(self) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        for tensor in [
            self.sse,
            self.count,
            self.weighted_sse,
            self.weighted_count,
            self.acc_num,
            self.acc_pred,
            self.acc_true,
        ]:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

    def compute(self) -> dict[str, list[float]]:
        rmse = torch.sqrt(self.sse / self.count.clamp_min(1.0))
        weighted_rmse = torch.sqrt(self.weighted_sse / self.weighted_count.clamp_min(1.0))
        acc = self.acc_num / torch.sqrt(self.acc_pred.clamp_min(1e-12) * self.acc_true.clamp_min(1e-12))
        return {
            "rmse": [float(v) for v in rmse.cpu()],
            "latitude_weighted_rmse": [float(v) for v in weighted_rmse.cpu()],
            "anomaly_correlation": [float(v) for v in acc.cpu()],
        }
