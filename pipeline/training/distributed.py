import os

import torch
import torch.distributed as dist

from pipeline.training.metrics import ForecastMetricAccumulator


def setup_distributed(backend: str | None = None):
    """Initialize the distributed process group.

    torchrun sets LOCAL_RANK, RANK, WORLD_SIZE as environment variables.
    NCCL backend uses AllReduce to average gradients across all ranks after
    each backward pass. init_process_group is a collective — all processes
    must call it before any can proceed.
    """
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return local_rank, backend


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, rank):
    model.train()
    # DistributedSampler must be re-shuffled each epoch so ranks get
    # different data orderings across epochs
    loader.sampler.set_epoch(epoch)

    total_loss = torch.tensor(0.0, device=device)
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        # DDP automatically AllReduces gradients here before optimizer.step()
        optimizer.step()

        total_loss += loss.detach()
        n_batches += 1

    # Aggregate loss across all ranks for accurate reporting
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / (n_batches * dist.get_world_size())).item()
    return avg_loss


def validate(model, loader, criterion, device, metric_context=None):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    n_batches = 0
    metrics = None
    if metric_context is not None:
        metrics = ForecastMetricAccumulator(metric_context.variable_names, device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            total_loss += criterion(pred, y).detach()
            n_batches += 1
            if metrics is not None:
                pred_for_metrics = metric_context.denormalize(pred)
                y_for_metrics = metric_context.denormalize(y)
                metrics.update(
                    pred_for_metrics,
                    y_for_metrics,
                    lat_weights=metric_context.lat_weights,
                    climatology=metric_context.climatology,
                )

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / (n_batches * dist.get_world_size())).item()
    result = {"loss": avg_loss}
    if metrics is not None:
        metrics.reduce()
        result.update(metrics.compute())
    return result


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model weights from DDP-wrapped model.

    Always save model.module.state_dict() (the underlying nn.Module, not the
    DDP wrapper) so the checkpoint is portable — loadable without DDP wrapping
    at inference time.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": loss,
        },
        path,
    )
