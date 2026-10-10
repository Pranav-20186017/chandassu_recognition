"""Weighted CE accumulation shared by production training and timing checks."""

import torch
from torch.nn import functional as F


def backward_window(model, batches, weights, weights_cpu, device, autocast):
    """Normalize once over an update, including its final partial window.

    Targets and normalization stay on CPU. The loss stays on the accelerator
    until progress reporting. Four-line poem grouping is supplied by the loader.
    """
    denominator = sum(float(weights_cpu[y].sum()) for _, y in batches)
    if denominator <= 0:
        raise ValueError("An optimizer window must have positive target weight")
    total = torch.zeros((), device=device)
    for ids, y in batches:
        with autocast():
            loss = F.cross_entropy(
                model(ids.to(device)).float(), y.to(device),
                weight=weights, reduction="sum",
            )
        (loss / denominator).backward()
        total += loss.detach()
    return total, denominator
