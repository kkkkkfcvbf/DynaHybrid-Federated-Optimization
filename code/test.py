"""Model evaluation."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        pred = logits.argmax(dim=1)
        total_loss += float(loss.item())
        total_correct += int((pred == y).sum().item())
        total_seen += int(y.size(0))
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


