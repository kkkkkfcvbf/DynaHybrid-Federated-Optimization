"""Construct the three unpretrained paper models with task-specific heads."""

import torch.nn as nn
from torchvision import models


def _build(constructor, num_classes: int) -> nn.Module:
    try:
        return constructor(weights=None, num_classes=num_classes)
    except TypeError:  # torchvision < 0.13
        return constructor(pretrained=False, num_classes=num_classes)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "alexnet":
        return _build(models.alexnet, num_classes)
    if model_name == "resnet18":
        return _build(models.resnet18, num_classes)
    if model_name == "googlenet":
        # Auxiliary heads return a named tuple during training and are not part
        # of the common single-logit interface used by all four FL algorithms.
        try:
            return models.googlenet(
                weights=None,
                num_classes=num_classes,
                aux_logits=False,
                init_weights=False,
            )
        except TypeError:  # older torchvision
            return models.googlenet(
                pretrained=False,
                num_classes=num_classes,
                aux_logits=False,
            )
    raise ValueError(
        f"Unsupported model {model_name!r}; choose alexnet, resnet18, or googlenet."
    )
