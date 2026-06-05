from __future__ import annotations

import timm
import torch.nn as nn


SUPPORTED_MODELS = {
    "efficientnet_b0": "tf_efficientnet_b0",
    "deit_tiny": "deit_tiny_patch16_224",
    "resnet18": "resnet18",
}


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name={model_name}.")
    timm_name = SUPPORTED_MODELS[model_name]
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

