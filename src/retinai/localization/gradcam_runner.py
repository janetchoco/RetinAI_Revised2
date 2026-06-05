from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from retinai.models.factory import create_model


def _get_target_layers(model, model_name: str) -> list:
    if model_name == "efficientnet_b0":
        return [model.conv_head]
    if model_name == "resnet18":
        return [model.layer4[-1]]
    if model_name == "deit_tiny":
        return [model.blocks[-1].norm1]
    raise ValueError(f"No Grad-CAM target layer defined for model_name={model_name}")


def _reshape_transform_deit(tensor, height: int = 14, width: int = 14):
    # Older timm: CLS + distillation token (skip 2). Newer timm: CLS only (skip 1).
    # Compute n_skip dynamically so this works with both versions.
    n_skip = tensor.size(1) - height * width
    result = tensor[:, n_skip:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def _load_image(image_path: str, image_size: int = 224):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    with Image.open(image_path).convert("RGB") as img:
        orig_size = img.size  # (width, height) — preserved for saving
        tensor = tfm(img).unsqueeze(0)
        rgb_img = (np.array(img.resize((image_size, image_size))) / 255.0).astype(np.float32)
    return tensor, rgb_img, orig_size


def generate_gradcam_maps(
    checkpoint_path: str,
    model_name: str,
    image_paths: list[str],
    output_dir: str,
    num_classes: int = 3,
    class_names: list[str] | None = None,
    image_size: int = 224,
    device: str = "cpu",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    target_layers = _get_target_layers(model, model_name)
    reshape_transform = _reshape_transform_deit if model_name == "deit_tiny" else None

    with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        for img_path in image_paths:
            tensor, rgb_img, orig_size = _load_image(img_path, image_size)
            tensor = tensor.to(device)

            with torch.no_grad():
                logits = model(tensor)
                pred_class = int(torch.argmax(logits, dim=1).item())
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            grayscale_cam = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(pred_class)],
            )[0]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            stem = Path(img_path).stem
            pred_label = class_names[pred_class]
            conf = float(probs[pred_class])
            # Resize back to original dimensions so landscape images are not squished
            Image.fromarray(visualization).resize(orig_size, Image.LANCZOS).save(
                out / f"{stem}_pred{pred_label}_conf{conf:.2f}.png"
            )

            with open(out / f"{stem}_scores.txt", "w", encoding="utf-8") as f:
                for cls, prob in zip(class_names, probs):
                    f.write(f"{cls}: {prob:.4f}\n")

    print(f"Grad-CAM maps saved to {output_dir}")
