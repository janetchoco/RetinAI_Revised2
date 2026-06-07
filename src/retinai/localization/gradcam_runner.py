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


def _find_original_path(preprocessed_path: str) -> str | None:
    """Locate the original source image in train/ or test/ for display purposes."""
    p = Path(preprocessed_path)
    class_name = p.parent.name
    filename = p.name
    if filename.startswith("train_"):
        orig = Path("train") / class_name / filename[6:]
    elif filename.startswith("test_"):
        orig = Path("test") / class_name / filename[5:]
    else:
        return None
    return str(orig) if orig.exists() else None


def _letterbox(img: Image.Image, size: int) -> np.ndarray:
    """Resize preserving aspect ratio, pad with black to size×size."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return (np.array(canvas) / 255.0).astype(np.float32)


def _load_image(image_path: str, image_size: int = 224):
    # Tensor: preprocessed 224×224 image — consistent with training
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    with Image.open(image_path).convert("RGB") as img:
        tensor = tfm(img).unsqueeze(0)

    # Visualization: use original image (correct aspect ratio) if available
    vis_path = _find_original_path(image_path) or image_path
    with Image.open(vis_path).convert("RGB") as vis_img:
        orig_size = vis_img.size
        rgb_img = _letterbox(vis_img, image_size)

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
            true_label = Path(img_path).parent.name
            pred_label = class_names[pred_class]
            conf = float(probs[pred_class])
            # Save at fixed square size — avoids distortion from non-square orig_size
            Image.fromarray(visualization).resize((512, 512), Image.LANCZOS).save(
                out / f"{stem}_true{true_label}_pred{pred_label}_conf{conf:.2f}.png"
            )

            with open(out / f"{stem}_scores.txt", "w", encoding="utf-8") as f:
                f.write(f"true_class: {true_label}\n")
                f.write(f"predicted:  {pred_label} ({conf:.4f})\n")
                f.write(f"correct:    {true_label == pred_label}\n\n")
                for cls, prob in zip(class_names, probs):
                    f.write(f"{cls}: {prob:.4f}\n")

    print(f"Grad-CAM maps saved to {output_dir}")
