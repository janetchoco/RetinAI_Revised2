"""Self-contained FastAPI inference server for HF Spaces.

Loads checkpoints from HF model repo janetkupt-22/retinai-multiclass.
Does NOT overwrite any previous model repo — uses a separate repo name.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import timm
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── constants ──────────────────────────────────────────────────────────────
HF_REPO = "janetkupt-22/ReinAI_Multiclass"
CLASS_NAMES = ["normal", "amd", "dr"]
IMAGE_SIZE = 224
DEVICE = "cpu"

MODEL_NAMES = ["efficientnet_b0", "deit_tiny", "resnet18"]
TIMM_NAMES = {
    "efficientnet_b0": "tf_efficientnet_b0",
    "deit_tiny": "deit_tiny_patch16_224",
    "resnet18": "resnet18",
}
LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "deit_tiny": "DeiT-Tiny",
    "resnet18": "ResNet18",
}

# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(title="RetinAI Multi-Model API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_models: dict = {}


def _target_layers(model, name: str) -> list:
    if name == "efficientnet_b0":
        return [model.conv_head]
    if name == "resnet18":
        return [model.layer4[-1]]
    if name == "deit_tiny":
        return [model.blocks[-1].norm1]
    return []


def _reshape_deit(tensor, h: int = 14, w: int = 14):
    n_skip = tensor.size(1) - h * w
    result = tensor[:, n_skip:, :].reshape(tensor.size(0), h, w, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


@app.on_event("startup")
def load_models() -> None:
    for name in MODEL_NAMES:
        try:
            ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=f"{name}/best.pt")
            model = timm.create_model(TIMM_NAMES[name], pretrained=False, num_classes=len(CLASS_NAMES))
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state["model_state"])
            model.eval()
            _models[name] = model
            print(f"Loaded {name}")
        except Exception as exc:
            print(f"WARN: could not load {name}: {exc}")


def _make_gradcam(model, name: str, tensor: torch.Tensor, rgb_np: np.ndarray, pred_class: int, orig_size: tuple) -> str | None:
    try:
        layers = _target_layers(model, name)
        if not layers:
            return None
        reshape = _reshape_deit if name == "deit_tiny" else None
        with GradCAM(model=model, target_layers=layers, reshape_transform=reshape) as cam:
            gray = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        overlay = show_cam_on_image(rgb_np, gray, use_rgb=True)
        buf = io.BytesIO()
        Image.fromarray(overlay).resize(orig_size, Image.LANCZOS).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        print(f"GradCAM error ({name}): {exc}")
        return None


@app.get("/health")
def health():
    return {"status": "ok", "models": list(_models.keys())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _models:
        raise HTTPException(503, "No models loaded — check HF repo has all 3 best.pt files")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_size = img.size  # (width, height) — preserved for Grad-CAM output
    rgb_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    tensor = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])(img).unsqueeze(0)

    results: dict = {}
    for name, model in _models.items():
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        results[name] = {
            "diagnosis": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": probs.tolist(),
            "gradcam": _make_gradcam(model, name, tensor, rgb_np, pred_idx, orig_size),
            "label": LABELS[name],
        }

    return {"models": results, "class_names": CLASS_NAMES}
