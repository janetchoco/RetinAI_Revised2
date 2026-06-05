from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from retinai.config import load_config
from retinai.localization.gradcam_runner import _get_target_layers, _reshape_transform_deit
from retinai.models.factory import create_model

MODEL_NAMES = ["efficientnet_b0", "deit_tiny", "resnet18"]
MODEL_LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "deit_tiny": "DeiT-Tiny",
    "resnet18": "ResNet18",
}

app = FastAPI(title="RetinAI Multi-Model API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cfg = None
_models: dict = {}


@app.on_event("startup")
def _load_models() -> None:
    global _cfg, _models
    _cfg = load_config("configs/base.yaml")
    for name in MODEL_NAMES:
        ckpt_path = Path(_cfg.runtime.artifact_root) / "final" / name / "best.pt"
        if not ckpt_path.exists():
            print(f"WARN: checkpoint not found at {ckpt_path} — {name} will be skipped")
            continue
        model = create_model(model_name=name, num_classes=len(_cfg.data.class_names), pretrained=False)
        state = torch.load(str(ckpt_path), map_location=_cfg.runtime.device)
        model.load_state_dict(state["model_state"])
        model.eval().to(_cfg.runtime.device)
        _models[name] = model
        print(f"Loaded {name} from {ckpt_path}")


def _make_gradcam(model, model_name: str, tensor: torch.Tensor, rgb_np: np.ndarray, pred_class: int) -> str | None:
    try:
        layers = _get_target_layers(model, model_name)
        reshape = _reshape_transform_deit if model_name == "deit_tiny" else None
        with GradCAM(model=model, target_layers=layers, reshape_transform=reshape) as cam:
            gray_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        overlay = show_cam_on_image(rgb_np, gray_cam, use_rgb=True)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        print(f"GradCAM failed for {model_name}: {exc}")
        return None


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _models:
        raise HTTPException(status_code=503, detail="No models loaded — check that artifacts/final/{model}/best.pt exist")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    size = _cfg.data.image_size

    rgb_np = np.array(img.resize((size, size))).astype(np.float32) / 255.0
    tensor = T.Compose([T.Resize((size, size)), T.ToTensor()])(img).unsqueeze(0).to(_cfg.runtime.device)

    results: dict = {}
    for name, model in _models.items():
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        results[name] = {
            "diagnosis": _cfg.data.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": probs.tolist(),
            "gradcam": _make_gradcam(model, name, tensor, rgb_np, pred_idx),
            "label": MODEL_LABELS[name],
        }

    return {"models": results, "class_names": _cfg.data.class_names}
