---
title: RetinAI Compare API
emoji: 👁️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# RetinAI Multi-Model API

FastAPI backend for the RetinAI multi-model retinal disease screening frontend.

Models loaded from [janetkupt-22/retinai-multiclass](https://huggingface.co/janetkupt-22/retinai-multiclass).

**POST /predict** — upload a retinal image, get predictions + Grad-CAM from all 3 models.
