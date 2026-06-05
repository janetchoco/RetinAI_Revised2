from __future__ import annotations

import time

import torch


def estimate_flops(model, input_size: int = 224) -> int | None:
    try:
        from thop import profile  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return int(flops)


def timed_call(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start

