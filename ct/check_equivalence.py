"""
Sanity check: TextPromptModel.forward (split image/text/grounding path) must
match forward_reference (infer_sam.py's model(batch) path), for SAM3 and MedSAM3.
Also reproduces Thesis A Fig 4.1 (truck demo) as results/sanity/truck_<model>.png.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ct.model import TextPromptModel, postprocess

img = np.array(Image.open("assets/images/truck.jpg").convert("L"))
out = Path("results/sanity")
out.mkdir(parents=True, exist_ok=True)
ok = True
for name, lora in [("sam3", None), ("medsam3", "/workspace/models/medsam3_lora/best_lora_weights.pt")]:
    m = TextPromptModel(lora_weights=lora)
    a, b = m.forward(img, "truck"), m.forward_reference(img, "truck")
    ds = (a["scores"] - b["scores"]).abs().max().item()
    dm = (a["mask_logits"] - b["mask_logits"]).abs().max().item()
    mask, n, s = postprocess(a, 0.5, 0.5)
    print(f"{name}: max|dscore|={ds:.2e} max|dmask|={dm:.2e} detections={n} max_score={s:.3f}")
    ok &= ds < 1e-3 and dm < 1e-2
    rgb = np.stack([img] * 3, -1).astype(float)
    rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array([255, 0, 0])
    Image.fromarray(rgb.astype(np.uint8)).save(out / f"truck_{name}.png")
    del m
    torch.cuda.empty_cache()
print("EQUIVALENCE_OK" if ok else "EQUIVALENCE_FAILED")
sys.exit(0 if ok else 1)
