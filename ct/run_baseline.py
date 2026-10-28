"""
Thesis A baseline reproduction on MSD Task07 (Tables 4.1 and 4.2).

For each model x patient x slice x prompt, one forward pass; every
(threshold, NMS IoU) setting is scored from that same output.

Outputs under --out:
  per_slice.csv   one row per model/patient/slice/prompt/setting
  summary.csv     per model/patient/prompt/setting: gt_slices, dsc, iou, fp_slices, missed_gt_slices
  overlays/       PNGs for --overlay-prompts at --overlay-setting, on slices with GT or a detection

Example:
  python -m ct.run_baseline --cases pancreas_004 pancreas_005 pancreas_006 \
      --models sam3 medsam3 --out results/baseline_windowed
"""

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ct.data import iter_slices, load_case, target_mask
from ct.metrics import dice_iou, summarise
from ct.model import TextPromptModel, postprocess

# prompt -> target label set (Thesis A section 3.4)
PROMPTS = {
    "pancreas": "pancreas",
    "pancreatic tumour": "tumour",
    "pancreatic tumor": "tumour",
    "pancreatic cancer": "tumour",
    "pancreatic mass": "tumour",
    "tumor": "tumour",
}
SETTINGS = [(0.5, 0.5), (0.6, 0.3)]
# infer_sam = Thesis A scoring; presence = stock SAM3 / ConceptBank scoring (for Phase 2 comparisons)
SCORINGS = ["infer_sam", "presence"]


def overlay(grey, gt, pred, path):
    """GT green outline-ish fill, prediction red, overlap yellow."""
    rgb = np.stack([grey] * 3, -1).astype(np.float32)
    colour = np.zeros_like(rgb)
    colour[gt] = (0, 255, 0)
    colour[pred] = (255, 0, 0)
    colour[gt & pred] = (255, 255, 0)
    m = (gt | pred)[..., None]
    rgb = np.where(m, 0.55 * rgb + 0.45 * colour, rgb)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/workspace/data/Task07_Pancreas"))
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["sam3", "medsam3"], choices=["sam3", "medsam3"])
    ap.add_argument("--lora-weights", default="/workspace/models/medsam3_lora/best_lora_weights.pt")
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    ap.add_argument("--no-window", action="store_true", help="min-max scale raw HU instead of W350/L40")
    ap.add_argument("--overlay-prompts", nargs="*", default=["pancreas", "pancreatic mass"])
    ap.add_argument("--overlay-setting", default="0.5/0.5")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ov_thr, ov_nms = map(float, args.overlay_setting.split("/"))
    rows = []
    for model_name in args.models:
        model = TextPromptModel(lora_weights=args.lora_weights if model_name == "medsam3" else None)
        text = {p: model.encode_text(p) for p in args.prompts}
        for case in args.cases:
            img, lbl = load_case(args.data, case)
            t0 = time.time()
            for z, grey, lbl_z in iter_slices(img, lbl, windowed=not args.no_window):
                image_state = model.encode_image(grey)
                for prompt in args.prompts:
                    target = PROMPTS[prompt]
                    gt = target_mask(lbl_z, target)
                    raw = model.ground(image_state, text[prompt])
                    for scoring, (thr, nms_iou) in product(SCORINGS, SETTINGS):
                        pred, n_det, max_score = postprocess(raw, thr, nms_iou, scoring)
                        dice, iou = dice_iou(pred, gt)
                        rows.append(dict(model=model_name, case=case, slice=z, prompt=prompt,
                                         target=target, scoring=scoring, thr=thr, nms=nms_iou,
                                         gt_present=bool(gt.any()), gt_px=int(gt.sum()),
                                         detected=n_det > 0, n_det=n_det, max_score=max_score,
                                         pred_px=int(pred.sum()), dice=dice, iou=iou))
                        if (prompt in args.overlay_prompts and scoring == "infer_sam"
                                and (thr, nms_iou) == (ov_thr, ov_nms) and (gt.any() or n_det > 0)):
                            d = args.out / "overlays" / model_name / case / prompt.replace(" ", "_")
                            d.mkdir(parents=True, exist_ok=True)
                            overlay(grey, gt, pred, d / f"slice{z:03d}.png")
            print(f"{model_name} {case}: {img.shape[2]} slices in {time.time() - t0:.0f}s", flush=True)
        del model

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_slice.csv", index=False)
    keys = ["model", "case", "prompt", "target", "scoring", "thr", "nms"]
    summary = pd.DataFrame([{**dict(zip(keys, k)), **summarise(g.to_dict("records"))}
                            for k, g in df.groupby(keys)])
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
