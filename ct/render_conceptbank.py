"""
Qualitative figures for the ConceptBank run: re-runs the saved banks (banks/<model>.pt
from run_conceptbank.py) on selected patients and writes one side-by-side panel per
GT-present slice, one tile per text condition (GT green, prediction red, overlap yellow,
per-slice DSC in the label).

Outputs under --run/overlays:
  <model>/<case>/<target>/sliceNNN.png   all GT-present slices (tiles at --tile px)
  highlights/<model>_<target>/...png     --n-highlights slices per model/target with the
                                         largest |DSC(cb_full) - DSC(base)|, full resolution
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ct.data import iter_slices, load_case, target_mask
from ct.metrics import dice_iou
from ct.model import TextPromptModel, postprocess
from ct.run_baseline import overlay

BASE = {"pancreas": "base_pancreas", "tumour": "base_pancreatic mass"}


def tile(grey, gt, pred, label, size):
    tmp = Path("/tmp/_tile.png")
    overlay(grey, gt, pred, tmp)
    im = Image.open(tmp).convert("RGB").resize((size, size), Image.BILINEAR)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(12, size // 22))
    except TypeError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, size, size // 13], fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(255, 255, 255), font=font)
    return im


def panel(tiles):
    w, h = tiles[0].size
    out = Image.new("RGB", (w * len(tiles), h))
    for i, t in enumerate(tiles):
        out.paste(t, (i * w, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/workspace/data/Task07_Pancreas"))
    ap.add_argument("--run", type=Path, default=Path("results/conceptbank_K10_J3"))
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["sam3", "medsam3"])
    ap.add_argument("--lora-weights", default="/workspace/models/medsam3_lora/best_lora_weights.pt")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--n-highlights", type=int, default=8)
    args = ap.parse_args()
    out = args.run / "overlays"

    for model_name in args.models:
        model = TextPromptModel(lora_weights=args.lora_weights if model_name == "medsam3" else None)
        bank = torch.load(args.run / "banks" / f"{model_name}.pt")
        conds = {t: {c: {k: v.to(model.device) for k, v in txt.items()} for c, txt in d.items()}
                 for t, d in bank.items()}
        order = {t: [BASE[t]] + [c for c in d if c.startswith("base_") and c != BASE[t]]
                 + [c for c in ["top1", "cb_lite", "cb_full"] if c in d] for t, d in conds.items()}
        cands = {t: [] for t in conds}  # (|delta|, case, z, frames) for highlights
        for case in args.cases:
            img, lbl = load_case(args.data, case)
            n = 0
            for z, grey, lbl_z in iter_slices(img, lbl):
                targets = [t for t in conds if target_mask(lbl_z, t).any()]
                if not targets:
                    continue
                st = model.encode_image(grey)
                for target in targets:
                    gt = target_mask(lbl_z, target)
                    frames, dsc = [], {}
                    for cond in order[target]:
                        pred, _, _ = postprocess(model.ground(st, conds[target][cond]), args.thr, args.nms)
                        dsc[cond] = dice_iou(pred, gt)[0]
                        frames.append((pred, f"{cond.replace('base_', '')}  DSC {dsc[cond]:.2f}"))
                    d = out / model_name / case / target
                    d.mkdir(parents=True, exist_ok=True)
                    panel([tile(grey, gt, p, lab, args.tile) for p, lab in frames]).save(d / f"slice{z:03d}.png")
                    delta = dsc["cb_full"] - dsc[BASE[target]]
                    cands[target].append((abs(delta), delta, case, z, grey, gt, frames))
                    cands[target] = sorted(cands[target], key=lambda c: -c[0])[:args.n_highlights]
                n += 1
            print(f"{model_name} {case}: {n} slices rendered", flush=True)
        for target, cs in cands.items():
            d = out / "highlights" / f"{model_name}_{target}"
            d.mkdir(parents=True, exist_ok=True)
            for _, delta, case, z, grey, gt, frames in cs:
                panel([tile(grey, gt, p, lab, grey.shape[0]) for p, lab in frames]).save(
                    d / f"{case}_slice{z:03d}_delta{delta:+.2f}.png")
        del model
        torch.cuda.empty_cache()
    print("RENDER_DONE", flush=True)


if __name__ == "__main__":
    main()
