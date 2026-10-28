"""
ConceptBank experiment (Thesis B conditions C1-C4 plus matched baselines).

For each model in --models: build one concept per class (pancreas, tumour) on the
support split in "lite" and "full" mode, then evaluate on --eval-split with every
text condition run on the same image features:
  base_<prompt>   Thesis A prompts unchanged ("pancreas"; "pancreatic tumour", "pancreatic mass")
  top1            single best candidate on support (prompt selection only)
  cb_lite         Stage III only, K random positive support slices
  cb_full         Stages I-III
Bank building uses the evaluation setting (--thr/--nms/--scoring) so selection and
evaluation agree; evaluation still records both settings and both scorings.

Outputs under --out: banks/<model>.pt, bank_log.json, per_slice.csv, summary.csv
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import pandas as pd
import torch

from ct.conceptbank import build_concept, collect_support
from ct.data import iter_slices, load_case, target_mask
from ct.metrics import dice_iou, summarise
from ct.model import TextPromptModel, postprocess

BASE_PROMPTS = {"pancreas": ["pancreas"], "tumour": ["pancreatic tumour", "pancreatic mass"]}
SETTINGS = [(0.5, 0.5), (0.6, 0.3)]
SCORINGS = ["infer_sam", "presence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/workspace/data/Task07_Pancreas"))
    ap.add_argument("--splits", type=Path, default=Path("results/splits.json"))
    ap.add_argument("--eval-split", nargs="+", default=["eval_core", "eval_heldout"])
    ap.add_argument("--models", nargs="+", default=["sam3", "medsam3"])
    ap.add_argument("--lora-weights", default="/workspace/models/medsam3_lora/best_lora_weights.pt")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--J", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--scoring", default="infer_sam")
    ap.add_argument("--support-max-per-case", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    splits = json.loads(args.splits.read_text())
    eval_cases = [(s, c) for s in args.eval_split for c in splits[s]]
    (args.out / "banks").mkdir(parents=True, exist_ok=True)
    bank_log, rows = {}, []

    for model_name in args.models:
        model = TextPromptModel(lora_weights=args.lora_weights if model_name == "medsam3" else None)
        conds = {}  # target -> {condition: text outputs}
        for target in ["pancreas", "tumour"]:
            t0 = time.time()
            support = collect_support(args.data, splits["support"], target,
                                      max_per_case=args.support_max_per_case)
            conds[target] = {f"base_{p}": model.encode_text(p) for p in BASE_PROMPTS[target]}
            for mode in ["lite", "full"]:
                c = build_concept(model, support, target, mode=mode, K=args.K, J=args.J, tau=args.tau,
                                  thr=args.thr, nms_iou=args.nms, scoring=args.scoring)
                conds[target][f"cb_{mode}"] = c["text"]
                if mode == "full":
                    conds[target]["top1"] = c["top1_text"]
                bank_log[f"{model_name}/{target}/{mode}"] = c["log"]
                print(f"{model_name} {target} {mode}: top1={c['log']['top1']} fused={c['log']['fused']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            del support
        torch.save({t: {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in d.items()}
                    for t, d in conds.items()}, args.out / "banks" / f"{model_name}.pt")
        (args.out / "bank_log.json").write_text(json.dumps(bank_log, indent=1))

        for split, case in eval_cases:
            t0 = time.time()
            img, lbl = load_case(args.data, case)
            for z, grey, lbl_z in iter_slices(img, lbl):
                st = model.encode_image(grey)
                for target, cmap in conds.items():
                    gt = target_mask(lbl_z, target)
                    for cond, text in cmap.items():
                        raw = model.ground(st, text)
                        for scoring, (thr, nms_iou) in product(SCORINGS, SETTINGS):
                            pred, n_det, max_score = postprocess(raw, thr, nms_iou, scoring)
                            dice, iou = dice_iou(pred, gt)
                            rows.append(dict(model=model_name, split=split, case=case, slice=z,
                                             target=target, condition=cond, scoring=scoring,
                                             thr=thr, nms=nms_iou, gt_present=bool(gt.any()),
                                             detected=n_det > 0, n_det=n_det, max_score=max_score,
                                             dice=dice, iou=iou))
            print(f"{model_name} eval {case}: {img.shape[2]} slices in {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_slice.csv", index=False)
    keys = ["model", "split", "case", "target", "condition", "scoring", "thr", "nms"]
    summary = pd.DataFrame([{**dict(zip(keys, k)), **summarise(g.to_dict("records"))}
                            for k, g in df.groupby(keys)])
    summary.to_csv(args.out / "summary.csv", index=False)
    means = (summary.groupby(["model", "split", "target", "condition", "scoring", "thr", "nms"])
             [["dsc", "iou", "fp_slices"]].mean().reset_index())
    means.to_csv(args.out / "summary_means.csv", index=False)
    print(means[(means.scoring == args.scoring) & (means.thr == args.thr)].to_string(index=False))


if __name__ == "__main__":
    main()
