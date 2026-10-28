"""
Per-slice 2D metrics as defined in Thesis A section 3.4: DSC and IoU are
averaged over slices where the target is present; a detection on a slice
with no target counts as a false-positive (FP) slice.
"""

import numpy as np


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    inter = np.logical_and(pred, gt).sum()
    p, g = pred.sum(), gt.sum()
    union = p + g - inter
    dice = 2 * inter / (p + g) if (p + g) > 0 else 1.0
    iou = inter / union if union > 0 else 1.0
    return float(dice), float(iou)


def summarise(rows):
    """rows: dicts with gt_present, detected, dice, iou. Returns one summary dict."""
    gt_rows = [r for r in rows if r["gt_present"]]
    return {
        "gt_slices": len(gt_rows),
        "dsc": float(np.mean([r["dice"] for r in gt_rows])) if gt_rows else float("nan"),
        "iou": float(np.mean([r["iou"] for r in gt_rows])) if gt_rows else float("nan"),
        "fp_slices": sum(1 for r in rows if not r["gt_present"] and r["detected"]),
        "missed_gt_slices": sum(1 for r in gt_rows if not r["detected"]),
    }
