"""
Fixed, disjoint patient splits for Thesis B (written once to results/splits.json):
  eval_core     pancreas_004/005/006 (Thesis A patients)
  support       15 volumes, stratified by tumour_pct tercile (5 per tercile)
  eval_heldout  20 volumes, stratified the same way (7/7/6)
Source metadata: dataset_analysis.csv (281 MSD Task07 training volumes).
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=Path, default=Path("/workspace/dataset_analysis.csv"))
ap.add_argument("--out", type=Path, default=Path("results/splits.json"))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

df = pd.read_csv(args.csv)
df["case"] = df.filename.str.replace(".nii.gz", "", regex=False)
core = ["pancreas_004", "pancreas_005", "pancreas_006"]
pool = df[~df.case.isin(core)].copy()
pool["tercile"] = pd.qcut(pool.tumour_pct, 3, labels=["small", "medium", "large"])
rng = random.Random(args.seed)
support, heldout = [], []
for t, n_held in zip(["small", "medium", "large"], [7, 7, 6]):
    cases = sorted(pool[pool.tercile == t].case)
    rng.shuffle(cases)
    support += cases[:5]
    heldout += cases[5:5 + n_held]
assert not (set(support) & set(heldout)) and not (set(core) & (set(support) | set(heldout)))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({"seed": args.seed, "eval_core": core, "support": sorted(support),
                                "eval_heldout": sorted(heldout)}, indent=2))
print(args.out.read_text())
