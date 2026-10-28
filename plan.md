# Thesis B plan: SAM3 + ConceptBank on pancreatic CT

Status as of 2026-10-28. Source of truth for what was done: `../Submitted Thesis A.pdf` (submitted 06/06/2026).
The Macquarie RTX 5090 server is gone; all GPU work moves to a Runpod Pod.

---

## 1. What Thesis A recorded (the baseline we must reproduce)

| Item | Recorded value |
|---|---|
| Dataset | MSD Task07 Pancreas, training split (labels: 0 bg, 1 pancreas, 2 tumour) |
| Patients | `pancreas_004`, `pancreas_005`, `pancreas_006`, all axial slices |
| Preprocessing | CT window centre 40 / width 350 HU (clip [-135, 215]), rescale to uint8, 512x512 greyscale PNG; slice **and** label transposed |
| Model input | resize to 1008x1008, to tensor [0,1], normalise mean = std = 0.5 |
| Prompting | text only (no box/point) |
| Models | Vanilla SAM3 (`facebook/sam3`); MedSAM3 (repo `Joey-S-Liu/MedSAM3`, LoRA `lal-Joey/MedSAM3` v1, r16, alpha 32, dropout 0.1, via `infer_sam.py --config configs/full_lora_config.yaml`) |
| Settings | threshold / NMS IoU = 0.5/0.5 and 0.6/0.3 |
| Pancreas target | prompt `"pancreas"` vs labels 1 and 2 combined |
| Tumour target | prompts `"pancreatic tumour"`, `"pancreatic tumor"`, `"pancreatic cancer"`, `"pancreatic mass"`, `"tumor"` at 0.5/0.5, vs label 2 |
| Metric | per-slice 2D DSC and IoU averaged over slices where the GT structure is present; detections on GT-empty slices counted as FP slices |

### Numbers to match

Table 4.1, pancreas, best setting per model and patient:

| Patient | Model | Thr/NMS | GT slices | IoU | DSC | FP slices |
|---|---|---|---|---|---|---|
| 004 | SAM3 | 0.6/0.3 | 35 | 0.552 | 0.688 | 14 |
| 004 | MedSAM3 | 0.6/0.3 | 35 | 0.555 | 0.671 | 6 |
| 005 | SAM3 | 0.5/0.5 | 40 | 0.587 | 0.716 | 10 |
| 005 | MedSAM3 | 0.6/0.3 | 40 | 0.465 | 0.563 | 5 |
| 006 | SAM3 | 0.5/0.5 | 25 | 0.455 | 0.572 | 8 |
| 006 | MedSAM3 | 0.5/0.5 | 25 | 0.236 | 0.326 | 42 |
| **Mean** | SAM3 / MedSAM3 | | | 0.537 / 0.418 | **0.665 / 0.520** | |

Table 4.2, tumour prompt sweep at 0.5/0.5, mean DSC over 004 to 006:

| Prompt | SAM3 | MedSAM3 |
|---|---|---|
| pancreatic tumour | 0.187 | 0.408 |
| pancreatic tumor | 0.183 | 0.411 |
| pancreatic cancer | 0.134 | 0.392 |
| pancreatic mass | **0.586** | 0.442 |
| tumor | 0.029 | 0.384 |

FP slices, SAM3 with "pancreatic mass": 76 / 69 / 66 against 14 / 10 / 10 GT tumour slices.

### Inconsistencies in Thesis A to fix in Thesis B

1. The abstract, section 5.1 point 1, section 2.7.1 and section 5.2.4 say vanilla SAM3 "produces no meaningful segmentation" and that "pancreas" and "pancreatic tumor" give identical output. Chapter 4 contradicts both (SAM3 pancreas DSC 0.665; the prompts differ in Table 4.2). These are leftovers from the first, unwindowed run.
2. The usable threshold range is quoted as 0.50 to 0.70 (abstract) and 0.55 to 0.70 (sections 2.7.5 and 5.1), but only 0.5 and 0.6 were tested.
3. Section 5.2.1 calls the evaluation "qualitative"; Chapter 4 is quantitative.
4. nnU-Net reference "0.69 pancreas / 0.21 tumour": check against Antonelli et al. 2022. The wiki note says nnU-Net tumour DSC was 0.52 and 0.21 was the median across teams.
5. Section 5.2.2 says "run on the test set". The 139 MSD test volumes have no public labels, so evaluation must use held-out **training** volumes.

---

## 2. Experiment design for Thesis B

Central comparison (Thesis A section 5.2.2): **MedSAM3 vs MedSAM3 + ConceptBank**, plus **SAM3 vs SAM3 + ConceptBank**.

### Data splits (fixed before any calibration, saved to `results/splits.json`)

| Split | Volumes | Use |
|---|---|---|
| `eval_core` | 004, 005, 006 | direct comparison with Thesis A tables |
| `support` | 15 volumes, seeded random from the other 278, stratified by tumour size (from `../dataset_analysis.csv`) | ConceptBank prototypes, mining, candidate scoring; positive slices only |
| `eval_heldout` | 20 more volumes, disjoint from both | main held-out result, so conclusions do not rest on 3 patients |

No patient appears in more than one split.

### Conditions

| ID | Condition | Purpose |
|---|---|---|
| B0 | SAM3, no CT window (raw HU min-max) | documents the "first run failed" claim with data |
| B1 | SAM3, W350/L40 | reproduce Thesis A |
| B2 | MedSAM3, W350/L40 | reproduce Thesis A |
| C1 | SAM3 + ConceptBank-lite (Stage III only: Dice-scored prompt candidates, softmax-fused text embeddings) | quick, low-risk result |
| C2 | MedSAM3 + ConceptBank-lite | central comparison, lite version |
| C3 | SAM3 + full ConceptBank (Stages I to III, with K and J tuned on support only) | full method |
| C4 | MedSAM3 + full ConceptBank | central comparison, full version |

Each condition runs on targets {pancreas vs labels 1+2, tumour vs label 2} at both settings {0.5/0.5, 0.6/0.3}.
Report DSC, IoU and FP slices per patient, plus the mean over `eval_core` and over `eval_heldout`.

ConceptBank candidate prompts: the 5 Thesis A tumour prompts plus "pancreas" variants, expanded by an LLM under the paper's rule of **label-faithful rewrites only** (no new visual concepts). The candidate list is frozen in `configs/conceptbank_candidates.yaml` before scoring.

---

## 3. Runpod Pod setup

All Runpod steps are done by Claude through the Runpod MCP tools, which call the same REST API as `runpodctl`. File transfer and the remote shell use plain SSH from this laptop. `runpodctl` is not installed locally and is not needed.

### Live reads on 2026-10-28 (redo these immediately before creating anything)

- Account: 0 Pods, 0 network volumes, 0 SSH keys registered.
- **NVIDIA A40, 48 GB: $0.59/hr Secure Cloud. Stock HIGH in CA-MTL-1 and EU-SE-1** (from `list-gpu-types`, product POD, CUDA 12.8 or newer).
- CA-MTL-1 supports network volumes (`HIGH_PERFORMANCE`); EU-SE-1 does not. **Use CA-MTL-1.**
- Alternatives:
  - RTX 5090, 32 GB (same as the old server): $1.19/hr Secure, stock LOW (EU-CZ-1, EU-RO-1, EUR-NO-1).
  - L40S, 48 GB: $1.09/hr Secure, stock MEDIUM.
- SAM3 is about 848M parameters, so inference fits comfortably in 48 GB, with room to cache ConceptBank features.

### Steps

1. **SSH key (laptop).** `~/.ssh/id_ed25519.pub` does not exist yet.
   - Run `ssh-keygen -t ed25519 -C gavinaiworkspace` on the laptop.
   - Register the public key on the account (`update-ssh-keys`).
2. **Network volume.**
   - 60 GB `HIGH_PERFORMANCE` volume in **CA-MTL-1**, name `sam3-thesisb`. Quote the price before creating it.
   - Expected contents: Task07 tar about 11 GB, extracted about 11 GB, SAM3 checkpoint 3.3 GB, MedSAM3 LoRA weights, outputs.
   - The volume bills from creation until it is deleted, whatever the Pod does.
3. **Pod.**
   - Secure Cloud, 1x A40, data center CA-MTL-1, volume mounted at `/workspace`.
   - `gpu.minCudaVersion = "12.8"`.
   - Container disk 30 GB.
   - Ports `22/tcp` and `8888/http` (Jupyter), `startSsh: true`.
   - Image: a pinned `runpod/pytorch` tag (CUDA 12.8, Python 3.12), resolved from the `list-templates` read at creation time, never from memory.
   - Environment: `HF_TOKEN` stored as a Runpod secret, never written to disk or committed.
4. **Verify.**
   - `get-pod` shows the Pod RUNNING, with `runtime.ports` mapped and the `ssh.direct` address present.
   - Then on the Pod: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`.
5. **Pod bootstrap**, run over SSH, everything under `/workspace` so it survives a stop:
   ```bash
   cd /workspace
   git clone https://github.com/gavinaiworkspace/sam3.git
   git clone https://github.com/Joey-S-Liu/MedSAM3.git
   git clone https://github.com/pgsmall/ConceptBank.git
   python -m venv venv && source venv/bin/activate
   pip install -e sam3 nibabel pandas scipy
   pip install -e MedSAM3            # check its requirements first
   huggingface-cli download facebook/sam3 --local-dir models/sam3                                 # gated, uses HF_TOKEN
   huggingface-cli download lal-Joey/MedSAM3_v1 --local-dir models/medsam3_lora                   # confirm exact repo id on HF
   mkdir -p data && cd data
   wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar
   tar -xf Task07_Pancreas.tar && rm Task07_Pancreas.tar
   ```
   Download data and weights on the Pod, which is faster than uploading 3.3 GB from the laptop.
6. **Cost control.**
   - **Stop** the Pod whenever it is idle. GPU billing ends; `/workspace` on the network volume is kept.
   - **Terminate** the Pod only at the end. Delete the volume only once results are committed to git.
   - Budget: about 10 to 20 GPU-hours for all phases, roughly $6 to $12 of A40 time plus the volume.

---

## 4. Work phases

### Phase 0: environment (about half a day)
- [ ] Create the SSH key; create the volume and Pod; bootstrap (section 3)
- [ ] Commit the pending local fix `sam3/model/position_encoding.py` (CPU fallback) and decide whether to keep `sam3/assets/images/test_image.jpg`
- [ ] Sanity check: run the existing truck demo on the Pod (reproduces Thesis A Fig 4.1)

### Phase 1: reproduce Thesis A (about 1 day)
- [ ] `sam3/ct/preprocess.py`: NIfTI to windowed, transposed uint8 slices plus aligned label slices (optional `--no-window` for B0)
- [ ] `sam3/ct/run_prompts.py`: model in {sam3, medsam3}, prompts, thr/NMS. Writes `results/<run>/overlays/*.png`, `per_slice.csv` (patient, slice, prompt, n_det, max_score, dsc, iou, gt_present, fp), `summary.csv`
- [ ] `sam3/ct/metrics.py`: DSC/IoU/FP-slice logic exactly as in Thesis A section 3.4
- [ ] Run B1 and B2 on `eval_core`; compare with Tables 4.1 and 4.2. Agreement within about ±0.03 DSC counts as reproduced; otherwise find the cause before going further
- [ ] Run B0
- [ ] Remove the PNG/CSV ignore rules for `results/` from `.gitignore`, then commit the results

### Phase 2: ConceptBank (about 2 to 3 days)
- [ ] Read `ConceptBank/` and map its hooks (text-encoder bypass, prototype pooling) onto the SAM3 fork and onto the MedSAM3 LoRA model
- [ ] Build the splits (`results/splits.json`)
- [ ] C1 and C2, lite version: score the frozen candidate list on `support`, fuse the top J, evaluate on `eval_core` and `eval_heldout`
- [ ] C3 and C4, full version: add Stage I prototypes and Stage II top-K mining; sweep K and J on `support` only
- [ ] Rerun B1 and B2 on `eval_heldout` so every condition has the same evaluation set

### Phase 3: write-up
- [ ] Tables in Thesis A format (4.1, 4.2, 4.3 and the FP section) for B0 to C4
- [ ] Fix the inconsistencies listed in section 1
- [ ] Overlays and CSVs committed; this plan updated with run IDs

---

## 5. Open items for Gavin
- Deadline, which decides whether C3 and C4 (full ConceptBank) are core results or an extension.
- Does the old MedSAM3 `infer_sam.py` / `full_lora_config.yaml` exist anywhere besides the dead server? If not, take it from the public repo.
- Hugging Face token with SAM3 gated access, to be stored as a Runpod secret.
- Go-ahead for the first billable step (volume and Pod).
