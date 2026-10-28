"""
SAM3 / MedSAM3 text-prompt inference on one image, reproducing the Thesis A
inference path: MedSAM3's infer_sam.py (Joey-S-Liu/MedSAM3). Both models go
through the same code; MedSAM3 differs only by the injected LoRA weights.

Scoring follows infer_sam.py exactly: score = max sigmoid(pred_logits) (no
presence-score product), keep score > threshold, torchvision NMS on boxes,
mask = sigmoid(low-res mask) > 0.5, bilinear-resized, > 0.5.

The forward pass runs once per (image, prompt); `postprocess` then applies any
number of (threshold, NMS IoU) settings to the same raw output.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image as PILImage
from torchvision.ops import nms

MEDSAM3_DIR = Path(os.environ.get("MEDSAM3_DIR", "/workspace/MedSAM3"))
# MedSAM3 bundles its own sam3 package; it must shadow any other installed copy.
sys.path.insert(0, str(MEDSAM3_DIR))

import sam3  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.geometry_encoders import Prompt  # noqa: E402
from sam3.model.utils.misc import copy_data_to_device  # noqa: E402
from sam3.train.data.collator import collate_fn_api  # noqa: E402
from sam3.train.data.sam3_image_dataset import (  # noqa: E402
    Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (  # noqa: E402
    ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI,
)
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights  # noqa: E402

assert Path(sam3.__file__).resolve().is_relative_to(MEDSAM3_DIR.resolve()), sam3.__file__


class TextPromptModel:
    def __init__(self, lora_weights: Optional[str] = None,
                 lora_config: str = str(MEDSAM3_DIR / "configs/full_lora_config.yaml"),
                 resolution: int = 1008, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = build_sam3_image_model(
            device=self.device.type, compile=False, load_from_HF=False,
            checkpoint_path=os.environ.get("SAM3_CKPT", "/workspace/models/sam3/sam3.pt"),
            bpe_path=str(MEDSAM3_DIR / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
            eval_mode=True,
        )
        if lora_weights:
            cfg = yaml.safe_load(open(lora_config))["lora"]
            self.model = apply_lora_to_model(self.model, LoRAConfig(
                rank=cfg["rank"], alpha=cfg["alpha"], dropout=0.0,
                target_modules=cfg["target_modules"],
                apply_to_vision_encoder=cfg["apply_to_vision_encoder"],
                apply_to_text_encoder=cfg["apply_to_text_encoder"],
                apply_to_geometry_encoder=cfg["apply_to_geometry_encoder"],
                apply_to_detr_encoder=cfg["apply_to_detr_encoder"],
                apply_to_detr_decoder=cfg["apply_to_detr_decoder"],
                apply_to_mask_decoder=cfg["apply_to_mask_decoder"],
            ))
            load_lora_weights(self.model, lora_weights)
        self.model.to(self.device).eval()
        self.transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=resolution, max_size=resolution, square=True,
                            consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _datapoint(self, pil: PILImage.Image, prompt: str) -> Datapoint:
        w, h = pil.size
        query = FindQueryLoaded(
            query_text=prompt, image_id=0, object_ids_output=[], is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=0, original_image_id=0, original_category_id=1,
                original_size=[w, h], object_id=0, frame_index=0,
            ),
        )
        return Datapoint(find_queries=[query],
                         images=[SAMImage(data=pil, objects=[], size=[h, w])])

    @torch.no_grad()
    def encode_image(self, grey: np.ndarray) -> dict:
        """Image features for one greyscale uint8 slice, reusable across prompts."""
        pil = PILImage.fromarray(grey).convert("RGB")
        # the query text is irrelevant here; text features are supplied per prompt in ground()
        batch = collate_fn_api([self.transform(self._datapoint(pil, "_"))], dict_key="input")["input"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)
        backbone_out = {"img_batch_all_stages": batch.img_batch}
        backbone_out.update(self.model.backbone.forward_image(batch.img_batch))
        return {"backbone_out": backbone_out, "batch": batch, "size": (pil.size[1], pil.size[0])}

    @torch.no_grad()
    def encode_text(self, prompt: str) -> dict:
        """language_features / language_mask / language_embeds for one prompt."""
        return self.model.backbone.forward_text([prompt], device=self.device)

    @torch.no_grad()
    def ground(self, image_state: dict, text_outputs: dict) -> dict:
        """Equivalent of Sam3Image.forward with precomputed image and text features
        (num_interactive_steps_val is 0 in eval, so a single grounding step)."""
        assert self.model.num_interactive_steps_val == 0
        batch = image_state["batch"]
        backbone_out = dict(image_state["backbone_out"])
        backbone_out.update(text_outputs)
        find_input = batch.find_inputs[0]
        geometric_prompt = Prompt(box_embeddings=find_input.input_boxes,
                                  box_mask=find_input.input_boxes_mask,
                                  box_labels=find_input.input_boxes_label)
        out = self.model.forward_grounding(backbone_out=backbone_out, find_input=find_input,
                                           find_target=batch.find_targets[0],
                                           geometric_prompt=geometric_prompt)
        H, W = image_state["size"]
        cx, cy, bw, bh = out["pred_boxes"][0].unbind(-1)
        boxes = torch.stack([(cx - bw / 2) * W, (cy - bh / 2) * H,
                             (cx + bw / 2) * W, (cy + bh / 2) * H], dim=-1)
        return {"scores": out["pred_logits"].sigmoid()[0].max(dim=-1)[0],
                "presence": out["presence_logit_dec"].sigmoid().view(-1)[0],
                "boxes": boxes, "mask_logits": out["pred_masks"][0], "size": (H, W)}

    def forward(self, grey: np.ndarray, prompt: str) -> dict:
        return self.ground(self.encode_image(grey), self.encode_text(prompt))

    @torch.no_grad()
    def forward_reference(self, grey: np.ndarray, prompt: str) -> dict:
        """Unsplit path exactly as infer_sam.py (model(batch)); used to check ground() matches."""
        pil = PILImage.fromarray(grey).convert("RGB")
        batch = collate_fn_api([self.transform(self._datapoint(pil, prompt))], dict_key="input")["input"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)
        out = self.model(batch)[-1]
        return {"scores": out["pred_logits"].sigmoid()[0].max(dim=-1)[0],
                "mask_logits": out["pred_masks"][0]}


def postprocess(raw: dict, threshold: float, nms_iou: float, scoring: str = "infer_sam"):
    """Returns (union mask bool (H, W), n_detections, max_score) for one setting.

    scoring="infer_sam": MedSAM3 infer_sam.py (Thesis A). scoring="presence":
    stock Sam3Processor / ConceptBank, score multiplied by the presence score.
    """
    H, W = raw["size"]
    scores = raw["scores"] * raw["presence"] if scoring == "presence" else raw["scores"]
    raw = {**raw, "scores": scores}
    keep = raw["scores"] > threshold
    if keep.sum() == 0:
        return np.zeros((H, W), bool), 0, 0.0
    scores, boxes = raw["scores"][keep], raw["boxes"][keep]
    keep_nms = nms(boxes, scores, nms_iou)
    masks_small = raw["mask_logits"][keep][keep_nms].sigmoid() > 0.5
    masks = F.interpolate(masks_small.unsqueeze(0).float(), size=(H, W),
                          mode="bilinear", align_corners=False).squeeze(0) > 0.5
    return masks.any(0).cpu().numpy(), int(len(keep_nms)), float(scores[keep_nms].max())
