import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def log(msg):
    print(msg, flush=True)

# Distinct colours (RGBA) cycled across detections
COLOURS = [
    (255,  60,  60, 120),   # red
    ( 60, 180,  60, 120),   # green
    ( 60, 120, 255, 120),   # blue
    (255, 200,   0, 120),   # yellow
    (200,   0, 255, 120),   # purple
    (  0, 220, 220, 120),   # cyan
    (255, 140,   0, 120),   # orange
]

def draw_results(image: Image.Image, masks, boxes, scores, prompt: str, out_path: str):
    """Overlay masks + bounding boxes + scores on the image and save."""
    vis = image.convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        colour = COLOURS[i % len(COLOURS)]
        border_colour = colour[:3] + (255,)

        # --- mask overlay ---
        # mask shape: (1, H, W) bool tensor → numpy bool array
        mask_np = mask[0].astype(bool)
        mask_img = Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize(image.size, Image.NEAREST)
        colour_layer = Image.new("RGBA", image.size, colour)
        overlay.paste(colour_layer, mask=mask_img)

        # --- bounding box ---
        x1, y1, x2, y2 = box.astype(int)
        draw.rectangle([x1, y1, x2, y2], outline=border_colour, width=3)

        # --- score label ---
        label = f"{prompt} {score:.2f}"
        text_x, text_y = x1 + 4, max(y1 - 20, 4)
        draw.rectangle([text_x - 2, text_y - 2, text_x + len(label) * 7 + 2, text_y + 16],
                       fill=(0, 0, 0, 160))
        draw.text((text_x, text_y), label, fill=(255, 255, 255, 255))

    vis = Image.alpha_composite(vis, overlay).convert("RGB")
    vis.save(out_path)
    log(f"Saved visualisation → {out_path}")
    vis.show()   # opens in default image viewer


# ── model ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Using device: {device}")

log("Loading SAM3 model...")
model = build_sam3_image_model(device=device)
processor = Sam3Processor(model, device=device)
log("Model loaded.")

# ── image ────────────────────────────────────────────────────────────────────
image_path = "assets/images/truck.jpg"
prompt     = "truck"

log(f"Loading image: {image_path}")
image = Image.open(image_path).convert("RGB")
log(f"Image size: {image.size}")

# ── inference ─────────────────────────────────────────────────────────────────
log(f'Running inference with prompt: "{prompt}"...')
state  = processor.set_image(image)
output = processor.set_text_prompt(prompt=prompt, state=state)

masks  = output["masks"].cpu().numpy()   # (N, 1, H, W) bool
boxes  = output["boxes"].cpu().numpy()   # (N, 4)  xyxy pixels
scores = output["scores"].cpu().numpy()  # (N,)

log(f"Detections: {len(masks)}")
for i, score in enumerate(scores):
    log(f"  [{i}] score={score:.4f}  box={boxes[i].astype(int).tolist()}")

if len(masks) == 0:
    log("No detections above confidence threshold — try a lower threshold or a different prompt.")
else:
    draw_results(image, masks, boxes, scores, prompt, out_path="output_mask.png")
