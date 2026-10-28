"""
MSD Task07 NIfTI loading and CT windowing, following Thesis A section 3.2:
window centre 40 HU / width 350 HU, rescale to uint8, transpose each axial
slice and its label slice identically.
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple

import nibabel as nib
import numpy as np

WINDOW_LEVEL = 40
WINDOW_WIDTH = 350


def load_case(data_dir: Path, case: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (image HU volume, label volume), both (X, Y, Z)."""
    img = nib.load(str(data_dir / "imagesTr" / f"{case}.nii.gz")).get_fdata(dtype=np.float32)
    lbl = np.asarray(nib.load(str(data_dir / "labelsTr" / f"{case}.nii.gz")).dataobj).astype(np.uint8)
    assert img.shape == lbl.shape, (case, img.shape, lbl.shape)
    return img, lbl


def window_slice(hu: np.ndarray, level: Optional[float] = WINDOW_LEVEL,
                 width: Optional[float] = WINDOW_WIDTH) -> np.ndarray:
    """HU slice -> uint8. With level/width None, min-max scale the raw slice (no windowing)."""
    if level is None or width is None:
        lo, hi = float(hu.min()), float(hu.max())
    else:
        lo, hi = level - width / 2, level + width / 2
    out = (np.clip(hu, lo, hi) - lo) / max(hi - lo, 1e-6)
    return (out * 255).round().astype(np.uint8)


def iter_slices(img: np.ndarray, lbl: np.ndarray, windowed: bool = True
                ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield (z, uint8 greyscale slice (H, W), label slice (H, W)), transposed as in Thesis A."""
    for z in range(img.shape[2]):
        hu = img[:, :, z].T
        grey = window_slice(hu) if windowed else window_slice(hu, None, None)
        yield z, grey, lbl[:, :, z].T


def target_mask(lbl_slice: np.ndarray, target: str) -> np.ndarray:
    """'pancreas' -> labels 1 and 2 combined; 'tumour' -> label 2 only."""
    if target == "pancreas":
        return lbl_slice > 0
    if target == "tumour":
        return lbl_slice == 2
    raise ValueError(target)
