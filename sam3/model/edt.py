# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Euclidean distance transform (EDT) — Triton kernel with CPU fallback for Windows."""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        for q in range(1, length):
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)


def _edt_cpu_fallback(data: torch.Tensor) -> torch.Tensor:
    """CPU fallback for EDT using scipy (triton is Linux/CUDA only)."""
    try:
        from scipy.ndimage import distance_transform_edt
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "scipy is required for the CPU EDT fallback on Windows. "
            "Install it with: pip install scipy"
        )
    B = data.shape[0]
    results = []
    arr = data.cpu().bool().numpy()
    for b in range(B):
        # distance_transform_edt measures distance to nearest zero pixel;
        # we want distance to nearest zero in the foreground (True) mask.
        dt = distance_transform_edt(arr[b])
        results.append(torch.from_numpy(dt.astype(np.float32)))
    out = torch.stack(results, dim=0).to(dtype=torch.float32, device=data.device)
    return out


def edt_triton(data: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
        Equivalent to a batched cv2.distanceTransform(input, cv2.DIST_L2, 0).
    """
    assert data.dim() == 3

    if not _TRITON_AVAILABLE or not data.is_cuda:
        return _edt_cpu_fallback(data)

    B, H, W = data.shape
    data = data.contiguous()

    output = torch.where(data, 1e18, 0.0)
    assert output.is_contiguous()

    parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
    parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, H)
    edt_kernel[grid](
        output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=True,
    )

    parabola_loc.zero_()
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, W)
    edt_kernel[grid](
        output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=False,
    )
    return output.sqrt()
