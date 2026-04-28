# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch
import torch.nn.functional as F


def addmm_act(activation, linear, mat1):
    """Fused linear + activation. Falls back to standard ops on non-Ampere GPUs.

    The original implementation forced BFloat16 to use a fused CUDA kernel
    (aten._addmm_activation) that is only efficient on Ampere+ GPUs. On Pascal
    and older hardware BFloat16 has no native support, causing dtype mismatches.
    This version respects the input dtype and uses standard PyTorch ops instead.
    """
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    x = F.linear(mat1, linear.weight, linear.bias)
    if activation in [F.relu, torch.nn.ReLU]:
        return F.relu(x)
    if activation in [F.gelu, torch.nn.GELU]:
        return F.gelu(x)
    raise ValueError(f"Unexpected activation {activation}")
