# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .decode_heads import BNHead, DPTHead
from ...losses import SigLoss, GradientLoss, L1Loss


def _make_dinov2_linear_depth_bins_head(
    *,
    embed_dim: int,
    cls_token: bool,
    layers: int,
    min_depth: float,
    max_depth: float,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")

    if layers == 1:
        in_index = [0]
    else:
        assert layers == 4
        in_index = [0, 1, 2, 3]

    return BNHead(
        classify=True,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
        upsample=4,
        in_channels=[embed_dim] * len(in_index),
        in_index=in_index,
        input_transform="resize_concat",
        channels=embed_dim * len(in_index) * (2 if cls_token else 1),
        use_cls_token=cls_token,
        align_corners=False,
        min_depth=min_depth,
        max_depth=max_depth,
        loss_decode=nn.ModuleList([
            SigLoss(valid_mask=True, loss_weight=1.0, warm_up=True, loss_name="loss_depth"),
            GradientLoss(valid_mask=True, loss_weight=0.5, loss_name="loss_grad"),
        ])
    )

def _make_dinov2_linear_depth_head(
    *,
    embed_dim: int,
    cls_token: bool,
    layers: int,
    min_depth: float,
    max_depth: float,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")

    if layers == 1:
        in_index = [0]
    else:
        assert layers == 4
        in_index = [0, 1, 2, 3]

    return BNHead(
        upsample=4,
        in_channels=[embed_dim] * len(in_index),
        in_index=in_index,
        input_transform="resize_concat",
        channels=embed_dim * len(in_index) * (2 if cls_token else 1),
        use_cls_token=cls_token,
        align_corners=False,
        min_depth=min_depth,
        max_depth=max_depth,
        loss_decode=nn.ModuleList([
            # SigLoss(valid_mask=True, loss_weight=1.0, warm_up=True, loss_name="loss_depth"),
            # GradientLoss(valid_mask=True, loss_weight=0.5, loss_name="loss_grad"),
            L1Loss(),
        ])
    )

def _make_dinov2_dpt_depth_head(*, embed_dim: int, patch_size: int, layers: int, min_depth: float, max_depth: float):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")

    if layers == 1:
        in_index = [0]
    else:
        assert layers == 4
        in_index = [0, 1, 2, 3]
        
    return DPTHead(
        in_channels=[embed_dim]  * len(in_index),
        channels=256,
        embed_dims=embed_dim,
        patch_size=patch_size,
        post_process_channels=[embed_dim // 2 ** (3 - i) for i in range(len(in_index))],
        readout_type="project",
        min_depth=min_depth,
        max_depth=max_depth,
        loss_decode=nn.ModuleList([
            # SigLoss(valid_mask=True, loss_weight=1.0, warm_up=True, loss_name="loss_depth"),
            # GradientLoss(valid_mask=True, loss_weight=0.5, loss_name="loss_grad"),
            L1Loss(),
        ])
    )