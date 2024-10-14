# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .decode_heads import BNHead
from ...losses import L1Loss
  

def _make_dinov2_linear_normal_head(
    *,
    embed_dim: int,
    cls_token: bool,
    layers: int,
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
        in_channels=[embed_dim] * len(in_index),
        in_index=in_index,
        input_transform="resize_concat",
        channels=embed_dim * len(in_index) * (2 if cls_token else 1),
        use_cls_token=cls_token,
        dropout_ratio=0,
        align_corners=False,
        loss_decode=nn.ModuleList([
            L1Loss(),
        ]),
        ignore_index=0,
    )