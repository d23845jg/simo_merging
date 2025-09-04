# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from ...losses import CrossEntropyLoss
from .decode_heads import BNHead, DPTHead


def _make_dinov2_linear_seg_head(
    *,
    embed_dim: int,
    cls_token: bool,
    layers: int,
    num_classes: int,
    loss_name="loss_seg",
    ignore_index=-1,
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
        num_classes=num_classes,
        align_corners=False,
        loss_decode=nn.ModuleList(
            [
                CrossEntropyLoss(
                    use_sigmoid=False,
                    loss_weight=1.0,
                    loss_name=loss_name,
                ),
            ]
        ),
        ignore_index=ignore_index,
    )


def _make_dinov2_dpt_seg_head(
    *,
    embed_dim: int,
    patch_size: int,
    cls_token: bool,
    layers: int,
    num_classes: int,
    loss_name="loss_seg",
    ignore_index=-1,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")

    if layers == 1:
        in_index = [0]
    else:
        assert layers == 4
        in_index = [0, 1, 2, 3]

    return DPTHead(
        in_channels=[embed_dim] * len(in_index),
        channels=256,
        embed_dims=embed_dim,
        patch_size=patch_size,
        post_process_channels=[embed_dim // 2 ** (3 - i) for i in range(len(in_index))],
        readout_type="project",
        dropout_ratio=0.0,
        num_classes=num_classes,
        align_corners=False,
        loss_decode=nn.ModuleList(
            [
                CrossEntropyLoss(
                    use_sigmoid=False,
                    loss_weight=1.0,
                    loss_name=loss_name,
                ),
            ]
        ),
        ignore_index=ignore_index,
    )


def _make_dinov2_dpt_small_seg_head(
    *,
    embed_dim: int,
    patch_size: int,
    cls_token: bool,
    layers: int,
    num_classes: int,
    loss_name="loss_seg",
    ignore_index=-1,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")

    if layers == 1:
        in_index = [0]
    else:
        assert layers == 4
        in_index = [0, 1, 2, 3]

    return DPTHead(
        in_channels=[embed_dim] * len(in_index),
        channels=128,
        embed_dims=embed_dim,
        patch_size=patch_size,
        post_process_channels=[embed_dim // 2 ** (5 - i) for i in range(len(in_index))],
        readout_type="project",
        dropout_ratio=0.0,
        num_classes=num_classes,
        align_corners=False,
        loss_decode=nn.ModuleList(
            [
                CrossEntropyLoss(
                    use_sigmoid=False,
                    loss_weight=1.0,
                    loss_name=loss_name,
                ),
            ]
        ),
        ignore_index=ignore_index,
    )


def _make_dinov2_dpt_add_small_seg_head(
    *,
    embed_dim: int,
    patch_size: int,
    cls_token: bool,
    layers: int,
    num_classes: int,
    loss_name="loss_seg",
    ignore_index=-1,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")

    if layers == 1:
        in_index = [0]
    else:
        assert layers == 4
        in_index = [0, 1, 2, 3]

    return DPTHead(
        in_channels=[embed_dim] * len(in_index),
        channels=128,
        embed_dims=embed_dim,
        patch_size=patch_size,
        post_process_channels=[embed_dim // 2 ** (5 - i) for i in range(len(in_index))],
        readout_type="add",
        dropout_ratio=0.0,
        num_classes=num_classes,
        align_corners=False,
        loss_decode=nn.ModuleList(
            [
                CrossEntropyLoss(
                    use_sigmoid=False,
                    loss_weight=1.0,
                    loss_name=loss_name,
                ),
            ]
        ),
        ignore_index=ignore_index,
    )
