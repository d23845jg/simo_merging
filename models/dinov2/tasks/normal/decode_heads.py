# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn

from ...layers.dpt import ConvModule, FeatureFusionBlock, HeadSeg, ReassembleBlocks
from ...layers.ops import resize


class NormalBaseDecodeHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        channels,
        out_channels=3,
        dropout_ratio=0.1,
        conv_layer=None,
        act_layer=nn.ReLU,
        in_index=-1,
        input_transform=None,
        loss_decode=(),
        ignore_index=255,
        sampler=None,
        align_corners=False,
        norm_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_layer = conv_layer
        self.act_layer = act_layer
        self.loss_decode = loss_decode
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.norm_layer = norm_layer
        self.out_channels = out_channels

        self.conv_normal = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def forward(self, inputs, img_metas):
        """Placeholder of forward function."""
        pass

    def forward_train(self, img, inputs, img_metas, img_gt, **kwargs):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): GT depth

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img_pred = self.forward(inputs, img_metas)
        losses = self.loss_by_feat(img_pred, img_metas, img_gt)
        return losses

    def normal_pred(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_normal(feat)
        return output

    def predict(self, inputs, batch_img_metas, test_cfg):
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        img_pred = self.forward(inputs)
        return self.predict_by_feat(img_pred, batch_img_metas)

    def loss_by_feat(self, img_logits, img_metas, img_gt):
        """Compute segmentation loss.

        Args:
            img_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        loss = dict()
        img_logits = resize(
            input=img_logits,
            size=img_gt.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
            warning=False,
        )
        loss.update({"pred": img_logits})

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        ignore_idx = img_metas.get("mask", self.ignore_index)
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    img_logits, img_gt, ignore_index=ignore_idx
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    img_logits, img_gt, ignore_index=ignore_idx
                )
        return loss

    def predict_by_feat(self, normal_logits, batch_img_metas):
        """Transform a batch of output normal_logits to the input shape.

        Args:
            normal_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        if isinstance(batch_img_metas[0]["img_shape"], torch.Size):
            # slide inference
            size = batch_img_metas[0]["img_shape"]
        elif "pad_shape" in batch_img_metas[0]:
            size = batch_img_metas[0]["pad_shape"][:2]
        else:
            size = batch_img_metas[0]["img_shape"]

        normal_logits = resize(
            input=normal_logits,
            size=size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return normal_logits


class BNHead(NormalBaseDecodeHead):
    """Just a batchnorm."""

    def __init__(
        self,
        input_transform="resize_concat",
        in_index=(0, 1, 2, 3),
        resize_factors=None,
        use_cls_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_transform = input_transform
        self.in_index = in_index
        self.use_cls_token = use_cls_token
        _channels = (
            sum(self.in_channels) * 2 if self.use_cls_token else sum(self.in_channels)
        )
        assert _channels == self.channels
        self.bn = nn.SyncBatchNorm(_channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs, img_metas=None, **kwargs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x) == 2 and self.use_cls_token:
                    x, cls_token = x[0], x[1]
                    if len(x.shape) == 2:
                        x = x[:, :, None, None]
                    cls_token = cls_token[:, :, None, None].expand_as(x)
                    inputs[i] = torch.cat((x, cls_token), 1)
                else:
                    x = x[0]
                    if len(x.shape) == 2:
                        x = x[:, :, None, None]
                    inputs[i] = x

            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (
                    len(self.resize_factors),
                    len(inputs),
                )
                inputs = [
                    resize(
                        input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area"
                    )
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs, img_metas=None, **kwargs):
        """Forward function."""
        output = self._forward_feature(inputs, img_metas=img_metas, **kwargs)
        output = self.normal_pred(output)
        return output


class DPTHead(NormalBaseDecodeHead):
    """Vision Transformers for Dense Prediction.
    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.
    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Out channels of post process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether expand the channels in post process
            block. Default: False.
    """

    def __init__(
        self,
        embed_dims=768,
        post_process_channels=[96, 192, 384, 768],
        readout_type="ignore",
        patch_size=16,
        expand_channels=False,
        **kwargs,
    ):
        super(DPTHead, self).__init__(**kwargs)

        self.in_channels = self.in_channels
        self.expand_channels = expand_channels

        self.reassemble_blocks = ReassembleBlocks(
            embed_dims, post_process_channels, readout_type, patch_size
        )

        self.post_process_channels = [
            channel * math.pow(2, i) if expand_channels else channel
            for i, channel in enumerate(post_process_channels)
        ]

        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(
                ConvModule(
                    channel,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    act_layer=None,
                    bias=False,
                )
            )

        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(
                FeatureFusionBlock(self.channels, self.act_layer, self.norm_layer)
            )
        self.fusion_blocks[0].res_conv_unit1 = None

        self.project = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_layer=self.norm_layer,
        )

        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels

        self.conv_normal = HeadSeg(self.channels, self.out_channels)

    def forward(self, inputs, img_metas):
        assert len(inputs) == self.num_reassemble_blocks
        x = [inp for inp in inputs]
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        out = self.project(out)
        out = self.normal_pred(out)
        return out
