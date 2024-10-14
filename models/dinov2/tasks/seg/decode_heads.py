# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn

from ..ops import resize
from ...losses import accuracy


class SegBaseDecodeHead(nn.Module):
    def __init__(
      self,
      *,
      in_channels,
      channels,
      num_classes,
      out_channels=None,
      threshold=None,
      dropout_ratio=0.1,
      conv_layer=None,
      act_layer=nn.ReLU,
      in_index=-1,
      input_transform=None,
      loss_decode=(),
      ignore_index=255,
      sampler=None,
      align_corners=False,
    ):
        super().__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        # self.in_channels = in_channels
        # self.input_transform = input_transform
        # self.in_index = in_index
        
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_layer = conv_layer
        self.act_layer = act_layer
        self.in_index = in_index
        self.loss_decode = loss_decode
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert `seg_logits` into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        self.conv_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def forward(self, inputs, img_metas):
        """Placeholder of forward function."""
        pass
      
    def forward_train(self, img, inputs, img_metas, seg_gt):
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
      seg_pred = self.forward(inputs, img_metas)
      losses = self.loss_by_feat(seg_pred, seg_gt)
      return losses

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
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
        seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def loss_by_feat(self, seg_logits, seg_gt):
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_label = seg_gt
        loss = dict()
        seg_logits = resize(
          input=seg_logits, size=seg_label.shape[1:], mode='bilinear', align_corners=self.align_corners, warning=False
        )
        loss.update({"pred": seg_logits})
        
        seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
          losses_decode = [self.loss_decode]
        else:
          losses_decode = self.loss_decode
        for loss_decode in losses_decode:
          if loss_decode.loss_name not in loss:
            loss[loss_decode.loss_name] = loss_decode(
              seg_logits,
              seg_label,
              weight=seg_weight,
              ignore_index=self.ignore_index
            )
          else:
            loss[loss_decode.loss_name] += loss_decode(
              seg_logits,
              seg_label,
              weight=seg_weight,
              ignore_index=self.ignore_index
            )

        # loss['acc_seg'] = accuracy(
        #   seg_logits, seg_label, ignore_index=self.ignore_index
        # )
        return loss

    def predict_by_feat(self, seg_logits, batch_img_metas):
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
          # slide inference
          size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
          size = batch_img_metas[0]['pad_shape'][:2]
        else:
          size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
          input=seg_logits, size=size, mode='bilinear', align_corners=self.align_corners
        )
        return seg_logits


class BNHead(SegBaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None, use_cls_token=True, **kwargs):
        super().__init__(**kwargs)
        self.use_cls_token = use_cls_token
        _channels = self.in_channels * 2 if self.use_cls_token else self.in_channels
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
          assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
          inputs = [
            resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
            for x, f in zip(inputs, self.resize_factors)
          ]
          # print("after", *(x.shape for x in inputs))
        upsampled_inputs = [
          resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
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
      output = self.cls_seg(output)
      return output
