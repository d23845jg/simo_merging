# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

<<<<<<< Updated upstream
from .ops import resize
from ..backbones import _make_dinov2_model
from ..tasks.depth.depther import _make_dinov2_linear_depth_head, _make_dinov2_linear_depth_bins_head, _make_dinov2_dpt_depth_head
from ..tasks.seg.segmentor import _make_dinov2_linear_seg_head
from ..tasks.normal.surface_normalizer import _make_dinov2_linear_normal_head
=======
from ..backbones import _make_dinov2_model
from ..layers.ops import resize
from ..tasks.depth.depther import (
    _make_dinov2_dpt_add_small_depth_head,
    _make_dinov2_dpt_depth_head,
    _make_dinov2_dpt_small_depth_head,
    _make_dinov2_linear_depth_bins_head,
    _make_dinov2_linear_depth_head,
)
from ..tasks.normal.surface_normalizer import (
    _make_dinov2_dpt_add_small_normal_head,
    _make_dinov2_dpt_normal_head,
    _make_dinov2_dpt_small_normal_head,
    _make_dinov2_linear_normal_head,
)
from ..tasks.seg.segmentor import (
    _make_dinov2_dpt_add_small_seg_head,
    _make_dinov2_dpt_seg_head,
    _make_dinov2_dpt_small_seg_head,
    _make_dinov2_linear_seg_head,
)
>>>>>>> Stashed changes
from ..utils import CenterPadding


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """
    outputs = dict()
    for name, value in inputs.items():
        outputs[f"{prefix}.{name}"] = value
    return outputs


class MTLDinoV2(nn.Module):
<<<<<<< Updated upstream
  def __init__(
    self,
    *,
    arch_name: str = "vit_base", # , "vit_small", "vit_base", "vit_large", "vit_giant2"
    ffn_layer = "mlp", # "mlp", "swiglufused"
    pretrained: bool = True,
    head_tasks,
    head_archs: str = "linear",
    out_index: list = [5, 7, 9, 11],
    cls_token: bool = False,
    **kwargs,
  ):
    super(MTLDinoV2, self).__init__()
    self.constructor_args = {
      "arch_name": arch_name,
      "ffn_layer": ffn_layer,
      "pretrained": pretrained,
      "head_tasks": head_tasks,
      "head_archs": head_archs,
      "out_index": out_index,
      "cls_token": cls_token,
      **kwargs
    }

    # Creating backbone
    self.backbone = _make_dinov2_model(arch_name=arch_name, ffn_layer=ffn_layer, pretrained=pretrained, **kwargs)
    
    # Define out_index layers
    # out_index = {
    #   "vit_small": [5, 7, 9, 11],
    #   "vit_base": [5, 7, 9, 11],
    #   "vit_large": [17, 19, 21, 23],
    #   "vit_giant2": [33, 35, 37, 39],
    # }[arch_name]
    self.backbone.forward = partial(
        self.backbone.get_intermediate_layers,
        n=out_index,
        reshape=True,
        return_class_token=True,
        norm=False,
    )
    self.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(self.backbone.patch_size)(x[0]))
    
    # Creating task-specific heads
    self.head_tasks = head_tasks
    self.decoders = nn.ModuleDict({
      task: self.create_task_specific_head(
        task, 
        head_archs,
        out_index,
        cls_token,
      )
      for task in self.head_tasks.keys()
    })
    
  
  def create_task_specific_head(self, task, head_archs, out_index, cls_token):
    if task == "depth":
      if head_archs == "linear_bins":
        return _make_dinov2_linear_depth_bins_head(
          embed_dim=self.backbone.embed_dim,
          cls_token=cls_token,
          layers=len(out_index),
          min_depth=self.head_tasks[task]["min_depth"],
          max_depth=self.head_tasks[task]["max_depth"],
        )
      elif head_archs == "linear":
        return _make_dinov2_linear_depth_head(
          embed_dim=self.backbone.embed_dim,
          cls_token=cls_token,
          layers=len(out_index),
          min_depth=self.head_tasks[task]["min_depth"],
          max_depth=self.head_tasks[task]["max_depth"],
        )
      elif head_archs == "dpt":
        return _make_dinov2_dpt_depth_head(
          embed_dim=self.backbone.embed_dim,
          patch_size=self.backbone.patch_size,
          layers=len(out_index),
          min_depth=self.head_tasks[task]["min_depth"],
          max_depth=self.head_tasks[task]["max_depth"],
        )
    elif task == "seg":
      if head_archs == "linear":
        return _make_dinov2_linear_seg_head(
          embed_dim=self.backbone.embed_dim,
          cls_token=cls_token,
          layers=len(out_index),
          num_classes=self.head_tasks[task]["num_classes"],
        )
    elif task == "normal":
      if head_archs == "linear":
        return _make_dinov2_linear_normal_head(
          embed_dim=self.backbone.embed_dim,
          cls_token=cls_token,
          layers=len(out_index),
        )
    else:
      raise NotImplementedError(f"Unsupported task: {task}")
  
  def _decode_heads_forward_train(self, img, x, img_metas, img_gt, **kwargs):
    """Run forward function and calculate loss for decode head in
    training."""
    losses = {
      task: {
        **(task_losses := self.decoders[task].forward_train(img, x, img_metas, img_gt[task], **kwargs)),
        "total_loss": sum(loss for loss_name, loss in task_losses.items() if "loss" in loss_name)
      }
      for task in self.head_tasks.keys()
    }
    
    # losses.update(add_prefix(loss_decode, "decode"))
    return losses
  
  def _decode_heads_forward_test(self, x, img_metas):
    """Run forward function and calculate loss for decode head in
    inference."""
    pred = dict()
    for task in self.head_tasks.keys():
      pred[task] = self.decoders[task].forward_test(x, img_metas)
    return pred
    
  def extract_feat(self, img):
    """Extract features from images."""
    return self.backbone(img)

  def encode_decode(self, img, img_metas, rescale=True, size=None):
    """Encode images with backbone and decode into a depth estimation
    map of the same size as input."""
    x = self.extract_feat(img)
    out = self._decode_heads_forward_test(x, img_metas)
    
    for task in self.head_tasks.items():
      if task == "depth":
        # crop the pred depth to the certain range.
        out = torch.clamp(out, min=self.decoders[task].min_depth, max=self.decoders[task].max_depth)
      
      if rescale:
        if size is None:
          if img_metas is not None:
            size = img_metas[0]["ori_shape"][:2]
          else:
            size = img.shape[2:]
        out = resize(input=out, size=size, mode="bilinear", align_corners=self.decoders[task].align_corners)
    return out
  
  def whole_inference(self, img, img_meta, rescale, size=None):
    """Inference with full image."""
    return self.encode_decode(img, img_meta, rescale, size=size)

  def slide_inference(self, img, img_meta, rescale, stride, crop_size):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.size()
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = img.new_zeros((batch_size, 1, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            depth_pred = self.encode_decode(crop_img, img_meta, rescale) # TODO: fix this
            preds += F.pad(depth_pred, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        # cast count_mat to constant while exporting to ONNX
        count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
    preds = preds / count_mat
    return preds
  
  def inference(self, img, img_meta, rescale, size=None, mode="whole"):
    """Inference with slide/whole style.

    Args:
        img (Tensor): The input image of shape (N, 3, H, W).
        img_meta (dict): Image info dict where each dict has: 'img_shape',
            'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `depth/datasets/pipelines/formatting.py:Collect`.
        rescale (bool): Whether rescale back to original shape.

    Returns:
        Tensor: The output depth map.
    """
    assert mode in ["slide", "whole"]
    if mode == "slide":
        img_pred = self.slide_inference(img, img_meta, rescale)
    else:
        img_pred = self.whole_inference(img, img_meta, rescale, size=size)
    return img_pred
  
  def mtl_loss(self, losses, mtl_weight="equal", **kwargs):
    """Calculate the total loss for multi-task learning."""
    if mtl_weight == "equal":
      total_loss = sum(loss["total_loss"] for loss in losses.values())
    elif mtl_weight == "uncert":
      # TODO: implement this
      total_loss = 1
    else:
      raise NotImplementedError(f"Unsupported mtl_weight: {mtl_weight}")
    
    return total_loss
  
  def forward_train(self, img, img_metas, img_gt, **kwargs):
    """Forward function for training.
=======
    def __init__(
        self,
        *,
        arch_name: str = "vit_base",  # , "vit_small", "vit_base", "vit_large", "vit_giant2"
        ffn_layer="mlp",  # "mlp", "swiglufused"
        pretrained: bool = True,
        head_tasks,
        head_archs: str = "linear",
        out_index: list = [5, 7, 9, 11],
        cls_token: bool = False,
        **kwargs,
    ):
        super(MTLDinoV2, self).__init__()
        self.constructor_args = {
            "arch_name": arch_name,
            "ffn_layer": ffn_layer,
            "pretrained": pretrained,
            "head_tasks": head_tasks,
            "head_archs": head_archs,
            "out_index": out_index,
            "cls_token": cls_token,
            **kwargs,
        }

        # Creating backbone
        self.backbone = _make_dinov2_model(
            arch_name=arch_name, ffn_layer=ffn_layer, pretrained=pretrained, **kwargs
        )
>>>>>>> Stashed changes

        # Define out_index layers
        self.backbone.forward = partial(
            self.backbone.get_intermediate_layers,
            n=out_index,
            reshape=True,
            return_class_token=True,
            norm=False,
        )
        self.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(self.backbone.patch_size)(x[0])
        )

        # Creating task-specific heads
        self.head_tasks = head_tasks
        self.decoders = nn.ModuleDict(
            {
                task: self.create_task_specific_head(
                    task,
                    head_archs,
                    out_index,
                    cls_token,
                )
                for task in self.head_tasks.keys()
            }
        )

    def create_task_specific_head(self, task, head_archs, out_index, cls_token):
        if task in ["depth", "disp", "depth_zbuffer"]:
            params = {
                "embed_dim": self.backbone.embed_dim,
                "cls_token": cls_token,
                "layers": len(out_index),
                "min_depth": self.head_tasks[task]["min_depth"],
                "max_depth": self.head_tasks[task]["max_depth"],
                "loss_name": f"loss_{task}",
            }
            if head_archs == "linear_bins":
                return _make_dinov2_linear_depth_bins_head(**params)
            elif "linear" in head_archs:
                return _make_dinov2_linear_depth_head(**params)
            elif head_archs == "dpt-add_small":
                return _make_dinov2_dpt_add_small_depth_head(
                    **params,
                    patch_size=self.backbone.patch_size,
                )
            elif head_archs == "dpt_small":
                return _make_dinov2_dpt_small_depth_head(
                    **params,
                    patch_size=self.backbone.patch_size,
                )
            elif head_archs == "dpt":
                return _make_dinov2_dpt_depth_head(
                    **params,
                    patch_size=self.backbone.patch_size,
                )
        elif task in ["seg", "part_seg", "segment_semantic"]:
            params = {
                "embed_dim": self.backbone.embed_dim,
                "cls_token": cls_token,
                "layers": len(out_index),
                "num_classes": self.head_tasks[task]["num_classes"],
                "loss_name": f"loss_{task}",
                "ignore_index": 0 if task == "segment_semantic" else -1,
            }
            if "linear" in head_archs:
                return _make_dinov2_linear_seg_head(**params)
            elif head_archs == "dpt-add_small":
                return _make_dinov2_dpt_add_small_seg_head(
                    **params,
                    patch_size=self.backbone.patch_size,
                )
            elif head_archs == "dpt_small":
                return _make_dinov2_dpt_small_seg_head(
                    **params,
                    patch_size=self.backbone.patch_size,
                )
            elif head_archs == "dpt":
                return _make_dinov2_dpt_seg_head(
                    **params,
                    patch_size=self.backbone.patch_size,
                )
        elif task in ["normal", "keypoints2d", "edge_texture"]:
            params = {
                "embed_dim": self.backbone.embed_dim,
                "cls_token": cls_token,
                "layers": len(out_index),
                "num_classes": self.head_tasks[task]["num_classes"],
                "loss_name": f"loss_{task}",
            }
            if "linear" in head_archs:
                return _make_dinov2_linear_normal_head(**params)
            elif head_archs == "dpt-add_small":
                return _make_dinov2_dpt_add_small_normal_head(
                    **params,
                    patch_size=self.backbone.patch_size,
                )
            elif head_archs == "dpt_small":
                return _make_dinov2_dpt_small_normal_head(
                    **params,
                    patch_size=self.backbone.patch_size,
                )
            elif head_archs == "dpt":
                return _make_dinov2_dpt_normal_head(
                    **params,
                    patch_size=self.backbone.patch_size,
                )
        else:
            raise NotImplementedError(f"Unsupported task: {task}")

<<<<<<< Updated upstream
  def freeze_shared_layers(self, requires_grad=False):
    for module in self.shared_modules():
      for param in module.parameters():
        param.requires_grad = requires_grad
        
    for module in self.decoders.values():
      for param in module.parameters():
        param.requires_grad = True
    
    
    
=======
    def _decode_heads_forward_train(self, img, x, img_metas, img_gt, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = {
            task: {
                **(
                    self.decoders[task].forward_train(
                        img, x, img_metas, img_gt[task], **kwargs
                    )
                ),
            }
            for task in self.head_tasks.keys()
        }
        return losses

    def _decode_heads_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        pred = dict()
        for task in self.head_tasks.keys():
            pred[task] = self.decoders[task].forward_test(x, img_metas)
        return pred

    def extract_feat(self, img):
        """Extract features from images."""
        return self.backbone(img)

    def encode_decode(self, img, img_metas, rescale=True, size=None):
        """Encode images with backbone and decode into a depth estimation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_heads_forward_test(x, img_metas)

        for task in self.head_tasks.items():
            if task == "depth":
                # crop the pred depth to the certain range.
                out = torch.clamp(
                    out,
                    min=self.decoders[task].min_depth,
                    max=self.decoders[task].max_depth,
                )

            if rescale:
                if size is None:
                    if img_metas is not None:
                        size = img_metas[0]["ori_shape"][:2]
                    else:
                        size = img.shape[2:]
                out = resize(
                    input=out,
                    size=size,
                    mode="bilinear",
                    align_corners=self.decoders[task].align_corners,
                )
        return out

    def whole_inference(self, img, img_meta, rescale, size=None):
        """Inference with full image."""
        return self.encode_decode(img, img_meta, rescale, size=size)

    def slide_inference(self, img, img_meta, rescale, stride, crop_size):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, 1, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                depth_pred = self.encode_decode(
                    crop_img, img_meta, rescale
                )  # TODO: fix this
                preds += F.pad(
                    depth_pred,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(
                device=img.device
            )
        preds = preds / count_mat
        return preds

    def inference(self, img, img_meta, rescale, size=None, mode="whole"):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output depth map.
        """
        assert mode in ["slide", "whole"]
        if mode == "slide":
            img_pred = self.slide_inference(img, img_meta, rescale)
        else:
            img_pred = self.whole_inference(img, img_meta, rescale, size=size)
        return img_pred

    def mtl_loss(self, losses, mtl_weight="equal", **kwargs):
        """Calculate the total loss for multi-task learning."""
        if mtl_weight == "equal":
            total_loss = sum(losses[task][f"loss_{task}"] for task in losses)
        elif mtl_weight == "uncert":
            logsigma = kwargs.get("logsigma", None)
            total_loss = sum(
                1 / (2 * torch.exp(logsigma[task])) * losses[task][f"loss_{task}"]
                + logsigma[task] / 2
                for task in losses
            )
        else:
            raise NotImplementedError(f"Unsupported mtl_weight: {mtl_weight}")

        return total_loss

    def forward_train(self, img, img_metas, img_gt, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): Depth gt
                used if the architecture supports depth estimation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        losses = self._decode_heads_forward_train(img, x, img_metas, img_gt, **kwargs)
        losses["total_loss"] = self.mtl_loss(losses, **kwargs)
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.inference(imgs, img_metas)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def shared_modules(self):
        return [
            self.backbone,
        ]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def freeze_shared_layers(self, requires_grad=False):
        for module in self.shared_modules():
            for param in module.parameters():
                param.requires_grad = requires_grad

    def freeze_head_layers(self, requires_grad=False):
        for module in self.decoders.values():
            for param in module.parameters():
                param.requires_grad = requires_grad
>>>>>>> Stashed changes
