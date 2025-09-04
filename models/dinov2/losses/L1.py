import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """
    Compute L1 loss.
    """

    def __init__(self, loss_weight=1, loss_name="loss_l1", **kwargs):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, prediction, target, ignore_index, **kwargs):
        if isinstance(ignore_index, int):
            return self.forward_ignore_index(prediction, target, ignore_index, **kwargs)
        return self.forward_mask(prediction, target, ignore_index, **kwargs)

    def forward_ignore_index(self, prediction, target, ignore_index, **kwargs):
        mask = (torch.sum(target, dim=1) != ignore_index).bool().unsqueeze(1)
        return self.forward_mask(prediction, target, mask, **kwargs)

    def forward_mask(self, prediction, target, mask, **kwargs):
        loss = torch.abs(prediction - target).masked_select(mask).mean()
        return self.loss_weight * loss
