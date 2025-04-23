import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndi

from isegm.utils import misc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi



class UnknownRegionDTLoss(nn.Module):
    def __init__(self, ignore_index=-1, reduction='mean', debug_print=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.debug = debug_print

    def forward(self, pred_logits, target):
        device = pred_logits.device
        dtype = pred_logits.dtype
        B, C, H, W = pred_logits.shape
        pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, H, W]
        pred_unknown = pred_probs[:, 1, :, :]       # [B, H, W]

        losses = []

        for i in range(B):
            target_i = target[i].cpu().numpy()
            gt_unknown_mask = (target_i == 1).astype(np.float32)

            if gt_unknown_mask.sum() == 0:
                losses.append(torch.tensor(0.0, dtype=dtype, device=device))
                continue

            dist_map = ndi.distance_transform_edt(gt_unknown_mask).astype(np.float32)
            dist_map = torch.from_numpy(dist_map).to(device=device, dtype=dtype)

            # The code is assigning the value of `pred_unknown[i]` to the variable `pred_i`. The
            # comment `# [H, W]` suggests that `pred_unknown[i]` is expected to be a 2D array with
            # dimensions Height (H) and Width (W).
            pred_i = pred_unknown[i]  # [H, W]
            gt_mask_tensor = torch.from_numpy(gt_unknown_mask).to(device=device,dtype=dtype)

            pred_error = (1.0 - pred_i) * gt_mask_tensor
            weighted_error = pred_error * dist_map 

            if self.debug:
                print(f"[DEBUG][{i}] pred_i.dtype: {pred_i.dtype}")
                print(f"[DEBUG][{i}] pred_error dtype: {pred_error.dtype}, dist_map dtype: {dist_map.dtype}, weighted_error dtype: {weighted_error.dtype}")
                print(f"[DEBUG][{i}] Unknown pixel count: {gt_unknown_mask.sum()}")
                print(f"[DEBUG][{i}] pred_error mean: {pred_error.mean().item():.6f}, dist_map max: {dist_map.max().item():.2f}")
                print(f"[DEBUG][{i}] weighted_error stats: min={weighted_error.min().item():.12f}, max={weighted_error.max().item():.12f}, mean={weighted_error.mean().item():.12f}")

            losses.append(weighted_error.mean())            

        loss_tensor = torch.stack(losses)

        if self.reduction == 'mean':
            return loss_tensor.mean()
        elif self.reduction == 'sum':
            return loss_tensor.sum()
        else:
            return loss_tensor




# class UnknownRegionDTLoss(nn.Module):
#     """
#     Distance Transform loss focused on the 'unknown' class (class index = 1 in trimap: 0=BG, 1=Unknown, 2=FG).
#     This loss penalizes errors in predicting the unknown region using a distance-based weighting.
#     """
#     def __init__(self, ignore_index=-1, reduction='mean', debug_print=True):
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.debug = debug_print

#     def forward(self, pred_logits, target):
#         """
#         Args:
#             pred_logits: [B, C, H, W] raw logits from model (before softmax)
#             target: [B, H, W] int tensor with values in {0, 1, 2} (0=BG, 1=Unknown, 2=FG)
#         Returns:
#             Distance Transform weighted loss for the unknown class
#         """
#         device = pred_logits.device
#         B, C, H, W = pred_logits.shape
#         pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, H, W]
#         pred_unknown = pred_probs[:, 1, :, :]       # [B, H, W] - channel 1 = unknown

#         losses = []

#         for i in range(B):
#             target_i = target[i].cpu().numpy()
#             pred_i = pred_unknown[i]  # [H, W]

#             # mask for unknown region
#             gt_unknown_mask = (target_i == 1).astype(np.uint8)
#             if np.sum(gt_unknown_mask) == 0:
#                 losses.append(torch.tensor(0.0, dtype=pred_i.dtype, device=pred_i.device))
#                 continue

#             # distance from non-unknown pixels to unknown
#             dist_map = ndi.distance_transform_edt(gt_unknown_mask)
#             dist_map = torch.tensor(dist_map, dtype=pred_i.dtype, device=pred_i.device)

#             # penalize low prediction in unknown region
#             pred_error = (1.0 - pred_i) * torch.tensor(gt_unknown_mask, dtype=pred_i.dtype, device=pred_i.device)
#             weighted_error = pred_error * dist_map

#             losses.append(weighted_error.mean())
            
#             if self.debug:
#                 print(f"[DEBUG][{i}] pred_i.dtype: {pred_i.dtype}")
#                 print(f"[DEBUG][{i}] pred_error dtype: {pred_error.dtype}, dist_map dtype: {dist_map.dtype}, weighted_error dtype: {weighted_error.dtype}")
#                 print(f"[DEBUG][{i}] Unknown pixel count: {gt_unknown_mask.sum()}")
#                 print(f"[DEBUG][{i}] pred_error mean: {pred_error.mean().item():.6f}, dist_map max: {dist_map.max().item():.2f}")
#                 print(f"[DEBUG][{i}] weighted_error stats: min={weighted_error.min().item():.12f}, max={weighted_error.max().item():.12f}, mean={weighted_error.mean().item():.12f}")

#         if self.reduction == 'mean':
#             return torch.stack(losses).mean()
#         elif self.reduction == 'sum':
#             return torch.stack(losses).sum()
#         else:
#             return torch.stack(losses)


class NormalizedFocalLossSoftmax(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=-1, reduction='mean', eps=1e-12, debug_print=False):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps
        self.debug_print = debug_print  # 디버깅 출력 여부 (False: 끔, True: 켬)

    def forward(self, inputs, targets):
        B, C, H, W = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)  # [(B*H*W), C]
        targets = targets.view(-1)                         # [(B*H*W)]

        if self.debug_print:
            try:
                class_counts = torch.bincount(targets)
            except:
                class_counts = "Error in bincount"
            print(f'[DEBUG] GT label counts: {class_counts}')

        valid_mask = targets != self.ignore_index
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        if self.debug_print:
            print(f'[DEBUG] Valid pixel count: {valid_mask.sum().item()}')

        if targets.numel() == 0:
            if self.debug_print:
                print("[DEBUG] No valid targets remaining.")
            return torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device, requires_grad=True)

        if self.debug_print:
            print(f'[DEBUG] Input logits stats: mean={inputs.mean().item():.6f}, std={inputs.std().item():.6f}')

        log_probs = F.log_softmax(inputs, dim=-1)                    # [N, C]
        probs = torch.exp(log_probs)                                # [N, C]
        pt = probs[torch.arange(len(targets)), targets]             # [N]
        log_pt = log_probs[torch.arange(len(targets)), targets]     # [N]

        focal_weight = (1.0 - pt).pow(self.gamma)
        focal_weight = focal_weight / (focal_weight.sum() + self.eps)
        focal_weight = focal_weight * targets.numel()  # scaling

        loss = -focal_weight * log_pt  # [N]

        if self.debug_print:
            print(f'[DEBUG] Loss stats: min={loss.min().item():.6f}, max={loss.max().item():.6f}, mean={loss.mean().item():.6f}')

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# class NormalizedFocalLossSoftmax(nn.Module):
#     def __init__(self, gamma=2.0, ignore_index=-1, reduction='mean', eps=1e-12):
#         """
#         Normalized Focal Loss (Softmax-based), as used in Click2Trimap.

#         Args:
#             gamma: focusing parameter
#             ignore_index: label to ignore
#             reduction: 'mean' | 'sum' | 'none'
#             eps: small epsilon to avoid division by zero
#         """
#         super().__init__()
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.eps = eps

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: [B, C, H, W] raw logits
#             targets: [B, H, W] with values in 0 .. C-1
#         """
#         B, C, H, W = inputs.shape
#         inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)  # [(B*H*W), C]
#         targets = targets.view(-1)                         # [(B*H*W)]

#         valid_mask = targets != self.ignore_index
#         inputs = inputs[valid_mask]
#         targets = targets[valid_mask]

#         if targets.numel() == 0:
#             return torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device, requires_grad=True)

#         log_probs = F.log_softmax(inputs, dim=-1)                    # [N, C]
#         probs = torch.exp(log_probs)                                # [N, C]
#         pt = probs[torch.arange(len(targets)), targets]             # [N]
#         log_pt = log_probs[torch.arange(len(targets)), targets]     # [N]

#         # Focal weight
#         focal_weight = (1.0 - pt).pow(self.gamma)                   # [N]

#         # Normalization as in Click2Trimap (sum to 1)
#         focal_weight = focal_weight / (focal_weight.sum() + self.eps)

#         # Loss
#         loss = -focal_weight * log_pt  # [N]

#         # Reduction
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) \
            / (torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


class BinaryDiceLoss(nn.Module):
    """ Dice Loss for binary segmentation
    """

    def forward(self, pred, label):
        batchsize = pred.size(0)

        # convert probability to binary label using maximum probability
        input_pred, input_label = pred.max(1)
        input_pred *= input_label.float()

        # convert to floats
        input_pred = input_pred.float()
        target_label = label.float()

        # convert to 1D
        input_pred = input_pred.view(batchsize, -1)
        target_label = target_label.view(batchsize, -1)

        # compute dice score
        intersect = torch.sum(input_pred * target_label, 1)
        input_area = torch.sum(input_pred * input_pred, 1)
        target_area = torch.sum(target_label * target_label, 1)

        sum = input_area + target_area
        epsilon = torch.tensor(1e-6)

        # batch dice loss and ignore dice loss where target area = 0
        batch_loss = torch.tensor(1.0) - (torch.tensor(2.0) * intersect + epsilon) / (sum + epsilon)
        loss = batch_loss.mean()

        return loss