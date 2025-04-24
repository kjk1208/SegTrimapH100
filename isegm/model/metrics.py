import torch
import numpy as np

from isegm.utils import misc


import torch
import numpy as np
from isegm.utils import misc
#from isegm.model.metrics import TrainMetric

class TrainMetric(object):
    def __init__(self, pred_outputs, gt_outputs):
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_epoch_value(self):
        raise NotImplementedError

    def reset_epoch_stats(self):
        raise NotImplementedError

    def log_states(self, sw, tag_prefix, global_step):
        pass

    @property
    def name(self):
        return type(self).__name__
class PerClassIoU(TrainMetric):
    def __init__(self, ignore_label=-1, pred_output='instances', gt_output='instances'):
        super().__init__(pred_outputs=(pred_output,), gt_outputs=(gt_output,))
        self.ignore_label = ignore_label
        self.class_iou_sum = [0.0, 0.0, 0.0]  # [BG, Unknown, FG]
        self.class_counts = [0, 0, 0]

    def update(self, pred, gt):
        if pred.dim() == 4 and pred.size(1) == 3:
            pred = torch.argmax(pred, dim=1)  # [B, H, W]

        for cls in range(3):
            cls_pred = pred == cls
            cls_gt = gt == cls
            ignore = gt == self.ignore_label

            iou = _compute_iou(cls_pred, cls_gt, ignore)
            if len(iou) > 0:
                self.class_iou_sum[cls] += float(np.mean(iou))
                self.class_counts[cls] += 1

    def get_epoch_value(self):
        return {
            'bg': self.class_iou_sum[0] / max(self.class_counts[0], 1),
            'unknown': self.class_iou_sum[1] / max(self.class_counts[1], 1),
            'fg': self.class_iou_sum[2] / max(self.class_counts[2], 1),
        }

    def reset_epoch_stats(self):
        self.class_iou_sum = [0.0, 0.0, 0.0]
        self.class_counts = [0, 0, 0]


class MultiClassIoU(TrainMetric):
    def __init__(self, ignore_label=-1, pred_output='instances', gt_output='instances'):
        super().__init__(pred_outputs=(pred_output,), gt_outputs=(gt_output,))
        self.ignore_label = ignore_label
        self.total_iou = 0.0
        self.num_batches = 0

    def update(self, pred, gt):
        if pred.dim() == 4 and pred.size(1) == 3:
            pred = torch.argmax(pred, dim=1)

        per_class_iou = []
        for cls in range(3):
            cls_pred = pred == cls
            cls_gt = gt == cls
            ignore = gt == self.ignore_label
            iou = _compute_iou(cls_pred, cls_gt, ignore)
            if len(iou) > 0:
                per_class_iou.append(np.mean(iou))

        if len(per_class_iou) == 3:
            self.total_iou += np.mean(per_class_iou)
            self.num_batches += 1

    def get_epoch_value(self):
        if self.num_batches == 0:
            return 0.0
        return self.total_iou / self.num_batches

    def reset_epoch_stats(self):
        self.total_iou = 0.0
        self.num_batches = 0


class UnknownIoU(TrainMetric):
    def __init__(self, ignore_label=-1, pred_output='instances', gt_output='instances'):
        super().__init__(pred_outputs=(pred_output,), gt_outputs=(gt_output,))
        self.iou_sum = 0.0
        self.count = 0
        self.ignore_label = ignore_label

    def update(self, pred, gt):
        if pred.dim() == 4 and pred.size(1) == 3:
            pred = torch.argmax(pred, dim=1)

        unknown_pred = pred == 1
        unknown_gt = gt == 1
        ignore = gt == self.ignore_label
        iou = _compute_iou(unknown_pred, unknown_gt, ignore)
        if len(iou) > 0:
            self.iou_sum += np.mean(iou)
            self.count += 1

    def get_epoch_value(self):
        return self.iou_sum / max(self.count, 1)

    def reset_epoch_stats(self):
        self.iou_sum = 0.0
        self.count = 0


def _compute_iou(pred_mask, gt_mask, ignore_mask=None):
    if ignore_mask is not None:
        pred_mask = torch.where(ignore_mask, torch.zeros_like(pred_mask), pred_mask)

    reduction_dims = misc.get_dims_with_exclusion(gt_mask.dim(), 0)
    union = torch.mean((pred_mask | gt_mask).float(), dim=reduction_dims).detach().cpu().numpy()
    intersection = torch.mean((pred_mask & gt_mask).float(), dim=reduction_dims).detach().cpu().numpy()
    nonzero = union > 0

    iou = intersection[nonzero] / union[nonzero]
    return iou





class AdaptiveIoU(TrainMetric):
    def __init__(self, init_thresh=0.4, thresh_step=0.025, thresh_beta=0.99, iou_beta=0.9,
                 ignore_label=-1, from_logits=True,
                 pred_output='instances', gt_output='instances'):
        super().__init__(pred_outputs=(pred_output,), gt_outputs=(gt_output,))
        self._ignore_label = ignore_label
        self._from_logits = from_logits
        self._iou_thresh = init_thresh
        self._thresh_step = thresh_step
        self._thresh_beta = thresh_beta
        self._iou_beta = iou_beta
        self._ema_iou = 0.0
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    # def update(self, pred, gt):
    #     gt_mask = gt > 0.5
    #     if self._from_logits:
    #         pred = torch.sigmoid(pred)

    #     gt_mask_area = torch.sum(gt_mask, dim=(1, 2)).detach().cpu().numpy()
    #     if np.all(gt_mask_area == 0):
    #         return

    #     ignore_mask = gt == self._ignore_label
    #     max_iou = _compute_iou(pred > self._iou_thresh, gt_mask, ignore_mask).mean()
    #     best_thresh = self._iou_thresh
    #     for t in [best_thresh - self._thresh_step, best_thresh + self._thresh_step]:
    #         temp_iou = _compute_iou(pred > t, gt_mask, ignore_mask).mean()
    #         if temp_iou > max_iou:
    #             max_iou = temp_iou
    #             best_thresh = t

    #     self._iou_thresh = self._thresh_beta * self._iou_thresh + (1 - self._thresh_beta) * best_thresh
    #     self._ema_iou = self._iou_beta * self._ema_iou + (1 - self._iou_beta) * max_iou
    #     self._epoch_iou_sum += max_iou
    #     self._epoch_batch_count += 1
    def update(self, pred, gt):
        # logits -> trimap class → foreground binary
        if pred.dim() == 4 and pred.size(1) == 3:
            pred = torch.argmax(pred, dim=1)  # [B, H, W]
        pred_mask = (pred == 2)  # foreground class만 IoU 평가

        gt_mask = (gt == 2)
        ignore_mask = (gt == self._ignore_label)

        gt_mask_area = torch.sum(gt_mask, dim=(1, 2)).detach().cpu().numpy()
        if np.all(gt_mask_area == 0):
            return

        max_iou = _compute_iou(pred_mask, gt_mask, ignore_mask).mean()
        best_thresh = self._iou_thresh

        # adaptive threshold logic 유지 (실제 효과 없음, dummy처럼 작동)
        for t in [best_thresh - self._thresh_step, best_thresh + self._thresh_step]:
            temp_iou = _compute_iou(pred_mask, gt_mask, ignore_mask).mean()
            if temp_iou > max_iou:
                max_iou = temp_iou
                best_thresh = t

        self._iou_thresh = self._thresh_beta * self._iou_thresh + (1 - self._thresh_beta) * best_thresh
        self._ema_iou = self._iou_beta * self._ema_iou + (1 - self._iou_beta) * max_iou
        self._epoch_iou_sum += max_iou
        self._epoch_batch_count += 1

    def get_epoch_value(self):
        if self._epoch_batch_count > 0:
            return self._epoch_iou_sum / self._epoch_batch_count
        else:
            return 0.0

    def reset_epoch_stats(self):
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def log_states(self, sw, tag_prefix, global_step):
        sw.add_scalar(tag=tag_prefix + '_ema_iou', value=self._ema_iou, global_step=global_step)
        sw.add_scalar(tag=tag_prefix + '_iou_thresh', value=self._iou_thresh, global_step=global_step)

    @property
    def iou_thresh(self):
        return self._iou_thresh


def _compute_iou(pred_mask, gt_mask, ignore_mask=None, keep_ignore=False):
    if ignore_mask is not None:
        pred_mask = torch.where(ignore_mask, torch.zeros_like(pred_mask), pred_mask)

    reduction_dims = misc.get_dims_with_exclusion(gt_mask.dim(), 0)
    union = torch.mean((pred_mask | gt_mask).float(), dim=reduction_dims).detach().cpu().numpy()
    intersection = torch.mean((pred_mask & gt_mask).float(), dim=reduction_dims).detach().cpu().numpy()
    nonzero = union > 0

    iou = intersection[nonzero] / union[nonzero]
    if not keep_ignore:
        return iou
    else:
        result = np.full_like(intersection, -1)
        result[nonzero] = iou
        return result
