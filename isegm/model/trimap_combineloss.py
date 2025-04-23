from isegm.model.losses import NormalizedFocalLossSoftmax, UnknownRegionDTLoss
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=1.0, dt_weight=0.1):
        super().__init__()
        self.focal = NormalizedFocalLossSoftmax(gamma=2.0, debug_print=False)
        self.dt = UnknownRegionDTLoss(debug_print=False)
        self.focal_weight = focal_weight
        self.dt_weight = dt_weight

    def forward(self, pred, target):
        loss_focal = self.focal(pred, target)
        loss_dt = self.dt(pred, target)
        return self.focal_weight * loss_focal + self.dt_weight * loss_dt, {
            'focal_loss': loss_focal.detach(),
            'dt_loss': loss_dt.detach(),
        }