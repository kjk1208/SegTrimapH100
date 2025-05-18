from isegm.model.losses import NormalizedFocalLossSoftmax, UnknownRegionDTLoss
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class CECombinedLoss(nn.Module):
    def __init__(self, ce_weight = 1.0, focal_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.focal = NormalizedFocalLossSoftmax(gamma=2.0, debug_print=False)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_focal = self.focal(pred, target)        
        return self.ce_weight * loss_ce + self.focal_weight * loss_focal, {
            'ce_loss': loss_ce.detach(),
            'focal_loss': loss_focal.detach(),            
        }