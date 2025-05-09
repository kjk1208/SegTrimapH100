import torch
import torch.nn as nn
from isegm.model.ops import BatchImageNormalize
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit_without_pos_embed import NoPosEmbedVisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

    def forward(self, x):
        return [
            self.down_4(x),
            self.down_8(x),
            self.down_16(x),
            self.down_32(x)
        ]


class NoPosEmbedTrimapPlainVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={},
        head_params={},
        random_split=False,
        norm_mean_std=([.485, .456, .406], [.229, .224, .225]),
        norm_radius=5,
        use_disks=False,
        with_prev_mask=False,
        cpu_dist_maps=False,
        **kwargs
    ):
        super().__init__(
            norm_radius=norm_radius,
            use_disks=use_disks,
            with_prev_mask=with_prev_mask,
            norm_mean_std=norm_mean_std,
            **kwargs
        )

        self.random_split = random_split

        self.patch_embed_mask = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            in_chans=1,  # segmentation mask (1 channel)
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = NoPosEmbedVisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

    def backbone_forward(self, image, mask_embed):
        backbone_features = self.backbone.forward_backbone(image, mask_embed, self.random_split)
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size
        backbone_features = backbone_features.transpose(1, 2).view(B, C, grid_size[0], grid_size[1])
        features = self.neck(backbone_features)
        return {'instances': self.head(features), 'instances_aux': None}

    def forward(self, image, seg_mask):
        """
        image:    [B, 3, H, W]
        seg_mask: [B, 1, H, W]
        Returns:  dict with key 'instances': [B, num_classes, H, W]
        """
        image = self.normalization(image)
        mask_embed = self.patch_embed_mask(seg_mask)
        return self.backbone_forward(image, mask_embed)