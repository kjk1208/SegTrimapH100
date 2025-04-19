import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import datetime

from isegm.data.datasets.aim500 import AIM500TrimapDataset
from isegm.model.is_trimap_plaintvit_model import TrimapPlainVitModel
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
from albumentations import Resize, Compose


def build_model():
    backbone_params = dict(
        img_size=(448, 448),
        patch_size=(14, 14),
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
    )
    neck_params = dict(in_dim=1280, out_dims=[240, 480, 960, 1920])
    head_params = dict(
        in_channels=[240, 480, 960, 1920],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=3,
        loss_decode=torch.nn.CrossEntropyLoss(),
        align_corners=False,
        upsample='x4',
        channels={'x1': 256, 'x2': 128, 'x4': 64}['x4'],
    )    

    model = TrimapPlainVitModel(
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=False,
        use_disks=False,
        with_prev_mask=False,
    )
    return model


def save_trimap(trimap_tensor, save_path):
    pred_mask = torch.argmax(trimap_tensor, dim=0).cpu().numpy()
    trimap_vis = np.zeros_like(pred_mask, dtype=np.uint8)
    trimap_vis[pred_mask == 0] = 0
    trimap_vis[pred_mask == 1] = 128
    trimap_vis[pred_mask == 2] = 255
    cv2.imwrite(save_path, trimap_vis)


def save_seg_mask(seg_tensor, save_path):
    seg = seg_tensor.squeeze().cpu().numpy()
    seg = (seg > 0.5).astype(np.uint8) * 255
    cv2.imwrite(save_path, seg)


def save_input_image(image_tensor, save_path):
    image = image_tensor.squeeze().cpu().numpy()
    image = np.transpose(image, (1, 2, 0)) * 255
    image = image.astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gt_trimap(gt_tensor, save_path):
    trimap = gt_tensor.squeeze().cpu().numpy()
    trimap_vis = np.zeros_like(trimap, dtype=np.uint8)
    trimap_vis[trimap == 0] = 0
    trimap_vis[trimap == 1] = 128
    trimap_vis[trimap == 2] = 255
    cv2.imwrite(save_path, trimap_vis)


def main():
    ckpt_path = './output/iter_mask/aim500_trimap_vit_huge448/005/checkpoints/last_checkpoint.pth'
    data_root = './datasets/3.AIM-500'
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('./inference', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 모델 로딩
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    interpolate_pos_embed_inference(model.backbone, infer_img_size=(448, 448), device='cpu')
    model.eval()

    test_augmentator = Compose([
        Resize(448, 448)
    ])

    testset = AIM500TrimapDataset(
        dataset_path=data_root,
        split='val',
        augmentator=test_augmentator,
        epoch_len=-1
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader)):
            image = batch['images']         # [1, 3, H, W]
            seg_mask = batch['seg_mask']    # [1, 1, H, W]
            gt_trimap = batch['instances']  # [1, H, W]
            sample_id = testset.dataset_samples[i]

            output = model(image, seg_mask)
            pred = output['instances'][0]  # [3, H, W]
            
            # pred = torch.nn.functional.interpolate(pred.unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False)[0]

            save_trimap(pred, os.path.join(save_dir, f'{sample_id}_pred_trimap.png'))
            save_gt_trimap(gt_trimap[0], os.path.join(save_dir, f'{sample_id}_gt_trimap.png'))
            save_seg_mask(seg_mask[0], os.path.join(save_dir, f'{sample_id}_seg_mask.png'))
            save_input_image(image[0], os.path.join(save_dir, f'{sample_id}_image.png'))


if __name__ == '__main__':
    main()
