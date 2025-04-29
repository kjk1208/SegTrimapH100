import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import datetime

from isegm.data.datasets.aim500 import AIM500TrimapDataset
from isegm.model.is_trimap_plaintvit_model import TrimapPlainVitModel
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
from albumentations import Compose, LongestMaxSize, PadIfNeeded


def build_model(infer_img_size):
    backbone_params = dict(
        img_size=infer_img_size,
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
    seg = (seg > 1e-5).astype(np.uint8) * 255 
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


def main(args):
    infer_img_size = (args.infer_img_size, args.infer_img_size)
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)    
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(infer_img_size)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    interpolate_pos_embed_inference(model.backbone, infer_img_size=infer_img_size, device='cpu')
    model.eval()

    test_augmentator = Compose([
        LongestMaxSize(max_size=args.infer_img_size),
        PadIfNeeded(min_height=args.infer_img_size, min_width=args.infer_img_size, border_mode=0)
    ])

    testset = AIM500TrimapDataset(
        dataset_path=args.data_root,
        split='val',
        augmentator=test_augmentator,
        epoch_len=-1
    )
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader)):
            image = batch['images']          # [B, 3, H, W]
            seg_mask = batch['seg_mask']     # [B, 1, H, W]
            gt_trimap = batch['instances']   # [B, H, W]
            output = model(image, seg_mask)
            pred = output['instances']      # [B, 3, H, W]

            for j in range(image.size(0)):   # loop over batch
                sample_id = testset.dataset_samples[i * args.batch_size + j]
                save_trimap(pred[j], os.path.join(save_dir, f'{sample_id}_pred_trimap.png'))
                save_gt_trimap(gt_trimap[j], os.path.join(save_dir, f'{sample_id}_gt_trimap.png'))
                save_seg_mask(seg_mask[j], os.path.join(save_dir, f'{sample_id}_seg_mask.png'))
                save_input_image(image[j], os.path.join(save_dir, f'{sample_id}_image.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./output/loss/composition_p3m10k_am2k_trimap_vit_huge448_focalloss_dtloss/001/checkpoints/020.pth', help='Path to the model checkpoint')
    parser.add_argument('--data_root', type=str, default='./datasets/3.AIM-500', help='Path to dataset root folder')
    parser.add_argument('--save_dir', type=str, default='./inference', help='Path to save predictions')
    parser.add_argument('--infer_img_size', type=int, default=448, help='Input size for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    args = parser.parse_args()

    main(args)
    
    