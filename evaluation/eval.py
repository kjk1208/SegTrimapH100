import torch
from torch.utils.data import DataLoader
import os
from datetime import datetime
from easydict import EasyDict as edict
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath('..')) 

from isegm.model.metrics import PerClassIoU, MultiClassIoU
from isegm.model.is_trimap_plaintvit_model import TrimapPlainVitModel
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
from tqdm import tqdm

# 1. 평가 함수
def evaluate_model(model, dataset, dataset_name, device, log_file, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    per_class_iou = PerClassIoU()
    mean_iou = MultiClassIoU()

    model.eval()
    with torch.no_grad():
        for sample in tqdm(loader, desc=f"\nEvaluating {dataset_name}"):
            images = sample['images'].to(device)         # [B, 3, H, W]
            seg_masks = sample['seg_mask'].to(device)    # [B, 1, H, W]
            trimap_labels = sample['instances'].to(device)  # [B, H, W]

            output = model(images, seg_masks)
            trimap_preds = output['instances']           # [B, 3, H_pred, W_pred]

            # Interpolate predictions to match ground truth size
            trimap_preds = F.interpolate(trimap_preds, size=trimap_labels.shape[-2:], mode='bilinear', align_corners=False)

            # Convert to class predictions
            pred_masks = torch.argmax(trimap_preds, dim=1)  # [B, H, W]

            for b in range(pred_masks.shape[0]):
                pred = pred_masks[b]
                gt = trimap_labels[b]
                per_class_iou.update(pred, gt)
                mean_iou.update(pred, gt)

    per_cls = per_class_iou.get_epoch_value()
    mean = mean_iou.get_epoch_value()

    print(f"\n=== {dataset_name} ===")
    print("Dataset Size:", len(dataset))
    print("Per-Class IoU:", per_cls)
    print("Mean IoU:", mean)

    with open(log_file, 'a') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Samples: {len(dataset)}\n")
        f.write(f"BG IoU\tUnknown IoU\tFG IoU\tMean IoU\n")
        f.write(f"{per_cls['bg']:.4f}\t{per_cls['unknown']:.4f}\t{per_cls['fg']:.4f}\t{mean:.4f}\n\n")

    per_class_iou.reset_epoch_stats()
    mean_iou.reset_epoch_stats()

# def evaluate_model(model, dataset, dataset_name, device, log_file):
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)

#     per_class_iou = PerClassIoU()
#     mean_iou = MultiClassIoU()

#     model.eval()
#     with torch.no_grad():
#         for sample in tqdm(loader, desc=f"\nEvaluating {dataset_name}"):
#             image = sample['images'].to(device)           # [B, 3, H, W]
#             seg_mask = sample['seg_mask'].to(device)      # [B, 1, H, W]
#             trimap_label = sample['instances'].to(device) # [B, H, W]

#             output = model(image, seg_mask)
#             trimap_pred = output['instances'][0]  # [3, H, W]

#             # Resize pred to match GT
#             trimap_pred = F.interpolate(trimap_pred.unsqueeze(0), size=trimap_label.shape[-2:], mode='bilinear', align_corners=False)[0]

#             # Convert to class indices
#             pred_mask = torch.argmax(trimap_pred, dim=0)
#             gt_mask = trimap_label.squeeze(0)

#             per_class_iou.update(pred_mask, gt_mask)
#             mean_iou.update(pred_mask, gt_mask)

#     per_cls = per_class_iou.get_epoch_value()
#     mean = mean_iou.get_epoch_value()

#     print(f"\n=== {dataset_name} ===")
#     print("Dataset Size:", len(dataset))
#     print("Per-Class IoU:", per_cls)
#     print("Mean IoU:", mean)

#     with open(log_file, 'a') as f:
#         f.write(f"Dataset: {dataset_name}\n")
#         f.write(f"Samples: {len(dataset)}\n")
#         f.write(f"BG IoU\tUnknown IoU\tFG IoU\tMean IoU\n")
#         f.write(f"{per_cls['bg']:.4f}\t{per_cls['unknown']:.4f}\t{per_cls['fg']:.4f}\t{mean:.4f}\n")
#         f.write(f"\n")

#     per_class_iou.reset_epoch_stats()
#     mean_iou.reset_epoch_stats()


# 2. 모델 생성 함수
def build_model(device="cpu"):
    backbone_params = dict(
        img_size=(448, 448),
        patch_size=(14,14),
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 1280,
        out_dims = [240, 480, 960, 1920],
    )

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
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=False,
    )

    # model.backbone.init_weights_from_pretrained(weight_path)
    model.to(device)
    return model


# 3. config 및 로그 파일 경로 정의
log_dir = './eval_logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S_eval_log.txt'))

#eval_weight_path = '/home/kjk/matting/SegTrimap/output/loss/aim500_am2k_trimap_vit_huge448_nfl_dtloss/000/checkpoints/last_checkpoint.pth'
#eval_weight_path = '/home/kjk/matting/SegTrimap/output/iter_mask/aim500_trimap_vit_huge448/008/checkpoints/last_checkpoint.pth'
eval_weight_path = '/home/kjk/matting/SegTrimap/output/loss/composition_p3m10k_am2k_trimap_vit_huge448_focalloss_dtloss/001/checkpoints/last_checkpoint.pth'
#eval_weight_path = '/home/kjk/matting/SegTrimap/output/loss/composition_p3m10k_am2k_trimap_vit_huge448_focalloss_dtloss/001/checkpoints/010.pth'
#eval_weight_path = '/home/kjk/matting/SegTrimap/output/loss/composition_p3m10k_am2k_trimap_vit_huge448_focalloss_dtloss/001/checkpoints/020.pth'

with open(log_file, 'a') as f:
    f.write(f"\nEval_weight_path: {eval_weight_path}\n")

now_device = "cpu"

# 4. 모델 생성 및 weight 로딩
model = build_model(device=now_device)
checkpoint = torch.load(eval_weight_path, map_location=now_device)
model.load_state_dict(checkpoint['state_dict'])
interpolate_pos_embed_inference(model.backbone, infer_img_size=(448, 448), device='cpu')
model.eval()


# 5. 데이터셋 불러오기
from isegm.data.datasets.composition import COMPOSITIONTrimapDataset
from isegm.data.datasets.p3m10k import P3M10KTrimapDataset
from isegm.data.datasets.aim500 import AIM500TrimapDataset
from isegm.data.datasets.am200 import AM200TrimapDataset
from albumentations import Compose, PadIfNeeded, LongestMaxSize

eval_augmentator = Compose([
    LongestMaxSize(max_size=448),
    PadIfNeeded(min_height=448, min_width=448, border_mode=0),
])

evaldataset1 = COMPOSITIONTrimapDataset('/home/kjk/matting/SegTrimap/datasets/Seg2TrimapDataset/Composition-1k-testset', augmentator=eval_augmentator)
evaldataset2 = P3M10KTrimapDataset('/home/kjk/matting/SegTrimap/datasets/2.P3M-10k/P3M-10k/validation/P3M-500-NP', augmentator=eval_augmentator)
evaldataset3 = AIM500TrimapDataset('/home/kjk/matting/SegTrimap/datasets/Seg2TrimapDataset/AIM-500', augmentator=eval_augmentator)
evaldataset4 = AM200TrimapDataset('/home/kjk/matting/SegTrimap/datasets/Seg2TrimapDataset/AM-200', augmentator=eval_augmentator)


# 6. 평가 수행
evaluate_model(model, evaldataset1, 'Composition-1K', now_device, log_file, batch_size=20)
evaluate_model(model, evaldataset2, 'P3M-500-P', now_device, log_file, batch_size=20)
evaluate_model(model, evaldataset3, 'AIM-500', now_device, log_file, batch_size=20)
evaluate_model(model, evaldataset4, 'AM-200', now_device, log_file, batch_size=20)