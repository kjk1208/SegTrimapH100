import os
import sys
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from albumentations import Compose, PadIfNeeded, LongestMaxSize
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from isegm.model.metrics import PerClassIoU, MultiClassIoU
from isegm.model.is_trimap_plaintvit_model_noposembed import NoPosEmbedTrimapPlainVitModel
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference_no_pos_embed
from isegm.data.datasets.composition import COMPOSITIONTrimapDataset
from isegm.data.datasets.p3m10k import P3M10KTrimapDataset
from isegm.data.datasets.aim500 import AIM500TrimapDataset
from isegm.data.datasets.am200 import AM200TrimapDataset


def build_model(device):
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

    neck_params = dict(
        in_dim=1280,
        out_dims=[240, 480, 960, 1920],
    )

    head_params = dict(
        in_channels=[240, 480, 960, 1920],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=3,
        loss_decode=torch.nn.CrossEntropyLoss(),
        align_corners=False,
        upsample='x4',
        channels=64,
    )

    model = NoPosEmbedTrimapPlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=False,
    )

    model.to(device)
    return model


def evaluate_model(model, dataset, dataset_name, device, log_file, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    per_class_iou = PerClassIoU()
    mean_iou = MultiClassIoU()

    model.eval()
    with torch.no_grad():
        for sample in tqdm(loader, desc=f"\nEvaluating {dataset_name}"):
            images = sample['images'].to(device)
            seg_masks = sample['seg_mask'].to(device)
            trimap_labels = sample['instances'].to(device)

            output = model(images, seg_masks)
            trimap_preds = output['instances']
            trimap_preds = F.interpolate(trimap_preds, size=trimap_labels.shape[-2:], mode='bilinear', align_corners=False)
            pred_masks = torch.argmax(trimap_preds, dim=1)

            for b in range(pred_masks.shape[0]):
                per_class_iou.update(pred_masks[b], trimap_labels[b])
                mean_iou.update(pred_masks[b], trimap_labels[b])

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--device', default='cpu', type=str, help='Device to use (cpu or cuda)')
    parser.add_argument('--batch_size', default=4, type=int, help='Evaluation batch size')
    parser.add_argument('--log_dir', default='./evaluation/eval_logs', type=str, help='Directory to save evaluation logs')
    parser.add_argument('--composition_path', default='./datasets/Composition-1k-testset', type=str, help='Path to composition')
    parser.add_argument('--p3m500_path', default='./datasets/2.P3M-10k/P3M-10k/validation/P3M-500-NP', type=str, help='Path to p3m500')
    parser.add_argument('--aim500_path', default='./datasets/AIM-500', type=str, help='Path to aim500')
    parser.add_argument('--am200_path', default='./datasets/AM-200', type=str, help='Path to am200')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d_%H%M%S_eval_log.txt'))

    with open(log_file, 'a') as f:
        f.write(f"\nEval_weight_path: {args.weight_path}\n")

    model = build_model(device=args.device)
    checkpoint = torch.load(args.weight_path, map_location=args.device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Checkpoint does not contain 'state_dict'")
    interpolate_pos_embed_inference_no_pos_embed(model.backbone, infer_img_size=(448, 448), device=args.device)
    model.eval()

    eval_augmentator = Compose([
        LongestMaxSize(max_size=448),
        PadIfNeeded(min_height=448, min_width=448, border_mode=0),
    ])

    evaldataset1 = COMPOSITIONTrimapDataset(args.composition_path, augmentator=eval_augmentator, do_aug=False)
    evaldataset2 = P3M10KTrimapDataset(args.p3m500_path, augmentator=eval_augmentator, do_aug=False)
    evaldataset3 = AIM500TrimapDataset(args.aim500_path, augmentator=eval_augmentator, do_aug=False)
    evaldataset4 = AM200TrimapDataset(args.am200_path, augmentator=eval_augmentator, do_aug=False)

    evaluate_model(model, evaldataset1, 'Composition-1K', args.device, log_file, batch_size=args.batch_size)
    evaluate_model(model, evaldataset2, 'P3M-500-P', args.device, log_file, batch_size=args.batch_size)
    evaluate_model(model, evaldataset3, 'AIM-500', args.device, log_file, batch_size=args.batch_size)
    evaluate_model(model, evaldataset4, 'AM-200', args.device, log_file, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
    
