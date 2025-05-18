from pathlib import Path
import numpy as np
import cv2
import torch
import albumentations as A
from torch.utils.data import DataLoader
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.data.augmentation import RandomEdgeNoise, JitterContourEdge, RandomHoleDrop

class AIM500TrimapDataset(ISDataset):
    def __init__(self, dataset_path, split='train', crop_size=(448, 448), do_aug=False, **kwargs):
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.crop_size = crop_size
        self.do_aug = do_aug
        self.eval_augmentator = kwargs.get('augmentator', None)

        self.image_dir = self.dataset_path / 'original'
        self.mask_dir = self.dataset_path / 'mask'
        self.trimap_dir = self.dataset_path / 'trimap'
        self.dataset_samples = sorted([p.stem for p in self.image_dir.glob('*.jpg')])

        if self.do_aug:
            self.geom_aug = A.ReplayCompose([
                A.HorizontalFlip(p=0.5),
                A.LongestMaxSize(max_size=crop_size[0]),
                A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0)
            ], additional_targets={'seg_mask': 'mask', 'gt_mask': 'mask'})

            self.color_aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
            ])

            self.seg_aug = [
                RandomEdgeNoise(erosion_prob=0.2),
                JitterContourEdge(prob=0.5),
                RandomHoleDrop(drop_prob=0.2, max_hole_area_ratio=0.05)
            ]

    def get_sample(self, index) -> DSample:
        sample_id = self.dataset_samples[index]
        image = cv2.imread(str(self.image_dir / f'{sample_id}.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        alpha = cv2.imread(str(self.mask_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        seg_mask = (alpha >= 30).astype(np.uint8)

        trimap = cv2.imread(str(self.trimap_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        trimap_mapped = np.zeros_like(trimap, dtype=np.int32)
        trimap_mapped[trimap <= 15] = 0        
        trimap_mapped[(trimap >= 15) & (trimap <= 229)] = 1
        trimap_mapped[trimap >= 230] = 2

        return DSample(
            image=image,
            encoded_masks=seg_mask[:, :, None],
            objects_ids=[1],
            sample_id=index,
            gt_mask=trimap_mapped
        )

    def augment_sample(self, sample: DSample) -> DSample:   
        if not self.do_aug:
            return sample     
        image = self.color_aug(image=sample.image)['image']

        # 2. geom_aug는 image와 mask를 함께 처리
        geom_out = self.geom_aug(
            image=image,
            seg_mask=sample._encoded_masks.squeeze(),
            gt_mask=sample.gt_mask
        )
        image = geom_out['image']
        seg_mask = geom_out['seg_mask']
        gt_mask = geom_out['gt_mask']

        for aug_fn in self.seg_aug:
            seg_mask = aug_fn(seg_mask=seg_mask)['seg_mask']
        seg_mask = (seg_mask > 0).astype(np.uint8)

        sample.image = image
        sample._encoded_masks = seg_mask[:, :, None]
        sample.gt_mask = gt_mask

        return sample

    def apply_eval_augmentator(self, sample: DSample) -> DSample:
        if self.eval_augmentator is None:
            return sample

        out = self.eval_augmentator(
            image=sample.image,
            masks=[sample._encoded_masks.squeeze(), sample.gt_mask]
        )
        sample.image = out['image']
        seg_mask, gt_mask = out['masks']
        sample._encoded_masks = (seg_mask > 0).astype(np.uint8)[:, :, None]
        sample.gt_mask = gt_mask
        return sample

    def __getitem__(self, index):
        sample = self.get_sample(index)
        if self.do_aug:
            sample = self.augment_sample(sample)
        elif self.eval_augmentator is not None:
            sample = self.apply_eval_augmentator(sample)

        seg_mask = sample._encoded_masks.squeeze()
        assert seg_mask.ndim == 2, f"[BUG] seg_mask should be 2D, got shape: {seg_mask.shape}"

        return {
            'images': self.to_tensor(sample.image),
            'seg_mask': torch.from_numpy(seg_mask).unsqueeze(0).float(),
            'instances': torch.from_numpy(sample.gt_mask).long()
        }


if __name__ == '__main__':
    dataset_path = '/home/work/SegTrimap/datasets/AIM-500'

    # 학습용 테스트
    dataset = AIM500TrimapDataset(
        dataset_path=dataset_path,
        split='train',
        crop_size=(448, 448),
        do_aug=True
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print("images:", batch['images'].shape)
        print("seg_mask:", batch['seg_mask'].shape)
        print("instances:", batch['instances'].shape)
        break

    save_dir = Path('./aug_test_outputs_aim500')
    save_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[2]

    image = sample['images'].permute(1, 2, 0).numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / 'image.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    seg_mask = sample['seg_mask'].squeeze().numpy()
    cv2.imwrite(str(save_dir / 'seg_mask.png'), (seg_mask * 255).astype(np.uint8))

    instances = sample['instances'].numpy()
    trimap_vis = np.zeros_like(instances, dtype=np.uint8)
    trimap_vis[instances == 1] = 127
    trimap_vis[instances == 2] = 255
    cv2.imwrite(str(save_dir / 'instances.png'), trimap_vis)

    # ========================
    # Evaluation Aug 시각화
    # ========================
    eval_augmentator = A.Compose([
        A.LongestMaxSize(max_size=448),
        A.PadIfNeeded(min_height=448, min_width=448, border_mode=0),
    ])

    eval_dataset = AIM500TrimapDataset(
        dataset_path=dataset_path,
        split='val',
        crop_size=(448, 448),
        do_aug=False,
        augmentator=eval_augmentator
    )

    eval_save_dir = Path('./aug_test_outputs_aim500_eval')
    eval_save_dir.mkdir(parents=True, exist_ok=True)

    eval_sample = eval_dataset[2]

    image_eval = eval_sample['images'].permute(1, 2, 0).numpy()
    image_eval = (image_eval * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(eval_save_dir / 'image.png'), cv2.cvtColor(image_eval, cv2.COLOR_RGB2BGR))

    seg_mask_eval = eval_sample['seg_mask'].squeeze().numpy()
    cv2.imwrite(str(eval_save_dir / 'seg_mask.png'), (seg_mask_eval * 255).astype(np.uint8))

    instances_eval = eval_sample['instances'].numpy()
    trimap_vis_eval = np.zeros_like(instances_eval, dtype=np.uint8)
    trimap_vis_eval[instances_eval == 1] = 127
    trimap_vis_eval[instances_eval == 2] = 255
    cv2.imwrite(str(eval_save_dir / 'instances.png'), trimap_vis_eval)

    print("\n[INFO] Evaluation augment sample saved to:", str(eval_save_dir.resolve()))

    concat_save_path = save_dir / 'concat_visualization.png'

    # original image: [H, W, 3], RGB
    vis_image = image.copy()

    # seg_mask (binary) → 3채널 grayscale
    vis_seg = cv2.cvtColor((seg_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # trimap (0,127,255) → 3채널 grayscale
    vis_trimap = cv2.cvtColor(trimap_vis, cv2.COLOR_GRAY2BGR)

    # concat
    concat_img = np.concatenate([vis_image, vis_seg, vis_trimap], axis=1)
    cv2.imwrite(str(concat_save_path), cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Concatenated visualization saved to: {concat_save_path}")

    concat_eval_path = eval_save_dir / 'concat_visualization.png'

    # original image
    vis_image_eval = image_eval.copy()

    # seg_mask (binary) → 3채널 grayscale
    vis_seg_eval = cv2.cvtColor((seg_mask_eval * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # trimap (0,127,255) → 3채널 grayscale
    vis_trimap_eval = cv2.cvtColor(trimap_vis_eval, cv2.COLOR_GRAY2BGR)

    # concat
    concat_eval_img = np.concatenate([vis_image_eval, vis_seg_eval, vis_trimap_eval], axis=1)
    cv2.imwrite(str(concat_eval_path), cv2.cvtColor(concat_eval_img, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Concatenated evaluation image saved to: {concat_eval_path}")
