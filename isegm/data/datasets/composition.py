from pathlib import Path
import numpy as np
import cv2
import torch
import albumentations as A

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.data.augmentation import RandomEdgeNoise, JitterContourEdge, RandomHoleDrop


class COMPOSITIONTrimapDataset(ISDataset):
    def __init__(self, dataset_path, split='train', crop_size=(448, 448), do_aug=True, **kwargs):
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.crop_size = crop_size
        self.do_aug = do_aug

        self.image_dir = self.dataset_path / 'original'
        self.mask_dir = self.dataset_path / 'mask'
        self.trimap_dir = self.dataset_path / 'trimap'
        self.dataset_samples = sorted([p.stem for p in self.image_dir.glob('*.jpg')])

        if self.do_aug:
            self.replay_aug = A.ReplayCompose(
                transforms=[
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
                    A.LongestMaxSize(max_size=crop_size[0]),
                    A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0)
                ],
                additional_targets={
                    'seg_mask': 'mask',
                    'gt_mask': 'mask'
                }
            )
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
            encoded_masks=seg_mask[:, :, None],  # [H, W, 1]
            objects_ids=[1],
            sample_id=index,
            gt_mask=trimap_mapped
        )

    def augment_sample(self, sample: DSample) -> DSample:
        if not self.do_aug:
            return sample

        seg_mask = sample._encoded_masks.squeeze()
        aug = self.replay_aug(
            image=sample.image,
            seg_mask=seg_mask,
            gt_mask=sample.gt_mask
        )

        sample.image = aug['image']
        seg_mask = aug['seg_mask']
        sample.gt_mask = aug['gt_mask']

        for aug_fn in self.seg_aug:
            seg_mask = aug_fn(seg_mask=seg_mask)['seg_mask']

        seg_mask = (seg_mask > 0).astype(np.uint8)
        sample._encoded_masks = seg_mask[:, :, None]
        return sample

    def __getitem__(self, index):
        sample = self.get_sample(index)
        sample = self.augment_sample(sample)

        seg_mask = sample._encoded_masks
        seg_mask = np.squeeze(seg_mask)
        assert seg_mask.ndim == 2, f"[BUG] seg_mask should be 2D, got shape: {seg_mask.shape}"

        return {
            'images': self.to_tensor(sample.image),
            'seg_mask': torch.from_numpy(seg_mask).unsqueeze(0).float(),  # [1, H, W]
            'instances': torch.from_numpy(sample.gt_mask).long()
        }

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset_path = '/home/work/SegTrimap/datasets/Composition-431k-png'

    dataset = COMPOSITIONTrimapDataset(
        dataset_path=dataset_path,
        split='train',
        crop_size=(448, 448),
        do_aug=True
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print("images:", batch['images'].shape)       # [B, 3, H, W]
        print("seg_mask:", batch['seg_mask'].shape)   # [B, 1, H, W]
        print("instances:", batch['instances'].shape) # [B, H, W]
        break

    import os
    import numpy as np
    import cv2
    from torchvision.utils import save_image
    from pathlib import Path

    # 저장할 디렉토리 설정
    save_dir = Path('./aug_test_outputs')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 하나의 sample 추출
    sample = dataset[1]

    # image 저장
    image = sample['images'].permute(1, 2, 0).numpy()  # [H, W, 3]
    image = (image * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / 'image.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # seg_mask 저장
    seg_mask = sample['seg_mask'].squeeze().numpy()
    print("seg_mask min/max:", seg_mask.min(), seg_mask.max(), "dtype:", seg_mask.dtype)
    print("unique:", np.unique(seg_mask))
    seg_mask_vis = (seg_mask * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / 'seg_mask.png'), seg_mask_vis)

    # instances 저장 (값은 있지만 너무 어두움 → 시각화용 변환)
    instances = sample['instances'].numpy().astype(np.uint8)
    print("instances unique:", np.unique(instances))
    trimap_vis = np.zeros_like(instances, dtype=np.uint8)
    trimap_vis[instances == 1] = 127
    trimap_vis[instances == 2] = 255
    cv2.imwrite(str(save_dir / 'instances.png'), trimap_vis)


# class COMPOSITIONTrimapDataset(ISDataset):
#     def __init__(self, dataset_path, split='train', **kwargs):
#         super().__init__(**kwargs)
#         self.dataset_path = Path(dataset_path)
#         self.split = split
#         self.image_dir = self.dataset_path / 'original'
#         self.mask_dir = self.dataset_path / 'mask'
#         self.trimap_dir = self.dataset_path / 'trimap'
#         self.dataset_samples = sorted([p.stem for p in self.image_dir.glob('*.jpg')])

#     def get_sample(self, index) -> DSample:
#         sample_id = self.dataset_samples[index]

#         # 1. Load image
#         image = cv2.imread(str(self.image_dir / f'{sample_id}.jpg'))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # 2. Load and binarize seg_mask from alpha matte
#         alpha = cv2.imread(str(self.mask_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
#         seg_mask = (alpha >= 30).astype(np.uint8)  # [H, W]
#         seg_mask = seg_mask[:, :, None]             # [H, W, 1]

#         # 3. Load GT trimap
#         trimap = cv2.imread(str(self.trimap_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
#         trimap_mapped = np.zeros_like(trimap, dtype=np.int32)
#         trimap_mapped[trimap <= 15] = 0        
#         trimap_mapped[(trimap >= 15) & (trimap <= 229)] = 1
#         trimap_mapped[trimap >= 230] = 2

#         sample = DSample(
#             image=image,
#             encoded_masks=seg_mask,
#             objects_ids=[1],
#             sample_id=index,
#             gt_mask=trimap_mapped
#         )
#         sample.gt_mask = trimap_mapped

#         return sample

#     def __getitem__(self, index):
#         sample = self.get_sample(index)
#         sample = self.augment_sample(sample)

#         return {
#             'images': self.to_tensor(sample.image),                       # [3, H, W]
#             'seg_mask': self.to_tensor(sample._encoded_masks).float(),   # [1, H, W]
#             'instances': torch.from_numpy(sample.gt_mask).long()         # [H, W]
#         }