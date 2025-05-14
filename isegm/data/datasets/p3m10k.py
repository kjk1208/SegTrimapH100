import os
import numpy as np
import cv2
import torch
import albumentations as A
from pathlib import Path
from torch.utils.data import DataLoader

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.data.augmentation import RandomEdgeNoise, JitterContourEdge, RandomHoleDrop


class P3M10KTrimapDataset(ISDataset):
    def __init__(self, dataset_path, split='train', crop_size=(448, 448), do_aug=False, **kwargs):
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
            self.geom_aug = A.ReplayCompose([
                A.HorizontalFlip(p=0.5),
                A.LongestMaxSize(max_size=crop_size[0]),
                A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0)
            ], additional_targets={
                'seg_mask': 'mask',
                'gt_mask': 'mask'
            })

            self.color_aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25),
                                           contrast_limit=(-0.15, 0.4), p=0.75),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
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

        seg_mask = sample._encoded_masks.squeeze()

        aug = self.geom_aug(
            image=sample.image,
            seg_mask=seg_mask,
            gt_mask=sample.gt_mask
        )

        sample.image = aug['image']
        seg_mask = aug['seg_mask']
        sample.gt_mask = aug['gt_mask']

        sample.image = self.color_aug(image=sample.image)['image']

        for aug_fn in self.seg_aug:
            seg_mask = aug_fn(seg_mask=seg_mask)['seg_mask']

        seg_mask = (seg_mask > 0).astype(np.uint8)
        sample._encoded_masks = seg_mask[:, :, None]

        return sample

    def __getitem__(self, index):
        sample = self.get_sample(index)
        sample = self.augment_sample(sample)

        seg_mask = np.squeeze(sample._encoded_masks)
        assert seg_mask.ndim == 2

        return {
            'images': self.to_tensor(sample.image),
            'seg_mask': torch.from_numpy(seg_mask).unsqueeze(0).float(),
            'instances': torch.from_numpy(sample.gt_mask).long()
        }



if __name__ == '__main__':
    dataset = P3M10KTrimapDataset(
        dataset_path='/home/work/SegTrimap/datasets/P3M-10k',
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

    save_dir = Path('./aug_test_outputs_p3m10k')
    save_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[0]
    image = sample['images'].permute(1, 2, 0).numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / 'image.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    seg_mask = sample['seg_mask'].squeeze().numpy()
    cv2.imwrite(str(save_dir / 'seg_mask.png'), (seg_mask * 255).astype(np.uint8))

    trimap = sample['instances'].numpy()
    trimap_vis = np.zeros_like(trimap, dtype=np.uint8)
    trimap_vis[trimap == 1] = 127
    trimap_vis[trimap == 2] = 255
    cv2.imwrite(str(save_dir / 'trimap.png'), trimap_vis)

# from pathlib import Path
# import numpy as np
# import cv2
# import torch
# from isegm.data.base import ISDataset
# from isegm.data.sample import DSample

# class P3M10KTrimapDataset(ISDataset):
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
        