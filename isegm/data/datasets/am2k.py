from pathlib import Path
import numpy as np
import cv2
import torch
import albumentations as A
from torch.utils.data import DataLoader
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.data.augmentation import RandomEdgeNoise, JitterContourEdge, RandomHoleDrop

class AM2KTrimapDataset(ISDataset):
    def __init__(self, dataset_path, split='train', crop_size=(448, 448), do_aug=False, **kwargs):
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.crop_size = crop_size
        self.do_aug = do_aug

        self.image_dir = self.dataset_path / 'original'
        self.mask_dir = self.dataset_path / 'mask'
        self.trimap_dir = self.dataset_path / 'trimap'
        self.dataset_samples = sorted([p.stem for p in self.mask_dir.glob('*.png')])

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
        if image is None:
            raise FileNotFoundError(f"Image not found: {sample_id}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        alpha = cv2.imread(str(self.mask_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            raise FileNotFoundError(f"Mask not found: {sample_id}")
        seg_mask = (alpha >= 30).astype(np.uint8)

        trimap = cv2.imread(str(self.trimap_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        if trimap is None:
            raise FileNotFoundError(f"Trimap not found: {sample_id}")
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

        geom_out = self.geom_aug(
            image=sample.image,
            seg_mask=sample._encoded_masks.squeeze(),
            gt_mask=sample.gt_mask
        )
        image = geom_out['image']
        seg_mask = geom_out['seg_mask']
        gt_mask = geom_out['gt_mask']

        image = self.color_aug(image=image)['image']

        for aug_fn in self.seg_aug:
            seg_mask = aug_fn(seg_mask=seg_mask)['seg_mask']
        seg_mask = (seg_mask > 0).astype(np.uint8)

        sample.image = image
        sample._encoded_masks = seg_mask[:, :, None]
        sample.gt_mask = gt_mask

        return sample

    def __getitem__(self, index):
        sample = self.get_sample(index)
        sample = self.augment_sample(sample)
        seg_mask = np.squeeze(sample._encoded_masks)
        assert seg_mask.ndim == 2, f"[BUG] seg_mask should be 2D, got shape: {seg_mask.shape}"

        return {
            'images': self.to_tensor(sample.image),
            'seg_mask': torch.from_numpy(seg_mask).unsqueeze(0).float(),
            'instances': torch.from_numpy(sample.gt_mask).long()
        }

        
if __name__ == '__main__':
    dataset = AM2KTrimapDataset(
        dataset_path='/home/work/SegTrimap/datasets/AM-2k',
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

    # 시각화 저장
    save_dir = Path('./aug_test_outputs_am2k')
    save_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[0]

    # image
    image = sample['images'].permute(1, 2, 0).numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / 'image.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # seg_mask
    seg_mask = sample['seg_mask'].squeeze().numpy()
    cv2.imwrite(str(save_dir / 'seg_mask.png'), (seg_mask * 255).astype(np.uint8))

    # instances
    instances = sample['instances'].numpy().astype(np.uint8)
    trimap_vis = np.zeros_like(instances, dtype=np.uint8)
    trimap_vis[instances == 1] = 127
    trimap_vis[instances == 2] = 255
    cv2.imwrite(str(save_dir / 'instances.png'), trimap_vis)


# from pathlib import Path
# import numpy as np
# import cv2
# import torch
# from isegm.data.base import ISDataset
# from isegm.data.sample import DSample

# class AM2KTrimapDataset(ISDataset):
#     def __init__(self, dataset_path, split='train', **kwargs):
#         super().__init__(**kwargs)
#         self.dataset_path = Path(dataset_path)
#         self.split = split
#         self.image_dir = self.dataset_path / 'original'
#         self.mask_dir = self.dataset_path / 'mask'
#         self.trimap_dir = self.dataset_path / 'trimap'

#         # mask 기준으로 sample ID 수집 (파일명 확장자 제거)
#         self.dataset_samples = sorted([p.stem for p in self.mask_dir.glob('*.png')])

#     def get_sample(self, index) -> DSample:
#         sample_id = self.dataset_samples[index]

#         # 1. Load original image (.jpg)
#         image = cv2.imread(str(self.image_dir / f'{sample_id}.jpg'))
#         if image is None:
#             raise FileNotFoundError(f"Original image not found: {self.image_dir / f'{sample_id}.jpg'}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # 2. Load seg_mask from alpha matte (.png)
#         alpha = cv2.imread(str(self.mask_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
#         if alpha is None:
#             raise FileNotFoundError(f"Segmentation mask not found: {self.mask_dir / f'{sample_id}.png'}")
#         seg_mask = (alpha >= 30).astype(np.uint8)[:, :, None]

#         # 3. Load GT trimap (.png)
#         trimap = cv2.imread(str(self.trimap_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
#         if trimap is None:
#             raise FileNotFoundError(f"Trimap not found: {self.trimap_dir / f'{sample_id}.png'}")

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

#         return sample

#     def __getitem__(self, index):
#         sample = self.get_sample(index)
#         sample = self.augment_sample(sample)

#         return {
#             'images': self.to_tensor(sample.image),                       # [3, H, W]
#             'seg_mask': self.to_tensor(sample._encoded_masks).float(),   # [1, H, W]
#             'instances': torch.from_numpy(sample.gt_mask).long()         # [H, W]
#         }