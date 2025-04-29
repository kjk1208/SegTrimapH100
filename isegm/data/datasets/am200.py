from pathlib import Path
import numpy as np
import cv2
import torch
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

class AM200TrimapDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_dir = self.dataset_path / 'original'
        self.mask_dir = self.dataset_path / 'mask'
        self.trimap_dir = self.dataset_path / 'trimap'

        # mask 기준으로 sample ID 수집 (파일명 확장자 제거)
        self.dataset_samples = sorted([p.stem for p in self.mask_dir.glob('*.png')])

    def get_sample(self, index) -> DSample:
        sample_id = self.dataset_samples[index]

        # 1. Load original image (.jpg)
        image = cv2.imread(str(self.image_dir / f'{sample_id}.jpg'))
        if image is None:
            raise FileNotFoundError(f"Original image not found: {self.image_dir / f'{sample_id}.jpg'}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load seg_mask from alpha matte (.png)
        alpha = cv2.imread(str(self.mask_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            raise FileNotFoundError(f"Segmentation mask not found: {self.mask_dir / f'{sample_id}.png'}")
        seg_mask = (alpha >= 30).astype(np.uint8)[:, :, None]

        # 3. Load GT trimap (.png)
        trimap = cv2.imread(str(self.trimap_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        if trimap is None:
            raise FileNotFoundError(f"Trimap not found: {self.trimap_dir / f'{sample_id}.png'}")

        trimap_mapped = np.zeros_like(trimap, dtype=np.int32)
        trimap_mapped[trimap <= 15] = 0        
        trimap_mapped[(trimap >= 15) & (trimap <= 229)] = 1
        trimap_mapped[trimap >= 230] = 2

        sample = DSample(
            image=image,
            encoded_masks=seg_mask,
            objects_ids=[1],
            sample_id=index,
            gt_mask=trimap_mapped
        )

        return sample

    def __getitem__(self, index):
        sample = self.get_sample(index)
        sample = self.augment_sample(sample)

        return {
            'images': self.to_tensor(sample.image),                       # [3, H, W]
            'seg_mask': self.to_tensor(sample._encoded_masks).float(),   # [1, H, W]
            'instances': torch.from_numpy(sample.gt_mask).long()         # [H, W]
        }