from pathlib import Path
import numpy as np
import cv2
import torch
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

class AIM500TrimapDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_dir = self.dataset_path / 'original'
        self.mask_dir = self.dataset_path / 'mask'
        self.trimap_dir = self.dataset_path / 'trimap'
        self.dataset_samples = sorted([p.stem for p in self.image_dir.glob('*.jpg')])

    def get_sample(self, index) -> DSample:
        sample_id = self.dataset_samples[index]

        # 1. Load image
        image = cv2.imread(str(self.image_dir / f'{sample_id}.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load and binarize seg_mask from alpha matte
        alpha = cv2.imread(str(self.mask_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        seg_mask = (alpha >= 30).astype(np.uint8)  # [H, W]
        seg_mask = seg_mask[:, :, None]             # [H, W, 1]

        # 3. Load GT trimap
        trimap = cv2.imread(str(self.trimap_dir / f'{sample_id}.png'), cv2.IMREAD_GRAYSCALE)
        trimap_mapped = np.zeros_like(trimap, dtype=np.int32)
        trimap_mapped[trimap == 0] = 0
        trimap_mapped[trimap == 128] = 1
        trimap_mapped[trimap == 255] = 2

        sample = DSample(
            image=image,
            encoded_masks=seg_mask,
            objects_ids=[1],
            sample_id=index,
            gt_mask=trimap_mapped
        )
        sample.gt_mask = trimap_mapped

        return sample

    def __getitem__(self, index):
        sample = self.get_sample(index)
        sample = self.augment_sample(sample)

        return {
            'images': self.to_tensor(sample.image),                       # [3, H, W]
            'seg_mask': self.to_tensor(sample._encoded_masks).float(),   # [1, H, W]
            'instances': torch.from_numpy(sample.gt_mask).long()         # [H, W]
        }
        