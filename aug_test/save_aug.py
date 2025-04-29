import cv2
from pathlib import Path
import albumentations as A
from albumentations import LongestMaxSize, PadIfNeeded, HorizontalFlip, RandomBrightnessContrast, RGBShift

# 저장 경로
save_root = Path('./aug_examples')
save_root.mkdir(exist_ok=True)

# 원본 이미지 경로
input_img_path = '/home/kjk/matting/SegTrimap/datasets/Seg2TrimapDataset/AM-2k/original/m_0b09726c.jpg'

# 원본 이미지 로드
image = cv2.imread(str(input_img_path))
assert image is not None, f"Cannot read image at {input_img_path}"

# 1. 원본 이미지 저장
cv2.imwrite(str(save_root / '01_original.jpg'), image)

# 2. Color Jittering 적용
color_transform = A.Compose([
    RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=1.0),
    RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0)
])
colored = color_transform(image=image)['image']
cv2.imwrite(str(save_root / '02_colored.jpg'), colored)

# 3. Flip 적용
flip_transform = A.Compose([
    HorizontalFlip(p=1.0)
])
flipped = flip_transform(image=colored)['image']
cv2.imwrite(str(save_root / '03_colored_flipped.jpg'), flipped)

# 4. Scaling 적용 (LongestMaxSize + PadIfNeeded)
scaling_transform = A.Compose([
    LongestMaxSize(max_size=448),
    PadIfNeeded(min_height=448, min_width=448, border_mode=0)
])
scaled = scaling_transform(image=flipped)['image']
cv2.imwrite(str(save_root / '04_colored_flipped_scaled.jpg'), scaled)