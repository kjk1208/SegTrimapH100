from PIL import Image
from pathlib import Path

def find_png_with_iccp(directory):
    directory = Path(directory)
    for img_path in directory.rglob('*.png'):
        try:
            with Image.open(img_path) as img:
                if 'icc_profile' in img.info:
                    print(f"[FOUND] iCCP chunk in seg_mask : {img_path}")
        except Exception as e:
            print(f"[ERROR] Failed to open {img_path}: {e}")

def find_png_with_iccp_trimap(directory):
    directory = Path(directory)
    for img_path in directory.rglob('*.png'):
        try:
            with Image.open(img_path) as img:
                if 'icc_profile' in img.info:
                    print(f"[FOUND] iCCP chunk in trimap : {img_path}")
        except Exception as e:
            print(f"[ERROR] Failed to open {img_path}: {e}")

# 예시: Composition 데이터셋의 mask / trimap 디렉토리 확인
find_png_with_iccp('/home/work/SegTrimap/datasets/Composition-431k-png/mask')
find_png_with_iccp_trimap('/home/work/SegTrimap/datasets/Composition-431k-png/trimap')

def find_png_with_iccp_original(directory):
    directory = Path(directory)
    for img_path in directory.rglob('*.jpg'):
        try:
            with Image.open(img_path) as img:
                if 'icc_profile' in img.info:
                    print(f"[FOUND] iCCP chunk in original : {img_path}")
        except Exception as e:
            print(f"[ERROR] Failed to open {img_path}: {e}")

# original 디렉토리도 검사
find_png_with_iccp_original('/home/work/SegTrimap/datasets/Composition-431k-png/original')
