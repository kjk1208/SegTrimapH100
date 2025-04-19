import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms as T
from isegm.model.is_trimap_plaintvit_model import TrimapPlainVitModel
from isegm.model.modeling.models_vit import VisionTransformer

# 1. 모델 구성
def build_model():
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
    neck_params = dict(in_dim=1280, out_dims=[240, 480, 960, 1920])
    head_params = dict(
        in_channels=[240, 480, 960, 1920],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=3,
        loss_decode=torch.nn.CrossEntropyLoss(),
        align_corners=False,
        upsample='x1',
        channels=256,
    )

    model = TrimapPlainVitModel(
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=False,
        use_disks=False,
        with_prev_mask=False,
    )
    return model

# 2. 모델 초기화 및 checkpoint 로딩
model = build_model()
ckpt_path = './output/iter_mask/aim500_trimap_vit_huge448/001/checkpoints/last_checkpoint.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

# 3. feature hook 등록
feature_map = {}

def hook_fn(module, input, output):
    feature_map['feat'] = output

model.backbone.blocks[0].register_forward_hook(hook_fn)

# 4. 실제 이미지 로딩
image_path = '/home/kjk/matting/SimpleClick/datasets/3.AIM-500/original/o_0a0ae43d.jpg'
mask_path = '/home/kjk/matting/SimpleClick/datasets/3.AIM-500/mask/o_0a0ae43d.png'

# 5. 전처리
transform_image = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor()
])

image = Image.open(image_path).convert('RGB')
image = transform_image(image).unsqueeze(0)  # [1, 3, 448, 448]

alpha = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
seg_mask = (alpha >= 200).astype(np.uint8)
seg_mask = cv2.resize(seg_mask, (448, 448), interpolation=cv2.INTER_NEAREST)
seg_mask = torch.from_numpy(seg_mask).unsqueeze(0).unsqueeze(0).float()  # [1, 1, 448, 448]

# 6. forward
with torch.no_grad():
    _ = model(image, seg_mask)

# 7. feature 추출 및 시각화
feat = feature_map['feat'][0].mean(dim=-1)  # 첫 번째 sample, 채널 평균
feat = feat[1:]  # CLS token 제외

# ViT 내부에서 실제 사용한 patch grid size를 기반으로 reshape
grid_h, grid_w = model.backbone.patch_embed.grid_size
assert feat.shape[0] == grid_h * grid_w, f"Mismatch: got {feat.shape[0]}, expected {grid_h * grid_w}"

feat = feat.reshape(grid_h, grid_w).cpu().numpy()
feat = (feat - feat.min()) / (feat.max() - feat.min())
heatmap = (feat * 255).astype(np.uint8)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 8. 시각화 출력
plt.imshow(heatmap[..., ::-1])  # BGR → RGB
plt.title("ViT Patch-wise Feature Heatmap")
plt.axis('off')
plt.show()
