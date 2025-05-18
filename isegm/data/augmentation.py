import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from copy import deepcopy


class RandomEdgeNoise:
    def __init__(self, erosion_prob=0.2, max_kernel_size=9, mode='mixed'):
        self.erosion_prob = erosion_prob
        self.max_kernel_size = max_kernel_size
        self.mode = mode

    def __call__(self, **data):
        seg_mask = data.get('seg_mask', None)
        if seg_mask is not None and random.random() < self.erosion_prob:
            seg_mask_np = seg_mask.squeeze().astype(np.uint8) * 255
            ksize = random.choice(range(3, self.max_kernel_size + 1, 2))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

            if self.mode == 'gradient':
                gradient = cv2.morphologyEx(seg_mask_np, cv2.MORPH_GRADIENT, kernel)
                mask_out = cv2.bitwise_and(seg_mask_np, cv2.bitwise_not(gradient))
            elif self.mode == 'erode':
                mask_out = cv2.erode(seg_mask_np, kernel, iterations=1)
            elif self.mode == 'dilate':
                mask_out = cv2.dilate(seg_mask_np, kernel, iterations=1)
            else:
                op = random.choice(['erode', 'dilate', 'gradient'])
                return self.__class__(1.0, self.max_kernel_size, op)(**data)

            seg_mask = (mask_out > 127).astype(np.uint8)[None, ...]
            data['seg_mask'] = seg_mask

        return data


class JitterContourEdge:
    def __init__(self, prob=0.5, jitter_px=2, point_drop_ratio=0.05, thickness=4, epsilon=7.0):
        self.prob = prob
        self.jitter_px = jitter_px
        self.point_drop_ratio = point_drop_ratio
        self.thickness = thickness
        self.epsilon = epsilon

    def __call__(self, **data):
        seg_mask = data.get('seg_mask', None)
        if seg_mask is None or random.random() > self.prob:
            return data

        seg = seg_mask.squeeze().astype(np.uint8) * 255
        h, w = seg.shape

        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        jitter_edge_mask = np.zeros_like(seg)

        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue

            num_points = contour.shape[0]
            keep_mask = np.random.rand(num_points) > self.point_drop_ratio
            contour = contour[keep_mask]

            jitter = np.random.randint(-self.jitter_px, self.jitter_px + 1, contour.shape)
            contour = np.clip(contour + jitter, 0, [[w - 1, h - 1]])

            approx = cv2.approxPolyDP(contour.astype(np.float32), self.epsilon, closed=True)

            #if approx.shape[0] >= 2:
            if approx is not None and hasattr(approx, 'shape') and approx.shape[0] >= 2:
                approx = approx.astype(np.int32)
                cv2.drawContours(jitter_edge_mask, [approx], -1, color=255, thickness=self.thickness)

        modified = cv2.subtract(seg, jitter_edge_mask)
        data['seg_mask'] = (modified > 127).astype(np.uint8)[None, ...]
        return data


class RandomHoleDrop:
    def __init__(self, drop_prob=0.2, max_hole_area_ratio=0.05):
        self.drop_prob = drop_prob
        self.max_area_ratio = max_hole_area_ratio

    def __call__(self, **data):
        seg_mask = data.get('seg_mask', None)
        if seg_mask is None or random.random() >= self.drop_prob:
            return data

        seg = seg_mask.squeeze().astype(np.uint8) * 255
        h, w = seg.shape

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(seg, kernel, iterations=1)
        eroded = cv2.erode(seg, kernel, iterations=1)
        edge_mask = cv2.subtract(dilated, eroded)

        ys, xs = np.where(edge_mask > 0)
        if len(xs) == 0:
            return data

        rand_idx = random.randint(0, len(xs) - 1)
        center_x, center_y = xs[rand_idx], ys[rand_idx]

        max_area = int(h * w * self.max_area_ratio)
        hole_area = random.randint(10, max_area)
        hole_radius = int(np.sqrt(hole_area / np.pi))

        num_vertices = random.randint(3, 6)
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        angles += np.random.uniform(-0.2, 0.2, size=angles.shape)

        points = []
        for a in angles:
            r = hole_radius * np.random.uniform(0.6, 1.0)
            x = int(center_x + r * np.cos(a))
            y = int(center_y + r * np.sin(a))
            points.append([np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)])

        poly = np.array([points], dtype=np.int32)
        cv2.fillPoly(seg, [poly], 0)

        data['seg_mask'] = (seg > 127).astype(np.uint8)[None, ...]
        return data

if __name__ == '__main__':
    mask_path = "/home/work/SegTrimap/datasets/Composition-1k-testset/mask/sea-sunny-person-beach_9.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    seg_mask = (mask >= 30).astype(np.uint8)
    seg_mask = cv2.resize(seg_mask, (448, 448), interpolation=cv2.INTER_NEAREST)
    #seg_mask = (alpha >= 30).astype(np.uint8)[None, ...]

    # 각 노이즈 인스턴스 생성
    edge_aug = RandomEdgeNoise(erosion_prob=1.0, max_kernel_size=10)
    jitter_aug = JitterContourEdge(prob=1.0, jitter_px=3, point_drop_ratio=0.05, thickness=4, epsilon=6.0)
    hole_aug = RandomHoleDrop(drop_prob=1.0, max_hole_area_ratio=0.03)

    # 결과 생성
    edge_result = edge_aug(image=None, seg_mask=deepcopy(seg_mask))['seg_mask']
    jitter_result = jitter_aug(image=None, seg_mask=deepcopy(seg_mask))['seg_mask']
    hole_result = hole_aug(image=None, seg_mask=deepcopy(seg_mask))['seg_mask']

    output_path = "/home/work/SegTrimap/seg_mask_noise_augmented.png"

    # 시각화
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(seg_mask.squeeze(), cmap='gray')
    axs[0].set_title("Original seg_mask")
    axs[1].imshow(edge_result.squeeze(), cmap='gray')
    axs[1].set_title("Erode")
    axs[2].imshow(jitter_result.squeeze(), cmap='gray')
    axs[2].set_title("Jitter Edge")
    axs[3].imshow(hole_result.squeeze(), cmap='gray')
    axs[3].set_title("Hole Drop")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    output_path



# class RandomEdgeNoise:
#     def __init__(self, erosion_prob=0.5, max_kernel_size=9, mode='mixed'):
#         self.erosion_prob = erosion_prob
#         self.max_kernel_size = max_kernel_size
#         self.mode = mode

#     def __call__(self, image, mask=None, seg_mask=None, label=None, **kwargs):
#         if seg_mask is not None and random.random() < self.erosion_prob:
#             ksize = random.choice(range(3, self.max_kernel_size + 1, 2))  # odd kernel
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
#             seg_mask_np = seg_mask.squeeze().astype(np.uint8) * 255

#             if self.mode == 'gradient':
#                 gradient = cv2.morphologyEx(seg_mask_np, cv2.MORPH_GRADIENT, kernel)
#                 mask_out = cv2.bitwise_and(seg_mask_np, cv2.bitwise_not(gradient))
#             elif self.mode == 'erode':
#                 mask_out = cv2.erode(seg_mask_np, kernel, iterations=1)
#             elif self.mode == 'dilate':
#                 mask_out = cv2.dilate(seg_mask_np, kernel, iterations=1)
#             else:  # mixed
#                 op = random.choice(['erode', 'dilate', 'gradient'])
#                 return self.__class__(1.0, self.max_kernel_size, op).__call__(image=image, mask=mask, seg_mask=seg_mask, label=label)

#             seg_mask = (mask_out > 127).astype(np.uint8)[None, ...]

#         return {
#             'image': image,
#             'mask': mask,
#             'seg_mask': seg_mask,
#             'label': label
#         }



# class JitterContourEdge:
#     def __init__(self, prob=0.5, jitter_px=2, point_drop_ratio=0.05):
#         self.prob = prob
#         self.jitter_px = jitter_px
#         self.point_drop_ratio = point_drop_ratio

#     def __call__(self, **data):
#         seg_mask = data.get('seg_mask', None)
#         if seg_mask is None or random.random() > self.prob:
#             return data

#         seg = seg_mask.squeeze().astype(np.uint8) * 255
#         h, w = seg.shape

#         contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         jitter_edge_mask = np.zeros_like(seg)

#         for contour in contours:
#             contour = contour.squeeze()
#             if contour.ndim != 2 or contour.shape[0] < 3:
#                 continue

#             num_points = contour.shape[0]
#             keep_mask = np.random.rand(num_points) > self.point_drop_ratio
#             contour = contour[keep_mask]

#             jitter = np.random.randint(-self.jitter_px, self.jitter_px + 1, contour.shape)
#             contour = np.clip(contour + jitter, 0, [[w - 1, h - 1]])

#             if contour.shape[0] >= 2:
#                 contour = contour.astype(np.int32)
#                 cv2.polylines(jitter_edge_mask, [contour], isClosed=True, color=255, thickness=3)

#         modified = cv2.subtract(seg, jitter_edge_mask)
#         data['seg_mask'] = (modified > 127).astype(np.uint8)[None, ...]
#         return data




# class RandomHoleDrop:
#     def __init__(self, drop_prob=0.2, max_hole_area_ratio=0.01):
#         self.drop_prob = drop_prob
#         self.max_area_ratio = max_hole_area_ratio

#     def __call__(self, image, mask=None, seg_mask=None, label=None, **kwargs):
#         if seg_mask is None or random.random() >= self.drop_prob:
#             return {'image': image, 'mask': mask, 'seg_mask': seg_mask, 'label': label}

#         h, w = seg_mask.shape[1:]
#         seg = seg_mask.squeeze().astype(np.uint8) * 255

#         # 1. edge mask 생성 (foreground 경계)
#         kernel = np.ones((3, 3), np.uint8)
#         dilated = cv2.dilate(seg, kernel, iterations=1)
#         eroded = cv2.erode(seg, kernel, iterations=1)
#         edge_mask = cv2.subtract(dilated, eroded)

#         # 2. edge 영역에서 좌표 샘플링
#         ys, xs = np.where(edge_mask > 0)
#         if len(xs) == 0:
#             return {'image': image, 'mask': mask, 'seg_mask': seg_mask, 'label': label}

#         rand_idx = random.randint(0, len(xs) - 1)
#         center_x, center_y = xs[rand_idx], ys[rand_idx]

#         # 3. 구멍 크기 및 모양 정의
#         max_area = int(h * w * self.max_area_ratio)
#         hole_area = random.randint(10, max_area)
#         hole_radius = int(np.sqrt(hole_area / np.pi))

#         num_vertices = random.randint(3, 6)
#         angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
#         angles += np.random.uniform(-0.2, 0.2, size=angles.shape)

#         points = []
#         for a in angles:
#             r = hole_radius * np.random.uniform(0.6, 1.0)
#             x = int(center_x + r * np.cos(a))
#             y = int(center_y + r * np.sin(a))
#             points.append([np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)])

#         poly = np.array([points], dtype=np.int32)
#         cv2.fillPoly(seg, [poly], 0)

#         seg_mask = (seg > 127).astype(np.uint8)[None, ...]

#         return {
#             'image': image,
#             'mask': mask,
#             'seg_mask': seg_mask,
#             'label': label
#         }