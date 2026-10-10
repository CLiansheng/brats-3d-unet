"""MSD Task01_BrainTumour 数据读取：训练随机 Patch，验证完整 MRI。"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class MSDBrainTumourDataset(Dataset):
    def __init__(self, data_root, patch_size=64, tumor_prob=0.7):
        self.data_root = Path(data_root)
        self.patch_size = patch_size  # None：验证时读取整个 MRI
        self.tumor_prob = tumor_prob

        if not 0 <= tumor_prob <= 1:
            raise ValueError("tumor_prob 必须在 [0, 1] 内")
        if patch_size is not None and (patch_size <= 0 or patch_size % 8 != 0):
            raise ValueError("patch_size 必须是正整数且能被 8 整除（适配 3 次下采样）")

        self.image_dir = self.data_root / "imagesTr"
        self.label_dir = self.data_root / "labelsTr"
        self.image_paths = sorted(self.image_dir.glob("BRATS_*.nii.gz"))
        if not self.image_paths:
            raise FileNotFoundError(f"未找到 MRI：{self.image_dir}")

        missing = [p.name for p in self.image_paths if not (self.label_dir / p.name).is_file()]
        if missing:
            raise FileNotFoundError(f"部分患者缺少标签，例如：{missing[:3]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_dir / image_path.name

        # MSD Task01 的 MRI 是 4D：(H, W, D, 4)
        image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        if image.ndim != 4 or image.shape[-1] != 4:
            raise ValueError(f"{image_path.name} MRI shape 异常：{image.shape}")
        image = np.moveaxis(image, -1, 0)  # -> (4, H, W, D)

        # 非背景区域逐通道做 Z-score 归一化
        for c in range(4):
            foreground = image[c] != 0
            if foreground.any():
                values = image[c][foreground]
                image[c][foreground] = (values - values.mean()) / (values.std() + 1e-8)

        # 标签应该为 0、1、2、3；不能直接把原始的 4 当作 CE 类别索引
        label = np.asarray(nib.load(str(label_path)).dataobj, dtype=np.int64)
        if image.shape[1:] != label.shape:
            raise ValueError(f"{image_path.name} MRI 和标签空间尺寸不一致")
        if label.min() < 0 or label.max() > 3:
            unique = np.unique(label)
            raise ValueError(f"{image_path.name} 标签值为 {unique}；请先核对 dataset.json")

        if self.patch_size is not None:
            p = self.patch_size
            h, w, d = label.shape
            if min(h, w, d) < p:
                raise ValueError(f"{image_path.name} 尺寸 {label.shape} 小于 Patch {p}")

            if np.random.rand() < self.tumor_prob and (label > 0).any():
                # 随机挑一个肿瘤体素，让当前 Patch 包含肿瘤
                tumor_positions = np.argwhere(label > 0)
                cx, cy, cz = tumor_positions[np.random.randint(len(tumor_positions))]
                x = int(np.clip(cx - p // 2, 0, h - p))
                y = int(np.clip(cy - p // 2, 0, w - p))
                z = int(np.clip(cz - p // 2, 0, d - p))
            else:
                x = np.random.randint(0, h - p + 1)
                y = np.random.randint(0, w - p + 1)
                z = np.random.randint(0, d - p + 1)

            # 两个采样分支都必须裁剪！且只对训练集裁剪
            image = image[:, x:x + p, y:y + p, z:z + p]
            label = label[x:x + p, y:y + p, z:z + p]

        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(label)),
        )
