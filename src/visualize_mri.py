
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 获取患者数据所在目录
ROOT = Path(__file__).resolve().parent.parent
patient = "BraTS-GLI-00046-000"
data_dir = ROOT / "data" / "raw" / patient

# 2. 读取 MRI 原图
mri_path = data_dir / f"{patient}-t2f.nii.gz"
mri_file = nib.load(str(mri_path))
mri = mri_file.get_fdata(dtype=np.float32)

# 3. 读取肿瘤分割标签
seg_path = data_dir / f"{patient}-seg.nii.gz"
seg_file = nib.load(str(seg_path))
seg = np.asanyarray(seg_file.dataobj)

print("MRI shape:", mri.shape)
print("Seg shape:", seg.shape)
print("标签种类:", np.unique(seg))

# 确保 MRI 和标签形状一致
assert mri.shape == seg.shape

# 4. 找到肿瘤面积最大的一层
# 对每一个 z，统计该层非零标签的 voxel 数量
tumor_counts = np.count_nonzero(seg != 0, axis=(0, 1))
z = int(np.argmax(tumor_counts))

# 5. 提取二维切片
mri_slice = mri[:, :, z].T
seg_slice = seg[:, :, z].T

# 6. 构造彩色透明标签层
# 最后一个维度是 RGBA：红、绿、蓝、透明度
overlay = np.zeros((*seg_slice.shape, 4), dtype=np.float32)

overlay[seg_slice == 1] = [1, 0, 0, 0.5]  # 红色
overlay[seg_slice == 2] = [0, 1, 0, 0.5]  # 绿色
overlay[seg_slice == 3] = [1, 1, 0, 0.5]  # 黄色

# 7. 并排显示 MRI 和标签叠加效果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(mri_slice, cmap="gray", origin="lower")
axes[0].set_title("Original MRI")
axes[0].axis("off")

axes[1].imshow(mri_slice, cmap="gray", origin="lower")
axes[1].imshow(overlay, origin="lower")
axes[1].set_title("Ground Truth Segmentation")
axes[1].axis("off")

plt.tight_layout()
plt.show()