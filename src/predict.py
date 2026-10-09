import torch
from pathlib import Path
from unet3d import UNet3D
import numpy as np
import nibabel as nib
# 1. 获取项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 2. 选择 GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 3. 创建模型
model = UNet3D().to(device)

# 4. 加载之前保存的权重
weight_path = ROOT / "checkpoints" / "unet3d_demo.pth"

state_dict = torch.load(
    weight_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

# 5. 切换为评估模式
model.eval()

print("模型权重加载成功！")
print("当前设备:", device)
# 6. 读取完整 MRI
patient = "BraTS-GLI-00046-000"
data_dir = ROOT / "data" / "raw" / patient

modalities = ["t1n", "t1c", "t2w", "t2f"]
images = []

for modality in modalities:
    path = data_dir / f"{patient}-{modality}.nii.gz"

    mri = nib.load(str(path))
    image = mri.get_fdata(dtype=np.float32)

    # 与训练阶段保持一致的归一化
    mask = image != 0

    if np.any(mask):
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / (std + 1e-8)

    images.append(image)

# 拼接四个通道，转换为 Tensor
volume = torch.from_numpy(
    np.stack(images, axis=0)
).unsqueeze(0)

print("完整 MRI Shape:", volume.shape)


# 7. 计算滑动窗口的起始位置
patch_size = 64
stride = 64

def get_starts(dim, patch_size, stride):
    # 按固定步长生成裁剪起点
    starts = list(range(0, dim - patch_size + 1, stride))

    # 确保最后一个 Patch 覆盖到末尾
    if starts[-1] != dim - patch_size:
        starts.append(dim - patch_size)

    return starts


# 获取三个空间维度
D, H, W = volume.shape[2:]

# 分别计算三个方向的裁剪起点
d_starts = get_starts(D, patch_size, stride)
h_starts = get_starts(H, patch_size, stride)
w_starts = get_starts(W, patch_size, stride)

print("D 方向:", d_starts)
print("H 方向:", h_starts)
print("W 方向:", w_starts)

print("Patch 总数量:",
      len(d_starts) * len(h_starts) * len(w_starts))


# 8. 创建容器，保存完整 MRI 的预测结果

# 累加每个 voxel 的四个类别分数
logits_sum = torch.zeros((4, D, H, W))

# 记录每个 voxel 被多少个 Patch 覆盖
count = torch.zeros((D, H, W))

# 9. 遍历全部 48 个 Patch
with torch.inference_mode():

    for d in d_starts:
        for h in h_starts:
            for w in w_starts:

                # 裁剪当前 Patch
                patch = volume[
                    :,
                    :,
                    d:d + patch_size,
                    h:h + patch_size,
                    w:w + patch_size
                ].to(device)

                # 3D U-Net 预测
                logits = model(patch)

                # 将预测结果移回 CPU
                logits = logits[0].cpu()

                # 累加到原始空间位置
                logits_sum[
                    :,
                    d:d + patch_size,
                    h:h + patch_size,
                    w:w + patch_size
                ] += logits

                # 对应位置的覆盖次数加 1
                count[
                    d:d + patch_size,
                    h:h + patch_size,
                    w:w + patch_size
                ] += 1

# 10. 合并重叠区域
assert torch.all(count > 0)

logits_sum /= count.unsqueeze(0)

# 11. 为每个 voxel 选择分数最高的类别
prediction = torch.argmax(logits_sum, dim=0)

print("完整预测 Shape:", prediction.shape)
print("预测标签:", torch.unique(prediction))

# 12. 保存完整分割结果
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

np.save(
    output_dir / "prediction.npy",
    prediction.numpy().astype(np.uint8)
)

print("完整分割结果保存成功！")
