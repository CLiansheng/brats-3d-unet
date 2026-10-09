from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import torch

class BraTSDataset(Dataset):

    def __init__(self, data_root):

        # 获取所有患者文件夹
        self.patient_dirs = sorted(
            p for p in Path(data_root).iterdir()
            if p.is_dir() and p.name.startswith("BraTS-GLI-")
        )
    def __len__(self):
        # 返回数据集中患者的数量
        return len(self.patient_dirs)
    def __getitem__(self, idx):

        # 1. 找到第 idx 名患者的文件夹
        patient_dir = self.patient_dirs[idx]
        patient = patient_dir.name

        # 2. 定义四种 MRI 模态
        modalities = ["t1n", "t1c", "t2w", "t2f"]
        images = []

        # 3. 依次读取四种 MRI
        for modality in modalities:
            path = patient_dir / f"{patient}-{modality}.nii.gz"

            mri = nib.load(str(path))
            image = mri.get_fdata(dtype=np.float32)
            # 只对非背景区域进行归一化
            mask = image != 0
    
            if np.any(mask):
                mean = image[mask].mean()
                std = image[mask].std()
    
                image[mask] = (image[mask] - mean) / (std + 1e-8)
            images.append(image)

        # 4. 将四个三维数组堆叠
        images = np.stack(images, axis=0)
       
        # 5. 转换为 PyTorch Tensor
        images = torch.from_numpy(images)

        # 6. 读取当前患者的分割标签
        seg_path = patient_dir / f"{patient}-seg.nii.gz"

        seg_file = nib.load(str(seg_path))
        label = seg_file.get_fdata().astype(np.int64)

        # 7. 将标签转换为 PyTorch Tensor
        label = torch.from_numpy(label)
        # 3D Patch 中心裁剪
        patch_size = 64

        # 计算三个维度的裁剪起点
        # 随机生成三个维度的裁剪起点
        x = np.random.randint(0, label.shape[0] - patch_size + 1)
        y = np.random.randint(0, label.shape[1] - patch_size + 1)
        z = np.random.randint(0, label.shape[2] - patch_size + 1)

        # 裁剪 MRI：保留全部四个通道
        images = images[
            :,
            x:x + patch_size,
            y:y + patch_size,
            z:z + patch_size
        ]

        # 裁剪 Seg：使用完全相同的空间范围
        label = label[
            x:x + patch_size,
            y:y + patch_size,
            z:z + patch_size
        ]

        # 8. 同时返回 MRI 和 Seg
        return images, label
if __name__ == "__main__":

    # 1. 获取数据目录
    ROOT = Path(__file__).resolve().parent.parent
    data_root = ROOT / "data" / "raw"

    # 2. 创建 Dataset 对象
    dataset = BraTSDataset(data_root)

    # 3. 查看患者数量
    print("患者数量:", len(dataset))

    # 4. 读取第一个患者
    image, label = dataset[0]

    # 5. 查看 Tensor 信息
    print("MRI shape:", image.shape)
    print("Seg shape:", label.shape)

    print("MRI dtype:", image.dtype)
    print("Seg dtype:", label.dtype)