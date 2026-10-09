from pathlib import Path
from torch.utils.data import DataLoader
from brats_dataset import BraTSDataset
import torch

# 1. 找到数据目录
ROOT = Path(__file__).resolve().parent.parent
data_root = ROOT / "data" / "raw"

# 2. 创建 Dataset
dataset = BraTSDataset(data_root)

# 3. 创建 DataLoader
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# 4. 取出第一个 Batch
images, labels = next(iter(loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
# 查看裁剪后的 Patch 包含哪些标签，以及各类体素数量
values, counts = torch.unique(labels, return_counts=True)

print("标签类别:", values)
print("各类别体素数量:", counts)