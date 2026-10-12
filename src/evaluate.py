
from pathlib import Path
import csv

import numpy as np
import torch

from msd_dataset import MSDBrainTumourDataset
from unet3d import UNet3D
from sliding_window import sliding_window_predict


# 1. 项目路径
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "Task01_BrainTumour"
WEIGHT_PATH = ROOT / "checkpoints" / "best.pth"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 计算设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("计算设备:", device)

# 3. 加载训练好的最佳模型
model = UNet3D().to(device)

state_dict = torch.load(
    WEIGHT_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("最佳模型加载成功！")


# 4. 计算某个肿瘤区域的 Dice
def region_dice(pred, target, classes):
    pred_mask = torch.isin(
        pred, torch.tensor(classes)
    )
    target_mask = torch.isin(
        target, torch.tensor(classes)
    )

    intersection = (pred_mask & target_mask).sum().item()
    total = pred_mask.sum().item() + target_mask.sum().item()

    # 双方都没有该区域，按 Dice=1 处理
    if total == 0:
        return 1.0

    return 2 * intersection / total


# 5. 读取完整 MRI，不进行随机裁剪
dataset = MSDBrainTumourDataset(
    DATA_ROOT,
    patch_size=None
)

results = []

# 6. 逐名患者评估
for i in range(len(dataset)):

    patient = dataset.image_paths[i].name
    print(f"\n正在评估: {patient}", flush=True)

    image, label = dataset[i]

    # (4,D,H,W) -> (1,4,D,H,W)
    volume = image.unsqueeze(0)

    # 7. 整脑滑窗推理
    prediction = sliding_window_predict(
        model,
        volume,
        device,
        patch_size=64,
        stride=64
    )

    # 8. 计算三个区域的 Dice
    wt = region_dice(prediction, label, [1, 2, 3])
    tc = region_dice(prediction, label, [2, 3])
    et = region_dice(prediction, label, [3])

    print(f"WT Dice: {wt:.4f}")
    print(f"TC Dice: {tc:.4f}")
    print(f"ET Dice: {et:.4f}")

    results.append({
        "patient": patient,
        "WT": wt,
        "TC": tc,
        "ET": et
    })

    # 9. 保存完整预测，用于后续 3D 可视化
    np.save(
        OUTPUT_DIR / f"{patient.replace('.nii.gz', '')}_pred.npy",
        prediction.numpy().astype(np.uint8)
    )


# 10. 保存评估表格
with open(
    OUTPUT_DIR / "metrics.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["patient", "WT", "TC", "ET"]
    )
    writer.writeheader()
    writer.writerows(results)


# 11. 计算三名患者的平均 Dice
print("\n===== 平均评估结果 =====")

for region in ["WT", "TC", "ET"]:
    mean = np.mean([r[region] for r in results])
    print(f"Mean {region} Dice: {mean:.4f}")

print("\n评估完成！")
