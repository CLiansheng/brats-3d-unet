
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "Task01_BrainTumour"
OUTPUT_ROOT = ROOT / "outputs" / "evaluation"

patients = ["BRATS_001", "BRATS_142", "BRATS_446"]


def hd95(pred_mask, target_mask, spacing):
    """计算两个三维二值区域的 HD95，单位为 mm"""

    # 处理空区域
    if not pred_mask.any() and not target_mask.any():
        return float("nan")

    if not pred_mask.any() or not target_mask.any():
        return float("inf")

    # 1. 提取两个区域的表面
    pred_surface = pred_mask & ~binary_erosion(pred_mask)
    target_surface = target_mask & ~binary_erosion(target_mask)

    # 2. 计算预测边界到真实边界的距离
    dist_to_target = distance_transform_edt(
        ~target_surface, sampling=spacing
    )
    d1 = dist_to_target[pred_surface]

    # 3. 计算真实边界到预测边界的距离
    dist_to_pred = distance_transform_edt(
        ~pred_surface, sampling=spacing
    )
    d2 = dist_to_pred[target_surface]

    # 4. 合并两个方向，取第 95 百分位距离
    distances = np.concatenate([d1, d2])

    return float(np.percentile(distances, 95))


results = []

for patient in patients:

    # 读取真实标签
    label_path = DATA_ROOT / "labelsTr" / f"{patient}.nii.gz"
    label_file = nib.load(str(label_path))

    label = np.asanyarray(label_file.dataobj).astype(np.uint8)

    # 读取模型预测
    pred = np.load(OUTPUT_ROOT / f"{patient}_pred.npy")

    # 读取每个体素的物理间距
    spacing = label_file.header.get_zooms()[:3]

    scores = {}

    # 分别计算 WT、TC、ET 的 HD95
    regions = {
        "WT": [1, 2, 3],
        "TC": [2, 3],
        "ET": [3]
    }

    for name, classes in regions.items():

        pred_mask = np.isin(pred, classes)
        target_mask = np.isin(label, classes)

        scores[name] = hd95(
            pred_mask,
            target_mask,
            spacing
        )

    results.append(scores)

    print(f"\n{patient}")
    for name, score in scores.items():
        print(f"{name} HD95: {score:.4f} mm")


print("\n===== 平均 HD95 =====")

for name in ["WT", "TC", "ET"]:
    values = [r[name] for r in results]
    print(f"Mean {name} HD95: {np.mean(values):.4f} mm")
