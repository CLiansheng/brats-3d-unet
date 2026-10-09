
import torch


def dice_score(pred, target):
    """
    计算整个肿瘤区域的 Dice Score

    pred: 模型预测的类别标签
    target: 真实 Seg 标签

    两者形状相同，例如 (1, 64, 64, 64)
    """

    # 1. 将多类别标签转换成二值 Mask
    # 0 表示非肿瘤，1/2/3 都视为肿瘤
    pred_mask = pred > 0
    target_mask = target > 0

    # 2. 计算预测和真实肿瘤的体素数量
    pred_sum = pred_mask.sum().item()
    target_sum = target_mask.sum().item()

    # 如果双方都没有肿瘤，暂时跳过该 Patch
    if pred_sum + target_sum == 0:
        return None

    # 3. 计算交集的体素数量
    intersection = (pred_mask & target_mask).sum().item()

    # 4. 计算 Dice
    dice = 2 * intersection / (pred_sum + target_sum)

    return dice
