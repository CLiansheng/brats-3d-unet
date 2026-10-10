"""使用重叠滑窗对完整 MRI 做 3D U-Net 推理。"""

import torch


def get_starts(dim, patch_size, stride):
    if dim < patch_size:
        raise ValueError(f"维度 {dim} 小于 patch_size={patch_size}")
    starts = list(range(0, dim - patch_size + 1, stride))
    if starts[-1] != dim - patch_size:
        starts.append(dim - patch_size)
    return starts


@torch.inference_mode()
def sliding_window_predict(model, volume, device, patch_size=64, stride=64):
    """
    输入 volume: CPU 上的 (1, 4, D, H, W)
    返回 prediction: CPU 上的 (D, H, W)，取值 0、1、2、3
    """
    if volume.ndim != 5 or volume.shape[0] != 1 or volume.shape[1] != 4:
        raise ValueError(f"要求输入形状 (1, 4, D, H, W)，实际为 {tuple(volume.shape)}")
    if not (0 < stride <= patch_size):
        raise ValueError("stride 必须在 (0, patch_size] 范围内")

    _, _, D, H, W = volume.shape
    d_starts = get_starts(D, patch_size, stride)
    h_starts = get_starts(H, patch_size, stride)
    w_starts = get_starts(W, patch_size, stride)

    # 完整预测分数留在 CPU，GPU 每次只处理一个 Patch
    logits_sum = torch.zeros((4, D, H, W), dtype=torch.float32)
    count = torch.zeros((D, H, W), dtype=torch.float32)

    for d in d_starts:
        for h in h_starts:
            for w in w_starts:
                patch = volume[:, :, d:d + patch_size,
                               h:h + patch_size, w:w + patch_size].to(device)
                logits = model(patch)[0].float().cpu()
                logits_sum[:, d:d + patch_size,
                           h:h + patch_size, w:w + patch_size] += logits
                count[d:d + patch_size,
                      h:h + patch_size, w:w + patch_size] += 1

    if not torch.all(count > 0):
        raise RuntimeError("有体素没有被滑窗覆盖")

    logits_sum /= count.unsqueeze(0)
    return logits_sum.argmax(dim=0)
