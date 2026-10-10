"""MSD BraTS 3D U-Net 正式训练：患者级划分、整脑验证、断点续训。"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from metrics import dice_score
from msd_dataset import MSDBrainTumourDataset
from sliding_window import sliding_window_predict
from unet3d import UNet3D

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("BRATS_DATA_ROOT", "/hy-tmp/Task01_BrainTumour"))
# 云端默认保存在系统盘，避免迁移时 /hy-tmp 临时盘清空。
# 注意：释放实例前仍需把权重下载或备份到外部持久存储。
DEFAULT_CKPT_DIR = "/root/brats-checkpoints" if Path("/hy-tmp").is_dir() else str(ROOT / "checkpoints")
SAVE_DIR = Path(os.environ.get("BRATS_CHECKPOINT_DIR", DEFAULT_CKPT_DIR))

EPOCHS = int(os.environ.get("BRATS_EPOCHS", "50"))
VAL_EVERY = int(os.environ.get("BRATS_VAL_EVERY", "5"))
PATCH_SIZE = 64
VAL_STRIDE = 64  # 正式高精度评估可改为 32，耗时也会明显增加


def save_last(path, epoch, model, optimizer, scaler, best_val_dice):
    """原子保存训练状态：中途关机可从上一完整 Epoch 继续。"""
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_dice": best_val_dice,
    }
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备:", device, flush=True)

    # 同一组患者，用不同的读取策略：训练随机 Patch、验证完整 MRI
    train_full = MSDBrainTumourDataset(DATA_ROOT, patch_size=PATCH_SIZE)
    val_full = MSDBrainTumourDataset(DATA_ROOT, patch_size=None)
    if [p.name for p in train_full.image_paths] != [p.name for p in val_full.image_paths]:
        raise RuntimeError("训练/验证患者文件列表不一致")

    n = len(train_full)
    if n < 2:
        raise ValueError("需要至少两名患者才能划分训练集和验证集")
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    n_train = int(n * 0.8)
    train_dataset = Subset(train_full, indices[:n_train])
    val_dataset = Subset(val_full, indices[n_train:])
    print(f"训练患者: {len(train_dataset)}，验证患者: {len(val_dataset)}", flush=True)

    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = UNet3D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # RTX 3090 支持混合精度训练；验证推理保持 FP32
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    last_path = SAVE_DIR / "last.pth"
    best_path = SAVE_DIR / "best.pth"
    start_epoch = 0
    best_val_dice = -1.0

    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        start_epoch = ckpt["epoch"]
        best_val_dice = ckpt.get("best_val_dice", -1.0)
        print(f"恢复训练：已经完成 {start_epoch} 轮，最佳 Dice={best_val_dice:.4f}", flush=True)

    for epoch in range(start_epoch, EPOCHS):
        begin = time.time()
        model.train()
        total_loss, total_dice, dice_count = 0.0, 0.0, 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()  # 每个 Patch 均计入 Loss
            with torch.no_grad():
                dice = dice_score(logits.argmax(dim=1), labels)
                if dice is not None:
                    total_dice += dice
                    dice_count += 1

        avg_loss = total_loss / len(loader)
        avg_dice = total_dice / dice_count if dice_count else float("nan")
        print(f"Epoch {epoch + 1}/{EPOCHS} 训练 Loss={avg_loss:.4f} "
              f"Patch Dice={avg_dice:.4f} 耗时={time.time() - begin:.1f}s", flush=True)

        # 先保存：即使稍后的整脑验证被中断，也不会丢失本轮训练
        save_last(last_path, epoch + 1, model, optimizer, scaler, best_val_dice)

        if (epoch + 1) % VAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            total_val_dice, count_val_dice = 0.0, 0
            print("开始全脑滑窗验证……", flush=True)

            for i, (images, labels) in enumerate(val_loader, start=1):
                prediction = sliding_window_predict(
                    model, images, device, patch_size=PATCH_SIZE, stride=VAL_STRIDE
                )
                dice = dice_score(prediction, labels[0])
                if dice is not None:
                    total_val_dice += dice
                    count_val_dice += 1
                if i % 10 == 0 or i == len(val_loader):
                    print(f"验证进度：{i}/{len(val_loader)}", flush=True)

            if not count_val_dice:
                raise RuntimeError("验证集中没有有效 Dice，请检查标签及数据")
            avg_val_dice = total_val_dice / count_val_dice
            print(f"Epoch {epoch + 1} 整脑 WT Dice={avg_val_dice:.4f}", flush=True)

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), best_path)
                print(f"保存最佳模型：{best_path}", flush=True)

            save_last(last_path, epoch + 1, model, optimizer, scaler, best_val_dice)

    print("训练完成！最佳整脑 WT Dice：", f"{best_val_dice:.4f}", flush=True)
    print("模型目录：", SAVE_DIR, flush=True)


if __name__ == "__main__":
    main()
