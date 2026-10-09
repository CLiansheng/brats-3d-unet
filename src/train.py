
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from metrics import dice_score
from brats_dataset import BraTSDataset
from unet3d import UNet3D


# 1. 选择计算设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("计算设备:", device)


# 2. 加载真实 MRI 数据
ROOT = Path(__file__).resolve().parent.parent
data_root = ROOT / "data" / "raw"

dataset = BraTSDataset(data_root)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)


# 3. 创建 3D U-Net
model = UNet3D().to(device)


# 4. 定义损失函数
criterion = nn.CrossEntropyLoss()

# 5. 定义 Adam 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 开始训练
model.train()

epochs = 3
steps_per_epoch = 10

for epoch in range(epochs):

    total_loss = 0.0
    total_dice = 0.0
    dice_count = 0
    for step in range(steps_per_epoch):

        # 每次重新随机裁剪一个 Patch
        images, labels = next(iter(loader))

        images = images.to(device)
        labels = labels.to(device)

        # 1. 清空梯度
        optimizer.zero_grad()

        # 2. 前向传播
        logits = model(images)

        # 3. 计算损失
        loss = criterion(logits, labels)

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()
        # 计算模型预测的分割标签
        with torch.no_grad():
            prediction = torch.argmax(logits, dim=1)
            if step == steps_per_epoch - 1:
                print("预测类别及数量:",
                    torch.unique(prediction, return_counts=True))

                print("真实类别及数量:",
                    torch.unique(labels, return_counts=True))
            # 计算当前 Patch 的 Dice
            dice = dice_score(prediction, labels)

            if dice is not None:
                total_dice += dice
                dice_count += 1
                # 累计当前 Epoch 的 Loss
                total_loss += loss.item()
    
    # 计算当前 Epoch 的平均 Loss
    avg_loss = total_loss / steps_per_epoch

    # 计算当前 Epoch 的平均 Dice
    avg_dice = total_dice / dice_count if dice_count > 0 else None

    print(
    f"Epoch {epoch + 1}/{epochs}, "
    f"Loss: {avg_loss:.4f}, "
    f"Dice: {avg_dice:.4f}" if avg_dice is not None
    else f"Epoch {epoch + 1}/{epochs}, "
         f"Loss: {avg_loss:.4f}, Dice: N/A"
    )
    # 保存模型权重
save_dir = ROOT / "checkpoints"
save_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    model.state_dict(),
    save_dir / "unet3d_demo.pth"
)

print("模型保存成功！")
