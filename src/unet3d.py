import torch
import torch.nn as nn
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            # 第一次 3D 卷积
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),

            # 第二次 3D 卷积
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            # 三维最大池化：空间尺寸减半
            nn.MaxPool3d(kernel_size=2, stride=2),

            # 双重卷积：提取特征并调整通道数
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一次卷积，不进行下采样
        self.inc = DoubleConv3D(4, 16)

        # 连续三次下采样
        self.down1 = Down3D(16, 32)
        self.down2 = Down3D(32, 64)
        self.down3 = Down3D(64, 128)

    def forward(self, x):
        # 每经过一个模块，保存对应的特征图
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 返回四个尺度的特征图
        return x1, x2, x3, x4


class Up3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # 将 D、H、W 各扩大两倍
        self.up = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=False
        )

        # 拼接后再进行双重卷积
        self.conv = DoubleConv3D(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        # 1. 上采样深层特征
        x = self.up(x)

        # 2. 拼接同分辨率的编码器特征
        x = torch.cat([skip, x], dim=1)

        # 3. 融合特征
        return self.conv(x)

class Decoder3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = Up3D(128, 64, 64)
        self.up2 = Up3D(64, 32, 32)
        self.up3 = Up3D(32, 16, 16)

    def forward(self, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return x


class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. 编码器
        self.encoder = Encoder3D()

        # 2. 解码器
        self.decoder = Decoder3D()

        # 3. 最终分类层：16通道 → 4个类别
        self.out_conv = nn.Conv3d(
            in_channels=16,
            out_channels=4,
            kernel_size=1
        )

    def forward(self, x):

        # 编码器提取四个尺度的特征
        x1, x2, x3, x4 = self.encoder(x)

        # 解码器恢复空间尺寸
        x = self.decoder(x1, x2, x3, x4)

        # 为每个 voxel 预测四个类别分数
        logits = self.out_conv(x)

        return logits

if __name__ == "__main__":

    # 1. 创建完整的 3D U-Net
    model = UNet3D()

    # 2. 模拟四通道 MRI 输入
    x = torch.randn(1, 4, 32, 32, 32)

    # 3. 切换到评估模式
    model.eval()

    # 4. 前向传播（暂时不计算梯度）
    with torch.no_grad():
        logits = model(x)

    # 5. 每个 voxel 选择分数最高的类别
    prediction = torch.argmax(logits, dim=1)

    print("输入 Shape:", x.shape)
    print("Logits Shape:", logits.shape)
    print("预测 Shape:", prediction.shape)

    