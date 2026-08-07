import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
    论文地址：https://arxiv.org/abs/2501.03775
    论文题目：DRPCA-Net: Make Robust PCA Great Again for Infrared Small Target Detection (2025 一区TOP) 
    中文题目：DRPCA-Net：让稳健主成分分析再次成为红外小目标检测的利器 (2025 一区TOP) 
    讲解视频：https://www.bilibili.com/video/BV14USkBWE74/
    残差通道–空间注意力（Residual Channel–Spatial Attention Block,RCSA）
    实际意义：①忽略了空间重要性差异：对于红外小目标任务，背景通常复杂、纹理强、小目标弱且分布稀疏，因此不同位置的重要性差异非常大。
            ②在复杂背景中，小目标需要更精细的空间选择性：传统通道注意力无法有效根据图像内容动态调整空间区域，难以将小目标从复杂背景中区分出来。
            ③静态权重无法适配不同背景的问题：使用静态卷积或固定池化策略，无法适配不同红外背景。
    实现方式：基础卷积提取局部特征→通道注意力用于筛选重要通道→动态空间注意力选择关键空间位置→残差连接增强稳定性。
"""

class DualPoolChannelAttention(nn.Module):
    """
    双池化通道注意力模块（原 ChannelAttention）
    AvgPool + MaxPool → MLP → 通道权重
    """
    def __init__(self, channels=32, reduction=16):
        super().__init__()

        # 全局平均池化 & 最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通道注意力 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=True)
        )

        self.activate = nn.Sigmoid()

    def forward(self, x):
        # [B, C, H, W]
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        weights = self.activate(avg_out + max_out)
        return x * weights


class DynamicKernelSpatialAttention(nn.Module):
    """
    动态卷积核空间注意力模块（原 DynamicSpatialAttention）
    为每个样本生成一个 k×k 卷积核，对空间位置加权
    """
    def __init__(self, channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        # 生成动态卷积核
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # [B, C, 1, 1]
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, kernel_size * kernel_size, 1)  # [B, k*k, 1, 1]
        )

        self.activate = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成每个样本对应的卷积核 → [B, 1, k, k]
        kernels = self.kernel_gen(x).view(B, 1, self.kernel_size, self.kernel_size)
        # 跨通道求平均，得到单通道特征 → [B, 1, H, W]
        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        # 分组卷积输入需要 [1, B, H, W]，groups=B
        x_mean = x_mean.view(1, B, H, W)
        # 使用 B 组卷积，每一组对应一个样本的 kernel
        att = F.conv2d(
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )  # [1, B, H, W]

        # reshape 回 [B, 1, H, W]
        att = att.view(B, 1, H, W)

        # 0~1 空间权重
        att = self.activate(att)
        # 空间注意力作用到原特征
        return x * att


class RCSABlock(nn.Module):
    """
        结构：Conv-BN-Act → Conv-BN → 通道注意力 → 空间注意力 → 残差
    """
    def __init__(self, channels, use_bn=True, act=nn.ReLU(True)):
        super().__init__()

        layers = []

        # 两个 3×3 卷积
        for i in range(2):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
            if use_bn:
                layers.append(nn.BatchNorm2d(channels))
            if i == 0:
                layers.append(act)  # 仅第一个卷积后加激活

        # 通道注意力
        layers.append(DualPoolChannelAttention(channels=channels))

        # 动态空间注意力
        layers.append(DynamicKernelSpatialAttention(channels=channels))

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        res = self.body(x)
        return x + res  # 残差连接

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = RCSABlock(channels=32)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
