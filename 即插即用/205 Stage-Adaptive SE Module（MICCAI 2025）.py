import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/pdf/2507.11415
    论文题目：U-RWKV: Lightweight medical image segmentation with direction-adaptive RWKV（MICCAI 2025）
    中文题目：U-RWKV：基于方向自适应 RWKV 的轻量化医学图像分割（MICCAI 2025）
    讲解视频：https://www.bilibili.com/video/BV1UNquBhEGk/
    自适应阶段性SE注意力模块（Stage-Adaptive SE Module，SASE）
        实际意义：①不同网络阶段的特征属性差异问题：传统SE或通道注意力模块在所有阶段使用同一压缩策略，无法同时兼顾“浅层的高分辨率细节保留”与“深层的高效语义建模”，造成浅层特征被压缩细节丢失,深层特征计算冗余效率下降。
                ②医疗图像中复杂的空间关系和多样模态带来的挑战：医疗图像（如CT、MRI、超声、内镜等）具有高度各向异性的特征，不同模态的空间相关性和语义信息差异很大。传统固定SE模块无法很好适应多样性，导致在不同数据集上的泛化能力不足。
        实现方式：通过通道压缩、奇偶拆分、逐级深度卷积处理、融合相加、通道恢复等方式，实现了多分支特征增强和残差连接。
"""
class DWConv(nn.Module):
    """
    深度可分离卷积（Depthwise Convolution）
    特点：每个输入通道单独做卷积，不进行通道间混合
    """
    def __init__(self, in_channels, out_channels, k=3):
        super(DWConv, self).__init__()
        # 深度卷积：groups=in_channels 表示“每个通道自己卷积”
        self.dwconv = nn.Conv2d(
            in_channels=in_channels,      # 输入通道数
            out_channels=out_channels,    # 输出通道数（通常等于输入通道数）
            kernel_size=k,                # 卷积核大小
            padding=(k - 1) // 2,          # padding 保证特征图尺寸不变
            groups=in_channels             # 核心：depthwise 卷积
        )
        self.act = nn.GELU()           # GELU 激活（Transformer 常用）


    def forward(self, x):
        # 先做深度卷积，再做激活
        return self.act(self.dwconv(x))


class ConvBnoptinalAct(nn.Module):
    """
    卷积 + BatchNorm + 可选激活函数模块
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBnoptinalAct, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        # 先经过 Conv + BN
        x = self.conv(x)
        x = F.gelu(x)
        return x

class SASE(nn.Module):
    """
    MultiSE 模块：
    - 通道压缩（reduction）
    - 通道分组与逐级深度卷积
    - 特征重组
    - 可选残差连接
    """
    def __init__(self, in_channels, out_channels, reduction=8, split=2):
        super(SASE, self).__init__()

        # 压缩后的通道数
        self.after_red_out_c = int(out_channels / reduction)

        # 判断是否可以使用残差连接
        self.add = (in_channels == out_channels)

        # Sigmoid（这里预留，当前 forward 中未显式使用）
        self.sigmoid = nn.Sigmoid()

        # 1×1 点卷积：用于通道压缩（SE 思想）
        self.pwconv1 = ConvBnoptinalAct(
            in_channels=in_channels,
            out_channels=out_channels // reduction,
            kernel_size=1,
            padding=0
        )

        # 1×1 点卷积：用于通道恢复
        self.pwconv2 = ConvBnoptinalAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            padding=0
        )

        # 多个深度卷积模块，形成“逐级特征增强”
        self.m = nn.ModuleList(
            DWConv(
                in_channels=self.after_red_out_c // split,
                out_channels=self.after_red_out_c // split,
                k=3
            )
            for _ in range(reduction - 1)
        )

    def forward(self, x):
        # 保存输入，用于残差连接
        x_residual = x

        # 通道压缩 32 -->4
        x = self.pwconv1(x)

        # 按通道维度进行奇偶拆分
        # x[:, 0::2] 取偶数通道
        # x[:, 1::2] 取奇数通道
        x = [
            x[:, 0::2, :, :],
            x[:, 1::2, :, :]
        ]

        # 逐级使用深度卷积处理后一个分支
        for m in self.m:
            x.append(m(x[-1]))
            # 偶数 奇数 递归结果1  递归结果2  递归结果3 ···

        # 将前两个分支相加，增强特征融合
        x[0] = x[0] + x[1]
        # 偶数+奇数 奇数 递归结果1  递归结果2  递归结果3 ···
        # 删除已融合的分支 删除 x[1]，所以x只剩下了 m处理的结果 和 相加的结果
        x.pop(1)  # 8个结果 4个通道
        # 偶数+奇数 递归结果1  递归结果2  递归结果3 ···

        # 在通道维度拼接所有特征
        y = torch.cat(x, dim=1) # 8个结果 4个通道 32个通道
        # 通道恢复
        y = self.pwconv2(y)
        # 如果通道数一致，使用残差连接
        return x_residual + y if self.add else y

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = SASE(in_channels=32, out_channels=32)
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")