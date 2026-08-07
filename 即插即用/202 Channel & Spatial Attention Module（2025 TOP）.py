import torch
import torch.nn as nn

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S1568494625008609
    论文题目：Striking a better balance between segmentation performance and computational costs with a minimalistic network design（2025 TOP）
    中文题目：通过极简网络设计，在分割性能与计算成本之间实现更优平衡（2025 TOP）
    讲解视频：https://www.bilibili.com/video/BV1sLq3BaEbX/
    通道与空间注意力模块（Channel & Spatial Attention Module，CSAM）
        实际意义：①网络感受野不足：传统卷积操作虽能提取局部和上下文特征，但受限于局部性，无法高效捕捉全局长程依赖。
                ②多通道表示和特征敏感度不足：在多通道特征图中，通道间和空间位置间的相关性往往被忽略，导致特征表示不充分。
        实现方式：并行通道注意力与空间注意力分支，对特征图在通道和空间维度上的权重信息进行建模，并将二者联合作用于输入特征，从而在不改变特征分辨率、
                且仅引入极少计算开销的前提下，增强网络对关键语义通道与显著空间区域的感知能力。
"""

class SpatialAttentionBlock(nn.Module):
    """空间注意力模块：通过可分离卷积生成查询、键和值，计算像素间的空间注意力权重"""
    def __init__(self, in_channels, ratio=2):
        super(SpatialAttentionBlock, self).__init__()
        # 查询分支：1x3 卷积，降低通道数以减少计算量
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True)
        )
        # 键分支：3x1 卷积，同样降低通道数
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace=True)
        )
        # 值分支：1x1 卷积，保持原始通道数
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 可学习的缩放参数，初始为0，实现渐进式注意力融合
        self.gamma = nn.Parameter(torch.zeros(1))
        # Softmax 用于归一化注意力权重
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播：计算空间注意力并加权融合到输入特征
        :param x: 输入特征图 (B, C, H, W)
        :return: 加入空间注意力后的特征图
        """
        B, C, H, W = x.size()

        # 生成查询特征：(B, C//ratio, H, W) -> (B, C//ratio, H*W) -> (B, H*W, C//ratio)
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        # 生成键特征：(B, C//ratio, H, W) -> (B, C//ratio, H*W)
        proj_key = self.key(x).view(B, -1, W * H)
        # 计算空间注意力图：(B, H*W, H*W)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)

        # 生成值特征：(B, C, H, W) -> (B, C, H*W)
        proj_value = self.value(x).view(B, -1, H * W)
        # 加权聚合：(B, C, H*W) x (B, H*W, H*W) -> (B, C, H*W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)

        # 残差连接 + 可学习缩放
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    """通道注意力模块：直接利用特征自身计算通道间的相关性"""
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        # 可学习的缩放参数，控制通道注意力强度
        self.gamma = nn.Parameter(torch.zeros(1))
        # Softmax 用于归一化通道注意力权重
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播：计算通道注意力并加权融合到输入特征
        :param x: 输入特征图 (B, C, H, W)
        :return: 加入通道注意力后的特征图
        """
        B, C, H, W = x.size()
        # 将空间维度展平：(B, C, H, W) -> (B, C, H*W)
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)

        # 计算通道间相关性矩阵：(B, C, C)
        affinity = torch.matmul(proj_query, proj_key)
        # 使用一种非标准方式增强注意力（原论文中的技巧）：取最大值减去原矩阵后再softmax
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)

        # 值直接使用展平后的输入特征
        proj_value = x.view(B, C, -1)
        # 加权聚合通道信息
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)

        # 残差连接 + 可学习缩放
        out = self.gamma * weights + x
        return out


class CSAM(nn.Module):
    """通道-空间注意力模块 (Channel-Spatial Attention Module)"""

    def __init__(self, in_channels, ratio=2):
        super(CSAM, self).__init__()
        # 空间注意力分支
        self.sab = SpatialAttentionBlock(in_channels, ratio)
        # 通道注意力分支
        self.cab = ChannelAttentionBlock(in_channels)

    def forward(self, x):
        """
        前向传播：分别计算空间注意力和通道注意力，然后相加融合
        :param x: 输入特征图
        :return: 融合双重注意力的输出特征图
        """
        sab_out = self.sab(x)  # 空间注意力增强特征
        cab_out = self.cab(x)  # 通道注意力增强特征
        out = sab_out + cab_out  # 简单相加（两种注意力互补）
        return out

if __name__ == "__main__":
    # batch=1, 通道=32, 高宽=50x50
    x = torch.randn(1, 32, 50, 50)
    model = CSAM(in_channels=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")