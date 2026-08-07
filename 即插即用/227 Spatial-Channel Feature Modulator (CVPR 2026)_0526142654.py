import torch
import torch.nn as nn
import numbers
from einops import rearrange

"""
    论文地址：https://arxiv.org/pdf/2602.21917
    论文题目：Scan Clusters, Not Pixels: A Cluster-Centric Paradigm for Efficient Ultra-high-definition Image Restoration（CVPR 2026）
    中文题目：扫描聚类，而非像素：面向高效超高清图像修复的聚类中心范式（CVPR 2026）
    讲解视频：https://www.bilibili.com/video/BV1Zv9zBgEbU/
    空间-通道特征调制器（Spatial-Channel Feature Modulator，SCFM）
            实际意义：①簇中心建模带来的高频细节丢失问题：高频信息（边缘、纹理、局部细节）可能在聚合过程中被削弱，甚至丢失。②全局建模强、局部保真不足的问题：全局聚类方法建模侧重语义与长程依赖，容易弱化局部细节，不擅长精细保留局部空间结构。
            实现方式：通过并行的空间注意力与通道注意力机制，对关键位置和重要通道进行自适应增强，从而补偿聚类全局建模过程中可能丢失局部细节信息。
"""

def feature_map_to_sequence(feature_map):
    # 将四维图像特征 [B, C, H, W] 展平成三维序列 [B, H*W, C]
    # B 表示 batch size，C 表示通道数，H 和 W 表示特征图高宽
    return rearrange(feature_map, 'batch channels height width -> batch (height width) channels')


def sequence_to_feature_map(feature_sequence, height, width):
    # 将三维序列特征 [B, H*W, C] 重新恢复为四维图像特征 [B, C, H, W]
    return rearrange(
        feature_sequence,
        'batch (height width) channels -> batch channels height width',
        height=height,
        width=width
    )


class RMSLayerNormWithBias(nn.Module):
    def __init__(self, normalized_channels):
        # 初始化带偏置的 RMS 风格 LayerNorm
        super(RMSLayerNormWithBias, self).__init__()

        # 如果输入的是整数，例如 64，则转换为元组形式，例如 (64,)
        if isinstance(normalized_channels, numbers.Integral):
            normalized_channels = (normalized_channels,)

        # 将归一化维度转换为 torch.Size 类型
        normalized_channels = torch.Size(normalized_channels)

        # 确保这里只对最后一个维度，也就是通道维度进行归一化
        assert len(normalized_channels) == 1

        # 定义可学习缩放参数，对归一化后的特征进行幅值调节
        self.scale = nn.Parameter(torch.ones(normalized_channels))

        # 定义可学习偏置参数，对归一化后的特征进行平移调节
        self.shift = nn.Parameter(torch.zeros(normalized_channels))

        # 保存归一化的通道维度
        self.normalized_channels = normalized_channels

    def forward(self, feature_sequence):
        # 计算最后一个维度上的均方值
        channel_mean_square = feature_sequence.pow(2).mean(-1, keepdim=True)

        # 对特征进行 RMS 归一化，并加入可学习缩放和平移
        normalized_sequence = feature_sequence * torch.rsqrt(channel_mean_square + 1e-6) * self.scale + self.shift

        # 返回归一化后的序列特征
        return normalized_sequence


class ImageLayerNorm(nn.Module):
    def __init__(self, channels):
        # 初始化适用于二维图像特征的归一化模块
        super(ImageLayerNorm, self).__init__()

        # 对展平后的通道维度做归一化
        self.sequence_norm = RMSLayerNormWithBias(channels)

    def forward(self, feature_map):
        # 获取输入图像特征的空间高度和宽度
        height, width = feature_map.shape[-2:]

        # 将图像特征转换成序列特征
        feature_sequence = feature_map_to_sequence(feature_map)

        # 对序列特征的通道维度进行归一化
        normalized_sequence = self.sequence_norm(feature_sequence)

        # 将归一化后的序列特征重新恢复为图像特征
        normalized_feature_map = sequence_to_feature_map(normalized_sequence, height, width)

        # 返回归一化后的图像特征
        return normalized_feature_map


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        # 初始化通道注意力模块
        super(ChannelAttention, self).__init__()

        # 使用全局平均池化提取每个通道的整体响应
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 使用全局最大池化提取每个通道的最强响应
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享通道映射网络，用于学习通道之间的依赖关系
        self.shared_channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )

        # 将通道响应压缩到 0 到 1 之间，作为通道注意力权重
        self.channel_gate = nn.Sigmoid()

    def forward(self, feature_map):
        # 平均池化分支：提取全局平均通道描述，并映射为通道权重
        avg_channel_descriptor = self.shared_channel_mlp(self.global_avg_pool(feature_map))
        # 最大池化分支：提取全局最大通道描述，并映射为通道权重
        max_channel_descriptor = self.shared_channel_mlp(self.global_max_pool(feature_map))
        # 融合平均响应和最大响应，得到综合通道注意力描述
        channel_attention_logits = avg_channel_descriptor + max_channel_descriptor
        # 通过 Sigmoid 得到最终通道注意力权重
        channel_attention_map = self.channel_gate(channel_attention_logits)
        # 返回通道注意力权重，形状为 [B, C, 1, 1]
        return channel_attention_map


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        # 初始化空间注意力模块
        super(SpatialAttention, self).__init__()

        # 通过卷积融合平均空间响应图和最大空间响应图
        self.spatial_fusion_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        # 将空间响应压缩到 0 到 1 之间，作为空间注意力权重
        self.spatial_gate = nn.Sigmoid()

    def forward(self, feature_map):
        # 沿通道维度求平均，得到空间平均响应图 [B, 1, H, W]
        avg_spatial_descriptor = torch.mean(feature_map, dim=1, keepdim=True)

        # 沿通道维度取最大值，得到空间最大响应图 [B, 1, H, W]
        max_spatial_descriptor, _ = torch.max(feature_map, dim=1, keepdim=True)

        # 将平均空间响应和最大空间响应在通道维度上拼接
        spatial_descriptor = torch.cat(
            [avg_spatial_descriptor, max_spatial_descriptor],
            dim=1
        )
        # 使用卷积融合两种空间描述信息
        spatial_attention_logits = self.spatial_fusion_conv(spatial_descriptor)
        # 通过 Sigmoid 得到最终空间注意力权重
        spatial_attention_map = self.spatial_gate(spatial_attention_logits)
        # 返回空间注意力权重，形状为 [B, 1, H, W]
        return spatial_attention_map


class SpatialChannelFeatureModulator(nn.Module):
    """
    空间-通道特征调制器 Spatial-Channel Feature Modulator, SCFM
    """

    def __init__(self, channels):
        # 初始化空间-通道特征调制模块
        super().__init__()

        # 图像特征归一化层，用于稳定特征分布
        self.input_norm = ImageLayerNorm(channels)

        # 通道注意力分支，用于建模不同通道的重要性
        self.channel_attention = ChannelAttention(channels)

        # 空间注意力分支，用于建模不同空间位置的重要性
        self.spatial_attention = SpatialAttention()

        # 通道注意力分支后的 1×1 投影卷积
        self.channel_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 空间注意力分支后的 1×1 投影卷积
        self.spatial_projection = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, input_feature):
        # 对输入特征进行归一化，得到稳定的特征表示
        normalized_feature = self.input_norm(input_feature)

        # 计算通道注意力权重，形状为 [B, C, 1, 1]
        channel_weight = self.channel_attention(normalized_feature)
        # 使用通道注意力权重对归一化特征进行通道调制
        channel_modulated_feature = channel_weight * normalized_feature
        # 对通道调制后的特征进行 1×1 卷积投影
        channel_branch_output = self.channel_projection(channel_modulated_feature)

        # 计算空间注意力权重，形状为 [B, 1, H, W]
        spatial_weight = self.spatial_attention(normalized_feature)
        # 使用空间注意力权重对归一化特征进行空间调制
        spatial_modulated_feature = spatial_weight * normalized_feature
        # 对空间调制后的特征进行 1×1 卷积投影
        spatial_branch_output = self.spatial_projection(spatial_modulated_feature)

        # 将通道分支和空间分支相加，得到空间-通道联合调制结果
        output_feature = channel_branch_output + spatial_branch_output

        # 返回最终输出特征
        return output_feature

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = SpatialChannelFeatureModulator(channels=32)
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")