import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/view/42457/46418
    论文题目：MODA: The First Challenging Benchmark for Multispectral Object Detection in Aerial Images（AAAI 2026）
    中文题目：MODA：首个用于航空影像多光谱目标检测的挑战性基准测试（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1NcRWBGENy/
    跨谱注意力特征细化模块（Cross-spectral Attention Feature Refinement，CAFR）
        实际意义：①跨层特征存在光谱偏差问题：高层特征语义强，但空间细节弱；低层特征空间细节丰富，但语义弱。不同层的光谱响应分布存在不一致，即“光谱偏差”。这会导致跨层融合时信息不对齐、特征被破坏、最终影响目标判别能力。
                ②多尺度特征融合缺乏“目标一致性”问题：特征融合通常是直接加和或拼接，缺少“目标导向”的约束，导致背景信息被放大，小目标特征被淹没，目标信息在不同层传播过程中逐渐衰减。
        实现方式：通过构建跨层光谱注意力，将低层目标细节与高层语义信息进行自适应融合，从而实现目标导向的特征精细化表达。
"""

# 跨光谱注意力特征精修模块
# 核心作用：利用高层语义特征与低层细节特征之间的“通道相关性”，实现自适应融合
class CrossSpectralAttentionFeatureRefinement(nn.Module):
    def __init__(self, in_channels):
        # in_channels：输入特征的通道数
        super().__init__()

        # 低层特征对齐卷积（用于增强局部表达 + 通道对齐）
        self.low_level_feature_alignment = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1
        )

        # 全局光谱投影层（建模通道之间的全局关系）
        self.global_spectral_projection = nn.Linear(
            in_channels,
            in_channels
        )

    # 前向传播函数
    def forward(self, high_level_feature, low_level_feature):
        # 获取高层特征的尺寸信息
        batch_size, channels, high_height, high_width = high_level_feature.shape

        # 对低层特征进行卷积对齐（增强表达能力）
        aligned_low_level_feature = self.low_level_feature_alignment(
            low_level_feature
        )
        # 将低层特征 resize 到高层特征尺寸（实现跨层对齐）
        aligned_low_level_feature = F.interpolate(
            aligned_low_level_feature,
            size=(high_height, high_width),
            mode="bilinear",
            align_corners=False
        )

        # 对高层特征进行全局平均池化，提取全局通道描述
        # [B, C, H, W] -> [B, C]
        high_level_global_descriptor = F.adaptive_avg_pool2d(
            high_level_feature,
            output_size=1
        ).flatten(1)

        # 对低层特征进行全局平均池化，提取全局通道描述
        low_level_global_descriptor = F.adaptive_avg_pool2d(
            aligned_low_level_feature,
            output_size=1
        ).flatten(1)

        # 将高层通道描述映射到光谱空间，并扩展维度【中间的线性层】
        high_level_spectral_vector = self.global_spectral_projection(high_level_global_descriptor).unsqueeze(-1)
        # 将低层通道描述映射到光谱空间，并扩展维度【中间的线性层】
        low_level_spectral_vector = self.global_spectral_projection(low_level_global_descriptor).unsqueeze(-1)

        # 计算跨光谱注意力矩阵（通道×通道相关性）【中间第一个乘号】
        cross_spectral_attention_map = torch.matmul(high_level_spectral_vector,low_level_spectral_vector.transpose(1, 2))
        # 对注意力矩阵做 softmax 归一化
        cross_spectral_attention_map = F.softmax(cross_spectral_attention_map,dim=-1)

        # 生成高层特征的融合权重【右侧两个乘号1】
        high_level_fusion_weight = torch.matmul(cross_spectral_attention_map.transpose(-1, -2),high_level_spectral_vector).unsqueeze(-1)
        # 生成低层特征的融合权重【右侧两个乘号2】
        low_level_fusion_weight = torch.matmul(cross_spectral_attention_map,low_level_spectral_vector).unsqueeze(-1)

        # 使用权重对高层特征和低层特征进行加权融合【最后一个加号】
        refined_feature = (high_level_fusion_weight * high_level_feature+ low_level_fusion_weight * aligned_low_level_feature)
        return refined_feature

if __name__ == "__main__":
    high_level_feature = torch.randn(2, 32, 25, 25)
    low_level_feature = torch.randn(2, 32, 50, 50)
    model = CrossSpectralAttentionFeatureRefinement(in_channels=32)
    output_feature = model(high_level_feature, low_level_feature)
    print(f"高层输入特征形状: {high_level_feature.shape}")
    print(f"低层输入特征形状: {low_level_feature.shape}")
    print(f"输出特征形状: {output_feature.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")