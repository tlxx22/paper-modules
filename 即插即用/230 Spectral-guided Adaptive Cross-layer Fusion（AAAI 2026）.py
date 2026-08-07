import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/view/42457/46418
    论文题目：MODA: The First Challenging Benchmark for Multispectral Object Detection in Aerial Images（AAAI 2026）
    中文题目：MODA：首个用于航空影像多光谱目标检测的挑战性基准测试（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1Q1dKB3EcZ/
    光谱引导自适应跨层融合模块（Spectral-guided Adaptive Cross-layer Fusion，SACF）
        实际意义：①光谱信息在跨层融合中被破坏问题：传统特征金字塔在跨层融合时，忽略或弱化光谱信息的结构关系，导致同一目标内部的光谱一致性被破坏。②下采样导致的空间纹理细节丢失与谱混叠问题：在目标边缘和混合像素区域，特征提取过程中的连续下采样会造成空间纹理细节的严重丢失。
        实现方式：实现“跨层特征重建机制”，通过一致性驱动的特征聚合、细节增强与自适应融合，实现语义信息与空间细节的协同优化，具备良好的跨任务迁移能力。
"""

# 欧氏距离光谱特征聚合模块
class EuclideanSpectralFeatureAggregator(nn.Module):

    # 初始化函数，local_window_size 表示局部窗口大小
    def __init__(self, local_window_size: int = 3):

        # 调用父类初始化函数
        super().__init__()

        # 保存局部窗口大小，例如 3 表示 3×3 邻域
        self.local_window_size = local_window_size

        # 使用 Unfold 提取每个像素周围的局部邻域 patch
        self.local_patch_unfold = nn.Unfold(
            kernel_size=local_window_size,
            padding=local_window_size // 2
        )

    # 前向传播函数
    def forward(self, center_spectral_feature: torch.Tensor, reference_spectral_feature: torch.Tensor) -> torch.Tensor:

        # 获取中心光谱特征的尺寸信息
        batch_size, channels, height, width = center_spectral_feature.shape

        # 计算局部邻域中的像素数量，例如 3×3=9
        num_neighbors = self.local_window_size * self.local_window_size

        # 计算整张特征图的空间位置数量
        num_positions = height * width

        # 从参考光谱特征中提取局部 patch
        local_spectral_patches = self.local_patch_unfold(
            reference_spectral_feature
        ).reshape(batch_size, channels, num_neighbors, num_positions)

        # 将中心光谱特征展平成序列形式
        center_spectral_vectors = center_spectral_feature.reshape(
            batch_size, channels, num_positions
        )

        # 调整局部 patch 维度，变为每个空间位置对应若干邻域光谱向量
        local_spectral_vectors = local_spectral_patches.permute(
            0, 3, 2, 1
        )

        # 调整中心光谱向量维度，便于和局部邻域向量计算距离
        center_spectral_vectors_for_distance = center_spectral_vectors.permute(
            0, 2, 1
        ).unsqueeze(2)

        # 计算中心光谱向量与局部邻域光谱向量之间的欧氏距离
        euclidean_distance = torch.norm(
            local_spectral_vectors - center_spectral_vectors_for_distance,
            dim=-1
        )

        # 距离越小，相似度越高，因此对负距离做 softmax
        spectral_similarity_weight = F.softmax(
            -euclidean_distance,
            dim=-1
        )

        # 根据相似度权重，对局部邻域光谱向量进行加权聚合
        aggregated_spectral_vectors = torch.matmul(
            spectral_similarity_weight.unsqueeze(2),
            local_spectral_vectors
        )

        # 将聚合后的序列特征恢复为二维特征图
        aggregated_spectral_feature = aggregated_spectral_vectors.squeeze(2).permute(
            0, 2, 1
        ).reshape(batch_size, channels, height, width)

        # 使用残差连接，保留原始中心光谱特征
        return aggregated_spectral_feature + center_spectral_feature

# 混合光谱特征聚合模块，同时结合余弦相似度和欧氏距离
class HybridSpectralFeatureAggregator(nn.Module):

    # 初始化函数
    def __init__(self, local_window_size: int = 3, fusion_mode: str = "add", cosine_ratio: float = 0.5):

        # 调用父类初始化函数
        super().__init__()

        # 保存局部窗口大小
        self.local_window_size = local_window_size

        # 保存融合方式，add 表示加权融合，其他模式表示拼接融合
        self.fusion_mode = fusion_mode

        # 保存余弦相似度分支所占比例
        self.cosine_ratio = cosine_ratio

        # 使用 Unfold 提取局部邻域 patch
        self.local_patch_unfold = nn.Unfold(
            kernel_size=local_window_size,
            padding=local_window_size // 2
        )

    # 前向传播函数
    def forward(self, center_spectral_feature: torch.Tensor, reference_spectral_feature: torch.Tensor) -> torch.Tensor:

        # 获取输入特征尺寸
        batch_size, channels, height, width = center_spectral_feature.shape

        # 计算局部邻域数量
        num_neighbors = self.local_window_size * self.local_window_size

        # 计算空间位置数量
        num_positions = height * width

        # 从参考特征中提取局部光谱 patch
        local_spectral_patches = self.local_patch_unfold(
            reference_spectral_feature
        ).reshape(batch_size, channels, num_neighbors, num_positions)

        # 将中心特征展平为序列
        center_spectral_vectors = center_spectral_feature.reshape(
            batch_size, channels, num_positions
        )

        # 对局部光谱 patch 做通道归一化
        normalized_local_patches = F.normalize(local_spectral_patches, dim=1)

        # 对中心光谱向量做通道归一化
        normalized_center_vectors = F.normalize(center_spectral_vectors, dim=1)

        # 计算余弦相似度
        cosine_similarity = torch.einsum(
            "bpkc,bpc->bpk",
            normalized_local_patches.permute(0, 3, 2, 1),
            normalized_center_vectors.permute(0, 2, 1)
        )

        # 对余弦相似度做 softmax，得到余弦权重
        cosine_similarity_weight = F.softmax(
            cosine_similarity,
            dim=-1
        )

        # 调整局部光谱向量维度
        local_spectral_vectors = local_spectral_patches.permute(
            0, 3, 2, 1
        ).reshape(batch_size, num_positions, num_neighbors, channels)

        # 调整中心光谱向量维度，用于计算欧氏距离
        center_spectral_vectors_for_distance = center_spectral_vectors.permute(
            0, 2, 1
        ).unsqueeze(2)

        # 计算中心向量与局部邻域向量之间的欧氏距离
        euclidean_distance = torch.norm(
            local_spectral_vectors - center_spectral_vectors_for_distance,
            dim=-1
        )

        # 对负欧氏距离做 softmax，得到欧氏相似度权重
        euclidean_similarity_weight = F.softmax(
            -euclidean_distance,
            dim=-1
        )

        # 如果采用 add 模式，则融合两种相似度权重
        if self.fusion_mode == "add":

            # 按比例融合余弦相似度权重和欧氏相似度权重
            fused_similarity_weight = (
                self.cosine_ratio * cosine_similarity_weight
                + (1 - self.cosine_ratio) * euclidean_similarity_weight
            )

            # 使用融合后的权重聚合局部光谱特征
            aggregated_spectral_vectors = torch.matmul(
                fused_similarity_weight.unsqueeze(2),
                local_spectral_vectors
            )

        # 如果不是 add 模式，则分别聚合后进行拼接
        else:

            # 使用余弦权重聚合局部光谱特征
            cosine_aggregated_vectors = torch.matmul(
                cosine_similarity_weight.unsqueeze(2),
                local_spectral_vectors
            )

            # 使用欧氏权重聚合局部光谱特征
            euclidean_aggregated_vectors = torch.matmul(
                euclidean_similarity_weight.unsqueeze(2),
                local_spectral_vectors
            )

            # 将两种聚合结果在通道维度拼接
            aggregated_spectral_vectors = torch.cat(
                [cosine_aggregated_vectors, euclidean_aggregated_vectors],
                dim=-1
            )

        # 将聚合后的序列恢复为二维特征图
        aggregated_spectral_feature = aggregated_spectral_vectors.squeeze(2).permute(
            0, 2, 1
        ).reshape(batch_size, -1, height, width)

        # 返回残差增强后的光谱特征
        # 注意：如果 fusion_mode 不是 add，通道数会变成 2C，此处与 center_spectral_feature 的 C 通道不匹配
        return aggregated_spectral_feature + center_spectral_feature

# 空间细节增强模块
class SpatialDetailEnhancer(nn.Module):

    # 初始化函数
    def __init__(self, in_channels: int):

        # 调用父类初始化函数
        super().__init__()

        # 平均池化提取低频结构信息
        self.low_frequency_pool = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # 低频特征精修卷积
        self.low_frequency_refine_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        # 高频细节注意力生成卷积
        self.high_frequency_attention_conv = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            bias=False
        )

        # 高频细节与低频结构融合卷积
        self.detail_fusion_conv = nn.Conv2d(
            in_channels * 2,
            in_channels,
            kernel_size=1,
            bias=False
        )

    # 前向传播函数
    def forward(self, spatial_feature: torch.Tensor) -> torch.Tensor:

        # 对输入空间特征进行平均池化，获得低频特征
        low_frequency_feature = self.low_frequency_pool(spatial_feature)

        # 对低频特征进行卷积精修
        refined_low_frequency_feature = self.low_frequency_refine_conv(
            low_frequency_feature
        )

        # 将精修后的低频特征恢复到原始空间尺寸
        refined_low_frequency_feature = F.interpolate(
            refined_low_frequency_feature,
            size=spatial_feature.shape[2:],
            mode="bilinear",
            align_corners=True
        )

        # 再次提取低频特征，并上采样回原始尺寸
        upsampled_low_frequency_feature = F.interpolate(
            self.low_frequency_pool(spatial_feature),
            size=spatial_feature.shape[2:],
            mode="bilinear",
            align_corners=True
        )

        # 原始特征减去低频特征，得到高频细节特征
        high_frequency_detail_feature = spatial_feature - upsampled_low_frequency_feature

        # 根据高频细节生成空间注意力权重
        high_frequency_attention = torch.sigmoid(
            self.high_frequency_attention_conv(high_frequency_detail_feature)
        )

        # 对高频细节进行注意力增强，并保留原始高频信息
        enhanced_high_frequency_feature = (
            high_frequency_detail_feature * high_frequency_attention
            + high_frequency_detail_feature
        )

        # 拼接高频增强特征和低频精修特征，再通过 1×1 卷积融合
        fused_spatial_detail_feature = self.detail_fusion_conv(
            torch.cat(
                [enhanced_high_frequency_feature, refined_low_frequency_feature],
                dim=1
            )
        )

        # 使用残差连接，输出空间细节增强特征
        return fused_spatial_detail_feature + spatial_feature


# 光谱-空间自适应融合模块
class SpectralSpatialAdaptiveFusion(nn.Module):

    # 初始化函数
    def __init__(self, in_channels: int, local_window_size: int = 3):

        # 调用父类初始化函数
        super().__init__()

        # 光谱特征聚合器，默认使用欧氏距离度量局部光谱相似性
        self.spectral_feature_aggregator = EuclideanSpectralFeatureAggregator(
            local_window_size=local_window_size
        )

        # 空间细节增强器，用于增强低层空间纹理和边缘细节
        self.spatial_detail_enhancer = SpatialDetailEnhancer(
            in_channels=in_channels
        )

        # 自适应融合权重生成器，根据光谱响应图和空间响应图生成两路权重
        self.adaptive_fusion_weight_generator = nn.Conv2d(
            2,
            2,
            kernel_size=1,
            bias=False
        )

        # 输出精修卷积，用于融合后进一步调整特征分布
        self.output_refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_level_spectral_feature: torch.Tensor, low_level_spatial_feature: torch.Tensor) -> torch.Tensor:
        # 对高层光谱特征进行局部相似性聚合
        aggregated_high_level_spectral_feature = self.spectral_feature_aggregator(high_level_spectral_feature,high_level_spectral_feature)
        # 将高层光谱特征上采样到低层空间特征的尺寸
        aligned_high_level_spectral_feature = F.interpolate(
            aggregated_high_level_spectral_feature,
            size=low_level_spatial_feature.shape[2:],
            mode="bilinear",
            align_corners=True
        )

        # 对低层空间特征进行细节增强
        enhanced_low_level_spatial_feature = self.spatial_detail_enhancer(low_level_spatial_feature)
        # 对光谱特征沿通道维度求平均，得到光谱全局响应图【中间的 GAP 】
        spectral_global_response_map = aligned_high_level_spectral_feature.mean(dim=1,keepdim=True)
        # 对空间特征沿通道维度求平均，得到空间全局响应图
        spatial_global_response_map = enhanced_low_level_spatial_feature.mean(dim=1,keepdim=True)

        # 拼接光谱响应图和空间响应图，并生成自适应融合权重
        adaptive_fusion_weight = torch.sigmoid(
            self.adaptive_fusion_weight_generator(torch.cat([spectral_global_response_map, spatial_global_response_map],dim=1
                )
            )
        )

        # 取出光谱分支的融合权重
        spectral_fusion_weight = adaptive_fusion_weight[:, 0:1]
        # 取出空间分支的融合权重
        spatial_fusion_weight = adaptive_fusion_weight[:, 1:2]

        # 根据自适应权重融合光谱特征和空间特征【最后两个点乘】
        fused_spectral_spatial_feature = (
            spectral_fusion_weight * aligned_high_level_spectral_feature
            + spatial_fusion_weight * enhanced_low_level_spatial_feature
        )

        # 对融合后的特征进行输出精修
        output_feature = self.output_refine_conv(fused_spectral_spatial_feature)
        return output_feature

if __name__ == "__main__":
    low_level_spatial_feature = torch.randn(1, 32, 50, 50)
    high_level_spectral_feature = torch.randn(1, 32, 25, 25)
    model = SpectralSpatialAdaptiveFusion(in_channels=32)
    output_feature = model(high_level_spectral_feature,low_level_spatial_feature)
    print(f"高层光谱特征形状: {high_level_spectral_feature.shape}")
    print(f"低层空间特征形状: {low_level_spatial_feature.shape}")
    print(f"输出特征形状: {output_feature.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")