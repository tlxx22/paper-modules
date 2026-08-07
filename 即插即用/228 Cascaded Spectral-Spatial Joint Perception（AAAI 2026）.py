import torch
import torch.nn as nn

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/view/42457/46418
    论文题目：MODA: The First Challenging Benchmark for Multispectral Object Detection in Aerial Images（AAAI 2026）
    中文题目：MODA：首个用于航空影像多光谱目标检测的挑战性基准测试（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1289mB1EQZ/
    级联光谱‑空间联合感知模块（Cascaded Spectral-Spatial Joint Perception，CSSP）
        实际意义：①高层特征中的空间混叠问题：多次下采样会引入空间混叠，使背景信息混入目标区域，削弱模型对目标关注能力。进而导致小目标在深层特征中被淹没，背景纹理被错误当作目标特征，模型注意力分散，检测性能下降。
                ②光谱信息与空间信息未被有效联合建模问题：传统方法存在两个典型问题：要么只关注空间信息；要么将光谱与空间信息解耦处理，导致光谱信息损失，计算复杂度增加。
        实现方式：通过光谱与空间特征的双向交互建模，实现目标区域的协同增强与背景抑制。
"""
# 光谱通道注意力模块：用于判断不同通道的重要性
class SpectralChannelAttention(nn.Module):

    # 初始化光谱通道注意力模块
    # in_channels：输入特征图的通道数
    # reduction_ratio：通道压缩比例，用于减少参数量和计算量
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        # 计算中间层通道数
        # 例如 in_channels=32，reduction_ratio=16，则 bottleneck_channels=2
        # max(1, ...) 防止通道数过小导致结果为 0
        bottleneck_channels = max(1, in_channels // reduction_ratio)

        # 全局平均池化
        # 将每个通道的 H×W 空间信息压缩成 1×1
        # 输入：[B, C, H, W]
        # 输出：[B, C, 1, 1]
        self.global_average_pool = nn.AdaptiveAvgPool2d(1)

        # 通道 MLP，用于生成每个通道的重要性权重
        self.channel_mlp = nn.Sequential(

            # 第一个 1×1 卷积：压缩通道数
            # 作用类似全连接层中的降维操作
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),

            # ReLU 激活函数：增强非线性表达能力
            nn.ReLU(inplace=True),

            # 第二个 1×1 卷积：恢复到原始通道数
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, bias=False)
        )

        # Sigmoid 函数：将通道权重压缩到 0~1 之间
        self.activation = nn.Sigmoid()

    # 前向传播函数
    # input_feature：输入特征图，形状为 [B, C, H, W]
    def forward(self, input_feature):

        # 生成光谱/通道注意力权重
        # global_average_pool 后形状为 [B, C, 1, 1]
        # channel_mlp 后形状仍为 [B, C, 1, 1]
        # Sigmoid 后得到每个通道的重要性权重
        spectral_weight = self.activation(
            self.channel_mlp(
                self.global_average_pool(input_feature)
            )
        )

        # 将通道权重乘回原始特征
        # spectral_weight 会通过广播机制扩展到 [B, C, H, W]
        spectral_enhanced_feature = input_feature * spectral_weight

        # 返回经过通道注意力增强后的特征图
        return spectral_enhanced_feature


# 空间区域注意力模块：用于判断不同空间位置的重要性
class SpatialRegionAttention(nn.Module):

    # 初始化空间区域注意力模块
    # attention_kernel_size：空间注意力卷积核大小，默认使用 7×7
    def __init__(self, attention_kernel_size=7):
        super().__init__()

        # 计算 padding 大小
        # 当 kernel_size=7 时，padding=3
        # 这样可以保证卷积前后空间尺寸 H 和 W 不变
        padding_size = attention_kernel_size // 2

        # 空间注意力卷积层
        # 输入通道为 1，因为前面会先对通道维度求平均
        # 输出通道仍为 1，表示每个空间位置的注意力权重
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=attention_kernel_size,
            padding=padding_size,
            bias=False
        )

        # Sigmoid 函数：将空间权重限制在 0~1 之间
        self.activation = nn.Sigmoid()

    # 前向传播函数
    # input_feature：输入特征图，形状为 [B, C, H, W]
    def forward(self, input_feature):

        # 沿通道维度求平均，得到空间响应图
        # 输入形状：[B, C, H, W]
        # 输出形状：[B, 1, H, W]
        # 每个位置的值表示该空间位置在所有通道上的平均响应强度
        channel_average_map = torch.mean(input_feature, dim=1, keepdim=True)

        # 使用卷积进一步建模局部空间区域关系
        # 再通过 Sigmoid 得到空间注意力权重
        # 输出形状：[B, 1, H, W]
        spatial_weight = self.activation(self.spatial_conv(channel_average_map))

        # 将空间权重乘回原始特征
        # spatial_weight 会广播到所有通道
        # 输出形状仍为 [B, C, H, W]
        spatial_enhanced_feature = input_feature * spatial_weight

        # 返回经过空间区域注意力增强后的特征图
        return spatial_enhanced_feature


# 级联式光谱-空间联合感知模块
# 核心思想：
# 1. 分别提取光谱/通道增强特征和空间增强特征
# 2. 计算两类特征之间的相关性
# 3. 通过相关性矩阵实现光谱信息与空间信息的交互
# 4. 最后融合两路特征，输出增强后的特征图
class CascadedSpectralSpatialJointPerception(nn.Module):
    # 初始化级联式光谱-空间联合感知模块
    # in_channels：输入特征通道数
    # reduction_ratio：光谱通道注意力中的通道压缩比例
    # spatial_kernel_size：空间注意力中的卷积核大小
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()

        # 光谱/通道注意力分支
        # 用于突出重要光谱通道或语义通道
        self.spectral_attention_branch = SpectralChannelAttention(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio
        )

        # 空间区域注意力分支
        # 用于突出重要空间位置或目标区域
        self.spatial_attention_branch = SpatialRegionAttention(
            attention_kernel_size=spatial_kernel_size
        )

        # 光谱-空间特征融合层
        self.spectral_spatial_fusion = nn.Sequential(

            # 使用 1×1 卷积融合拼接后的特征
            # 输入通道数为 2C，输出通道数恢复为 C
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),

            # 批归一化，加速训练并稳定特征分布
            nn.BatchNorm2d(in_channels),

            # ReLU 激活函数，增强非线性表达能力
            nn.ReLU(inplace=True)
        )

    # 对相关性矩阵进行全局 L2 归一化
    # correlation_matrix：输入的相关性矩阵
    # eps：防止除零的小常数
    def _global_l2_normalize(self, correlation_matrix, eps=1e-6):

        # 计算整个相关性矩阵的 L2 范数
        # 注意：这里没有指定 dim，因此是对整个张量做全局归一化
        l2_norm = torch.norm(correlation_matrix, p=2)

        # 将相关性矩阵除以其 L2 范数
        # 使相关性数值更加稳定，避免过大
        normalized_matrix = correlation_matrix / (l2_norm + eps)

        # 返回归一化后的相关性矩阵
        return normalized_matrix

    def forward(self, input_feature):
        # 读取输入特征图的批量大小、通道数、高度和宽度
        batch_size, channels, height, width = input_feature.shape
        # 计算空间 token 数量，每一个像素位置可以看作一个 token，因此 token 数量为 H×W
        num_spatial_tokens = height * width

        # 得到光谱感知特征：原始信息+通道增强特征，并将其展平成Token形式 [B, C, H, W]===>新形状：[B, C, H×W]
        spectral_aware_feature = input_feature + self.spectral_attention_branch(input_feature)
        spectral_tokens = spectral_aware_feature.view(batch_size, channels, num_spatial_tokens)

        # 得到空间感知特征：保留原始信息+空间增强特征，并将其展平成Token形式 [B, C, H, W]===>新形状：[B, C, H×W]
        spatial_aware_feature = input_feature + self.spatial_attention_branch(input_feature)
        spatial_tokens = spatial_aware_feature.view(batch_size, channels, num_spatial_tokens)

        # 计算光谱感知特征与空间感知特征的相关性矩阵
        spectral_to_spatial_correlation = torch.matmul(spectral_tokens.transpose(1, 2),spatial_tokens)
        # 先对相关性矩阵进行 L2 归一化，稳定数值：通过 tanh 将值限制在 -1 到 1 之间
        spectral_to_spatial_correlation = torch.tanh(self._global_l2_normalize(spectral_to_spatial_correlation))
        # 使用光谱到空间的相关性矩阵调制特征【下加号 与 相乘】
        spatial_modulated_tokens = torch.matmul(spectral_tokens,spectral_to_spatial_correlation) + spatial_tokens

        # 计算空间调制特征与光谱感知特征的相关性矩阵
        spatial_to_spectral_correlation = torch.matmul(spatial_modulated_tokens.transpose(1, 2),spectral_tokens)
        # 对第二个相关性矩阵进行归一化和 tanh 限幅：进一步稳定跨分支交互过程
        spatial_to_spectral_correlation = torch.tanh(self._global_l2_normalize(spatial_to_spectral_correlation))
        # 使用空间到光谱的相关性矩阵进一步调制特征【上加号 与 相乘】
        spectral_modulated_tokens = torch.matmul(spatial_modulated_tokens,spatial_to_spectral_correlation) + spectral_tokens

        # 将光谱调制后的 token 恢复为二维特征图
        spectral_modulated_feature = spectral_modulated_tokens.view(batch_size, channels, height, width)
        # 将空间调制后的 token 恢复为二维特征图
        spatial_modulated_feature = spatial_modulated_tokens.view(batch_size, channels, height, width)
        # 在通道维度拼接光谱调制特征和空间调制特征
        fused_spectral_spatial_feature = torch.cat([spectral_modulated_feature, spatial_modulated_feature],dim=1)
        # 使用 1×1 卷积 + BN + ReLU 融合两路特征，将通道数从 2C 压缩回 C
        output_feature = self.spectral_spatial_fusion(fused_spectral_spatial_feature)
        # 返回最终的光谱-空间联合增强特征图
        return output_feature

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = CascadedSpectralSpatialJointPerception(in_channels=32)
    output_tensor = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")