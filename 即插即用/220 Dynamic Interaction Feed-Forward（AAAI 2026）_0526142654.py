import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/download/38042/42004
    论文题目：Gradient as Conditions: Rethinking HOG for All-in-one Image Restoration（AAAI 2026）
    中文题目：梯度作为条件：为一体化图像恢复重新思考方向梯度直方图（HOG）的应用（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1QxDhBYEi2/
        动态交互前馈网络（Dynamic Interaction Feed-Forward，DIFF）
        实际意义：①传统 FFN 缺乏内容自适应能力问题：在标准 Transformer 中，FFN通常是固定的，不同输入区域基本采用相似的处理方式。但在图像恢复任务中，不同区域的退化程度、类型和结构往往并不一样，传统 FFN 难以做到“按区域、按内容”进行有针对性的增强。
                ②难以同时兼顾空间信息与通道信息的动态交互问题：模型不仅要知道“哪里退化严重”，还要知道“哪些通道特征更重要”。普通FFN对空间与通道维度之间的联合建模能力不足，因此不利于复杂场景下，调整精细特征。
        实现方式：通过多尺度空间建模、像素级动态门控和跨通道交互，实现了面向不同区域的自适应特征增强。
"""

def channel_shuffle_by_groups(feature_map, num_groups):
    """
    通道打乱
    作用：增强不同通道组之间的信息交互
    输入:
        feature_map: [B, C, H, W]
        num_groups : 分组数
    输出:
        shuffled_feature: [B, C, H, W]
    """
    batch_size, num_channels, height, width = feature_map.size()
    channels_per_group = num_channels // num_groups

    grouped_feature = feature_map.view(
        batch_size, num_groups, channels_per_group, height, width
    )
    shuffled_feature = torch.transpose(grouped_feature, 1, 2).contiguous()
    shuffled_feature = shuffled_feature.view(batch_size, -1, height, width)

    return shuffled_feature


class LearnableChannelScaler(nn.Module):
    """
    可学习的逐通道缩放因子
    作用：对输入特征进行逐元素/逐通道自适应缩放
    """

    def __init__(self, num_channels, init_scale=0.0, requires_grad=True):
        super(LearnableChannelScaler, self).__init__()
        self.channel_scale = nn.Parameter(
            init_scale * torch.ones((1, num_channels, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, feature_map):
        return feature_map * self.channel_scale


class DynamicInteractionFFN(nn.Module):
    """
    Dynamic Interaction Feed-Forward (DIFF)

    核心思路：
    1. 先用 1x1 卷积做通道扩展
    2. 用 PixelShuffle 进行空间重排，增强空间交互
    3. 分成两支，分别提取不同感受野的空间特征
    4. 通过门控相乘实现动态特征交互
    5. 通过特征分解模块进一步做内容校正
    6. 用 PixelUnshuffle 恢复尺度，再投影回原始通道数
    """

    def __init__(self, in_channels, expansion_ratio=2, bias=False):
        super(DynamicInteractionFFN, self).__init__()

        expanded_channels = int(in_channels * expansion_ratio)

        # 用于特征分解校正的可学习缩放参数
        self.feature_residual_scale = LearnableChannelScaler(
            num_channels=expanded_channels // 4,
            init_scale=1e-5,
            requires_grad=True
        )

        # 特征分解：将通道特征压缩为单通道响应图
        self.feature_decompose_conv = nn.Conv2d(
            in_channels=expanded_channels // 4,
            out_channels=1,
            kernel_size=1
        )
        self.feature_decompose_act = nn.GELU()

        # 输入投影：通道扩展
        self.input_projection = nn.Conv2d(
            in_channels, expanded_channels * 2, kernel_size=1, bias=bias
        )

        # 分支1：5x5 深度卷积，偏向局部较大感受野
        self.branch_large_kernel_dwconv = nn.Conv2d(
            expanded_channels // 4,
            expanded_channels // 4,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=expanded_channels // 4,
            bias=bias
        )

        # 分支2：3x3 空洞深度卷积，偏向稀疏感受野建模
        self.branch_dilated_dwconv = nn.Conv2d(
            expanded_channels // 4,
            expanded_channels // 4,
            kernel_size=3,
            stride=1,
            padding=2,
            groups=expanded_channels // 4,
            bias=bias,
            dilation=2
        )

        # 空间重排
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # 输出投影：恢复到原始通道数
        self.output_projection = nn.Conv2d(
            expanded_channels, in_channels, kernel_size=1, bias=bias
        )

    def feature_decomposition(self, fused_feature):
        """
        特征分解与残差校正
        作用：
            利用单通道响应图提取公共/冗余信息，
            再通过可学习缩放进行残差式特征增强
        """
        decomposed_response = self.feature_decompose_act(
            self.feature_decompose_conv(fused_feature)
        )
        refined_feature = fused_feature + self.feature_residual_scale(
            fused_feature - decomposed_response
        )
        return refined_feature

    def forward(self, input_feature):
        # 1. 输入投影，先做通道扩展
        expanded_feature = self.input_projection(input_feature)

        # 2. PixelShuffle：把一部分通道信息重排到空间维度
        spatially_rearranged_feature = self.pixel_shuffle(expanded_feature)

        # 3. 通道打乱，增强跨通道交互
        shuffled_feature = channel_shuffle_by_groups(
            spatially_rearranged_feature, num_groups=1
        )

        # 4. 分成两支，分别建模不同空间模式【两个DW卷积】
        local_branch_feature, context_branch_feature = shuffled_feature.chunk(2, dim=1)
        local_branch_feature = self.branch_large_kernel_dwconv(local_branch_feature)
        context_branch_feature = self.branch_dilated_dwconv(context_branch_feature)

        # 5. 动态门控交互
        gated_fused_feature = F.mish(context_branch_feature) * local_branch_feature

        # 6. 特征分解与校正
        refined_feature = self.feature_decomposition(gated_fused_feature)

        # 7. PixelUnshuffle：恢复原始空间尺度对应的通道布局
        restored_feature = self.pixel_unshuffle(refined_feature)

        # 8. 输出投影，映射回输入通道数
        output_feature = self.output_projection(restored_feature)
        return output_feature

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = DynamicInteractionFFN(in_channels=32)
    output_tensor = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")