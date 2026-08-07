import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/11419859/
    论文题目：Visual-in-Visual: A Unified and Efficient Baseline for Image Restoration（TPAMI 2026）
    中文题目：视觉嵌套：一种统一且高效的图像复原基线（TPAMI 2026）
    讲解视频：https://www.bilibili.com/video/BV1BRLK69EYj/
    类脑启发模块（Brain-Inspired Module，BIM）
        实际意义：①高性能与高效率难以兼顾的问题：Transformer、Mamba等方法会带来较高的参数量、计算量。对于真实应用场景，图像复原模型不仅要效果好，还要足够轻量、易部署。
                ②多尺度视觉信息提取不足的问题：局部细节和大范围上下文都非常重要。例如：去雨、去雪需要关注细小条纹和局部噪声；去雾、低光增强需要理解全局亮度和大范围结构。如果只依赖单一尺度卷积，很容易出现局部细节不足或全局上下文感知不充分的问题。
                ③冗余特征和无效响应干扰复原问题：图像退化中往往包含大量冗余信息或干扰信息，例如背景、雨纹、雪花、噪声等。如果模型不能区分哪些特征重要、哪些特征冗余，就容易把退化信息也保留下来。
        实现方式：通过“多尺度卷积编码 + 相似性感知加权 + 高阶特征交互”，以轻量方式模拟人类视觉处理过程，实现有效特征增强、冗余信息抑制。
"""

class BrainInspiredModule(nn.Module):
    def __init__(self, channels):
        super(BrainInspiredModule, self).__init__()

        # 局部分支：1×1 卷积调整通道
        self.local_branch_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # 局部分支：3×3 深度卷积提取局部纹理和边缘
        self.local_depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

        # 上下文分支：1×1 卷积调整通道
        self.context_branch_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # 上下文分支：9×9 深度卷积提取大范围信息
        self.context_depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=9, padding=4, groups=channels
        )

        # MLP 隐藏层通道数，设置为输入通道数的 1.2 倍
        hidden_channels = int(1.2 * channels)

        # 根据通道相似度生成注意力权重
        self.similarity_weight_mlp = nn.Sequential(
            nn.Linear(2 * channels, hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_channels, 1)
        )

        # 高阶交互中的第一个 3×3 深度卷积
        self.high_order_depthwise_conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

        # 高阶交互中的第二个 3×3 深度卷积
        self.high_order_depthwise_conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

        # 1×1 卷积整合最终输出特征
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # 获取输入特征图的形状：[批量大小, 通道数, 高, 宽]
        batch_size, channels, height, width = x.shape

        # 生成局部分支特征，对应小感受野信息：3×3 深度卷积提取局部纹理和边缘
        local_feature = self.local_depthwise_conv(self.local_branch_proj(x))

        # 生成上下文分支特征，对应大感受野信息：9×9 深度卷积提取大范围信息
        context_feature = self.context_depthwise_conv(self.context_branch_proj(x))

        # 拼接两个分支，通道数由 C 变为 2C
        multi_scale_feature = torch.cat([local_feature, context_feature], dim=1)
        # 展平空间维度：[B, 2C, H, W] → [B, 2C, HW]
        flattened_feature = multi_scale_feature.view(batch_size, 2 * channels, -1)
        # 对每个通道的空间响应做 L2 归一化
        normalized_feature = F.normalize(flattened_feature, p=2, dim=-1)
        # 计算通道间余弦相似度：[B, 2C, 2C]
        channel_similarity_matrix = torch.bmm(
            normalized_feature,
            normalized_feature.transpose(1, 2)
        )
        # 将相似度矩阵映射为通道注意力权重：[B, 2C, 1]
        channel_attention = self.similarity_weight_mlp(channel_similarity_matrix)
        # 调整权重形状，方便与特征图相乘：[B, 2C, 1, 1]
        channel_attention = channel_attention.view(batch_size, 2 * channels, 1, 1)

        # 将注意力权重拆成局部分支权重和上下文分支权重
        local_attention, context_attention = torch.split(
            channel_attention, channels, dim=1
        )

        # 对局部分支进行加权增强
        weighted_local_feature = local_feature * local_attention

        # 对上下文分支进行加权增强
        weighted_context_feature = context_feature * context_attention

        # 一阶交互：两个加权特征直接逐元素相乘
        first_order_feature = weighted_local_feature * weighted_context_feature
        # 二阶交互：卷积后再与上下文特征相乘
        second_order_feature = (self.high_order_depthwise_conv1(first_order_feature)* weighted_context_feature)
        # 三阶交互：卷积后再与局部特征相乘
        third_order_feature = (self.high_order_depthwise_conv2(second_order_feature)* weighted_local_feature)

        # 通过 1×1 卷积得到最终输出
        output = self.output_proj(third_order_feature)
        return output

if __name__ == "__main__":
    # 构造一个模拟输入：[B, C, H, W] = [2, 32, 50, 50]
    input_tensor = torch.randn(2, 32, 50, 50)
    # 创建 BIM 模块，输入通道数为 32
    model = BrainInspiredModule(channels=32)
    # 前向传播，得到输出特征
    output_tensor = model(input_tensor)
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")