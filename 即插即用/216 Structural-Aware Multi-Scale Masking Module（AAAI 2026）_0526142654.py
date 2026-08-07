import math
import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/abs/2512.05494
    论文题目：Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation（AAAI 2026）
    中文题目：基于结构感知的解码：融合方向、频域‑空间与结构注意力的医学图像分割（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1MqXwBnEi1/
    结构感知多尺度掩码模块（Structural-Aware Multi-scale Masking Module，SMMM）
        实际意义：①传统跳连接（skip connections）引入大量冗余和无关信息：U-Net等经典架构的跳连接通常采用简单加法融合多层特征，这会导致编码器中大量冗余、低相关特征直接传递到解码器，干扰语义交互，降低分割精度。
                ②空间细节丢失与全局-局部特征难以平衡：简单跳连接容易造成空间细节（尤其是边缘）丢失，同时难以有效整合多尺度上下文，导致模型表现不佳。
                ③跳连接缺乏结构显著性过滤与多尺度优化：传统方法缺少对关键结构区域的针对性筛选，无法动态抑制不重要特征，也未能充分利用多尺度上下文来强化边界。
        实现方式：通过多尺度卷积 + 结构显著性掩码来过滤冗余，并结合膨胀融合优化跳跃连接，有效强化边界与语义交互。
"""

def compute_same_padding(kernel_size, padding=None, dilation=1):
    """根据卷积核大小和空洞率自动计算 same padding。"""
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [
            dilation * (k - 1) + 1 for k in kernel_size
        ]
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
    return padding


class BasicConv(nn.Module):
    """
    标准卷积模块：卷积 + BN + 激活函数
    """

    default_activation = nn.SiLU()

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        activation=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            compute_same_padding(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = (
            self.default_activation
            if activation is True
            else activation if isinstance(activation, nn.Module)
            else nn.Identity()
        )

    def forward(self, input_feature):
        return self.activation(self.batch_norm(self.conv(input_feature)))


class DepthwiseConv(BasicConv):
    """深度卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, activation=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=math.gcd(in_channels, out_channels),
            dilation=dilation,
            activation=activation
        )


class MultiScaleExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 第一阶段多尺度提取
        self.stage1_dwconv3 = DepthwiseConv(channels, channels, kernel_size=3)
        self.stage1_dwconv5 = DepthwiseConv(channels, channels, kernel_size=5)

        # 第二阶段多尺度提取
        self.stage2_dwconv3 = DepthwiseConv(2 * channels, channels, kernel_size=3)
        self.stage2_dwconv5 = DepthwiseConv(2 * channels, channels, kernel_size=5)

        # 特征融合
        self.channel_fusion_conv = nn.Conv2d(
            2 * channels, channels, kernel_size=1, stride=1, padding=0
        )

        self.channel_layer_norm = nn.LayerNorm(channels)

    def forward(self, input_feature):
        # [B, C, H, W] -> [B, H, W, C]
        normalized_feature = input_feature.permute(0, 2, 3, 1)
        normalized_feature = self.channel_layer_norm(normalized_feature)

        # [B, H, W, C] -> [B, C, H, W]
        normalized_feature = normalized_feature.permute(0, 3, 1, 2)

        stage1_feature_k3 = self.stage1_dwconv3(normalized_feature)
        stage1_feature_k5 = self.stage1_dwconv5(normalized_feature)
        stage1_concat_feature = torch.cat([stage1_feature_k3, stage1_feature_k5], dim=1)

        stage2_feature_k3 = self.stage2_dwconv3(stage1_concat_feature)
        stage2_feature_k5 = self.stage2_dwconv5(stage1_concat_feature)
        stage2_concat_feature = torch.cat([stage2_feature_k3, stage2_feature_k5], dim=1)

        fused_feature = self.channel_fusion_conv(stage2_concat_feature)
        return fused_feature


class MaskLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pointwise_mask_conv = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, groups=channels
        )
        self.depthwise_mask_conv_k3 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )
        self.depthwise_mask_conv_k5 = nn.Conv2d(
            channels, channels, kernel_size=5, stride=1, padding=2, groups=channels
        )
        self.channel_softmax = nn.Softmax(dim=1)

    def forward(self, input_feature):
        mask_feature_k1 = self.pointwise_mask_conv(input_feature)
        mask_feature_k3 = self.depthwise_mask_conv_k3(input_feature)
        mask_feature_k5 = self.depthwise_mask_conv_k5(input_feature)

        fused_mask = self.channel_softmax(mask_feature_k1 + mask_feature_k3 + mask_feature_k5)
        return input_feature * fused_mask


class SMMM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 编码器/解码器投影
        self.encoder_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.decoder_projection = nn.Conv2d(channels, channels, kernel_size=1)

        # 共享多尺度特征提取器
        self.shared_multiscale_extractor = MultiScaleExtractor(channels)

        # 掩码层
        self.mask_layer = MaskLayer(channels)

        # 输出融合
        self.dilated_fusion_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2
        )
        self.group_norm = nn.GroupNorm(4, channels)
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, encoder_feature, decoder_feature):
        projected_encoder_feature = self.encoder_projection(encoder_feature)
        projected_decoder_feature = self.decoder_projection(decoder_feature)

        encoder_multiscale_feature = self.shared_multiscale_extractor(projected_encoder_feature)
        decoder_multiscale_feature = self.shared_multiscale_extractor(projected_decoder_feature)

        masked_fusion_feature = (
            self.mask_layer(encoder_multiscale_feature)
            + self.mask_layer(decoder_multiscale_feature)
        )

        output_feature = self.dilated_fusion_conv(masked_fusion_feature)
        output_feature = self.group_norm(output_feature)
        output_feature = self.output_projection(output_feature)

        return output_feature

if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)
    y = torch.randn(1, 32, 50, 50)
    model = SMMM(32)
    output_feature = model(x, y)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")