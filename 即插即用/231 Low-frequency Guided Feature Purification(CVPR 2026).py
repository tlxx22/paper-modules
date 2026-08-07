import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

"""
    论文地址：https://arxiv.org/abs/2508.06878
    论文题目：Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective（CVPR 2026）
    中文题目：穿透噪声：从噪声抑制角度提升红外小目标检测与分割性能（CVPR 2026）
    讲解视频：https://www.bilibili.com/video/BV1BZ5s66EyH/
    低频引导特征提纯模块（Low-frequency Guided Feature Purification，LFP）
        实际意义：①高频特征虽有用，但容易放大噪声问题：红外小目标通常很小、很暗、形状不明显，因此需要依赖高频信息来捕获目标边缘和局部细节突变。但高频成分不仅包含目标细节，也包含大量背景噪声。直接增强高频特征，同时增强目标、噪声导致误检。②特征增强与噪声抑制之间的矛盾问题：传统方法往往难以在“增强目标特征”和“抑制背景噪声”之间取得平衡。只关注特征增强，噪声会被放大；过度平滑，丢失目标细节。
        实现方式：通过“低频定位引导 + 高频细节净化”的方式，既保留红外小目标边缘细节，也可以的抑制背景噪声。
"""

class DiscreteWaveletTransform2D(nn.Module):
    """
    单层二维离散小波正变换模块：
    将输入特征分解为低频子带 LL 和三个高频子带 LH、HL、HH。
    输入:  [B, C, H, W]
    输出:  [B, 4C, H/2, W/2]
    """

    def __init__(self, wavelet_type='haar', padding_mode='zero'):
        super(DiscreteWaveletTransform2D, self).__init__()

        self.forward_wavelet_transform = DWTForward(
            J=1,
            wave=wavelet_type,
            mode=padding_mode
        )

    def forward(self, input_feature):
        batch_size, channels, height, width = input_feature.shape

        with torch.cuda.amp.autocast(enabled=False):
            if input_feature.dtype != torch.float32:
                input_feature = input_feature.float()

            low_frequency_feature, high_frequency_list = self.forward_wavelet_transform(input_feature)

        # high_frequency_list[0]: [B, C, 3, H/2, W/2]
        # 将三个方向的高频子带合并到通道维，得到 [B, 3C, H/2, W/2]
        high_frequency_feature = high_frequency_list[0].transpose(1, 2).reshape(
            batch_size,
            -1,
            high_frequency_list[0].shape[3],
            high_frequency_list[0].shape[4]
        )

        # 拼接低频 LL 与高频 LH/HL/HH，得到 [B, 4C, H/2, W/2]
        wavelet_feature = torch.cat(
            (low_frequency_feature, high_frequency_feature),
            dim=1
        )

        wavelet_feature = F.interpolate(
            wavelet_feature,
            size=(height // 2, width // 2),
            mode='bilinear',
            align_corners=False
        )

        return wavelet_feature


class InverseDiscreteWaveletTransform2D(nn.Module):
    """
    单层二维离散小波逆变换模块：
    根据低频子带 LL 和高频子带 LH/HL/HH 重建空间域特征。
    输入:
        low_frequency_feature:  [B, C, H/2, W/2]
        high_frequency_feature: [B, 3C, H/2, W/2]
    输出:
        reconstructed_feature:  [B, C, H, W]
    """

    def __init__(self, wavelet_type='haar', padding_mode='zero'):
        super(InverseDiscreteWaveletTransform2D, self).__init__()

        self.inverse_wavelet_transform = DWTInverse(
            wave=wavelet_type,
            mode=padding_mode
        )

    def forward(self, low_frequency_feature, high_frequency_feature):
        batch_size, channels, half_height, half_width = low_frequency_feature.shape

        # [B, 3C, H/2, W/2] -> [B, C, 3, H/2, W/2]
        high_frequency_feature = high_frequency_feature.reshape(
            batch_size,
            channels,
            3,
            half_height,
            half_width
        )

        with torch.cuda.amp.autocast(enabled=False):
            reconstructed_feature = self.inverse_wavelet_transform(
                (low_frequency_feature, [high_frequency_feature.float()])
            )

        reconstructed_feature = F.interpolate(
            reconstructed_feature,
            size=(2 * half_height, 2 * half_width),
            mode='bilinear',
            align_corners=False
        )

        return reconstructed_feature


class SpatialAttentionMapGenerator(nn.Module):
    """
    空间注意力图生成模块：
    通过通道平均池化和通道最大池化生成空间权重图。
    输入:  [B, C, H, W]
    输出:  [B, 1, H, W]
    """

    def __init__(self, kernel_size=7, use_bn_before_sigmoid=False):
        super(SpatialAttentionMapGenerator, self).__init__()

        assert kernel_size in (3, 7), 'kernel_size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1
        self.use_bn_before_sigmoid = use_bn_before_sigmoid

        self.spatial_context_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        if use_bn_before_sigmoid:
            self.attention_bn = nn.BatchNorm2d(1)
            self.attention_bn.bias.data.fill_(0)
            self.attention_bn.bias.requires_grad = False

        self.activation = nn.Sigmoid()

    def forward(self, input_feature):
        average_response_map = torch.mean(input_feature, dim=1, keepdim=True)
        maximum_response_map, _ = torch.max(input_feature, dim=1, keepdim=True)

        spatial_descriptor = torch.cat(
            [average_response_map, maximum_response_map],
            dim=1
        )

        attention_logit = self.spatial_context_conv(spatial_descriptor)

        if self.use_bn_before_sigmoid:
            attention_logit = self.attention_bn(attention_logit)

        spatial_attention_map = self.activation(attention_logit)

        return spatial_attention_map

class LearnableGaussianSmoothingBank(nn.Module):
    """
    可学习高斯平滑滤波器组：
    用可学习 sigma 生成高斯核，对高频特征进行深度卷积平滑。
    输入:  [B, C, H, W]
    输出:  [B, num_filters * C, H, W]
    """

    def __init__(self, kernel_size, num_filters, num_channels):
        super(LearnableGaussianSmoothingBank, self).__init__()

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.padding_size = kernel_size // 2

        self.learnable_sigmas = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            for _ in range(num_filters)
        ])

    def forward(self, input_feature):
        gaussian_kernels = [
            self._build_gaussian_kernel(
                kernel_size=self.kernel_size,
                sigma=sigma
            ).repeat(self.num_channels, 1, 1, 1)
            for sigma in self.learnable_sigmas
        ]

        smoothed_feature_list = [
            F.conv2d(
                F.pad(
                    input_feature,
                    (
                        self.padding_size,
                        self.padding_size,
                        self.padding_size,
                        self.padding_size
                    ),
                    mode='replicate'
                ),
                weight=kernel.to(input_feature.device),
                groups=self.num_channels
            )
            for kernel in gaussian_kernels
        ]

        smoothed_feature = torch.cat(smoothed_feature_list, dim=1)

        return smoothed_feature

    def _build_gaussian_kernel(self, kernel_size, sigma):
        gaussian_kernel = torch.zeros(
            1,
            1,
            kernel_size,
            kernel_size,
            dtype=sigma.dtype,
            device=sigma.device
        )

        kernel_center = kernel_size // 2

        for row_index in range(kernel_size):
            for col_index in range(kernel_size):
                gaussian_kernel[:, :, row_index, col_index] = torch.exp(
                    -(
                        (row_index - kernel_center) ** 2
                        + (col_index - kernel_center) ** 2
                    )
                    / (2 * sigma ** 2)
                )

        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        return gaussian_kernel


class LowFrequencyGuidedFeaturePurification(nn.Module):
    """
    低频引导特征净化模块，LFP：

    核心流程：
    1. 使用 DWT 将输入特征分解为低频 LL 和高频 LH/HL/HH；
    2. 使用低频 LL 生成潜在目标位置的空间注意力图；
    3. 用该注意力图调制高频特征，增强目标相关高频，抑制背景噪声高频；
    4. 对低置信高频响应进行门控高斯平滑；
    5. 使用 IDWT 重建空间域特征。

    输入:  [B, C, H, W]
    输出:  [B, C, H, W]
    """

    def __init__(
        self,
        in_channels,
        wavelet_type='haar',
        padding_mode='symmetric',
        enable_gaussian_smoothing=True,
        high_frequency_threshold=0.5
    ):
        super(LowFrequencyGuidedFeaturePurification, self).__init__()

        self.wavelet_decomposition = DiscreteWaveletTransform2D(
            wavelet_type=wavelet_type,
            padding_mode=padding_mode
        )

        self.wavelet_reconstruction = InverseDiscreteWaveletTransform2D(
            wavelet_type=wavelet_type,
            padding_mode=padding_mode
        )

        self.enable_gaussian_smoothing = enable_gaussian_smoothing
        self.high_frequency_threshold = high_frequency_threshold

        self.low_frequency_spatial_attention = SpatialAttentionMapGenerator()

        if self.enable_gaussian_smoothing:
            self.high_frequency_gaussian_smoothing = LearnableGaussianSmoothingBank(
                kernel_size=3,
                num_filters=1,
                num_channels=3 * in_channels
            )

    def forward(self, input_feature):
        batch_size, channels, height, width = input_feature.shape
        # [B, 4C, H/2, W/2] ：单层二维离散小波正变换模块
        wavelet_feature = self.wavelet_decomposition(input_feature)

        # 低频 LL: [B, C, H/2, W/2]
        low_frequency_feature = wavelet_feature[:, :channels, :, :]
        # 高频 LH/HL/HH: [B, 3C, H/2, W/2]
        high_frequency_feature = wavelet_feature[:, channels:, :, :]

        # 由低频特征生成空间注意力图
        target_spatial_weight_map = self.low_frequency_spatial_attention(low_frequency_feature)
        # 低频引导高频调制 【点乘】
        modulated_high_frequency_feature = (high_frequency_feature * target_spatial_weight_map)

        if self.enable_gaussian_smoothing:
            smoothed_high_frequency_feature = self.high_frequency_gaussian_smoothing(
                modulated_high_frequency_feature
            )

            # 小幅值高频响应通常更可能是不稳定噪声，因此使用高斯平滑结果替代
            low_confidence_high_frequency_mask = (
                modulated_high_frequency_feature.abs() < self.high_frequency_threshold
            ).float()

            purified_high_frequency_feature = (
                modulated_high_frequency_feature
                * (1.0 - low_confidence_high_frequency_mask)
                + smoothed_high_frequency_feature
                * low_confidence_high_frequency_mask
            )
        else:
            purified_high_frequency_feature = modulated_high_frequency_feature
        #  单层二维离散小波逆变换模块
        purified_feature = self.wavelet_reconstruction(low_frequency_feature,purified_high_frequency_feature)
        return purified_feature

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = LowFrequencyGuidedFeaturePurification(
        in_channels=32,
        wavelet_type='haar',
        padding_mode='symmetric',
        enable_gaussian_smoothing=True,
        high_frequency_threshold=0.5
    )
    output_tensor = model(input_tensor)
    print("输入特征维度：", input_tensor.shape)
    print("输出特征维度：", output_tensor.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")