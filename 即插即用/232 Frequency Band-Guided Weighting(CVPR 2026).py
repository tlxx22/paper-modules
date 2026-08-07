import torch
import torch.nn as nn


"""
    论文地址：https://arxiv.org/pdf/2603.18834
    论文题目：Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging（CVPR 2026）
    中文题目：统计特征引导的快速高分辨透射电子显微成像去噪方法（CVPR 2026）
    讲解视频：https://www.bilibili.com/video/BV1zP5T69EBP/
    频带引导加权模块（Frequency Band-Guided Weighting，FBGW）
        实际意义：①强噪声下原子信号被淹没的问题：随机噪声、点噪声、柱状噪声会严重干扰图像，使模型难以判断哪些响应是真实信号，哪些是噪声。
                ②普通空间卷积难以利用频率分布差异问题：普通卷积主要在空间域处理图像，更擅长提取局部纹理和结构信息。图像和噪声在频率域具有明显差异：目标信号具有周期性，会在特定频带形成较强响应；而噪声通常表现为不稳定的高频干扰。如果只依赖空间卷积，模型很难充分利用这种“频带分布差异”。
                ③不同频带重要性不同但被统一处理问题：已有方法通常对所有频率成分采用统一处理方式，无法根据频带内容动态决定增强还是抑制。
        实现方式：通过将图像特征映射到频率域，结合频带内容与位置生成动态权重，自适应增强信号频带并抑制噪声频带，从而提升图像去噪效果。
"""

class FrequencyBandAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(FrequencyBandAttention, self).__init__()

        # 局部频带特征建模
        self.local_band_conv = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 全局平均池化：提取每个频带通道的平均响应
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全局最大池化：提取每个频带通道的最大响应
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(1, num_channels // reduction_ratio)

        # 频带权重生成网络
        self.band_weight_mlp = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, frequency_band_feature):
        # 对频带特征先做一次局部建模
        refined_band_feature = self.local_band_conv(frequency_band_feature)

        # 基于平均响应生成频带权重
        avg_band_weight = self.band_weight_mlp(
            self.global_avg_pool(refined_band_feature)
        )

        # 基于最大响应生成频带权重
        max_band_weight = self.band_weight_mlp(
            self.global_max_pool(refined_band_feature)
        )

        # 两种全局统计共同决定最终频带权重
        frequency_band_weight = avg_band_weight + max_band_weight

        return frequency_band_weight


class FrequencyBandGuidedUnit(nn.Module):
    """
    频带引导增强模块

    对应论文中的 Frequency Band-Guided Weighting 思想：
    1. 使用 RFFT 将空域特征转换到频域；
    2. 将复数频域特征拆分为实部和虚部；
    3. 拼接频带位置编码，显式提供频带位置信息；
    4. 通过 1×1 卷积进行频带解耦；
    5. 使用频带注意力增强信号频带、抑制噪声频带；
    6. 通过 1×1 卷积进行频带重耦合；
    7. 使用 IRFFT 转换回空域特征。
    """

    def __init__(self, in_channels, out_channels):
        super(FrequencyBandGuidedUnit, self).__init__()

        # 频带解耦卷积：
        # 输入为 实部/虚部拼接后的 2C 通道 + 2 个频带位置编码通道
        self.band_decoupling_conv = nn.Conv2d(
            in_channels=in_channels * 2 + 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # 频域特征归一化
        self.band_norm = nn.BatchNorm2d(out_channels * 2)

        # 频域非线性激活
        self.band_activation = nn.ReLU(inplace=True)

        # 频带注意力，用于生成不同频带的增强/抑制权重
        self.frequency_band_attention = FrequencyBandAttention(out_channels * 2)

        # 频带重耦合卷积：
        # 将加权后的频带特征重新融合为完整频域表示
        self.band_coupling_conv = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, spatial_feature):
        batch_size = spatial_feature.shape[0]
        # 在 H 和 W 两个空间维度上做傅里叶变换
        fft_dims = (-2, -1)
        # 记录原始空间尺寸，用于后续 IRFFT 恢复
        original_spatial_size = spatial_feature.shape[-2:]

        # RFFT：空域特征 -> 频域复数特征
        complex_frequency_feature = torch.fft.rfftn(spatial_feature,dim=fft_dims,norm='ortho')

        # 拆分复数频域特征的实部和虚部
        real_imag_frequency_feature = torch.stack((complex_frequency_feature.real,complex_frequency_feature.imag), dim=-1)
        # [B, C, H, W_freq, 2] -> [B, C, 2, H, W_freq]
        real_imag_frequency_feature = real_imag_frequency_feature.permute(0, 1, 4, 2, 3).contiguous()
        # 将实部/虚部维度合并到通道维度
        # [B, C, 2, H, W_freq] -> [B, 2C, H, W_freq]
        frequency_content_feature = real_imag_frequency_feature.view( (batch_size, -1) + real_imag_frequency_feature.size()[3:])
        frequency_height, frequency_width = frequency_content_feature.shape[-2:]

        # 构造垂直方向频带位置编码
        vertical_band_position = torch.linspace(0, 1, frequency_height)[None, None, :, None].expand(batch_size, 1, frequency_height, frequency_width).to(frequency_content_feature)
        # 构造水平方向频带位置编码
        horizontal_band_position = torch.linspace(0, 1, frequency_width)[None, None, None, :].expand(batch_size, 1, frequency_height, frequency_width).to(frequency_content_feature)
        # 拼接频带内容特征（实部和虚部）与频带位置编码
        # [B, 2C, H, W_freq] + [B, 2, H, W_freq]
        band_characteristic_feature = torch.cat((vertical_band_position,horizontal_band_position,frequency_content_feature),dim=1)

        # 频带解耦：将不同频带内容映射到不同特征通道
        decoupled_band_feature = self.band_decoupling_conv(band_characteristic_feature)
        decoupled_band_feature = self.band_activation(self.band_norm(decoupled_band_feature))
        # 生成频带权重
        band_importance_weight = self.frequency_band_attention(decoupled_band_feature)
        # 频带加权：增强有效信号频带，抑制噪声频带
        weighted_band_feature = decoupled_band_feature * band_importance_weight

        # 频带重耦合：重新融合为完整频域特征
        recoupled_frequency_feature = self.band_coupling_conv(weighted_band_feature)
        # 将通道重新拆分为 实部/虚部
        recoupled_frequency_feature = recoupled_frequency_feature.view((batch_size, -1, 2) + recoupled_frequency_feature.size()[2:])
        # [B, C, 2, H, W_freq] -> [B, C, H, W_freq, 2]
        recoupled_frequency_feature = recoupled_frequency_feature.permute( 0, 1, 3, 4, 2).contiguous()
        # 恢复为复数频域特征
        complex_recoupled_frequency_feature = torch.complex(recoupled_frequency_feature[..., 0],recoupled_frequency_feature[..., 1])
        # IRFFT：频域特征 -> 空域特征
        enhanced_spatial_feature = torch.fft.irfftn(complex_recoupled_frequency_feature,s=original_spatial_size,dim=fft_dims,norm='ortho')

        return enhanced_spatial_feature


if __name__ == "__main__":
    input_feature = torch.randn(1, 32, 50, 50)
    model = FrequencyBandGuidedUnit(in_channels=32,out_channels=32)
    output_feature = model(input_feature)
    print("输入特征维度：", input_feature.shape)
    print("输出特征维度：", output_feature.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")