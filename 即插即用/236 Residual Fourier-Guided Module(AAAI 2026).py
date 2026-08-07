import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/view/38276/42238
    论文题目：Beyond Illumination: Fine-Grained Detail Preservation in Extreme Dark Image Restoration(AAAI 2026)
    中文题目：超越光照局限：极致暗光图像复原中的细粒度细节保留(AAAI 2026)
    讲解视频：https://www.bilibili.com/video/BV1vtLy6dEyf/
    残差傅里叶引导模块（Residual Fourier-Guided Module，RFGM）
        实际意义：①频域处理中的阶段间信息流失问题：现有频域方法采用卷积处理，早期有效信息会被弱化，导致光照恢复不准、结构信息丢失。
                ②单阶段内通道间信息孤立、未充分融合的问题：频域特征各通道包含互补结构信息，各通道独立处理的话，造成结构分散问题。
                ③依赖不可靠先验、导致错误累积的问题：现有方法常依赖预训练模型或人工设计先验，一旦先验不准，误差会不断放大。
                ④频域信息冗余、计算效率低的问题：传统频域方法多次重复卷积，造成频域信息冗余、计算开销大。
                ⑤频域相位结构补偿不足的问题：相位对应结构细节，传统方法对相位处理简单，导致结构特征偏弱。
        实现方式：通过傅里叶实现频域分离幅度（亮度）与相位（结构、轮廓），对幅度分量进行亮度先验引导、对相位分量进行结构残差补偿，从而实现极暗图像的全局亮度重建与结构信息维护。
"""

class AmplitudeGuidedFusion(nn.Module):
    # 定义幅度引导融合模块
    def __init__(self, channels):
        # 初始化父类 nn.Module
        super(AmplitudeGuidedFusion, self).__init__()

        # 保存通道数
        self.channels = channels

        # 将单通道幅度先验扩展为多通道幅度权重
        self.prior_expand_conv = nn.Conv2d(
            1, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, previous_amplitude, current_amplitude):
        # 获取幅度特征的形状
        batch_size, channels, height, width = previous_amplitude.shape

        # 将 previous_amplitude 从 B×C×H×W 展平为 B×C×HW
        previous_amplitude_flat = previous_amplitude.view(
            batch_size, channels, -1
        )

        # 将 current_amplitude 从 B×C×H×W 展平为 B×C×HW
        current_amplitude_flat = current_amplitude.view(
            batch_size, channels, -1
        )

        # 对 previous_amplitude 在空间维度做归一化，便于计算相似度
        previous_amplitude_norm = F.normalize(
            previous_amplitude_flat,
            dim=-1
        )

        # 对 current_amplitude 在空间维度做归一化，便于计算相似度
        current_amplitude_norm = F.normalize(
            current_amplitude_flat,
            dim=-1
        )

        # 计算通道之间的相似度矩阵，结果形状为 B×C×C
        channel_similarity_matrix = torch.bmm(
            previous_amplitude_norm,
            current_amplitude_norm.transpose(1, 2)
        )

        # 对每个通道与其他通道的相似度求平均，得到每个通道的重要性分数
        channel_similarity_score = channel_similarity_matrix.mean(dim=-1)

        # 选择相似度最高的通道索引，作为最可靠的幅度先验通道
        top1_channel_index = torch.argmax(
            channel_similarity_score,
            dim=-1
        )

        # 根据 top1_channel_index，从 previous_amplitude 中取出对应通道
        selected_amplitude_prior = torch.stack(
            [
                previous_amplitude[b, top1_channel_index[b]]
                for b in range(batch_size)
            ],
            dim=0
        ).unsqueeze(1)

        # 将单通道幅度先验扩展为 C 通道
        expanded_amplitude_prior = self.prior_expand_conv(
            selected_amplitude_prior
        )

        # 通过 sigmoid 得到 0 到 1 之间的幅度引导权重
        amplitude_guidance_weight = torch.sigmoid(
            expanded_amplitude_prior
        )

        # 用幅度引导权重调制当前幅度，并加入残差连接
        guided_amplitude = current_amplitude * amplitude_guidance_weight + current_amplitude

        # 返回引导后的幅度，以及对应的引导权重
        return guided_amplitude, amplitude_guidance_weight

class SpatialResidualBlock(nn.Module):
    # 定义一个空间域残差模块，用普通卷积提取局部空间特征
    def __init__(self, channels):
        # 初始化父类 nn.Module
        super(SpatialResidualBlock, self).__init__()

        # 定义空间特征细化分支
        self.spatial_refine = nn.Sequential(
            # 3×3 卷积，用于提取局部邻域特征，输入输出通道数不变
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),

            # LeakyReLU 激活函数，增加非线性表达能力
            nn.LeakyReLU(0.1, inplace=True),

            # 再使用一个 3×3 卷积继续提取空间细节
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),

            # 再次使用 LeakyReLU 激活
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input_feature):
        # 将输入特征送入空间细化分支
        refined_feature = self.spatial_refine(input_feature)

        # 残差连接：输出 = 原始输入 + 细化后的特征
        return input_feature + refined_feature

class FourierAmplitudePhaseBlock(nn.Module):
    # 定义傅里叶幅度-相位处理模块
    def __init__(self, channels, top_k):
        # 初始化父类 nn.Module
        super(FourierAmplitudePhaseBlock, self).__init__()

        # 保存通道数
        self.channels = channels

        # 保存 top_k 参数，这里实际代码中暂时没有直接使用
        self.top_k = top_k

        # 输入投影层，用 1×1 卷积调整输入特征
        self.input_projection = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

        # 幅度分支，用于增强傅里叶幅度信息
        self.amplitude_refine = nn.Sequential(
            # 1×1 卷积，用于通道间信息交互
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),

            # LeakyReLU 激活函数
            nn.LeakyReLU(0.1, inplace=True),

            # 再使用 1×1 卷积进一步处理幅度特征
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        )

        # 相位分支，用于增强傅里叶相位信息
        self.phase_refine = nn.Sequential(
            # 1×1 卷积，用于通道间信息交互
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),

            # LeakyReLU 激活函数
            nn.LeakyReLU(0.1, inplace=True),

            # 再使用 1×1 卷积进一步处理相位特征
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        )

        # 相位残差融合模块，用于融合原始相位和增强后的相位
        self.phase_residual_fusion = nn.Sequential(
            # 输入是原始相位和增强相位拼接后的 2C 通道，输出压缩回 C 通道
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0),

            # LeakyReLU 激活函数
            nn.LeakyReLU(0.1, inplace=True),

            # 再使用 1×1 卷积得到最终补偿后的相位
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        )

        # 幅度引导融合模块，用原始幅度信息引导当前幅度恢复
        self.amplitude_guided_fusion = AmplitudeGuidedFusion(
            channels=channels
        )

        # 输出投影层，这里定义了但当前 forward 中没有使用
        self.output_projection = nn.Conv2d(
            channels * 2, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, input_feature):
        # 获取输入特征的形状：B 表示批大小，C 表示通道数，H/W 表示高和宽
        batch_size, channels, height, width = input_feature.shape
        # 使用 1×1 卷积对输入特征进行投影
        projected_feature = self.input_projection(input_feature)
        # 对特征做二维实数快速傅里叶变换，进入频域
        frequency_feature = torch.fft.rfft2(projected_feature, norm="backward")

        # 获取频域特征的【幅度分量】，幅度通常和亮度、能量强度有关，对原始幅度进行卷积增强
        original_amplitude = torch.abs(frequency_feature)
        refined_amplitude = self.amplitude_refine(original_amplitude)
        # 【使用原始幅度作为先验，引导增强幅度的融合】
        guided_amplitude, amplitude_guidance_weight = self.amplitude_guided_fusion(
            original_amplitude,
            refined_amplitude
        )

        # 获取频域特征的【相位分量】，相位通常和结构、轮廓、位置有关，原始相位进行卷积增强
        original_phase = torch.angle(frequency_feature)
        refined_phase = self.phase_refine(original_phase)
        # 将原始相位和增强后的相位在通道维度拼接
        phase_for_fusion = torch.cat([original_phase, refined_phase],dim=1)
        # 融合原始相位和增强相位，得到补偿后的相位
        compensated_phase = self.phase_residual_fusion(phase_for_fusion)

        # 根据【卷积后的幅度】和【卷积后的相位】计算复数频域特征的【实部】
        real_part = guided_amplitude * torch.cos(compensated_phase)
        # 根据【卷积后的幅度】和【卷积后的相位】计算复数频域特征的【虚部】
        imag_part = guided_amplitude * torch.sin(compensated_phase)
        # 将实部和虚部重新组合成复数形式的频域特征
        reconstructed_frequency = torch.complex(real_part, imag_part)

        # 通过逆傅里叶变换，将频域特征转换回空间域
        restored_feature = torch.fft.irfft2(reconstructed_frequency,s=(height, width),norm="backward")

        # 返回恢复后的空间域特征
        return restored_feature


class SpatialFrequencyFusionBlock(nn.Module):
    # 定义空间域与频域融合模块
    def __init__(self, channels, use_spatial_branch=True):
        # 初始化父类 nn.Module
        super(SpatialFrequencyFusionBlock, self).__init__()

        # 是否启用空间分支
        self.use_spatial_branch = use_spatial_branch

        # 如果启用空间分支，则使用 SpatialResidualBlock；否则使用恒等映射
        self.spatial_branch = (
            SpatialResidualBlock(channels)
            if use_spatial_branch
            else nn.Identity()
        )

        # 定义频域分支，用于在傅里叶域处理幅度和相位信息
        self.frequency_branch = FourierAmplitudePhaseBlock(
            channels=channels,
            top_k=channels
        )

        # 如果同时使用空间分支和频域分支，拼接后通道数会变成 2 倍
        if use_spatial_branch:
            # 使用 1×1 卷积将 2C 通道压缩回 C 通道
            self.fusion_conv = nn.Conv2d(
                channels * 2, channels, kernel_size=1, stride=1, padding=0
            )

        # 如果不使用空间分支，则只处理频域特征，通道数仍为 C
        else:
            # 使用 1×1 卷积对频域特征进行融合
            self.fusion_conv = nn.Conv2d(
                channels, channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, input_feature):
        # 保存原始输入，用于最后的残差连接
        identity = input_feature
        # 频域分支：通过 FFT 处理幅度和相位信息
        frequency_feature = self.frequency_branch(input_feature)

        # 空间分支：通过卷积提取局部空间特征
        spatial_feature = self.spatial_branch(input_feature)

        # 如果启用空间分支，则将空间特征和频域特征在通道维度拼接
        if self.use_spatial_branch:
            # dim=1 表示沿通道维度拼接，形状从 C 变为 2C
            fused_feature = torch.cat([spatial_feature, frequency_feature], dim=1)

            # 使用 1×1 卷积融合空间和频域信息，并压缩回 C 通道
            output_feature = self.fusion_conv(fused_feature)
        # 如果没有启用空间分支，则只使用频域分支结果
        else:
            # 对频域特征进行 1×1 卷积处理
            output_feature = self.fusion_conv(frequency_feature)

        # 残差连接：输出 = 融合特征 + 原始输入
        return output_feature + identity

if __name__ == "__main__":
    input_tensor = torch.randn(2, 32, 50, 50)
    model =  SpatialFrequencyFusionBlock(32)
    output_tensor = model(input_tensor)
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")