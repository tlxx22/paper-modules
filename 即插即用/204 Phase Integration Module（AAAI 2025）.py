import torch
import torch.nn as nn

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/download/32297/34452
    论文题目：Guided Real Image Dehazing Using YCbCr Color Space（AAAI 2025）
    中文题目：使用ycbcr颜色空间的引导式真实的图像去雾（AAAI 2025）
    讲解视频：https://www.bilibili.com/video/BV1PNikB2Eec/
    相位集成模块（Phase Integration Module，PIM）
        实际意义：①真实浓雾场景下,RGB 特征结构信息严重退化（纹理与边缘）：RGB 图像中的高频纹理被雾散射严重抑制；
                    CNN主要依赖局部空间卷积，在输入结构已被影响的前提下，难以重建真实边缘与轮廓。
                ②幅度谱的失真问题：在雾霾图像的频率域表示中，幅度谱容易受到对比度失真和噪声的影响，导致特征恢复不准确。
                ③频率域中的特征融合不足：传统方法在频率域中难以有效整合双颜色空间的优势。
        实现方式：在真实雾霾条件下，利用相位不变性，把更可靠的跨颜色空间的结构信息重新整合到RGB 特征中。（特征融合）
"""
class PIM(nn.Module):
    def __init__(self, channel):
        # 初始化父类 nn.Module
        super(PIM, self).__init__()

        # 幅值（magnitude）处理分支
        # 使用 1×1 卷积，仅在通道维度上进行特征重组，不改变空间尺寸
        self.processmag = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),   # 通道间线性映射
            nn.LeakyReLU(0.1, inplace=True),        # 引入非线性，避免梯度消失
            nn.Conv2d(channel, channel, 1, 1, 0)    # 再次通道映射，增强表达能力
        )

        # 相位（phase）处理分支
        # 结构与幅值分支完全一致，但语义上用于处理相位信息
        self.processpha = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0)
        )

    def forward(self, rgb_x, ycbcr_x):
        # 对 RGB 特征在空间维度做二维 FFT，进入频域
        rgb_fft = torch.fft.rfft2(rgb_x, norm='backward')
        # 对 YCbCr 特征在空间维度做二维 FFT，进入频域
        ycbcr_fft = torch.fft.rfft2(ycbcr_x, norm='backward')

        # 计算 RGB 频谱的幅值（反映能量强度）
        rgb_amp = torch.abs(rgb_fft)
        # 计算 RGB 频谱的相位（反映结构与位置信息）
        rgb_phase = torch.angle(rgb_fft)

        # 计算 YCbCr 频谱的幅值
        ycbcr_amp = torch.abs(ycbcr_fft)
        # 计算 YCbCr 频谱的相位
        ycbcr_phase = torch.angle(ycbcr_fft)

        # 使用可学习模块对 RGB 幅值进行建模与调整
        rgb_amp = self.processmag(rgb_amp)
        # 使用可学习模块对 RGB 相位进行建模与调整
        rgb_phase = self.processmag(rgb_phase)

        # 使用可学习模块对 YCbCr 幅值进行建模与调整
        ycbcr_amp = self.processmag(ycbcr_amp)
        # 使用可学习模块对 YCbCr 相位进行建模与调整
        ycbcr_phase = self.processmag(ycbcr_phase)

        # 将两种模态的相位信息进行融合（相位叠加）
        # 相位主要决定空间结构，因此用于跨模态信息交互
        mix_phase = rgb_phase + ycbcr_phase

        # 使用 RGB 的幅值 + 融合后的相位，重建 RGB 空间特征
        out_rgb = torch.fft.irfft2(
            rgb_amp * torch.exp(1j * mix_phase),
            norm='backward'
        )

        # 使用 YCbCr 的幅值 + 融合后的相位，重建 YCbCr 空间特征
        out_ycbcr = torch.fft.irfft2(
            ycbcr_amp * torch.exp(1j * mix_phase),
            norm='backward'
        )
        # 加起来就是特征融合
        # return out_rgb+out_ycbcr

        # 分开起来就是特征分解
        return out_rgb, out_ycbcr

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    y = torch.randn(1, 32, 50, 50)
    block = PIM(32)
    output_rgb, output_ycbcr = block(x, y)
    print(f"输入张量X形状: {x.shape}")
    print(f"输入张量Y形状: {y.shape}")
    print(f"输出张量1形状: {output_rgb.shape}")
    print(f"输出张量2形状: {output_ycbcr.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")