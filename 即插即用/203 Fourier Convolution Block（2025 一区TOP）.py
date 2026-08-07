import torch
import torch.nn as nn
import numpy as np
from numpy.random import RandomState

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S1361841524002482
    论文题目：Fourier Convolution Block with global receptive field for MRI reconstruction（2025一区TOP ）
    中文题目：用于磁共振成像（MRI）重建的全局感受野的傅里叶卷积块（2025一区TOP ）
    讲解视频：https://www.bilibili.com/video/BV1wyvpBsEQD/
    傅里叶卷积模块（Fourier Convolution Block，FCB）
        实际意义：①传统CNN的有限感受野问题：CNN 依赖局部空间卷积（如3×3或5×5核），导致感受野受限，无法有效捕捉全局特征。
                ②Vision Transformer和大核CNN的计算复杂度和训练困难：ViT通过自注意力捕捉全局信息，但计算需求高、训练难；大核CNN（如7×7或31×31核）参数量大、计算开销高。
                ②傅里叶卷积的参数过多和应用限制：早期傅里叶卷积参数量大，导致网络深度不足，无法用于现代深度CNN。
        实现方式：将特征图变到频域，在频域做一次可学习的逐点乘法（等价于空域的全局卷积），再变回空域，从而让卷积具备全局感受野。
"""

def complexinit(weights_real, weights_imag, criterion):
    # 获取卷积核（频域权重）的形状
    output_chs, input_chs, num_rows, num_cols = weights_real.shape

    # fan_in 表示输入通道数，用于初始化尺度计算
    fan_in = input_chs

    # fan_out 表示输出通道数，用于初始化尺度计算
    fan_out = output_chs

    # 根据初始化准则选择尺度参数 s（复数权重的幅值尺度）
    if criterion == 'glorot':
        # Glorot 初始化：考虑输入和输出通道
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':
        # He 初始化：只考虑输入通道
        s = 1. / np.sqrt(fan_in) / 4.
    else:
        # 若初始化方式非法，直接报错
        raise ValueError('Invalid criterion: ' + criterion)

    # 创建随机数生成器
    rng = RandomState()

    # 保存权重的整体形状
    kernel_shape = weights_real.shape

    # 使用 Rayleigh 分布采样复数的“模长”
    modulus = rng.rayleigh(scale=s, size=kernel_shape)

    # 使用均匀分布采样复数的“相位角”
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    # 根据极坐标形式计算复数的实部
    weight_real = modulus * np.cos(phase)

    # 根据极坐标形式计算复数的虚部
    weight_imag = modulus * np.sin(phase)

    # 将 numpy 数组转换为 PyTorch Tensor，并赋值给实部参数
    weights_real.data = torch.Tensor(weight_real)

    # 将 numpy 数组转换为 PyTorch Tensor，并赋值给虚部参数
    weights_imag.data = torch.Tensor(weight_imag)

class FCB(nn.Module):
    def __init__(self, input_chs: int, output_chs: int, num_rows: int, num_cols: int, stride=1, init='he'):
        # 初始化父类 nn.Module
        super(FCB, self).__init__()

        # 定义频域卷积核的实部参数（rFFT 只保留一半频谱）
        self.weights_real = nn.Parameter(
            torch.Tensor(1, input_chs, num_rows, int(num_cols // 2 + 1))
        )

        # 定义频域卷积核的虚部参数
        self.weights_imag = nn.Parameter(
            torch.Tensor(1, input_chs, num_rows, int(num_cols // 2 + 1))
        )

        # 使用复数初始化方法初始化频域权重
        complexinit(self.weights_real, self.weights_imag, init)

        # 保存原始空间尺寸，用于反 FFT 还原
        self.size = (num_rows, num_cols)

        # 保存步长（是否下采样）
        self.stride = stride

    def forward(self, x):
        # 对输入特征图在空间维度做实数 FFT（进入频域）
        x = torch.fft.rfftn(x, dim=(-2, -1), norm=None)

        # 分离 FFT 结果的实部
        x_real = x.real

        # 分离 FFT 结果的虚部
        x_imag = x.imag

        # 复数乘法：实部计算公式
        y_real = torch.mul(x_real, self.weights_real) - torch.mul(x_imag, self.weights_imag)

        # 复数乘法：虚部计算公式
        y_imag = torch.mul(x_real, self.weights_imag) + torch.mul(x_imag, self.weights_real)

        # 将处理后的复数频谱通过逆 FFT 还原回空间域
        x = torch.fft.irfftn(
            torch.complex(y_real, y_imag),
            s=self.size,
            dim=(-2, -1),
            norm=None
        )

        # 若 stride=2，则在空间域进行下采样
        if self.stride == 2:
            x = x[..., ::2, ::2]

        # 返回最终输出特征图
        return x

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = FCB(32, 32, 50, 50)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")