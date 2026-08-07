import torch
import torch.nn as nn

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2026/html/Cai_PFGNet_A_Fully_Convolutional_Frequency-Guided_Peripheral_Gating_Network_for_Efficient_CVPR_2026_paper.html
    论文题目：PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning (CVPR 2026)
    中文题目：PFGNet：面向高效时空预测学习的全卷积频域引导外围门控网络(CVPR 2026)
    讲解视频：https://www.bilibili.com/video/BV1F8E56REU9/
    多尺度初始化模块（Multi-Scale Initialization ,MSInit）
        实际意义：①缺少多尺度候选特征的问题：如果输入特征本身只有单一尺度时，即使存在门控机制，也只能在有限的信息基础上进行选择。
                ②单一小核特征长程建模能力不足的问题：小卷积虽然适合局部细节，但缺少较大范围的上下文建模能力，难以捕捉视频预测中的大范围运动、结构延续和区域变化。
                ③全尺度大核处理计算成本高的问题：如果一开始就使用很大卷积核或完整大尺度特征处理，会带来较高计算量。
        实现方式：通过 3/5/7 轻量可分解卷积分支构建多尺度初始特征，提供稳定的低、中、高频特征的表达。
"""

class _RepDWLite(nn.Module):
    # 定义 MSInit 内部使用的轻量深度卷积分支
    def __init__(self, channels: int, kernel_size: int, stride: int = 1) -> None:
        # 调用父类 nn.Module 的初始化函数
        super().__init__()

        # 判断卷积核大小是否为奇数，因为奇数卷积核更容易通过对称 padding 保持特征对齐
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial alignment.")

        # 定义水平方向的深度卷积，卷积核大小为 1×k，只在宽度方向提取长范围信息
        self.dw_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, kernel_size // 2),
            groups=channels,
            bias=False,
        )

        # 定义垂直方向的深度卷积，卷积核大小为 k×1，只在高度方向提取长范围信息
        self.dw_v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(kernel_size // 2, 0),
            groups=channels,
            bias=False,
        )

        # 定义 3×3 深度卷积分支，用于补充局部中频纹理和局部结构信息
        self.dw_mid = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=channels,
            bias=False,
        )

        # 定义 1×1 深度卷积分支，用于模拟论文公式中的 identity 项
        self.dw_identity = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=stride,
            groups=channels,
            bias=False,
        )

        # 将 1×1 深度卷积分支初始化为近似恒等映射，使其初始阶段尽量保留原始输入信息
        nn.init.dirac_(self.dw_identity.weight)

    # 定义当前轻量深度卷积分支的前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先经过 1×k 水平深度卷积，再经过 k×1 垂直深度卷积，近似实现 k×k 大核感受野
        large = self.dw_v(self.dw_h(x))

        # 使用 3×3 深度卷积提取局部中频特征
        mid = self.dw_mid(x)

        # 使用 1×1 深度卷积保留输入自身的信息，类似残差或恒等映射
        identity = self.dw_identity(x)

        # 将大核响应、中频响应和恒等响应相加，得到当前尺度的初始化特征
        return large + mid + identity

class MSInit(nn.Module):
    # 定义 Multi-Scale Initialization 模块，输入和输出均为 BCHW 格式
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        stride: int = 1,
        use_group_norm: bool = True,
    ) -> None:
        # 调用父类 nn.Module 的初始化函数
        super().__init__()

        # 判断 kernel_sizes 是否为空，如果没有任何尺度分支，则模块无法正常构建
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty.")

        # 如果没有指定输出通道数，则默认输出通道数等于输入通道数，方便即插即用
        out_channels = in_channels if out_channels is None else out_channels

        # 计算每个尺度分支分配到的输出通道数
        branch_channels = out_channels // len(kernel_sizes)

        # 计算不能被平均分配的剩余通道数
        remainder = out_channels - branch_channels * len(kernel_sizes)

        # 构建多个尺度分支，每个分支包含一个 _RepDWLite 和一个 1×1 通道投影卷积
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    # 当前尺度的轻量深度卷积分支，用于提取对应尺度的空间响应
                    _RepDWLite(in_channels, kernel_size=k, stride=stride),

                    # 使用 1×1 卷积将当前分支的通道数投影为 branch_channels
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                )
                for k in kernel_sizes
            ]
        )

        # 提前声明 tail 分支，tail 用于补齐无法被平均分配的剩余通道
        self.tail: nn.Module

        # 如果没有剩余通道，则 tail 不需要做任何处理
        if remainder == 0:
            self.tail = nn.Identity()

        # 如果存在剩余通道，则额外构建一个补充分支
        else:
            self.tail = nn.Sequential(
                # 使用第一个尺度的 _RepDWLite 提取补充分支特征
                _RepDWLite(in_channels, kernel_size=kernel_sizes[0], stride=stride),

                # 使用 1×1 卷积将补充分支输出为 remainder 个通道
                nn.Conv2d(in_channels, remainder, kernel_size=1, bias=False),
            )

        # 如果启用 GroupNorm，则使用 GroupNorm(1, out_channels) 对输出特征进行归一化
        self.norm = nn.GroupNorm(1, out_channels) if use_group_norm else nn.Identity()

        # 使用 GELU 激活函数增强非线性表达能力
        self.act = nn.GELU()

    # 定义 MSInit 的前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入 x 分别送入不同尺度分支，得到多个尺度的输出特征
        parts = [branch(x) for branch in self.branches]

        # 如果 tail 不是 Identity，说明存在剩余通道，需要额外计算 tail 分支
        if not isinstance(self.tail, nn.Identity):
            parts.append(self.tail(x))

        # 将所有尺度分支的输出沿通道维度拼接，得到多尺度初始化特征
        y = torch.cat(parts, dim=1)

        # 先进行归一化，再进行 GELU 激活，输出最终 MSInit 特征
        return self.act(self.norm(y))


if __name__ == "__main__":
    # 随机生成一个 BCHW 格式的输入特征图
    x = torch.randn(1, 32, 50, 50)
    # 创建 MSInit 模块，默认输入输出通道数相同，默认使用 3、5、7 三个尺度
    module = MSInit(in_channels=32)
    y = module(x)
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")