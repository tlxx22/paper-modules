import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
    论文地址：https://ieeexplore.ieee.org/abstract/document/10906597/
    论文题目：AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation (2025 一区TOP) 
    中文题目：面向弱监督少样本语义分割的自适应频域感知网络 (2025 一区TOP) 
    讲解视频：https://www.bilibili.com/video/BV18XUbBnEcF/
        基于池化机制的频域感知模块（Frequency-Aware Module,FAM）
            实际意义：①RGB域无法提供足够的语义信息：在训练中，模型只能获得极少的标注（仅图像级标签+极少样本），RGB域只能提供颜色、纹理等低层次视觉线索，频域是有效手段，解决“图像表达不足”是目的。
                    ②卷积特征中高低频信息混合导致的冗余：引用别人的论文。（在弱监督/少样本/极端条件下）
            实现方式：多层特征提取 → OctaveConv 拆成高频/低频 → 跨层融合，得到清晰结构频域特征。
"""

class FirstOctaveConv(nn.Module):  # 初始卷积：从单频特征拆分为高频与低频
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride                                         # 保存步长用于空间下采样判断
        kernel_size = kernel_size[0]                                 # 提取卷积核大小（单值使用）
        self.h2g_pool = nn.AvgPool2d(2, 2)                           # 用池化获得低频特征

        self.h2l = nn.Conv2d(                                        # 高 → 低频分支
            in_channels, int(alpha * in_channels),
            kernel_size, 1, padding, dilation, groups, bias
        )
        self.h2h = nn.Conv2d(                                        # 高 → 高频分支
            in_channels, in_channels - int(alpha * in_channels),
            kernel_size, 1, padding, dilation, groups, bias
        )

    def forward(self, x):  # 输入为单路特征：N,C,H,W
        if self.stride == 2:                        # 若 stride=2，则对输入整体执行下采样
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)                   # 池化获得低频输入
        X_h = x                                     # 高频直接使用原始特征

        X_h = self.h2h(X_h)                         # 高频卷积映射
        X_l = self.h2l(X_h2l)                       # 低频卷积映射

        return X_h, X_l                             # 输出高频与低频特征


class OctaveConv(nn.Module):  # 主体卷积：高低频输入 → 高低频输出
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]                        # 卷积核尺寸
        self.h2g_pool = nn.AvgPool2d(2, 2)                 # 下采样用于产生低频特征
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样用于恢复空间尺寸
        self.stride = stride                               # 存储步长信息

        self.l2l = nn.Conv2d(                              # 低 → 低频
            int(alpha * in_channels), int(alpha * out_channels),
            kernel_size, 1, padding, dilation, groups, bias
        )
        self.l2h = nn.Conv2d(                              # 低 → 高频
            int(alpha * in_channels), out_channels - int(alpha * out_channels),
            kernel_size, 1, padding, dilation, groups, bias
        )
        self.h2l = nn.Conv2d(                              # 高 → 低频
            in_channels - int(alpha * in_channels), int(alpha * out_channels),
            kernel_size, 1, padding, dilation, groups, bias
        )
        self.h2h = nn.Conv2d(                              # 高 → 高频
            in_channels - int(alpha * in_channels),
            out_channels - int(alpha * out_channels),
            kernel_size, 1, padding, dilation, groups, bias
        )

    def forward(self, x):
        X_h, X_l = x                                        # 输入两个分支：高频 & 低频

        if self.stride == 2:                               # 若设置步长为 2，则整体下采样
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)                         # 从高频产生低频信息

        X_h2h = self.h2h(X_h)                              # 高 → 高通道卷积
        X_l2h = self.l2h(X_l)                              # 低 → 高通道卷积

        X_l2l = self.l2l(X_l)                              # 低 → 低
        X_h2l = self.h2l(X_h2l)                            # 高 → 低

        X_l2h = F.interpolate(                             # 上采样低频→高频空间尺寸对齐
            X_l2h, (X_h2h.size(2), X_h2h.size(3)), mode='bilinear'
        )

        X_h = X_l2h + X_h2h                                 # 高频 = 来自低频上采样 + 原高频卷积
        X_l = X_h2l + X_l2l                                 # 低频 = 来自高频下采样 + 原低频卷积

        return X_h, X_l                                     # 返回两路特征


class LastOctaveConv(nn.Module):  # 末层卷积：高低频合并输出
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride                               # 保存步长信息
        kernel_size = kernel_size[0]                       # 提取卷积核大小
        self.h2g_pool = nn.AvgPool2d(2, 2)                 # 用于下采样低频

        self.l2h = nn.Conv2d(                              # 低频 → 单一路输出
            int(alpha * out_channels), out_channels,
            kernel_size, 1, padding, dilation, groups, bias
        )
        self.h2h = nn.Conv2d(                              # 高频 → 单一路输出
            out_channels - int(alpha * out_channels),
            out_channels,
            kernel_size, 1, padding, dilation, groups, bias
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 用于空间尺寸对齐

    def forward(self, x):
        X_h, X_l = x                                        # 输入高低频

        if self.stride == 2:                               # 下采样操作（若定义）
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h)                              # 高频输出映射
        X_l2h = self.l2h(X_l)                              # 低频输出映射

        X_l2h = F.interpolate(                             # 上采样低频映射，匹配高频特征尺寸
            X_l2h, (X_h2h.size(2), X_h2h.size(3)), mode='bilinear'
        )

        X_h = X_h2h + X_l2h                                # 高频结果与低频贡献汇合
        return X_h                                         # 结束卷积结构，输出单路特征图


class Octave(nn.Module):  # 完整 Octave 卷积模块：包含拆分→处理→融合三阶段
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)    # 第一阶段：拆分频率
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)         # 中间卷积 1（保持通道）
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)        # 中间卷积 2（修改通道）
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)     # 第三阶段：融合输出

    def forward(self, x):
        x_h, x_l = self.fir(x)                     # 高低频初次拆分
        x_h_1, x_l_1 = self.mid1((x_h, x_l))       # 第一次高低频卷积
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1))   # 第二次高低频卷积
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2))   # 改变通道数量的卷积
        x_ret = self.lst((x_h_5, x_l_5))           # 最后的频率融合
        return x_ret                                # 输出单路特征图

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = Octave(in_channels=32, out_channels=32)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")