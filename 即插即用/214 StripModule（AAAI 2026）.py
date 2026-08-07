import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2501.03775
    论文题目：Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection（AAAI 2026）
    中文题目：条带区域卷积神经网络：面向遥感目标检测的大条带卷积算法（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1WiAjzCEY7/
        条带卷积模块（Strip Module）
            实际意义：①方形卷积难以有效构建细长目标结构特征问题：传统卷积（如3×3、5×5或更大的方形卷积）在水平和垂直方向上具有相同的感受野。当目标呈现细长结构（如桥梁、船舶、跑道等）时，方形卷积覆盖大量背景区域，使得有效目标结构信息被背景噪声稀释，导致特征表达不充分。
                    ②细长目标长程空间依赖建模不足问题：细长目标的关键特征往往分布在远距离位置，普通小卷积（如 3×3）只能建模局部关系，如果想获得长程信息，需要堆叠很多层卷积或使用大方核卷积，但这两种方法都会带来计算开销。
            实现方式：通过“局部方形卷积 + 横向与纵向大条带卷积”的顺序结构建模方向性长距离依赖特征，并利用特征重标定机制（加权）增强细长目标结构表达能力。
"""
class StripSpatialAttention(nn.Module):
    # 定义条带空间注意力模块，用于生成空间注意力权重
    def __init__(self, channels, kernel_h, kernel_w):
        # 初始化父类 nn.Module
        super().__init__()

        # 定义一个深度卷积层，用于提取局部空间上下文信息
        # groups=channels 表示每个通道独立卷积（深度卷积）
        self.local_context_conv = nn.Conv2d(
            channels,          # 输入通道数
            channels,          # 输出通道数（保持不变）
            kernel_size=5,     # 卷积核大小 5×5
            padding=2,         # padding=2 保证输出尺寸与输入一致
            groups=channels    # 深度卷积（每个通道独立计算）
        )

        # 定义水平方向条带卷积，用于捕获水平方向的长距离依赖
        self.horizontal_band_conv = nn.Conv2d(
            channels,                       # 输入通道
            channels,                       # 输出通道
            kernel_size=(kernel_h, kernel_w),  # 条带卷积核，例如 (1,19)
            stride=1,                       # 步长为1
            padding=(kernel_h // 2, kernel_w // 2),  # padding保持尺寸不变
            groups=channels                 # 深度卷积
        )

        # 定义垂直方向条带卷积，用于捕获垂直方向的长距离依赖
        self.vertical_band_conv = nn.Conv2d(
            channels,                       # 输入通道
            channels,                       # 输出通道
            kernel_size=(kernel_w, kernel_h),  # 与水平卷积相反的方向
            stride=1,                       # 步长为1
            padding=(kernel_w // 2, kernel_h // 2),  # padding保持尺寸一致
            groups=channels                 # 深度卷积
        )

        # 定义1×1卷积，用于融合不同通道信息并生成最终注意力图
        self.attention_projection = nn.Conv2d(
            channels,       # 输入通道
            channels,       # 输出通道
            kernel_size=1   # 1×1卷积用于通道信息融合
        )

    def forward(self, x):
        # 输入特征先通过局部上下文卷积，提取基础空间信息
        attention_map = self.local_context_conv(x)

        # 在水平方向进行条带卷积，增强水平结构信息
        attention_map = self.horizontal_band_conv(attention_map)

        # 在垂直方向进行条带卷积，增强垂直结构信息
        attention_map = self.vertical_band_conv(attention_map)

        # 使用1×1卷积融合通道信息，生成最终空间注意力图
        attention_map = self.attention_projection(attention_map)

        # 将生成的注意力图与输入特征逐元素相乘，实现特征重标定
        return x * attention_map


class StripAttentionBlock(nn.Module):
    # 定义条带注意力残差模块
    def __init__(self, channels, kernel_h, kernel_w):
        # 初始化父类
        super().__init__()

        # 定义1×1卷积，用于进行通道投影（类似特征变换）
        self.channel_expand = nn.Conv2d(channels, channels, kernel_size=1)

        # 定义GELU激活函数，增加网络的非线性表达能力
        self.activation = nn.GELU()

        # 定义条带空间注意力模块
        self.strip_attention = StripSpatialAttention(
            channels,   # 通道数
            kernel_h,   # 条带卷积高度
            kernel_w    # 条带卷积宽度
        )

        # 定义1×1卷积，将特征再次映射回原通道空间
        self.channel_reduce = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x):
        # 保存输入特征，用于后面的残差连接
        residual = x

        # 输入特征通过1×1卷积进行通道变换
        x = self.channel_expand(x)

        # 通过GELU激活函数增加非线性
        x = self.activation(x)

        # 通过条带空间注意力模块进行空间特征增强
        x = self.strip_attention(x)

        # 再通过1×1卷积进行通道融合
        x = self.channel_reduce(x)

        # 残差连接：将原始输入与当前特征相加
        x = x + residual

        # 返回最终输出
        return x

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = StripAttentionBlock(
        channels=32,
        kernel_h=1,
        kernel_w=19
    )
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")