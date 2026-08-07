import torch
import torch.nn as nn
from torch import Tensor

"""
    论文地址：https://arxiv.org/pdf/2502.01303
    论文题目：Partial Channel Network: Compute Fewer, Perform Better（AAAI 2026）
    中文题目：部分通道网络：少算多得，性能更优（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV13AfsBvECx/
    局部通道注意力块（Partial Channel-Attention block ，PCAB）
        实际意义：①通道信息的全局空间交互不足问题：传统的卷积操作在处理特征图时，通常仅仅关注局部的空间特征，并未充分考虑不同通道之间的全局交互。
                ②冗余信息：在卷积神经网络中，特征图中的不同通道往往存在冗余信息。
        实现方式：PCAB通过3×3卷积提取局部空间信息，高斯-自注意力机制利用通道的均值和标准差优化通道特征的表示，从而增强通道间的全局信息交互，减少冗余，提高模型的表示能力和性能。
"""

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel_count):
        super().__init__()
        # 用于生成通道注意力的卷积层
        self.conv1x2 = nn.Conv2d(channel_count, channel_count, (1, 2), bias=False)
        self.batch_norm = nn.BatchNorm2d(channel_count)
        self.activation = nn.Hardsigmoid()

    def forward(self, input_tensor):
        batch_size, channel_count, height, width = input_tensor.shape
        # 计算通道的均值和标准差
        mean = input_tensor.reshape(batch_size, channel_count, -1).mean(-1).view(batch_size, channel_count, 1, 1)
        std_dev = input_tensor.reshape(batch_size, channel_count, -1).std(-1).view(batch_size, channel_count, 1, 1)
        # 将均值和标准差拼接
        combined_features = torch.cat([mean, std_dev], dim=-1)
        # 通过卷积层和激活函数生成通道注意力
        attention_map = self.conv1x2(combined_features)
        attention_map = self.activation(attention_map).view(batch_size, channel_count, 1, 1)
        # 将输入乘以生成的通道注意力
        return input_tensor * attention_map

class PartialAttentionCH(nn.Module):
    def __init__(self, input_channels, split_factor=4):
        super().__init__()
        self.input_channels = input_channels
        self.split_factor = split_factor
        self.conv3x3_channels = input_channels // split_factor
        self.untouched_channels = input_channels - self.conv3x3_channels

        # 定义卷积操作
        self.partial_conv3x3 = nn.Conv2d(self.conv3x3_channels, self.conv3x3_channels, 3, 1, 1, bias=False)

        # 定义通道注意力模块
        self.channel_attention = ChannelAttentionModule(self.untouched_channels)
        self.batch_norm = nn.BatchNorm2d(self.untouched_channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        # 分割输入张量
        conv_part, attention_part = torch.split(input_tensor, [self.conv3x3_channels, self.untouched_channels], dim=1)
        # 对部分通道进行卷积操作
        conv_part = self.partial_conv3x3(conv_part)
        # 对剩余部分应用通道注意力
        attention_part = self.batch_norm(self.channel_attention(attention_part))
        # 合并卷积部分和注意力部分
        return torch.cat((conv_part, attention_part), 1)

if __name__ == "__main__":
    x = torch.rand(1, 32, 50, 50)
    model = PartialAttentionCH(input_channels=32)
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")