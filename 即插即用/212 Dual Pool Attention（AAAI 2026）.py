import torch.nn as nn
import torch

"""
    论文地址：https://arxiv.org/pdf/2511.18888
    论文题目：MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model（AAAI 2026）
    中文题目：MFmamba: 基于状态空间模型的全色图像分辨率恢复多功能网络（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1MycZzaE2a/
        双池化注意力（Dual Pool Attention , DPA）
            实际意义：①特征提取的全面性不足问题：单一池化方式难以同时捕捉图像全局特征和局部显著特征的问题。
                    ②特征通道的权重分配不合理问题：传统方法难以同时捕获“整体统计信息”和“显著峰值特征”，容易忽略关键细节，并引入冗余噪声。
            实现方式：通过平均池化+最大池化双流通道注意力，动态加权重要特征通道，有效提升模型保留图像细节的能力。
"""
class DualPoolingAttention(nn.Module):  # 定义一个双池化通道注意力模块，继承自 PyTorch 的 nn.Module
    def __init__(self, in_channels, reduction_ratio=16):  # 初始化函数，in_channels 为输入特征通道数，reduction_ratio 为通道压缩比例
        super(DualPoolingAttention, self).__init__()  # 调用父类 nn.Module 的初始化方法

        self.avg_pool_layer = nn.AdaptiveAvgPool2d(1)  # 定义自适应平均池化层，将每个通道的 H×W 压缩为 1×1（提取全局平均信息）
        self.max_pool_layer = nn.AdaptiveMaxPool2d(1)  # 定义自适应最大池化层，将每个通道的 H×W 压缩为 1×1（提取全局最大响应）

        self.avg_attention_mlp = nn.Sequential(  # 构建平均池化分支的注意力权重生成网络（MLP）
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),  # 第一层全连接：压缩通道维度，减少计算量
            nn.ReLU(inplace=True),  # ReLU 激活函数，引入非线性能力
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),  # 第二层全连接：恢复原始通道维度
            nn.Sigmoid()  # Sigmoid 激活函数，将输出映射到 0~1，用作通道权重
        )

        self.max_attention_mlp = nn.Sequential(  # 构建最大池化分支的注意力权重生成网络
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),  # 压缩通道维度
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),  # 恢复通道维度
            nn.Sigmoid()  # 输出 0~1 的通道注意力权重
        )

    def forward(self, input_feature):  # 前向传播函数，input_feature 为输入特征图
        batch_size, channels, _, _ = input_feature.size()  # 获取输入特征的维度：batch 大小、通道数、空间尺寸

        # 对输入特征进行全局平均池化，得到每个通道的平均值描述向量
        # 输出形状从 (B, C, 1, 1) 变为 (B, C)
        avg_descriptor = self.avg_pool_layer(input_feature).view(batch_size, channels)
        # 将平均池化得到的通道描述向量输入 MLP
        # 生成每个通道的注意力权重，并 reshape 成 (B, C, 1, 1)
        avg_attention = self.avg_attention_mlp(avg_descriptor).view(batch_size, channels, 1, 1)

        # 对输入特征进行全局最大池化
        # 获取每个通道最强响应值作为特征描述
        max_descriptor = self.max_pool_layer(input_feature).view(batch_size, channels)
        # 将最大池化得到的通道描述向量输入 MLP
        # 得到最大池化分支的通道注意力权重
        max_attention = self.max_attention_mlp(max_descriptor).view(batch_size, channels, 1, 1)

        # 将平均池化分支生成的通道权重扩展到特征图大小
        # 然后与输入特征逐元素相乘，实现通道加权
        avg_refined_feature = input_feature * avg_attention.expand_as(input_feature)
        # 将最大池化分支生成的通道权重扩展到特征图大小
        # 与输入特征逐元素相乘，实现另一种通道加权
        max_refined_feature = input_feature * max_attention.expand_as(input_feature)

        # 将两种注意力加权特征与原始特征进行残差相加
        # 这样可以保留原始信息，同时增强重要通道
        fused_feature = avg_refined_feature + max_refined_feature + input_feature

        return fused_feature

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = DualPoolingAttention(32)
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")