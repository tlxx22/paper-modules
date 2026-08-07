import torch.nn as nn
import torch

"""
    论文地址：https://arxiv.org/pdf/2511.18888
    论文题目：MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model（AAAI 2026）
    中文题目：MFmamba: 基于状态空间模型的全色图像分辨率恢复多功能网络（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1iZPmzXEF4/
    多尺度混合交叉模块（Multi-scale Hybrid Cross Block ,MHCB）
        实际意义：①局部与多尺度特征提取不足问题：现有超分辨率方法在浅层特征提取时，往往难以同时有效捕捉局部细节（如边缘、纹理）和全局多尺度信息（如大范围结构），这导致复杂细节区域容易产生错误纹理并丢失精细信息。
                ②细节特征提取能力弱问题：传统方法在复杂纹理或高频细节处上提取特征不充分，容易造成细节模糊。
                ③梯度流与特征持久性问题：深层网络中常见的梯度消失或关键特征在传播过程中逐渐衰减问题。
        实现方式：通过多尺度并行卷积与稠密残差连接，实现高效浅层特征提取，提升图像分辨率重建性能。
"""
class MultiScaleHybridCrossBlock(nn.Module):  # 定义一个多尺度混合交叉特征块（用于提取和融合不同尺度特征）
    def __init__(self, in_channels, out_channels, bias=True, activation=nn.ReLU(inplace=True)):  # 初始化模块参数
        super().__init__()  # 调用父类 nn.Module 的初始化函数
        k3 = 3  # 定义3×3卷积核大小
        k5 = 5  # 定义5×5卷积核大小
        self.conv3_stage1 = nn.Conv2d(in_channels, out_channels, k3, padding=k3 // 2, bias=bias)  # 第一阶段：使用3×3卷积提取局部特征
        self.conv5_stage1 = nn.Conv2d(in_channels, out_channels, k5, padding=k5 // 2, bias=bias)  # 第一阶段：使用5×5卷积提取更大感受野特征

        self.conv3_stage2 = nn.Conv2d(in_channels, out_channels, k3, padding=k3 // 2, bias=bias)  # 第二阶段：再次使用3×3卷积对融合特征进行细化
        self.conv5_stage2 = nn.Conv2d(in_channels, out_channels, k5, padding=k5 // 2, bias=bias)  # 第二阶段：再次使用5×5卷积进行多尺度特征增强

        self.fusion3 = nn.Conv2d(in_channels * 3, out_channels, 1, bias=True)  # 使用1×1卷积融合三个特征（identity + 3×3特征 + 5×5特征）
        self.fusion5 = nn.Conv2d(in_channels * 3, out_channels, 1, bias=True)  # 使用1×1卷积进行另一分支的特征融合

        self.bottleneck_fusion = nn.Conv2d(in_channels * 3 + out_channels * 2, out_channels, 1, bias=True)  # 最终融合层：将所有阶段特征压缩并融合

        self.activation = activation  # 保存激活函数（默认 ReLU）

    def forward(self, x):  # 前向传播函数
        identity = x  # 保存输入特征，用于后续残差连接

        # 前半段  上
        feat3_stage1 = self.activation(self.conv3_stage1(identity))  # 通过3×3卷积提取第一阶段特征并进行激活
        feat3_stage1 = feat3_stage1 + identity  # 将卷积结果与原始输入相加，形成残差增强特征
        # 前半段  下
        feat5_stage1 = self.activation(self.conv5_stage1(identity))  # 通过5×5卷积提取第一阶段特征并进行激活
        feat5_stage1 = feat5_stage1 + identity  # 同样进行残差连接，增强特征表达
        # 第一个 C
        stage1_concat = torch.cat([identity, feat3_stage1, feat5_stage1], dim=1)  # 在通道维度拼接三个特征，形成多尺度组合特征

        # 后半部分
        fusion_feat3 = self.fusion3(stage1_concat)  # 使用1×1卷积对拼接后的特征进行融合，得到第一条融合分支特征
        fusion_feat5 = self.fusion5(stage1_concat)  # 使用1×1卷积得到第二条融合分支特征
        feat3_stage2 = self.activation(self.conv3_stage2(fusion_feat3))  # 对融合特征进行3×3卷积细化
        feat5_stage2 = self.activation(self.conv5_stage2(fusion_feat5))  # 对融合特征进行5×5卷积细化
        stage2_concat = torch.cat(  # 将第一阶段和第二阶段的特征全部拼接
            [identity, feat3_stage1, feat5_stage1, feat3_stage2, feat5_stage2],  # 包含原始特征和四种不同尺度处理后的特征
            dim=1  # 在通道维度进行拼接
        )

        # 最后一个卷积
        out = self.bottleneck_fusion(stage2_concat)  # 使用1×1卷积对所有特征进行最终融合和压缩

        # 最后一个 + 号
        out = out + identity  # 最终输出与输入做残差连接，提升梯度传播能力
        return out  # 返回输出特征

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = MultiScaleHybridCrossBlock(32, 32)
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")