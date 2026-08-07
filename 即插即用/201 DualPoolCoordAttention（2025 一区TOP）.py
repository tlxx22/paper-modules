import torch
import torch.nn as nn

"""
    论文地址：https://www.sciencedirect.com/science/article/pii/S0168169924012250
    论文题目：Flora-NET: Integrating dual coordinate attention with adaptive kernel based convolution network for medicinal flower identification（2025 一区TOP）
    中文题目：Flora-NET：融合双坐标注意力与自适应核卷积网络的药用花卉识别模型（2025 一区TOP）
    讲解视频：https://www.bilibili.com/video/BV1ojmyBKEwr/
        双坐标注意力特征提取模块（Dual Coordinate Attention Feature Extraction ,DCAFE）
            实际意义：①复杂和噪声背景的影响：实时花卉识别受复杂背景和噪声干扰的影响，导致模型捕获的区分特征较少，分类准确率低下。
                    ②现有注意力机制无法捕获长距离和敏感特征：传统注意力机制仅关注通道间信息，忽略位置信息；使用池化操作仅捕获局部相关性，而无法捕获长距离特征。
                    ③现有注意力机制的局限性：平均池化会使得最重要特征变得模糊（因取平均值，导致模糊信息聚合），而最大池化能保留显著特征和锐度。
            实现方式：沿特征图的水平与垂直方向分别进行池化，作为位置信息并建立长程空间依赖；分别基于平均池化与最大池化构建两条并行的坐标注意力分支，
                    捕获全局上下文信息与局部显著判别特征。最终进行拼接，得到方向增强的特征表示。
"""

# 定义基于均值池化 + 最大池化的坐标注意力模块
class DualPoolCoordAttention(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(DualPoolCoordAttention, self).__init__()

        # 定义沿高度方向的均值池化（宽度压缩为1）
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        # 定义沿宽度方向的均值池化（高度压缩为1）
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))

        # 定义沿高度方向的最大池化
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        # 定义沿宽度方向的最大池化
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        # 通道压缩比例，保证最小通道数不低于8
        mip = max(8, inp // groups)

        # -------------------- 均值分支 Mean Branch ---------------------
        # 第一次1x1卷积用于通道压缩
        self.conv1_mean = nn.Conv2d(inp, mip, kernel_size=1)
        # BN用于稳定训练
        self.bn1_mean = nn.BatchNorm2d(mip)
        # 第二次1x1卷积恢复到输出维度
        self.conv2_mean = nn.Conv2d(mip, oup, kernel_size=1)

        # -------------------- 最大分支 Max Branch ----------------------
        self.conv1_max = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1_max = nn.BatchNorm2d(mip)
        self.conv2_max = nn.Conv2d(mip, oup, kernel_size=1)

        # 激活函数 ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 保留原输入用于做注意力加权
        identity = x
        # 读取 batch、通道、高、宽
        n, c, h, w = x.size()

        # -------------------- 均值池化分支 --------------------
        # 对高度方向进行池化
        x_h_mean = self.pool_h_mean(x)
        # 对宽度方向池化并交换维度，使其与高度池化结果对齐
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        # 在高度维度拼接
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        # 通道压缩卷积
        y_mean = self.conv1_mean(y_mean)
        # BN归一化
        y_mean = self.bn1_mean(y_mean)
        # ReLU激活
        y_mean = self.relu(y_mean)
        # 根据高度和宽度分为两个张量
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        # 宽度特征维度恢复
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)

        # -------------------- 最大池化分支 --------------------
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.conv1_max(y_max)
        y_max = self.bn1_max(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)

        # -------------------- Sigmoid 注意力生成 --------------------
        # 高度方向的均值权重
        x_h_mean = self.conv2_mean(x_h_mean).sigmoid()
        # 宽度方向的均值权重
        x_w_mean = self.conv2_mean(x_w_mean).sigmoid()

        # 高度方向的最大权重
        x_h_max = self.conv2_max(x_h_max).sigmoid()
        # 宽度方向的最大权重
        x_w_max = self.conv2_max(x_w_max).sigmoid()

        # 扩展图到原始尺度
        x_h_mean = x_h_mean.expand(-1, -1, h, w)
        x_w_mean = x_w_mean.expand(-1, -1, h, w)
        x_h_max = x_h_max.expand(-1, -1, h, w)
        x_w_max = x_w_max.expand(-1, -1, h, w)

        # -------------------- 融合注意力 --------------------
        # 均值分支对原特征加权
        attention_mean = identity * x_w_mean * x_h_mean

        # 最大分支对原特征加权
        attention_max = identity * x_w_max * x_h_max

        # 两个注意力结果相加
        return attention_mean + attention_max

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = DualPoolCoordAttention(inp=32, oup=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")