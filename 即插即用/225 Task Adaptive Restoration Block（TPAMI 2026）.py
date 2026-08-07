import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11419884
    论文题目：AIRPNet: Adaptive Image Restoration with Privacy Protection in Steganographic Domain（TPAMI 2026）
    中文题目：AIRPNet：隐写域中面向隐私保护的自适应图像修复方法（TPAMI 2026）
    讲解视频：https://www.bilibili.com/video/BV1nUoVBjEux/
    基于任务的自适应重建模块（Task Adaptive Restoration Block，TARB）
        实际意义：①单一重建模型无法适配多种失真类型的问题：传统图像重建网络只能针对去噪、去模糊、超分的其中一种任务进行设计，然而，现实场景图像往往遭受多种失真混合影响（例如，同时存在噪声和模糊）。
                ②混合失真场景下不同退化难以区分和权重难以分配的问题：一张图像里可能同时存在噪声、模糊和分辨率下降，不同退化对图像的影响程度并不相同，因此需要判断“当前主要是什么退化、各种退化各占多大权重”。
                ③重建任务自适应调节能力弱的问题：面对未知失真时，网络不能按需分配计算与特征权重。（加权）
        实现方式：通过多退化分支建模与退化感知加权机制，实现对不同及混合图像退化的自适应重建。
"""

# 定义一个基础的 3×3 卷积模块
class BasicConv3x3(nn.Module):
    # 初始化模块，channels 表示输入和输出通道数
    def __init__(self, channels):
        # 调用父类初始化
        super().__init__()
        # 使用顺序容器组合卷积和激活函数
        self.block = nn.Sequential(
            # 3×3 普通卷积，保持通道数不变
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            # 使用 LeakyReLU 激活函数增强非线性表达能力
            nn.LeakyReLU(inplace=True)
        )

    # 前向传播函数
    def forward(self, x):
        # 输入特征经过卷积模块后返回
        return self.block(x)


# 定义一个带膨胀率的 3×3 卷积模块
class DilatedConv3x3(nn.Module):
    # dilation 表示膨胀卷积的膨胀率，默认是 2
    def __init__(self, channels, dilation=2):
        # 调用父类初始化
        super().__init__()
        # 构建卷积和激活组成的顺序模块
        self.block = nn.Sequential(
            # 使用膨胀卷积扩大感受野
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            # 激活函数
            nn.LeakyReLU(inplace=True)
        )

    # 前向传播
    def forward(self, x):
        # 返回处理后的结果
        return self.block(x)


# 定义一个深度可分离卷积模块
class SepConv3x3(nn.Module):
    # 初始化模块
    def __init__(self, channels):
        # 调用父类初始化
        super().__init__()
        # 深度卷积：每个通道单独卷积，不做通道间融合
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )
        # 点卷积：使用 1×1 卷积做通道融合
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 激活函数
        self.act = nn.LeakyReLU(inplace=True)

    # 前向传播
    def forward(self, x):
        # 先做深度卷积
        x = self.depthwise(x)
        # 再做点卷积
        x = self.pointwise(x)
        # 再做激活
        x = self.act(x)
        # 返回结果
        return x


# 定义一个“平均池化 + 1×1卷积”模块
class AvgPoolConv(nn.Module):
    # 初始化模块
    def __init__(self, channels):
        # 调用父类初始化
        super().__init__()
        # 先使用平均池化提取局部平滑信息
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # 使用 1×1 卷积做通道映射
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 激活函数
        self.act = nn.LeakyReLU(inplace=True)

    # 前向传播
    def forward(self, x):
        # 先进行平均池化
        x = self.pool(x)
        # 再进行卷积
        x = self.conv(x)
        # 再进行激活
        x = self.act(x)
        # 返回结果
        return x


# 定义一个“恒等映射风格”的 1×1 卷积模块
class IdentityConv(nn.Module):
    # 初始化模块
    def __init__(self, channels):
        # 调用父类初始化
        super().__init__()
        # 使用 1×1 卷积调整特征
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 激活函数
        self.act = nn.LeakyReLU(inplace=True)

    # 前向传播
    def forward(self, x):
        # 先进行 1×1 卷积
        x = self.conv(x)
        # 再进行激活
        x = self.act(x)
        # 返回结果
        return x


# 定义 SE 注意力模块
class SeModule(nn.Module):
    # reduction 表示通道压缩比例
    def __init__(self, in_channels, reduction=1):
        # 调用父类初始化
        super().__init__()
        # 计算中间隐藏层通道数，至少为 1
        hidden_channels = max(1, in_channels // reduction)
        # 构建 SE 模块
        self.se = nn.Sequential(
            # 对每个通道做全局平均池化，得到通道描述向量
            nn.AdaptiveAvgPool2d(1),
            # 第一层 1×1 卷积，进行通道压缩
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            # 激活函数
            nn.LeakyReLU(inplace=True),
            # 第二层 1×1 卷积，恢复通道数
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
            # Sigmoid 将权重压缩到 0~1
            nn.Sigmoid()
        )

    # 前向传播
    def forward(self, x):
        # 将输入特征与通道注意力权重逐元素相乘
        return x * self.se(x)


# 定义任务适配器模块
class TaskAdaptor(nn.Module):
    # in_channels 是输入通道数，out_channels 是输出通道数
    def __init__(self, in_channels, out_channels, semodule=None):
        # 调用父类初始化
        super().__init__()
        # 保存 SE 模块
        self.se = semodule

        # 第一层：3×3 卷积，将输入映射到 32 个通道
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # 第一层激活函数
        self.act1 = nn.LeakyReLU(inplace=True)

        # 第二层：1×1 卷积，将 32 通道映射到 64 通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # 第二层激活函数
        self.act2 = nn.LeakyReLU(inplace=True)

        # 第三层：深度卷积，每个通道单独做 3×3 卷积
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        # 第三层激活函数
        self.act3 = nn.LeakyReLU(inplace=True)

        # 第四层：1×1 卷积，将 64 通道变成指定输出通道数
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # 如果输入通道数和输出通道数不同，就用 1×1 卷积做捷径分支匹配
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            # 如果通道数相同，就直接恒等映射
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 输入先经过第一层卷积和激活
        out = self.act1(self.conv1(x))
        # 再经过第二层卷积和激活
        out = self.act2(self.conv2(out))
        # 再经过第三层深度卷积和激活
        out = self.act3(self.conv3(out))
        # 最后经过第四层卷积输出
        out = self.conv4(out)

        # 如果定义了 SE 注意力模块，则进一步进行通道加权
        if self.se is not None:
            out = self.se(out)

        # 主分支输出与捷径分支相加，形成残差连接
        out = out + self.shortcut(x)
        # 返回结果
        return out


# 定义操作注意力模块，用来给不同操作分配权重
class OperationAttn(nn.Module):
    # task_channels 是输入任务特征通道数
    # num_steps 是总共需要执行多少步操作
    # num_operations 是每一步中可选操作的数量
    def __init__(self, task_channels, num_steps, num_operations):
        # 调用父类初始化
        super().__init__()
        # 保存步数
        self.num_steps = num_steps
        # 保存每步的操作数
        self.num_operations = num_operations
        # 输出维度 = 步数 × 每步操作数
        self.output_dim = num_steps * num_operations

        # 全局平均池化，把空间信息压缩成通道描述
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两层全连接网络，用于生成操作权重
        self.fc = nn.Sequential(
            # 第一层线性映射，扩大特征维度
            nn.Linear(task_channels, self.output_dim * 2),
            # ReLU 激活
            nn.ReLU(inplace=True),
            # 第二层线性映射，输出最终权重
            nn.Linear(self.output_dim * 2, self.output_dim)
        )

    # 前向传播
    def forward(self, x):
        # 对输入特征做全局平均池化
        pooled = self.avg_pool(x)
        # 将 [B, C, 1, 1] 拉平成 [B, C]
        pooled = pooled.view(x.size(0), -1)
        # 通过全连接层生成权重
        weights = self.fc(pooled)
        # 将权重 reshape 成 [B, num_steps, num_operations]
        weights = weights.view(-1, self.num_steps, self.num_operations)
        # 在操作维度上做 softmax，使每一步中各操作权重之和为 1
        weights = F.softmax(weights, dim=-1)
        # 返回每一步各操作的权重
        return weights


# 定义单层操作模块
class OperationLayer(nn.Module):
    # channels 表示输入输出通道数
    def __init__(self, channels):
        # 调用父类初始化
        super().__init__()

        # 定义 5 种候选操作
        self.operations = nn.ModuleList([
            BasicConv3x3(channels),
            DilatedConv3x3(channels, dilation=2),
            SepConv3x3(channels),
            AvgPoolConv(channels),
            IdentityConv(channels)
        ])

        # 将多个操作的输出拼接后，再用 1×1 卷积融合
        self.fuse = nn.Sequential(
            # 输入通道数是 channels × 操作数
            nn.Conv2d(channels * len(self.operations), channels, kernel_size=1, padding=0, bias=False),
            # ReLU 激活
            nn.ReLU(inplace=True)
        )

    # 前向传播
    def forward(self, x, weights):
        # 将权重从 [B, num_operations] 转成 [num_operations, B]
        weights = weights.transpose(1, 0)
        # 用列表保存每个操作的输出
        outputs = []

        # 遍历每个操作及其对应权重
        for weight, op in zip(weights, self.operations):
            # 当前操作的输出乘以对应权重
            op_out = op(x) * weight.view(-1, 1, 1, 1)
            # 保存结果
            outputs.append(op_out)

        # 在通道维上拼接所有操作结果
        out = torch.cat(outputs, dim=1)
        # 用 1×1 卷积融合多个操作输出
        out = self.fuse(out)
        # 返回融合结果
        return out


# 定义多步操作模块
class GroupOLs(nn.Module):
    # num_steps 表示一共堆叠多少个 OperationLayer
    def __init__(self, num_steps, channels):
        # 调用父类初始化
        super().__init__()
        # 保存步数
        self.num_steps = num_steps
        # 按照步数构建多个操作层
        self.layers = nn.ModuleList([OperationLayer(channels) for _ in range(num_steps)])

    # 前向传播
    def forward(self, x, weights):
        # 依次执行每一步操作
        for i in range(self.num_steps):
            # 第 i 步使用对应的权重
            x = self.layers[i](x, weights[:, i, :])
        # 返回最终结果
        return x


# 定义完整的 TARBlock 模块
class TARBlock(nn.Module):
    # channels 是输入输出特征通道数
    # num_tasks 是任务适配器输出的任务通道数
    # num_steps 是操作层的步数
    # use_residual 表示是否使用残差连接
    def __init__(self, channels, num_tasks=3, num_steps=2, use_residual=True):
        # 调用父类初始化
        super().__init__()
        # 保存输入通道数
        self.channels = channels
        # 保存任务数
        self.num_tasks = num_tasks
        # 保存步数
        self.num_steps = num_steps
        # 保存是否使用残差连接
        self.use_residual = use_residual

        # 构建任务适配器，用于生成任务相关特征
        self.task_adaptor = TaskAdaptor(
            in_channels=channels,
            out_channels=num_tasks,
            semodule=SeModule(num_tasks)
        )

        # 构建操作注意力模块，用于生成每一步各操作的权重
        self.operation_attn = OperationAttn(
            task_channels=num_tasks,
            num_steps=num_steps,
            num_operations=5
        )

        # 构建多步操作层
        self.operation_layers = GroupOLs(
            num_steps=num_steps, # 默认循环两次
            channels=channels
        )

        # 最终再用一个 1×1 卷积进行融合
        self.final_fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    # 前向传播
    def forward(self, x):
        # 保存输入，后面做残差连接
        residual = x

        # 第一步：通过任务适配器提取任务相关特征【四个卷积激活函数+SE注意力】
        task_logits = self.task_adaptor(x)
        # 第二步：根据任务特征生成每一步中各操作的权重【全局平均池化+softmax得到特征权重】
        op_weights = self.operation_attn(task_logits)
        # 第三步：根据权重执行多步操作 【遍历多次，将权重与输入特征x进行相乘加权！】
        restored = self.operation_layers(x, op_weights)
        # 第四步：进行最终特征融合
        restored = self.final_fuse(restored)

        # 如果启用残差连接，则把输入加回去
        if self.use_residual:
            restored = restored + residual

        return restored

if __name__ == "__main__":
    # 构造一个随机输入张量，形状为 [1, 32, 50, 50]
    x = torch.randn(1, 32, 50, 50)
    model = TARBlock(channels=32, num_tasks=3, num_steps=2, use_residual=True)
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")