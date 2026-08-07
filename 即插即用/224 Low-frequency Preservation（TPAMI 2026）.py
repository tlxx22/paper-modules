import torch
import torch.nn as nn
import torch.nn.init as init

"""
    论文地址：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11419884
    论文题目：AIRPNet: Adaptive Image Restoration with Privacy Protection in Steganographic Domain（TPAMI 2026）
    中文题目：AIRPNet：隐写域中面向隐私保护的自适应图像修复方法（TPAMI 2026）
    讲解视频：https://www.bilibili.com/video/BV1EAd2BAE3q/
    低频分量保留模块（Low-frequency Preservation，LP）
        实际意义：①仅恢复高频细节、不足以保证整体图像质量的问题：重要结构信息隐藏在低频子带中，如果只关注高频恢复、忽略低频信息，最终图像虽然细节可能有所增强，但整体结构、轮廓和内容一致性仍然会受到破坏。
                ②低频结构信息未被充分利用，导致恢复保真度不足的问题：低频分量中不仅包含亮度信息，还包含图像整体结构内容。如果这些信息不能被有效提取和恢复，就会对图像保真度产生影响。
        实现方式：通过“低频注意力定位 + 主干特征”的方式，显式挖掘低频子带中隐藏的结构信息，从而提升恢复图像的整体结构保真度。
"""

def initialize_weights(net_l, scale=1):
    # 定义一个权重初始化函数，用来初始化网络中的不同层
    if not isinstance(net_l, list):
        # 如果传入的不是列表，就把单个网络层包装成列表，便于统一处理
        net_l = [net_l]

    for net in net_l:
        # 遍历列表中的每一个网络或层
        for m in net.modules():
            # 遍历当前网络中的所有子模块

            if isinstance(m, nn.Conv2d):
                # 如果当前层是二维卷积层，则使用 Kaiming 正态分布初始化权重
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')

                # 对权重再乘一个缩放系数，常用于残差结构中控制初始输出幅度
                m.weight.data *= scale

                if m.bias is not None:
                    # 如果卷积层带偏置，则将偏置初始化为 0
                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                # 如果当前层是全连接层，同样使用 Kaiming 正态分布初始化
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')

                # 对全连接层权重也进行缩放
                m.weight.data *= scale

                if m.bias is not None:
                    # 如果全连接层有偏置，则置为 0
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                # 如果当前层是二维批归一化层，则把缩放系数 gamma 初始化为 1
                init.constant_(m.weight, 1)

                # 把偏移系数 beta 初始化为 0
                init.constant_(m.bias.data, 0.0)

class SELayer(nn.Module):
    # 定义 SE 通道注意力模块
    def __init__(self, channel, reduction=16):
        # channel 表示输入通道数，reduction 表示通道压缩比例
        super(SELayer, self).__init__()

        # 定义全局平均池化，把每个通道的空间信息压缩成一个数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 定义两层全连接网络，用来学习每个通道的重要性权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x 的形状一般为 [B, C, H, W]
        b, c, _, _ = x.size()

        # 先做全局平均池化，输出形状变为 [B, C, 1, 1]
        # 再通过 view 变成 [B, C]，方便送入全连接层
        y = self.avg_pool(x).view(b, c)

        # 通过全连接网络生成每个通道的权重
        # 然后再恢复成 [B, C, 1, 1] 的形状
        y = self.fc(y).view(b, c, 1, 1)

        # 将得到的通道权重逐通道乘到原特征图上
        # expand_as(x) 的作用是把权重扩展到和 x 相同的形状
        return x * y.expand_as(x)

class AttentionBlock(nn.Module):
    # 定义注意力分支模块，用来生成引导主干分支的注意力图
    def __init__(self, input=3, output=3, bias=True):
        # input 表示输入通道数，output 表示输出通道数
        super(AttentionBlock, self).__init__()

        # 第一层卷积：从输入中提取第一阶段特征
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)

        # 第二层卷积：输入为原始输入和第一层输出拼接后的特征
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)

        # 第三层卷积：输入为原始输入、第一层输出、第二层输出拼接后的特征
        # 最终输出注意力图
        self.conv3 = nn.Conv2d(input + 2 * 32, output, 3, 1, 1, bias=bias)

        # 定义 LeakyReLU 激活函数
        self.lrelu = nn.LeakyReLU(inplace=True)

        # 定义 SE 通道注意力模块，对拼接后的特征进行通道加权
        self.senet = SELayer(channel=input + 2 * 32)

        # 对最后一层卷积做特殊初始化
        # scale=0 表示把卷积权重初始化得非常小，使模块初始输出接近 0
        initialize_weights([self.conv3], 0.0)

    def forward(self, x):
        # 输入 x 先经过第一层卷积和激活，得到第一阶段特征 x1
        x1 = self.lrelu(self.conv1(x))

        # 将原始输入 x 和 x1 在通道维拼接后，再经过第二层卷积和激活，得到第二阶段特征 x2
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))

        # 将原始输入、第一阶段特征、第二阶段特征再次拼接
        x = torch.cat((x, x1, x2), 1)

        # 对拼接后的特征做 SE 通道注意力加权
        x = self.senet(x)
        # 通过最后一层卷积输出注意力图
        x3 = self.conv3(x)
        # 返回生成的注意力图
        return x3

class LPM(nn.Module):
    # 该模块由一个注意力分支和一个主干分支组成
    def __init__(self, in_channel=3, att_channel=3, width=16, bias=True):
        # in_channel 表示输入通道数
        # att_channel 表示注意力图通道数
        # width 表示主干特征通道宽度
        super(LPM, self).__init__()

        # 定义注意力分支，用来生成引导图 imp_map
        self.attn_block = AttentionBlock(input=in_channel, output=att_channel)

        # 主干分支第一层卷积，把输入映射到更高维特征空间
        self.conv1 = nn.Conv2d(in_channel, width, 3, 1, 1, bias=bias)

        # 主干分支第二层卷积
        # 输入为主干特征和注意力图拼接后的结果
        self.conv2 = nn.Conv2d(width + att_channel, width, 3, 1, 1, bias=bias)

        # 第一处 PReLU 激活函数
        self.prelu1 = nn.PReLU()

        # 主干分支第三层卷积
        # 同样再次融合注意力图
        self.conv3 = nn.Conv2d(width + att_channel, width, 3, 1, 1, bias=bias)

        # 第二处 PReLU 激活函数
        self.prelu2 = nn.PReLU()

        # 主干分支第四层卷积，用于进一步提取和融合特征
        self.conv4 = nn.Conv2d(width, width, 3, 1, 1, bias=bias)

        # 最后一层 1×1 卷积，把通道数恢复到输入通道数
        self.conv5 = nn.Conv2d(width, in_channel, 1, 1, 0, bias=bias)

    def forward(self, x):
        # 第一步：通过注意力分支生成引导图 imp_map
        imp_map = self.attn_block(x)

        # 第二步：输入先经过主干第一层卷积，得到基础特征 x1
        x1 = self.conv1(x)

        # 第三步：把 x1 和 imp_map 拼接后送入第二层卷积，再经过 PReLU 激活
        x2 = self.prelu1(self.conv2(torch.cat((x1, imp_map), 1)))

        # 将 x2 与 x1 做残差相加，保留原始主干特征信息
        x2 = x2 + x1

        # 第四步：再次把 x2 和 imp_map 拼接，进行第二次注意力引导融合
        x3 = self.prelu2(self.conv3(torch.cat((x2, imp_map), 1)))

        # 再次与 x1 做残差相加，增强信息传递和训练稳定性
        x3 = x3 + x1

        # 第五步：将 x3 送入第四层卷积做进一步变换
        x4 = self.conv4(x3)

        # 再次加入残差连接
        x4 = x4 + x1

        # 第六步：用 1×1 卷积把通道恢复成与输入相同
        x5 = self.conv5(x4)
        return x5

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = LPM(32, 2)
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")