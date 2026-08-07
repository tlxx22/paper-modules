import torch
import torch.nn as nn

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/view/37241/41203
    论文题目：SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition（AAAI 2026）
    中文题目：SEMC：面向超声标准平面识别的结构增强混合专家对比学习方法（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1bFDWBMEFF/
    结构感知多上下文块（Structure-Aware Multi-Context Block，SAMC）
        实际意义：①融合后特征的判别能力仍然不足问题：仅仅特征融合融合还不够，模型并不能够有效区分不同样本间的细微差别。
                ②单一尺度难以刻画复杂解剖结构问题：目标具有尺度变化大，局部细节复杂的特点，如果只用单一感受野，很难兼顾局部细节和整体结构。
                ③关键区域和关键通道没有被充分突出问题：并不是所有区域、所有通道都同样重要，很多背景信息和噪声会干扰识别。 
                ④背景干扰和噪声会削弱结构表达的问题：超声图像本身就存在散斑噪声、低对比度、边界模糊等问题，导致重要结构细节容易被淹没。
        实现方式：在特征融合之后进一步增强关键结构表达，通过多尺度上下文和通道-空间联合注意力，提升细粒度判别能力并抑制噪声干扰。
"""

def get_activation(name="relu6", inplace=True, negative_slope=0.2):
    """
    根据名称返回激活函数
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU(inplace=inplace)
    else:
        raise ValueError(f"Unsupported activation type: {name}")


def gcd(a, b):
    """
    计算最大公约数，供 channel shuffle 分组使用
    """
    while b:
        a, b = b, a % b
    return a


def channel_shuffle(x, groups):
    """
    通道混洗操作
    输入:
        x: [B, C, H, W]
        groups: 分组数
    输出:
        打乱通道后的特征图
    """
    b, c, h, w = x.size()
    assert c % groups == 0, "通道数必须能被 groups 整除"

    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    对应论文 SAMC 中的公式 (8)

    结构:
        AvgPool + MLP
        MaxPool + MLP
        两支相加后 Sigmoid
    """
    def __init__(self, in_channels, reduction=16, activation="relu6"):
        super().__init__()

        reduced_channels = max(1, in_channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.act = get_activation(activation)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        attn = self.sigmoid(avg_out + max_out)
        return attn

class SpatialAttention(nn.Module):
    """
    空间注意力模块
    对应论文 SAMC 中的公式 (9)

    结构:
        沿通道维做 mean / max
        拼接后卷积
        Sigmoid 得到空间注意图
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7, 11), "kernel_size 必须是 3 / 7 / 11"

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return attn


class MultiScaleDepthwiseConv(nn.Module):
    """
    多尺度深度卷积
    对应论文中并行多尺度卷积提取 {F_k}

    每个分支:
        Depthwise Conv + BN + Activation
    """
    def __init__(self, in_channels, kernel_sizes=(1, 3, 5), stride=1, activation="relu6"):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=k,
                    stride=stride,
                    padding=k // 2,
                    groups=in_channels,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                get_activation(activation)
            )
            for k in kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        return outputs


class MultiScaleContextBlock(nn.Module):
    """
    多尺度上下文块
    对应论文公式 (10)

    步骤:
        1) 1x1 升维
        2) 并行多尺度 depthwise conv
        3) 多分支融合（加和或拼接）
        4) channel shuffle
        5) 1x1 压缩输出
    """
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_sizes=(1, 3, 5),
        expansion_factor=2,
        stride=1,
        fuse_mode="add",     # "add" 或 "concat"
        activation="relu6",
        use_residual=True
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        assert fuse_mode in ["add", "concat"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.expanded_channels = int(in_channels * expansion_factor)
        self.fuse_mode = fuse_mode
        self.use_residual = use_residual and (stride == 1)

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expanded_channels),
            get_activation(activation)
        )

        self.ms_dwconv = MultiScaleDepthwiseConv(
            in_channels=self.expanded_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            activation=activation
        )

        if fuse_mode == "add":
            fused_channels = self.expanded_channels
        else:
            fused_channels = self.expanded_channels * len(kernel_sizes)

        self.project_conv = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if self.use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x):
        identity = x

        x = self.expand_conv(x)
        multi_scale_feats = self.ms_dwconv(x)

        if self.fuse_mode == "add":
            fused = 0
            for feat in multi_scale_feats:
                fused = fused + feat
        else:
            fused = torch.cat(multi_scale_feats, dim=1)

        groups = gcd(fused.shape[1], self.out_channels)
        groups = max(1, groups)
        fused = channel_shuffle(fused, groups=groups)

        out = self.project_conv(fused)

        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity

        return out


class SAMC(nn.Module):
    """
    Structure-Aware Multi-Context Block (SAMC)

    对应论文中的整体流程:
        1) 通道注意力
        2) 空间注意力
        3) 多尺度上下文重构

    输入:
        x: [B, C, H, W]
    输出:
        out: [B, C_out, H, W]
    """
    def __init__(
        self,
        in_channels,
        out_channels=None,
        reduction=16,
        spatial_kernel=7,
        multi_scale_kernels=(1, 3, 5),
        expansion_factor=2,
        fuse_mode="add",
        activation="relu6",
        use_residual=True
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            reduction=reduction,
            activation=activation
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel
        )

        self.multi_context = MultiScaleContextBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=multi_scale_kernels,
            expansion_factor=expansion_factor,
            stride=1,
            fuse_mode=fuse_mode,
            activation=activation,
            use_residual=use_residual
        )

    def forward(self, x):
        # Step 1: 通道注意力
        ca = self.channel_attention(x)
        x = ca * x

        # Step 2: 空间注意力
        sa = self.spatial_attention(x)
        x = sa * x

        # Step 3: 多尺度上下文重构
        out = self.multi_context(x)
        return out

if __name__ == "__main__":
    x = torch.randn(2, 32, 64, 64)
    model = SAMC(
        in_channels=32,
        out_channels=32
    )
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")