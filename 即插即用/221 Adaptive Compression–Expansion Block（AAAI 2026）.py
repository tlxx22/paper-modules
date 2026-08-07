import torch
import torch.nn as nn

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/view/37241/41203
    论文题目：SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition（AAAI 2026）
    中文题目：SEMC：面向超声标准平面识别的结构增强混合专家对比学习方法（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1rHDzBuEvP/
    自适应压缩-扩展块（Adaptive Compression–Expansion Block，ACE）【下采样】
        实际意义：①浅层特征与深层特征之间的尺度不匹配问题：浅层特征通常分辨率更高，而深层特征经过多次下采样后分辨率更低。如果直接融合，就会因为尺寸不同而无法有效对齐。
                ②浅层特征与深层特征之间的通道不匹配问题：浅层特征偏底层纹理和边缘信息，通道表达能力与深层高语义特征不同。
        实现方式：浅层结构特征自适应对齐到深层语义空间，实现高效的浅深层特征融合。
"""

def get_activation(name='relu'):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'relu6':
        return nn.ReLU6(inplace=True)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Unsupported activation: {name}')

class ACE(nn.Module):
    """
    核心流程:
        1) 深度卷积下采样
        2) 1×1卷积调整通道
        3) 重复 down_times 次
        4) 最后再用 1×1 卷积映射到目标通道
    """

    def __init__(self, in_channels, out_channels, down_times=1, activation='relu'):
        super().__init__()

        act = get_activation(activation)
        stages = []

        channel_list = [in_channels * (2 ** i) for i in range(down_times + 1)]
        channel_list[-1] = out_channels

        for i in range(down_times):
            in_ch = channel_list[i]
            out_ch = channel_list[i + 1]

            stages.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=in_ch,
                        bias=False
                    ),
                    nn.BatchNorm2d(in_ch),
                    act,
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_ch),
                    act
                )
            )

        self.stages = nn.Sequential(*stages)

        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(activation)
        )

    def forward(self, x):
        x = self.stages(x)
        x = self.out_proj(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 32, 50, 50)
    ace = ACE(in_channels=32, out_channels=32, down_times=1, activation='relu')
    y = ace(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")