import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2502.01303
    论文题目：Partial Channel Network: Compute Fewer, Perform Better（AAAI 2026）
    中文题目：部分通道网络：少算多得，性能更优（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1iMfYBAEPD/
    局部空间注意力块（Partial Spatial-Attention block，PSAB）
        实际意义：①MLP 层通道信息混合不足问题：在许多现代CNN架构中，MLP 层负责通道间的特征混合，但单纯1×1卷积在全局通道信息交互上仍然有限，当模型追求高效时，MLP层往往成为制约准确率瓶颈之一。
                ②传统空间注意力对推理速度的负面影响较大：在计算注意力图时，常规空间注意力机制需要对原始特征进行逐元素乘法，增加内存访问和计算延迟，在移动端/边缘设备场景下不友好。
        实现方式：PSAB通过将1×1 卷积与 Hard-Sigmoid 激活函数结合，生成空间注意力图，并与 MLP 层后的 Conv1×1 卷积合并，增强模型对空间信息的处理能力，提高计算效率。
"""

class PartialAttentionSA(nn.Module):
    def __init__(self, dim, partial=0.5):
        super().__init__()
        self.dim = dim
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv

        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 1)
        self.conv_attn = nn.Conv2d(self.dim_untouched, self.dim_conv, 1)

        self.norm = nn.BatchNorm2d(self.dim_untouched)
        self.norm2 = nn.BatchNorm2d(self.dim_conv)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_untouched, self.dim_conv], 1)
        weight = self.act(self.conv_attn(x1))
        x1 = self.norm(x1 * weight)
        x2 = self.norm2(self.conv(x2))
        return torch.cat((x1, x2), 1)

if __name__ == "__main__":
    x = torch.rand(1, 32, 50, 50)
    model = PartialAttentionSA(dim=32)
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")