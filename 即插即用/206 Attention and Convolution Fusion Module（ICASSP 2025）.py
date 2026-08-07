import torch
import torch.nn as nn
from einops import rearrange
"""
    论文地址：https://arxiv.org/pdf/2408.01897
    论文题目：CAF-YOLO: A Robust Framework for Multi-Scale Lesion Detection in Biomedical Imagery（MICCAI 2025）
    中文题目：CAF-YOLO：一种用于生物医学影像多尺度病变检测的稳健框架（ICASSP 2025）
    讲解视频：https://www.bilibili.com/video/BV1Lpr8B5Egt/
    注意力与卷积融合模块（Attention and Convolution Fusion Module，ACFM）
        实际意义：①卷积核的长距离信息交互能力不足：卷积核受限于局部感受野，无法有效处理长距离依赖关系，在处理全局上下文的生物医学图像时表现不足。
                ②纯注意力机制忽视局部细节问题：Transformer类方法具备强大的全局建模能力，微小病灶往往依赖局部纹理、边缘和形态变化；纯注意力结构在噪声较强、样本有限场景中，特征泛化能力不足。
                ③计算开销与效率平衡：虽然 Transformer 擅长提取全局特征，但其传统的注意力机制计算负担较重。
        实现方式：在同一特征层内，通过并行的注意力分支与卷积分支，分别建模全局依赖关系与局部空间细节，在低计算复杂度约束下实现二者的有效融合。
"""
class ACFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(ACFM, self).__init__()

        # 多头注意力的头数，用于将通道拆分成多个子空间
        self.num_heads = num_heads

        # 可学习的温度参数，用于缩放注意力分数，控制注意力分布的“尖锐程度”
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 使用 1×1×1 的 3D 卷积生成 Q、K、V（通道维度扩展为 3 倍）
        self.qkv = nn.Conv3d(
            in_channels=dim,
            out_channels=dim * 3,
            kernel_size=(1, 1, 1),
            bias=bias
        )

        # 深度可分离 3D 卷积：在每个通道上独立提取局部空间信息
        self.qkv_dwconv = nn.Conv3d(
            in_channels=dim * 3,
            out_channels=dim * 3,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias
        )

        # 用于将注意力输出投影回原始通道维度
        self.project_out = nn.Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(1, 1, 1),
            bias=bias
        )

        # 将 Q/K/V 的多头信息映射为 9 个方向/权重（用于局部卷积）
        self.fc = nn.Conv3d(
            in_channels=3 * self.num_heads,
            out_channels=9,
            kernel_size=(1, 1, 1),
            bias=True
        )

        # 深度分组卷积，用于执行自适应的局部卷积建模
        self.dep_conv = nn.Conv3d(
            in_channels=9 * dim // self.num_heads,
            out_channels=dim,
            kernel_size=(3, 3, 3),
            groups=dim // self.num_heads,
            padding=1,
            bias=True
        )

    def forward(self, x):
        # x 的输入格式为 [B, C, H, W]
        b, c, h, w = x.shape
        temp = x
        # 在第 2 维增加一个“伪深度维度”，以适配 3D 卷积
        x = x.unsqueeze(2)  # [B, C, 1, H, W]
        # 先生成 QKV，再进行深度可分离卷积以增强局部建模能力
        qkv = self.qkv_dwconv(self.qkv(x))  # [B, 3C, 1, H, W]
        # 去掉伪深度维度，回到 2D 特征表示
        qkv = qkv.squeeze(2)  # [B, 3C, H, W]

        # 按空间位置展开，并按多头方式组织 QKV 特征
        f_all = qkv.reshape(
            b, h * w, 3 * self.num_heads, -1
        ).permute(0, 2, 1, 3)
        # 通过 1×1×1 卷积生成局部卷积所需的 9 个权重
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)
        # 将权重重新组织为局部卷积输入格式
        f_conv = f_all.permute(0, 3, 1, 2).reshape(
            b, 9 * c // self.num_heads, h, w
        )
        # 增加伪深度维度，送入 3D 深度分组卷积
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)
        # 去掉伪深度维度，得到局部卷积分支输出
        out_conv = out_conv.squeeze(2)  # [B, C, H, W]

        # 将 QKV 拆分为 Q、K、V
        q, k, v = qkv.chunk(3, dim=1)
        # 将通道拆分成多个注意力头，并展平成序列
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对 Q 和 K 做 L2 归一化，稳定注意力计算
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 计算自注意力矩阵，并用 temperature 调整尺度
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 对注意力权重做 softmax 归一化
        attn = attn.softmax(dim=-1)
        # 使用注意力权重加权 V，得到全局特征
        out = attn @ v
        # 将多头特征重新拼接回原始空间结构
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)

        # 增加伪深度维度，进行通道投影
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        out = temp + out

        # 将全局注意力结果与局部卷积结果相加，完成特征融合
        output = out + out_conv
        return output

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = ACFM(dim=32)
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")