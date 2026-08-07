import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

""" 
    论文地址：https://arxiv.org/abs/2501.03775
    论文题目：Linear Attention with Global Context: A Multipole Attention Mechanism for Vision and Physics (ICCV 2025) 
    中文题目：全局上下文的线性注意力机制：一种面向视觉与物理任务的多极注意力方法 (ICCV 2025) 
    讲解视频：https://www.bilibili.com/video/BV1eVSaBqEaW/
    多极注意力（Multipole Attention,MA）
        实际意义：①标准Transformer的计算复杂度问题：传统Transformer自注意力机制的复杂度为O(N²)，在处理高分辨率图像或网格数据（如物理模拟中的高分辨率网格）时，导致运行时间长和内存高。
                ②高效变体中细节丢失的问题：现有Transformer变体虽然降低计算成本，但往往牺牲性能。
                ③全局上下文与局部细节的平衡问题：在计算机视觉和物理模拟任务中，需要同时捕捉全局上下文和局部精细信息，现有方法难以在保持线性复杂度的前提下实现全局感受野。
        实现方式：把自注意力当成多体问题，通过多尺度分解（粗到细分解）来计算远/近距离的注意力，从而在保持全局感受野，将复杂度降为 O(N)。
"""

class AttentionBlock(nn.Module):
    # 全局多头自注意力模块（用于序列特征），支持可选的输出投影
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        # 调用父类 nn.Module 的初始化
        super().__init__()
        # 计算多头注意力内部使用的总通道数 = 每个头的维度 * 头数
        inner_dim = dim_head * heads
        # 如果只有 1 个头 且 该头的维度刚好等于输入维度 dim，则不需要额外的线性投影
        project_out = not (heads == 1 and dim_head == dim)

        # 记录多头数量
        self.heads = heads
        # 缩放系数 = 1 / sqrt(dim_head)，用于稳定点积注意力的数值
        self.scale = dim_head ** -0.5

        # 对最后一个维度 C 做 LayerNorm，提升训练稳定性
        self.norm = nn.LayerNorm(dim)

        # 沿着最后一个维度做 softmax，用于计算注意力权重
        self.attend = nn.Softmax(dim=-1)
        # dropout 用于对注意力权重进行随机丢弃，防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 线性层一次性映射出 Q、K、V 三个矩阵（拼在一起），再用 chunk 拆分
        # 输入维度 dim，输出维度为 inner_dim * 3（分别对应 Q/K/V）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 如果需要输出投影，则使用 Linear + Dropout；否则直接使用恒等映射
        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        # 期望输入形状为 [B, L, C]，B=批次，L=序列长度，C=通道维度
        # 先对输入做 LayerNorm
        x = self.norm(x)

        # 使用线性层得到拼接的 QKV，然后沿最后一个维度拆分成 3 份
        # qkv 是一个包含 3 个张量的元组 (Q, K, V)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # 使用 einops 的 rearrange，将每个张量从 [B, L, inner_dim] 变为 [B, H, L, D]
        # 其中 H = heads，D = dim_head，满足 inner_dim = H * D
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            qkv
        )

        # 计算注意力分数：Q @ K^T，得到形状 [B, H, L, L]，再乘以缩放因子
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 对最后一个维度做 softmax，得到归一化的注意力权重
        attn = self.attend(dots)
        # 对注意力权重使用 dropout
        attn = self.dropout(attn)

        # 使用注意力权重加权 V，得到加权后的特征 [B, H, L, D]
        out = torch.matmul(attn, v)
        # 将多头维度 H 和特征维度 D 合并，恢复为 [B, L, H*D]
        out = rearrange(out, "b h n d -> b n (h d)")
        # 如果设置了输出投影，则通过线性层映射回原始通道维度 dim
        return self.to_out(out)


class LocalAttention2D(nn.Module):
    # 在二维特征图上做局部窗口注意力的模块，通过 unfold/fold 实现
    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        # 调用父类 nn.Module 的初始化
        super().__init__()
        # 窗口大小（方形窗口 kernel_size x kernel_size）
        self.kernel_size = kernel_size
        # 窗口滑动步长（通常可设置为等于 kernel_size，实现不重叠窗口）
        self.stride = stride
        # 每个像素点的通道维度 C
        self.dim = dim
        # 这里定义 padding 变量（如果需要可以扩展成带 padding 的 unfold）
        padding = 0  # 当前未使用，可在需要时扩展

        # 对每个 patch 的通道维度做 LayerNorm
        self.norm = nn.LayerNorm(dim)

        # 内部使用前面定义的 AttentionBlock，对展开后的每个窗口做自注意力
        self.Attention = AttentionBlock(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )

        # nn.Unfold 用于将 [B, C, H, W] 展开为若干局部 2D patch
        # 输出形状为 [B, C*K*K, L]，其中 K=kernel_size，L=patch 数量
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding
        )

    def forward(self, x):
        # 输入 x 形状为 [B, H, W, C]（NHWC 格式）
        B, H, W, C = x.shape

        # 先将特征调整为 [B, C, H, W]，以适配 nn.Unfold 的输入格式
        x = rearrange(x, "B H W C -> B C H W")

        # 使用 unfold 将整幅图像拆成局部窗口 patch
        # 形状为 [B, C*K*K, L]，其中 L 为所有窗口的数量
        patches = self.unfold(x)

        # 将 patch 重新排列为 [B*L, K*K, C]
        # 解释：
        #   - 每个 patch 内有 K*K 个位置
        #   - 每个位置有 C 维特征
        #   - 将批次维 B 和窗口索引 L 合并，方便送入 AttentionBlock
        patches = rearrange(
            patches,
            "B (C K1 K2) L -> (B L) (K1 K2) C",
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        # 对每个 patch 的通道维度做 LayerNorm
        patches = self.norm(patches)

        # 在每个窗口内部做自注意力，输出形状仍为 [B*L, K*K, C]
        out = self.Attention(patches)

        # 将输出重新排列回 [B, C*K*K, L]，以便后续 fold 回原始空间
        out = rearrange(
            out,
            "(B L) (K1 K2) C -> B (C K1 K2) L",
            B=B,
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        # 使用 nn.Fold 将展开的窗口重新折叠回 [B, C, H, W]
        # 当 stride < kernel_size 时会有重叠区域，需要后续归一化
        fold = nn.Fold(
            output_size=(H, W),
            kernel_size=self.kernel_size,
            stride=self.stride
        )
        out = fold(out)

        # 计算每个位置被多少个窗口覆盖，用于重叠区域的归一化
        # 首先构造一个全 1 的特征图 [B, 1, H, W]，用相同方式 unfold+fold
        norm = self.unfold(torch.ones((B, 1, H, W), device=x.device))
        norm = fold(norm)  # 形状为 [B, 1, H, W]

        # 将输出除以覆盖次数，实现重叠区域的平均
        out = out / norm

        # 再将通道维调整回 [B, H, W, C] 格式
        out = rearrange(out, "B C H W -> B H W C")

        # 返回局部注意力增强后的特征图
        return out


class Multipole_Attention(nn.Module):
    # 多极注意力模块：在多个尺度上采用局部注意力，并通过上下采样进行特征融合
    def __init__(
        self,
        in_channels,                        # 输入特征通道数 C
        image_size,                         # 输入图像的空间尺寸（假设为正方形 H = W）
        local_attention_kernel_size=2,      # 局部注意力窗口大小（K）
        local_attention_stride=2,           # 窗口滑动步长（通常与 K 相同）
        downsampling="conv",                # 下采样方式：'avg_pool' 或 'conv'
        upsampling="conv",                  # 上采样方式：'avg_pool' 或 'conv'
        sampling_rate=2,                    # 金字塔每一层的缩放倍率（通常为 2）
        heads=4,                            # 多头注意力的头数
        dim_head=16,                        # 每个注意力头的维度
        dropout=0.1,                        # dropout 概率
        channel_scale=1,                    # 每一层通道数的扩展系数（>1 时多尺度通道会变宽）
    ):
        # 调用父类 nn.Module 的初始化
        super().__init__()

        # 根据图像尺寸和缩放倍率，计算金字塔的层数
        # 例如 image_size=64, sampling_rate=2，则 levels=log2(64)=6 层
        self.levels = int(math.log(image_size, sampling_rate))

        # 定义每一层的通道数列表
        # 这里简单地设置为 in_channels * (channel_scale ** i)
        channels_conv = [in_channels * (channel_scale ** i) for i in range(self.levels)]

        # 定义局部注意力模块
        # 注意：这里只使用了 channels_conv[0] 作为 dim，表示各层通道数相同
        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=channels_conv[0],
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )

        # ---------------- 下采样模块 ---------------- #
        if downsampling == "avg_pool":
            # 使用平均池化作为下采样方式
            self.down = nn.Sequential(
                # NHWC -> NCHW，适配 2D 池化
                Rearrange("B H W C -> B C H W"),
                # 按照 sampling_rate 进行下采样
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                # 再从 NCHW 转回 NHWC
                Rearrange("B C H W -> B H W C"),
            )

        elif downsampling == "conv":
            # 使用卷积（stride>1）实现下采样
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],   # 输入通道数
                    out_channels=channels_conv[0],  # 输出通道保持不变
                    kernel_size=sampling_rate,      # 卷积核大小等于缩放倍率
                    stride=sampling_rate,           # 步长等于缩放倍率，实现下采样
                    bias=False
                ),
                Rearrange("B C H W -> B H W C"),
            )
        else:
            # 如果给出了不支持的下采样方式，则抛出异常
            raise ValueError(f"Unsupported downsampling type: {downsampling}")

        # ---------------- 上采样模块 ---------------- #
        if upsampling == "avg_pool":
            # 对输入尺寸进行检查，确保能被每一层的缩放倍率整除
            current = image_size
            for _ in range(self.levels):
                assert current % sampling_rate == 0, "图像尺寸不能被采样倍率整除"
                current = current // sampling_rate

            # 使用最近邻插值进行上采样
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )

        elif upsampling == "conv":
            # 使用转置卷积（反卷积）进行上采样
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False
                ),
                Rearrange("B C H W -> B H W C"),
            )
        else:
            # 不支持的上采样方式同样抛出异常
            raise ValueError(f"Unsupported upsampling type: {upsampling}")

    def forward(self, x):
        # 输入 x 形状为 [B, C, H, W]，先调整为 [B, H, W, C]（NHWC）
        x = x.permute(0, 2, 3, 1)
        # 保存当前尺度的特征用于逐层下采样
        x_in = x

        # 用于存储每一层（每个尺度）的注意力输出
        x_out = []

        # 第 0 层（原始分辨率）直接做局部注意力
        x_out.append(self.Attention(x_in))

        # 从第 1 层开始，依次进行下采样 + 注意力
        for l in range(1, self.levels):
            # 对特征图做一次下采样 每次尺度大小除以2，64-->32-->16-->8（多尺度分解）
            x_in = self.down(x_in)
            # 在当前尺度上做局部注意力
            x_out_down = self.Attention(x_in)
            # 将结果保存到列表中
            x_out.append(x_out_down)

        # 从列表末尾取出最后一层（最小尺度）的特征作为初始融合结果
        res = x_out.pop()

        # 逐层向上（从粗到细）融合特征
        # x_out[::-1] 表示从高层到低层反向遍历
        for l, out_down in enumerate(x_out[::-1]):
            # 将当前融合结果上采样到上一尺度，并按权重缩放再与该尺度特征相加
            # 这里的 (1 / (l + 1)) 是一个简单的权重衰减策略
            res = out_down + (1 / (l + 1)) * self.up(res)

        # 最后将特征从 [B, H, W, C] 调整回 [B, C, H, W] 格式
        return res.permute(0, 3, 1, 2)

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 64, 64)
    model = Multipole_Attention(in_channels=32, image_size=64)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")