import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2512.21542
    论文题目：Vision Transformers are Circulant Attention Learners（AAAI 2026）
    中文题目：MFmamba: 基于状态空间模型的全色图像分辨率恢复多功能网络（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1gCwazcEkD/
    基于傅里叶变换的循环注意力（Circulant Attention）
        实际意义：①自注意力计算复杂度过高的问题：在标准 Transformer中，自注意力需要计算所有Token之间的相关性，若输入包含𝑁个Token，则需要构建一个𝑁×𝑁的注意力矩阵，因此计算复杂度为：𝑂(𝑁2) 。
                ②高分辨率视觉任务效率不足的问题：在很多计算机视觉任务中（如目标检测、语义分割、医学图像分析等），需要处理大尺寸特征图。传统自注意力会出现：推理速度下降，GPU显存占用过高，难以扩展到更高分辨率。
                    因此，需要一种更高效的注意力计算方式。
        实现方式：通过将注意力矩阵约束为块循环结构，并利用 FFT 在频域完成注意力计算，将 Transformer 的注意力复杂度从𝑂(𝑁2)降至 𝑂( 𝑁log(𝑁) )，实现高效全局建模。
"""

class ComplexLinear(nn.Linear):
    # 定义一个复数线性层，用于处理频域中的复数特征
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # 调用父类 nn.Linear 初始化线性层（不使用 bias）
        super(ComplexLinear, self).__init__(in_features, out_features, False, device, dtype)

    def forward(self, complex_tensor):
        # 将复数张量转换为实部和虚部分离的表示形式
        # 原复数 tensor → [..., 2]，其中最后一维表示 [real, imag]
        real_imag = torch.view_as_real(complex_tensor).transpose(-2, -1)

        # 对拆分后的实部/虚部执行线性变换
        # linear 运算只作用于最后一维
        projected = torch.nn.functional.linear(real_imag, self.weight).transpose(-2, -1)

        # 若数据类型不是 float32，则转换为 float32
        if projected.dtype != torch.float32:
            projected = projected.to(torch.float32)

        # 将实部和虚部重新组合成复数张量
        complex_output = torch.view_as_complex(projected.contiguous())

        # 返回复数形式的输出
        return complex_output


class CirculantAttention(nn.Module):

    def __init__(self, channels, proj_drop=0.):
        # 初始化父类
        super().__init__()

        # QKV投影层：输入C维 → 输出3C维（分别对应Q、K、V）
        # 由于输入在频域中为复数，因此使用ComplexLinear
        self.qkv_projection = ComplexLinear(channels, channels * 3)

        # 门控层：用于动态调节特征重要性
        # Linear + SiLU 激活
        self.gating_layer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU()
        )

        # 输出线性投影层，用于融合注意力结果
        self.output_projection = nn.Linear(channels, channels)

        # Dropout层，用于防止过拟合
        self.output_dropout = nn.Dropout(proj_drop)

    def forward(self, feature_map):
        # 获取输入特征图尺寸
        batch_size, channels, height, width = feature_map.shape
        # 计算空间token数量（H×W）
        num_tokens = height * width
        # 将输入特征从 [B,C,H,W] 转换为 token序列 [B,N,C]
        token_features = feature_map.reshape(batch_size, channels, num_tokens).permute(0, 2, 1)
        # 通过门控层生成调制权重
        modulation_gate = self.gating_layer(token_features)

        # ·············[Transfomer部分开始]······················
        # 将token序列重新恢复为二维空间结构 [B,H,W,C]
        spatial_features = token_features.reshape(batch_size, height, width, channels)
        # 对空间特征进行二维快速傅里叶变换（FFT）
        # 将特征从时域转换到频域
        freq_features = torch.fft.rfft2(
            spatial_features,
            dim=(1, 2),
            norm='ortho'
        )
        # 在频域上生成 Q、K、V
        qkv_freq = self.qkv_projection(freq_features)
        query_freq, key_freq, value_freq = torch.chunk(qkv_freq, chunks=3, dim=-1)

        # 在频域计算注意力：共轭Q × K
        # 等价于时域中的相关性计算
        attention_freq = torch.conj(query_freq) * key_freq

        # 将频域注意力结果转换回空间域
        attention_spatial = torch.fft.irfft2(
            attention_freq,
            s=(height, width),
            dim=(1, 2),
            norm='ortho'
        )
        # 将空间注意力 reshape 为 token形式并进行 softmax 归一化
        attention_weights = (
            attention_spatial.reshape(batch_size, num_tokens, channels)
            .softmax(dim=1)
            .reshape(batch_size, height, width, channels)
        )

        # 将注意力权重再次转换到频域
        attention_freq = torch.fft.rfft2(attention_weights, dim=(1, 2))
        # 在频域中对 value 进行加权
        value_weighted_freq = torch.conj(attention_freq) * value_freq
        # 将加权后的特征转换回空间域
        value_weighted_spatial = torch.fft.irfft2(
            value_weighted_freq,
            s=(height, width),
            dim=(1, 2),
            norm='ortho'
        )
        # ·············[Transfomer部分结束]······················

        # 将输出重新整理为 token序列 [B,N,C]
        output_tokens = value_weighted_spatial.reshape(batch_size, num_tokens, channels)
        # 使用门控权重对输出特征进行调制
        output_tokens = output_tokens * modulation_gate

        # 线性投影融合通道信息
        output_tokens = self.output_projection(output_tokens)

        # 将 token序列恢复为特征图结构 [B,C,H,W]
        output_feature_map = output_tokens.permute(0, 2, 1).reshape(
            batch_size, channels, height, width
        )

        # Dropout防止过拟合
        output_feature_map = self.output_dropout(output_feature_map)

        # 返回最终输出
        return output_feature_map

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = CirculantAttention(32)
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")