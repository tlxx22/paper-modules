import torch                                  # 导入 PyTorch 主库
import torch.nn as nn                         # 导入神经网络模块
import torch.fft as fft                       # 导入傅里叶变换相关函数
"""
    论文地址：https://papers.miccai.org/miccai-2025/paper/3694_paper.pdf
    论文题目：PFESA: FFT-based Parameter-Free Edge and Structure Attention for Medical Image Segmentation（MICCAI 2026）
    中文题目：PFESA：基于 FFT 的无参数边缘与结构注意力医学图像分割方法（MICCAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1f4dBBCEvW/
    无参数的边缘与结构注意力（Parameter-Free Edge-Structure Attention，PFESA）
        实际意义：①浅层跳跃连接中的噪声干扰问题：浅层特征包含背景噪声和低信噪比信息，它会干扰语义重建过程，影响分割结果。
                ②下采样过程中边缘细节逐渐衰减的问题：医学图像分割对边界质量要求很高，在下采样过程中，高频边缘信息会逐渐减弱，导致分割结果容易出现边界模糊。③现有参数化注意力模块容易过拟合的问题：像 SE、CBAM这类注意力方法通常带有可学习参数。由于医学图像数据量往往有限，这些参数模块容易产生过拟合，削弱泛化能力。④现有注意力机制可解释性不足的问题：传统注意力模块虽然能提升性能，难以说明“为什么某些区域权重大”，这种黑盒特性不利于临床可信性。
        实现方式：先把特征“边缘细节”和“主体结构”拆开，再分别增强，最后融合指导原始特征的重标定。
"""

class PFESA(nn.Module):
    # base_ratio 用于控制低频掩码的范围大小
    def __init__(self, base_ratio=0.1):
        # 调用父类 nn.Module 的初始化函数
        super(PFESA, self).__init__()

        # 定义 Sigmoid 激活函数
        # 用于将注意力权重压缩到 0~1 范围内
        self.activation = nn.Sigmoid()

        # 保存基础频率比例参数
        self.base_ratio = base_ratio

        # 设置极小值，防止后续除法出现分母为 0 的情况
        self.eps = 1e-5

    # 定义高频边缘注意力分支：输入是高频特征，输出是边缘注意力图
    def _edge_attention(self, high_freq_feature):
        # 在空间维度 H 和 W 上计算均值
        # 得到每个通道对应的全局平均响应
        spatial_mean = high_freq_feature.mean(dim=[2, 3], keepdim=True)

        # 计算每个位置与均值的平方差
        # 平方差越大，表示当前位置变化越剧烈，更可能是边缘区域
        squared_deviation = (high_freq_feature - spatial_mean).pow(2)

        # 计算每个通道在空间维度上的方差
        # 方差用于衡量该通道整体的波动情况
        feature_variance = high_freq_feature.var(dim=[2, 3], keepdim=True)

        # 用平方差除以方差，得到归一化后的边缘响应图
        # 这样能够突出相对变化更加明显的局部区域
        edge_attention_map = squared_deviation / (feature_variance + self.eps)

        # 返回高频边缘注意力图
        return edge_attention_map

    # 定义低频结构注意力分支：输入是低频特征，输出是结构注意力图
    def _structure_attention(self, low_freq_feature):
        # 对低频特征逐元素平方，得到低频能量图
        # 低频部分通常更关注整体结构和主体轮廓
        low_freq_energy = torch.pow(low_freq_feature, 2)

        # 在空间维度上计算低频能量的均值
        # 表示每个通道整体的平均能量水平
        low_freq_energy_mean = torch.mean(low_freq_energy, dim=[2, 3], keepdim=True)

        # 在空间维度上计算低频能量的方差
        # 用于刻画低频能量分布的离散程度
        low_freq_energy_var = torch.var(low_freq_energy, dim=[2, 3], keepdim=True)

        # 对低频能量做归一化
        # 能量高于平均水平的位置会得到更大的响应
        structure_attention_map = (low_freq_energy - low_freq_energy_mean) / (low_freq_energy_var + self.eps)

        # 通过 Sigmoid 将注意力值映射到 0~1 范围
        structure_attention_map = self.activation(structure_attention_map)

        # 返回低频结构注意力图
        return structure_attention_map

    # 定义低频掩码生成函数：该掩码是一个二维高斯分布，中心位置对应低频区域
    def _create_low_freq_mask(self, height, width, device='cpu'):
        # 根据输入特征图的高宽比例调整掩码半径
        # 这样在非方形特征图下也能保持掩码分布合理
        mask_ratio = self.base_ratio * min(height, width) / max(height, width)

        # 在高度方向生成从 -1 到 1 的均匀坐标
        y_coords = torch.linspace(-1, 1, height, device=device)

        # 在宽度方向生成从 -1 到 1 的均匀坐标
        x_coords = torch.linspace(-1, 1, width, device=device)

        # 根据横纵坐标生成二维网格
        # grid_y 表示每个位置的纵向坐标
        # grid_x 表示每个位置的横向坐标
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # 生成二维高斯低频掩码
        # 越接近中心位置，值越大；越远离中心，值越小
        low_freq_mask = torch.exp(-(grid_y ** 2 + grid_x ** 2) / (2 * mask_ratio ** 2))

        # 返回低频掩码
        return low_freq_mask

    def forward(self, input_feature):
        # 获取输入特征的形状
        # batch_size 表示批大小
        # channels 表示通道数
        # height 和 width 表示特征图空间尺寸
        batch_size, channels, height, width = input_feature.size()

        # 对输入特征在 H 和 W 两个空间维度上做傅里叶变换：将特征从空间域转换到频域
        freq_feature = fft.fftn(input_feature, dim=(-2, -1))

        # 将频谱中心移动到中间位置，使得低频区域位于中心，更方便高低频分离
        freq_feature = fft.fftshift(freq_feature, dim=(-2, -1))

        # 根据当前特征图大小构造低频掩码，掩码中心区域值更高，对应低频成分
        low_freq_mask = self._create_low_freq_mask(height, width, device=freq_feature.device)
        # 用低频掩码提取频域中的低频部分
        low_freq_spectrum = freq_feature * low_freq_mask
        # 对低频频谱执行逆傅里叶变换，恢复到空间域：再取绝对值，得到低频特征
        low_freq_feature = torch.abs(fft.ifftn(low_freq_spectrum, dim=(-2, -1)))
        # 将低频特征送入结构注意力分支：得到低频结构注意力图
        structure_attention_map = self._structure_attention(low_freq_feature)

        # 通过 1 减去低频掩码，得到互补的高频掩码
        high_freq_mask = 1 - low_freq_mask
        # 用高频掩码提取频域中的高频部分
        high_freq_spectrum = freq_feature * high_freq_mask
        # 对高频频谱执行逆傅里叶变换，恢复到空间域：再取绝对值，得到高频特征
        high_freq_feature = torch.abs(fft.ifftn(high_freq_spectrum, dim=(-2, -1)))
        # 将高频特征送入边缘注意力分支：得到高频边缘注意力图
        edge_attention_map = self._edge_attention(high_freq_feature)

        # 将结构注意力和边缘注意力直接相加进行融合
        fused_attention_map = structure_attention_map + edge_attention_map
        # 对融合后的注意力图再做一次 Sigmoid 归一化：使最终注意力权重更加稳定
        fused_attention_map = self.activation(fused_attention_map)
        # 将最终注意力图与原始输入特征逐元素相乘：从而增强重要区域并抑制无关区域
        output_feature = fused_attention_map * input_feature
        return output_feature

if __name__ == "__main__":
    input_feature = torch.randn(1, 32, 50, 50)
    model = PFESA()
    output_feature = model(input_feature)
    print(f"输入张量形状: {input_feature.shape}")
    print(f"输出张量形状: {output_feature.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")