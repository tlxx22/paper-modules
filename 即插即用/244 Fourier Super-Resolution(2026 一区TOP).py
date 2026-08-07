import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络相关模块
import torch.nn.functional as F  # 导入常用函数接口，比如 relu

"""
    论文地址：https://arxiv.org/pdf/2503.10043
    论文题目：FourierSR: A Fourier Token-based Plugin for Efficient Image Super-Resolution(2026 一区TOP)
    中文题目：FourierSR：一种基于傅里叶 Token 的高效图像超分辨率插件模块(2026 一区TOP)
    讲解视频：https://www.bilibili.com/video/BV1zmKG6nEmh/
    基于傅里叶变换的特征超分模块（Fourier Super-Resolution，FourierSR）
        实际意义：①普通卷积感受野有限：高效超分辨率网络通常大量使用 3×3 卷积。虽然卷积计算稳定、部署方便，但只能在局部邻域内提取特征，需要堆叠较多层才能建立长距离联系。
                ②Transformer自注意力计算量较大：高效超分辨率模型通常采用窗口注意力降低复杂度，但窗口之间的信息连接有限，因此仍不能获得真正充分的全局感受野。
        实现方式：通过“傅里叶变换—通道令牌混合—实虚部调制—逆傅里叶变换”，以轻量化频域运算模拟全局卷积，从而增强模型的长距离特征建模能力。
"""
class Frequency_Convolution(nn.Module):  # 定义频域卷积模块，继承自 nn.Module
    def __init__(self, channels, num_blocks=8, sparsity_threshold=0.01):  # 初始化函数
        super().__init__()  # 调用父类初始化

        assert channels % num_blocks == 0, f"channels {channels} 必须能被 num_blocks {num_blocks} 整除"
        # 断言：通道数必须能被分组数整除，否则后面无法均匀分组

        self.channels = channels  # 保存输入通道数
        self.sparsity_threshold = sparsity_threshold  # 保存软阈值参数，用于抑制小的无效频域响应
        self.num_blocks = num_blocks  # 保存分组数量
        self.block_size = channels // self.num_blocks  # 计算每组包含多少个通道
        self.scale = 0.02  # 权重初始化的缩放系数，防止初始值过大

        self.w = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size, 2))
        # 定义频域通道混合权重
        # 形状为 [分组数, 输入组通道数, 输出组通道数, 2]
        # 最后一个维度 2 表示复数的实部和虚部

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        # 定义第一个频域滤波权重
        # 这里的 2 表示这组参数会分别参与实部和虚部的计算

        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        # 定义第二个频域滤波权重
        # 用于另一条分支的频域特征调制

        self.b = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        # 定义偏置项
        # 形状中的 2 同样对应两条分支

    def forward(self, x):  # 定义前向传播
        bias = x  # 保存输入，后面做残差连接
        dtype = x.dtype  # 记录输入的数据类型，便于最后恢复
        x = x.float()  # 将输入转成 float 类型，保证 FFT 运算稳定
        B, C, H, W = x.shape  # 读取输入特征图的 batch、通道、高、宽

        # 对输入做二维实数快速傅里叶变换，dim=(2,3) 表示沿着高和宽两个维度做 FFT，变换后x从空域特征变成频域特征
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")

        # 将通道维重新整理成“分组形式”，[B, C, Hf, Wf]===>[B, num_blocks, block_size, Hf, Wf]
        # [1, 64, 32, 32] ===> [1, 8, 8, 32, 32]
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])
        # 将 self.w 从“最后一维长度为2的实数表示”转成真正的复数张量，这样后面就可以直接参与复数乘法
        weight = torch.view_as_complex(self.w.contiguous())
        # 用爱因斯坦求和做分组内的通道混合，本质上是在每个组内做一次通道线性变换
        x = torch.einsum('bkihw,kio->bkohw', x, weight)

        # 计算新的实部特征，x.real 取复数特征的实部，x.imag 取复数特征的虚部
        # 这里按照复数运算思想，对实部和虚部进行线性组合，最后加偏置，再经过 ReLU 激活
        o1_real = F.relu(
            torch.mul(x.real, self.w1[0].unsqueeze(dim=0)) -
            torch.mul(x.imag, self.w1[1].unsqueeze(dim=0)) +
            self.b[0, :, :, None, None]
        )

        # 计算新的虚部特征
        # 同样是对原来的实部和虚部做组合，再加偏置，最后经过 ReLU
        o1_imag = F.relu(
            torch.mul(x.imag, self.w2[0].unsqueeze(dim=0)) +
            torch.mul(x.real, self.w2[1].unsqueeze(dim=0)) +
            self.b[1, :, :, None, None]
        )
        # 将新的实部和虚部重新堆叠起来
        x = torch.stack([o1_real, o1_imag], dim=-1)

        # 对频域响应做软阈值收缩，绝对值较小的元素会被压缩甚至变成0，减少噪声和无效频率成分
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        # 将分组后的通道重新拼接回原始通道数
        x = x.reshape(B, C, x.shape[3], x.shape[4])
        # 将频域特征重新还原回空域特征
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")

        x = x.type(dtype)
        # 残差连接：输出 = 频域增强结果 + 原始输入，有助于稳定训练，也能保留原始信息
        return x + bias

if __name__ == "__main__":  # 当前脚本直接运行时，执行下面的测试代码
    x = torch.randn(1, 64, 32, 32)
    # batch=1，通道数=64，高=32，宽=32
    model = Frequency_Convolution(64)
    y = model(x)
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")