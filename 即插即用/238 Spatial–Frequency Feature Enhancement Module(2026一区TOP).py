import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

"""
    论文地址：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11363305
    论文题目：A Cross-Modality Feature Adaptive Interaction Approach for RGB-Infrared Object Detection in Aerial Imagery (2026一区TOP)
    中文题目：面向航拍图像 RGB - 红外目标检测的跨模态特征自适应交互方法 (2026一区TOP)
    讲解视频：https://www.bilibili.com/video/BV1hCVw6ZERq/
    空间-频域特征增强模块模块（Spatial–Frequency Feature Enhancement Module，SFFEM）
        实际意义：①RGB模态在复杂空中场景下的特征不鲁棒性：空中图像（尤其是无人机视角）存在光照变化、运动模糊、背景杂波等问题，导致RGB特征容易受干扰而退化。
                ②传统方法仅在空间域（Spatial Domain）操作，忽略频率域信息：大多数现有方法仅使用空间卷积，缺失频率域的全局结构线索。而频率信息对捕捉目标的整体形状、纹理和全局上下文非常重要。
        实现方式：通过空间边缘增强与频率结构建模的双分支融合，同时强化 RGB 图像的小目标局部细节和全局结构表征。

"""

def autopad(k, p=None, d=1):
    # 当膨胀率大于 1 时，卷积核在空间上的实际覆盖范围会变大
    if d > 1:
        # 若卷积核尺寸为整数，则计算单个等效卷积核尺寸
        # 例如：k=3、d=2 时，等效卷积核尺寸为 2*(3-1)+1=5
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]

    # 当用户未主动指定 padding 时，自动计算保持尺寸所需的填充值
    if p is None:
        # 奇数卷积核通常取一半作为 padding，从而尽可能保持输入输出尺寸一致
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]

    # 返回最终计算得到的 padding 参数
    return p


class Conv(nn.Module):
    # 将 SiLU 设置为该卷积模块默认使用的激活函数
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        # 调用父类初始化方法，注册当前模块中的网络层
        super().__init__()

        # 构建二维卷积层：
        # c1 表示输入通道数，c2 表示输出通道数
        # k 表示卷积核尺寸，s 表示卷积步长
        # g 表示分组数量，d 表示膨胀率
        # bias=False 是因为后续 BatchNorm 已包含可学习的偏移能力
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False
        )

        # 使用批归一化稳定特征分布，加速网络训练过程
        self.bn = nn.BatchNorm2d(c2)

        # 当 act=True 时，使用默认的 SiLU 激活函数
        if act is True:
            self.act = self.default_act

        # 当用户传入一个具体激活层时，直接使用该激活层
        elif isinstance(act, nn.Module):
            self.act = act

        # 当 act=False 时，不进行非线性变换
        else:
            self.act = nn.Identity()

    def forward(self, x):
        # 训练阶段的标准计算顺序：卷积 -> 批归一化 -> 激活函数
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # 当卷积层与 BatchNorm 在推理阶段完成融合后，
        # 只需要执行卷积操作和激活函数操作
        return self.act(self.conv(x))


class ScharrConv(nn.Module):
    def __init__(self, channel):
        # 调用父类初始化方法
        super(ScharrConv, self).__init__()

        # 定义水平方向的 Scharr 卷积核，用于感知左右方向的灰度或特征变化
        scharr_kernel_x = np.array(
            [
                [3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]
            ],
            dtype=np.float32
        )

        # 定义垂直方向的 Scharr 卷积核，用于感知上下方向的灰度或特征变化
        scharr_kernel_y = np.array(
            [
                [3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]
            ],
            dtype=np.float32
        )

        # 将水平方向的 numpy 卷积核转换为 PyTorch 张量
        scharr_kernel_x = torch.tensor(scharr_kernel_x, dtype=torch.float32)

        # 将卷积核由 [3, 3] 扩展为卷积权重所需的 [1, 1, 3, 3] 形状
        scharr_kernel_x = scharr_kernel_x.unsqueeze(0).unsqueeze(0)

        # 将垂直方向的 numpy 卷积核转换为 PyTorch 张量
        scharr_kernel_y = torch.tensor(scharr_kernel_y, dtype=torch.float32)

        # 将卷积核由 [3, 3] 扩展为卷积权重所需的 [1, 1, 3, 3] 形状
        scharr_kernel_y = scharr_kernel_y.unsqueeze(0).unsqueeze(0)

        # 将单通道的水平 Scharr 卷积核复制到每一个输入通道
        # 最终形状为 [channel, 1, 3, 3]
        self.scharr_kernel_x = scharr_kernel_x.expand(channel, 1, 3, 3)

        # 将单通道的垂直 Scharr 卷积核复制到每一个输入通道
        # 最终形状为 [channel, 1, 3, 3]
        self.scharr_kernel_y = scharr_kernel_y.expand(channel, 1, 3, 3)

        # 构建水平方向的深度卷积层
        # groups=channel 表示每个通道独立提取边缘，不发生通道混合
        self.scharr_kernel_x_conv = nn.Conv2d(
            channel,
            channel,
            kernel_size=3,
            padding=1,
            groups=channel,
            bias=False
        )

        # 构建垂直方向的深度卷积层
        # padding=1 用于保持特征图空间尺寸不变
        self.scharr_kernel_y_conv = nn.Conv2d(
            channel,
            channel,
            kernel_size=3,
            padding=1,
            groups=channel,
            bias=False
        )

        # 将水平方向卷积层的参数替换为预定义的 Scharr 水平梯度核
        self.scharr_kernel_x_conv.weight.data = self.scharr_kernel_x.clone()

        # 将垂直方向卷积层的参数替换为预定义的 Scharr 垂直梯度核
        self.scharr_kernel_y_conv.weight.data = self.scharr_kernel_y.clone()

        # 固定水平方向 Scharr 核，使其仅作为边缘算子使用而不参与训练更新
        self.scharr_kernel_x_conv.weight.requires_grad = False

        # 固定垂直方向 Scharr 核，使其仅作为边缘算子使用而不参与训练更新
        self.scharr_kernel_y_conv.weight.requires_grad = False

    def forward(self, x):
        # 使用水平方向 Scharr 卷积核计算输入特征的水平梯度响应
        grad_x = self.scharr_kernel_x_conv(x)

        # 使用垂直方向 Scharr 卷积核计算输入特征的垂直梯度响应
        grad_y = self.scharr_kernel_y_conv(x)

        # 将两个方向的梯度响应进行加权融合
        # 该结果能够同时描述水平边缘和垂直边缘信息
        edge_magnitude = grad_x * 0.5 + grad_y * 0.5

        # 返回融合后的局部边缘特征
        return edge_magnitude


class FreqSpatial(nn.Module):
    def __init__(self, in_channels):
        # 调用父类初始化方法
        super(FreqSpatial, self).__init__()

        # 构建 Scharr 边缘提取模块，用于空间分支中的局部结构增强
        self.sed = ScharrConv(in_channels)

        # 对 Scharr 提取到的边缘信息进行第一次可学习卷积映射
        self.spatial_conv1 = Conv(in_channels, in_channels)

        # 在加入原始输入的残差信息后，再次进行卷积细化
        self.spatial_conv2 = Conv(in_channels, in_channels)

        # 傅里叶变换后的每个通道都包含实部和虚部
        # 因此频域卷积的输入输出通道数均为原始通道数的两倍
        self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)

        # 频域特征经过逆傅里叶变换回到空间域后，
        # 使用卷积进一步恢复和增强空间表达
        self.fft_conv2 = Conv(in_channels, in_channels, 3)

        # 使用 1×1 卷积融合空间分支和频域分支输出的特征
        self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        # 读取输入特征图的形状：
        # batch 为批量大小，c 为通道数，h 和 w 为空间尺寸
        batch, c, h, w = x.size()

        # ==========================================================
        # 空间分支：提取局部边缘、纹理和轮廓信息
        # ==========================================================
        # 利用固定 Scharr 算子提取输入特征中的局部梯度信息【局部特征】
        spatial_feat = self.sed(x)
        # 使用可学习卷积进一步编码边缘响应
        spatial_feat = self.spatial_conv1(spatial_feat)
        # 将边缘增强特征与原始输入特征进行残差相加
        # 这样可以避免仅强调边缘而丢失原始内容信息
        spatial_feat = spatial_feat + x
        # 使用第二个卷积模块对融合后的空间特征进行细化
        spatial_feat = self.spatial_conv2(spatial_feat)

        # ==========================================================
        # 频域分支：提取更大范围的全局结构和频率模式信息
        # ==========================================================
        # 对输入特征执行二维实数傅里叶变换
        # 输出为复数频谱，包含实部和虚部
        # norm='ortho' 表示采用正交归一化，使正逆变换尺度更稳定
        fft_feat = torch.fft.rfft2(x, norm='ortho')

        # 取出频域复数特征的实部，并在最后增加一个维度
        # 形状由 [B, C, H, Wf] 变为 [B, C, H, Wf, 1]
        x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
        # 取出频域复数特征的虚部，并在最后增加一个维度
        # 形状由 [B, C, H, Wf] 变为 [B, C, H, Wf, 1]
        x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
        # 在最后一个维度上拼接实部与虚部
        # 最后一个维度中的两个数分别表示一个复数的实部和虚部
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)

        # 将“实部/虚部”维度合并到通道维度中
        # 这样普通二维卷积就可以处理频域中的复数表示
        # 通道数由 C 变为 2C
        fft_feat = rearrange(fft_feat,'b c h w d -> b (c d) h w').contiguous()
        # 使用卷积在频率域中对全局结构模式进行可学习变换
        fft_feat = self.fft_conv(fft_feat)

        # 将通道维重新拆分为“原始通道数”和“实部/虚部”两个维度
        # 该操作用于后续重新构造复数频谱
        fft_feat = rearrange(fft_feat,'b (c d) h w -> b c h w d',d=2).contiguous()
        # 将最后一维中的实部与虚部重新组合成 PyTorch 复数张量
        fft_feat = torch.view_as_complex(fft_feat)
        # 对处理后的频域特征执行二维逆傅里叶变换，s=(h, w) 用于确保恢复后的特征尺寸与原始输入完全一致
        fft_feat = torch.fft.irfft2(fft_feat,s=(h, w),norm='ortho')

        # 对返回空间域的频率增强特征进行进一步卷积细化
        fft_feat = self.fft_conv2(fft_feat)

        # ==========================================================
        # 双分支融合：整合局部边缘信息与全局频率结构信息
        # ==========================================================
        # 将空间分支特征与频域分支特征逐元素相加
        # spatial_feat 更关注局部轮廓，fft_feat 更关注全局结构
        out = spatial_feat + fft_feat

        # 通过 1×1 卷积对融合特征进行通道整合并输出最终增强特征
        return self.final_conv(out)

if __name__ == "__main__":
    # 1 表示批量大小，64 表示通道数，32×32 表示空间分辨率
    x = torch.randn(1, 64, 32, 32)
    model = FreqSpatial(64)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")