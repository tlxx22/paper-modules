import torch
import torch.nn as nn
from typing import Optional, Callable, Any
from torch import Tensor
import numpy as np
import math

"""
    论文地址：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11305185
    论文题目：DCCS-Det: Directional Context and Cross-Scale-Aware Detector for Infrared Small Target (2026一区TOP)
    中文题目：DCCS-Det：面向红外小目标的方向感知与跨尺度检测器 (2026一区TOP)
    讲解视频：https://www.bilibili.com/video/BV1VZLs6vEaT/
    潜在语义感知与聚合模块（Latent-Aware Semantic Extraction and Aggregation，LaSEA）
        实际意义：①深层网络中小目标语义逐渐退化问题：小目标本身只占少量像素，经过多次下采样后，在深层特征图中的空间分辨率会进一步降低，目标响应变得非常弱，甚至被背景特征淹没。
                ②解码器融合阶段缺少可靠的语义引导：如果深层目标语义已经退化，那么解码器在融合浅层细节和深层语义时就缺少有效指导，容易出现：小目标漏检；背景被误识别为目标；边界不稳定。
                ③深层特征中存在信息冗余和背景噪声问题：深层特征中不仅目标语义可能被削弱，还可能存在大量冗余背景信息。这些冗余特征会干扰模型判断。
        实现方式：通过跨尺度卷积提取深层目标语义，并利用随机池化生成注意力权重，从而增强弱小目标表示、抑制背景噪声。
"""

# 定义一个函数，用于将通道数调整为 divisor 的整数倍
def make_divisible(value: float, divisor: int, min_value: Optional[int] = None) -> int:

    # 如果没有设置最小通道数，则默认最小值等于 divisor
    if min_value is None:

        # 将 divisor 作为最小通道数
        min_value = divisor

    # 将 value 调整为最接近 divisor 整数倍的数，同时不能小于 min_value
    adjusted_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

    # 如果调整后的数值比原值下降超过 10%，说明压缩过多
    if adjusted_value < 0.9 * value:

        # 再加一个 divisor，避免通道数被压得太小
        adjusted_value += divisor

    # 返回调整后的通道数
    return adjusted_value


# 定义一个函数，用字符串名称从模块中获取对应的子模块
def get_module_by_name(module: nn.Module, module_name: str):

    # getattr 可以根据字符串名字访问对象中的属性
    return getattr(module, module_name)


# 定义一个函数，用字符串名称给模块动态添加子模块
def set_module_by_name(module: nn.Module, module_name: str, sub_module: nn.Module):

    # setattr 可以根据字符串名字给对象设置属性
    return setattr(module, module_name, sub_module)


# 定义随机空间打乱函数，用于打乱特征图中的空间位置
def random_spatial_shuffle(features: Tensor, mode: int = 1) -> Tensor:

    # 如果输入是一个 Tensor，则将其包装成列表，方便后续统一处理
    if isinstance(features, Tensor):

        # 将单个特征图放入列表中
        features = [features]

    # 初始化随机索引，后面用于记录打乱顺序
    shuffle_indices = None

    # 创建一个列表，用于保存打乱后的特征图
    shuffled_features = []

    # 遍历输入的每一个特征图
    for feature_map in features:

        # 获取特征图的形状：batch_size、通道数、高度、宽度
        batch_size, channels, height, width = feature_map.shape

        # 如果 mode 等于 1，则对整个空间维度 H×W 进行整体随机打乱
        if mode == 1:

            # 将特征图从 [B, C, H, W] 拉平成 [B, C, H*W]
            feature_map = feature_map.flatten(2)

            # 如果还没有生成随机索引，则生成一次
            if shuffle_indices is None:

                # 生成长度为 H*W 的随机排列索引
                shuffle_indices = torch.randperm(feature_map.shape[-1], device=feature_map.device)

            # 按照随机索引重新排列空间位置
            feature_map = feature_map[:, :, shuffle_indices.to(feature_map.device)]

            # 将特征图恢复成原来的 [B, C, H, W] 形状
            feature_map = feature_map.reshape(batch_size, channels, height, width)

        # 如果 mode 不等于 1，则分别对高度和宽度两个方向进行打乱
        else:

            # 如果还没有生成随机索引，则分别为高度和宽度生成索引
            if shuffle_indices is None:

                # 生成高度方向和宽度方向的随机排列索引
                shuffle_indices = [
                    torch.randperm(height, device=feature_map.device),
                    torch.randperm(width, device=feature_map.device)
                ]

            # 按照随机索引打乱高度方向
            feature_map = feature_map[:, :, shuffle_indices[0].to(feature_map.device)]

            # 按照随机索引打乱宽度方向
            feature_map = feature_map[:, :, :, shuffle_indices[1].to(feature_map.device)]

        # 将打乱后的特征图保存到列表中
        shuffled_features.append(feature_map)

    # 返回打乱后的特征图列表
    return shuffled_features


# 定义带复杂度统计接口的自适应平均池化层
class ProfileAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):

    # 初始化函数，指定池化后的输出尺寸
    def __init__(self, output_size: int or tuple = 1):

        # 调用父类 nn.AdaptiveAvgPool2d 的初始化函数
        super(ProfileAdaptiveAvgPool2d, self).__init__(output_size=output_size)

    # 定义 profile_module 函数，用于兼容复杂度统计接口
    def profile_module(self, input_tensor: Tensor):

        # 对输入特征进行自适应平均池化
        output_tensor = self.forward(input_tensor)

        # 返回输出结果、参数量和计算量；池化层这里暂时记为 0
        return output_tensor, 0.0, 0.0


# 定义带复杂度统计接口的自适应最大池化层
class ProfileAdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):

    # 初始化函数，指定池化后的输出尺寸
    def __init__(self, output_size: int or tuple = 1):

        # 调用父类 nn.AdaptiveMaxPool2d 的初始化函数
        super(ProfileAdaptiveMaxPool2d, self).__init__(output_size=output_size)

    # 定义 profile_module 函数，用于兼容复杂度统计接口
    def profile_module(self, input_tensor: Tensor):

        # 对输入特征进行自适应最大池化
        output_tensor = self.forward(input_tensor)

        # 返回输出结果、参数量和计算量；池化层这里暂时记为 0
        return output_tensor, 0.0, 0.0


# 定义一个基础卷积模块，包含卷积、可选 BN 和可选激活函数
class ConvBNAct(nn.Module):

    # 初始化卷积模块需要的各种参数
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        padding: Optional[int] = None,
        groups: Optional[int] = 1,
        bias: Optional[bool] = None,
        use_bn: bool = False,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
        bn_momentum: Optional[float] = 0.1,
        **kwargs: Any
    ) -> None:

        # 调用 nn.Module 的初始化函数
        super(ConvBNAct, self).__init__()

        # 如果没有手动设置 padding，则根据卷积核大小和膨胀率自动计算 padding
        if padding is None:

            # 该 padding 设置通常可以让卷积前后的特征图尺寸保持不变
            padding = int((kernel_size - 1) // 2 * dilation)

        # 如果没有手动设置 bias，则根据是否使用 BN 来决定
        if bias is None:

            # 如果使用 BN，卷积层通常不需要 bias；如果不用 BN，则保留 bias
            bias = not use_bn

        # 保存输入通道数，方便后续查看模块结构
        self.in_channels = in_channels

        # 保存输出通道数
        self.out_channels = out_channels

        # 保存卷积核大小
        self.kernel_size = kernel_size

        # 保存卷积步长
        self.stride = stride

        # 保存 padding 大小
        self.padding = padding

        # 保存分组卷积的组数
        self.groups = groups

        # 保存是否使用 bias
        self.bias = bias

        # 定义二维卷积层
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            **kwargs
        )

        # 如果 use_bn 为 True，则使用 BatchNorm2d；否则使用 Identity 占位
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=0.001,
            momentum=bn_momentum
        ) if use_bn else nn.Identity()

        # 如果传入了激活函数，则创建激活层
        if activation_layer is not None:

            # 判断激活函数是否是 Sigmoid 类型
            if isinstance(list(activation_layer().named_modules())[0][1], nn.Sigmoid):

                # Sigmoid 一般不需要 inplace 参数
                self.activation = activation_layer()

            # 如果不是 Sigmoid，例如 ReLU
            else:

                # 对 ReLU 等激活函数使用 inplace=True，节省显存
                self.activation = activation_layer(inplace=True)

        # 如果没有传入激活函数
        else:

            # 不使用激活层
            self.activation = None

        # 对当前模块中的参数进行初始化
        self.apply(init_weights)

    # 定义前向传播过程
    def forward(self, input_tensor: Tensor) -> Tensor:

        # 输入先经过卷积层
        output = self.conv(input_tensor)

        # 再经过 BN 或 Identity
        output = self.bn(output)

        # 如果定义了激活函数
        if self.activation is not None:

            # 对特征进行非线性激活
            output = self.activation(output)

        # 返回处理后的特征
        return output


# 定义常见的归一化层类型，后面初始化参数时会用到
NORM_LAYER_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.GroupNorm,
    nn.BatchNorm3d,
)


# 定义通用权重初始化函数
def init_weights(module):

    # 如果传入的模块为空，则直接返回
    if module is None:

        # 不做任何操作
        return

    # 如果模块是卷积层或反卷积层
    elif isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):

        # 使用 Kaiming 均匀分布初始化卷积权重
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

        # 如果卷积层存在偏置项
        if module.bias is not None:

            # 计算 fan_in，用于确定偏置初始化范围
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)

            # 如果 fan_in 不为 0
            if fan_in != 0:

                # 根据 fan_in 计算初始化边界
                bound = 1 / math.sqrt(fan_in)

                # 使用均匀分布初始化偏置
                nn.init.uniform_(module.bias, -bound, bound)

    # 如果模块是归一化层
    elif isinstance(module, NORM_LAYER_TYPES):

        # 如果归一化层有 weight 参数
        if module.weight is not None:

            # 将归一化层的缩放系数初始化为 1
            nn.init.ones_(module.weight)

        # 如果归一化层有 bias 参数
        if module.bias is not None:

            # 将归一化层的偏置初始化为 0
            nn.init.zeros_(module.bias)

    # 如果模块是全连接层
    elif isinstance(module, nn.Linear):

        # 使用 Kaiming 均匀分布初始化全连接层权重
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

        # 如果全连接层存在偏置项
        if module.bias is not None:

            # 计算 fan_in，用于确定偏置初始化范围
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)

            # 根据 fan_in 计算初始化边界
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

            # 使用均匀分布初始化偏置
            nn.init.uniform_(module.bias, -bound, bound)

    # 如果模块是 Sequential 或 ModuleList 容器
    elif isinstance(module, (nn.Sequential, nn.ModuleList)):

        # 遍历容器中的每一个子模块
        for sub_module in module:

            # 递归初始化子模块
            init_weights(sub_module)

    # 如果当前模块还有子模块
    elif list(module.children()):

        # 遍历当前模块的所有子模块
        for sub_module in module.children():

            # 递归初始化子模块
            init_weights(sub_module)


# 定义随机池化通道注意力模块
class RandomPoolingChannelAttention(nn.Module):

    # 初始化注意力模块
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        reduction_ratio: int = 4,
        pool_scales: list = [1, 2, 3],
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
        scale_activation_layer: Callable[..., nn.Module] = nn.Sigmoid,
        enable_spatial_shuffle: bool = True,
        **kwargs: Any,
    ) -> None:

        # 调用 nn.Module 的初始化函数
        super().__init__()

        # 如果没有指定注意力中间层通道数
        if hidden_channels is None:

            # 根据输入通道数和压缩比例自动计算隐藏通道数，且至少为 32
            hidden_channels = max(make_divisible(in_channels // reduction_ratio, 8), 32)

        # 如果候选池化尺度中没有 1，则额外加入 1，保证推理阶段可以使用全局池化
        all_pool_scales = pool_scales + [1] if 1 not in pool_scales else pool_scales

        # 遍历所有候选池化尺度
        for pool_size in all_pool_scales:

            # 创建对应输出尺寸的自适应平均池化层
            pooling_layer = ProfileAdaptiveAvgPool2d(pool_size)

            # 将池化层动态注册到当前模块中，例如 pool_1、pool_2、pool_3
            set_module_by_name(self, f"pool_{pool_size}", pooling_layer)

        # 定义注意力生成器，用两个 1×1 卷积生成通道注意力权重
        self.attention_generator = nn.Sequential(
            # 第一个 1×1 卷积用于压缩通道，并加入 ReLU 激活
            ConvBNAct(in_channels, hidden_channels, 1, activation_layer=activation_layer),

            # 第二个 1×1 卷积用于恢复通道数，并用 Sigmoid 得到 0 到 1 的权重
            ConvBNAct(hidden_channels, in_channels, 1, activation_layer=scale_activation_layer),
        )

        # 保存候选池化尺度，例如 [1, 2, 3]
        self.pool_scales = pool_scales

        # 保存是否启用空间随机打乱
        self.enable_spatial_shuffle = enable_spatial_shuffle

    # 定义随机池化采样函数
    def random_pooling_sample(self, input_features: Tensor) -> Tensor:
        # 如果当前模型处于训练模式
        if self.training:
            # 从候选池化尺度中随机选择一个尺度
            selected_pool_scale = np.random.choice(self.pool_scales)
            # 如果启用空间随机打乱
            if self.enable_spatial_shuffle:
                # 对输入特征做空间位置打乱，并取出打乱后的第一个特征图
                shuffled_features = random_spatial_shuffle(input_features)[0]
            # 如果不启用空间随机打乱
            else:
                # 直接使用原始输入特征
                shuffled_features = input_features
            # 根据随机选中的池化尺度，动态调用对应的池化层
            pooled_features: Tensor = get_module_by_name(
                self,
                f"pool_{selected_pool_scale}"
            )(shuffled_features)
            # 如果池化后的空间尺寸大于 1×1
            if pooled_features.shape[-1] > 1:
                # 将空间维度拉平成一维，例如 [B, C, 3, 3] 变成 [B, C, 9]
                pooled_features = pooled_features.flatten(2)

                # 从所有空间位置中随机选择一个位置
                selected_position = torch.randperm(
                    pooled_features.shape[-1],
                    device=pooled_features.device
                )[0]
                # 取出被选中的空间位置对应的特征
                pooled_features = pooled_features[:, :, selected_position]
                # 恢复成 [B, C, 1, 1]，方便后续生成通道注意力
                pooled_features = pooled_features[:, :, None, None]
        # 如果当前模型处于推理模式
        else:
            # 推理阶段固定使用 1×1 全局平均池化，保证输出稳定
            pooled_features: Tensor = get_module_by_name(self, "pool_1")(input_features)

        # 返回池化采样后的上下文特征
        return pooled_features

    # 定义前向传播
    def forward(self, input_features: Tensor) -> Tensor:
        # 通过随机池化采样得到通道上下文信息
        pooled_context = self.random_pooling_sample(input_features)
        # 将池化后的上下文输入注意力生成器，得到通道注意力权重
        channel_attention = self.attention_generator(pooled_context)
        # 用通道注意力权重对输入特征进行逐通道加权
        return input_features * channel_attention


# 定义通道混洗函数，用于增强不同分支之间的通道交互
def channel_shuffle(input_features: Tensor, groups: int) -> Tensor:
    # 获取输入特征图的形状
    batch_size, num_channels, height, width = input_features.size()
    # 计算每个分组中包含多少个通道
    channels_per_group = num_channels // groups
    # 将通道维拆分成 groups 和 channels_per_group 两个维度
    input_features = input_features.view(
        batch_size,
        groups,
        channels_per_group,
        height,
        width
    )
    # 交换 groups 和 channels_per_group 两个维度，实现不同分组之间的信息混合
    input_features = torch.transpose(input_features, 1, 2).contiguous()
    # 将特征图恢复为 [B, C, H, W] 的形状
    input_features = input_features.view(batch_size, -1, height, width)
    # 返回通道混洗后的特征
    return input_features

# 定义 LaSEA 模块
class LaSEA(nn.Module):

    # 初始化 LaSEA 模块
    def __init__(self, in_channels: int):

        # 调用 nn.Module 的初始化函数
        super(LaSEA, self).__init__()

        # 保存输入通道数
        self.in_channels = in_channels

        # LaSEA 的输出通道数与输入通道数保持一致，方便残差连接
        self.out_channels = in_channels

        # 定义第一个分支：普通 3×3 卷积，膨胀率为 1，关注局部细节
        self.dilation_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 定义第二个分支：3×3 膨胀卷积，膨胀率为 2，扩大感受野
        self.dilation_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 定义第三个分支：3×3 膨胀卷积，膨胀率为 3，进一步扩大感受野
        self.dilation_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 定义第四个分支：3×3 膨胀卷积，膨胀率为 4，获取更大范围上下文
        self.dilation_conv_4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 定义多尺度融合层，将四个分支拼接后的 4 倍通道压缩回原通道数
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 定义随机池化通道注意力模块，用于增强重要语义通道
        self.random_pool_attention = RandomPoolingChannelAttention(
            in_channels=in_channels,
            hidden_channels=16
        )

    def forward(self, input_features: Tensor) -> Tensor:
        # 保存原始输入，用于最后的残差连接
        identity = input_features
        # 使用膨胀率为 1 的卷积分支提取局部语义特征
        feat_dilation_1 = self.dilation_conv_1(input_features)
        # 使用膨胀率为 2 的卷积分支提取稍大感受野特征
        feat_dilation_2 = self.dilation_conv_2(input_features)
        # 使用膨胀率为 3 的卷积分支提取更大感受野特征
        feat_dilation_3 = self.dilation_conv_3(input_features)
        # 使用膨胀率为 4 的卷积分支提取大范围上下文特征
        feat_dilation_4 = self.dilation_conv_4(input_features)

        # 将四个不同感受野分支的特征在通道维度上拼接
        multi_scale_features = torch.cat(
            [
                feat_dilation_1,
                feat_dilation_2,
                feat_dilation_3,
                feat_dilation_4
            ],
            dim=1
        )
        # 对拼接后的多尺度特征进行通道混洗，促进不同分支之间的信息交互
        shuffled_features = channel_shuffle(multi_scale_features, groups=4)
        # 使用 1×1 卷积融合多尺度特征，并将通道数压缩回 in_channels
        fused_features = self.multi_scale_fusion(shuffled_features)
        # 使用【随机池化注意力】对融合后的特征进行通道加权
        attended_features = self.random_pool_attention(fused_features)
        # 将注意力增强后的特征与原始输入相加，形成残差增强输出
        output_features = attended_features + identity
        # 返回 LaSEA 模块的输出特征
        return output_features

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 50, 50)
    model = LaSEA(in_channels=32)
    output_tensor = model(input_tensor)
    print("输入特征维度：", input_tensor.shape)
    print("输出特征维度：", output_tensor.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")