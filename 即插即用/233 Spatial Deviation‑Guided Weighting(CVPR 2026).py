import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/pdf/2603.18834
    论文题目：Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging（CVPR 2026）
    中文题目：统计特征引导的快速高分辨透射电子显微成像去噪方法（CVPR 2026）
    讲解视频：https://www.bilibili.com/video/BV1eD556vEcn/
    空间域偏差引导加权模块（Spatial Deviation‑Guided Weighting，SDGW）
        实际意义：①普通卷积 “全局统一处理”，无法适配局部噪声差异问题：传统卷积对图像所有位置用相同卷积操作，图像不同区域的信号波动、噪声强度差异极大，统一处理会模糊边缘并保留噪声。②不同空间位置的噪声强度和信号波动不一致问题：由于短曝光导致信噪比极低，图像中不同区域的波动情况并不相同。有些位置可能是真实结构，有些位置可能主要是随机噪声，还有一些位置是信号与噪声混合区域。
        实现方式：通过局部标准差感知每个位置的波动特征，并生成像素级动态权重来调节卷积响应，从而自适应增强结构特征、抑制局部噪声。
"""

class LocalWindowStdExtractor(nn.Module):
    """
    局部窗口标准差特征提取模块

    对应论文中的 Deviation Characteristic Extraction：
    对输入特征图的每个通道，计算每个空间位置局部窗口内的标准差，
    用于描述该位置附近的局部波动程度。
    """

    def __init__(self, window_size=3, num_channels=None, eps=1e-5):
        super(LocalWindowStdExtractor, self).__init__()

        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            assert len(window_size) == 2, "window_size 必须是整数或二元组"
            self.window_size = window_size

        self.num_channels = num_channels
        self.eps = eps

        self.pad_size = (
            self.window_size[0] // 2,
            self.window_size[1] // 2
        )

        if self.num_channels is not None:
            self._build_mean_filter()

    def _build_mean_filter(self):
        """
        构建固定的局部均值卷积核。

        该卷积核不参与训练，只用于按通道独立计算局部窗口均值。
        """
        window_height, window_width = self.window_size
        window_area = window_height * window_width

        base_mean_filter = torch.ones(
            1, 1, window_height, window_width
        ) / window_area

        channelwise_mean_filter = base_mean_filter.repeat(
            self.num_channels, 1, 1, 1
        )

        self.register_buffer(
            "channelwise_mean_filter",
            channelwise_mean_filter
        )

    def forward(self, input_feature):
        assert input_feature.dim() == 4, "输入必须是四维张量 [B, C, H, W]"

        batch_size, channels, height, width = input_feature.shape

        if self.num_channels is None:
            self.num_channels = channels
            self._build_mean_filter()
        else:
            assert self.num_channels == channels, (
                f"输入通道数 {channels} 与初始化通道数 {self.num_channels} 不一致"
            )

        # 使用 reflect padding，保证边缘位置的局部统计更自然
        padded_feature = F.pad(
            input_feature,
            pad=(
                self.pad_size[1], self.pad_size[1],
                self.pad_size[0], self.pad_size[0]
            ),
            mode="reflect"
        )

        # 计算局部一阶矩：E[x]
        local_mean = F.conv2d(
            padded_feature,
            weight=self.channelwise_mean_filter,
            bias=None,
            stride=1,
            padding=0,
            groups=channels
        )

        # 计算输入特征的平方
        squared_feature = input_feature ** 2

        padded_squared_feature = F.pad(
            squared_feature,
            pad=(
                self.pad_size[1], self.pad_size[1],
                self.pad_size[0], self.pad_size[0]
            ),
            mode="reflect"
        )

        # 计算局部二阶矩：E[x²]
        local_mean_square = F.conv2d(
            padded_squared_feature,
            weight=self.channelwise_mean_filter,
            bias=None,
            stride=1,
            padding=0,
            groups=channels
        )

        # 根据 std = sqrt(E[x²] - E[x]² + eps) 计算局部标准差
        local_deviation_map = torch.sqrt(
            torch.clamp(
                local_mean_square - local_mean ** 2,
                min=self.eps
            )
        )

        return local_deviation_map


class SpatialDeviationGuidedWeighting(nn.Module):
    """
    Spatial Deviation-Guided Weighting, SDGW 即插即用模块

    核心流程：
    1. 通过 3×3 卷积提取空间特征；
    2. 通过局部窗口标准差提取 Deviation Characteristic；
    3. 通过 1×1 卷积 + Sigmoid 生成空间动态权重；
    4. 用空间动态权重对卷积特征进行逐像素加权。
    """

    def __init__(
        self,
        channels,
        window_size=3,
        eps=1e-5,
        use_residual=False
    ):
        super(SpatialDeviationGuidedWeighting, self).__init__()

        self.use_residual = use_residual

        # 主分支：提取基础空间特征
        self.spatial_feature_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        # 统计分支：提取局部标准差特征，即 Deviation Characteristic
        self.local_deviation_extractor = LocalWindowStdExtractor(
            window_size=window_size,
            num_channels=channels,
            eps=eps
        )

        # 权重生成分支：根据局部波动特征生成逐像素空间权重
        self.deviation_weight_generator = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, input_feature):
        # 使用 3×3 卷积提取空间增强特征
        spatial_feature = self.spatial_feature_conv(input_feature)

        # 计算局部标准差，得到局部波动特征
        local_deviation_map = self.local_deviation_extractor(input_feature)
        # 根据局部标准差生成动态空间权重
        spatial_deviation_weight = self.deviation_weight_generator(local_deviation_map)

        # 使用动态权重对空间特征进行逐像素加权
        weighted_spatial_feature = spatial_feature * spatial_deviation_weight

        # 可选残差连接，便于插入到已有 CNN、UNet、YOLO、Restormer 等网络中
        if self.use_residual:
            output_feature = weighted_spatial_feature + input_feature
        else:
            output_feature = weighted_spatial_feature

        return output_feature

if __name__ == "__main__":
    input_feature = torch.randn(1, 32, 50, 50)
    model = SpatialDeviationGuidedWeighting(channels=32,window_size=3,use_residual=True)
    output_feature = model(input_feature)
    print("Input shape:", input_feature.shape)
    print("Output shape:", output_feature.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")