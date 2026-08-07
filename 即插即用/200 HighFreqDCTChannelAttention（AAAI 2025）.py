import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络构建模块
import torch.nn.functional as F  # 导入常用的函数式接口（例如池化等）
import torch_dct as DCT  # 导入离散余弦变换（DCT）库，用于频域变换

"""
    
    论文地址：https://arxiv.org/abs/2501.03775
    论文题目：HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection（AAAI 2025）
    中文题目：全局上下文的线性注意力机制：一种面向视觉与物理任务的多极注意力方法（AAAI 2025 图像通用）
    讲解视频：https://www.bilibili.com/video/BV1N725BcEHW/
        高频通道路径感知模块（High Frequency Perception Channel Path,HFPC）
        实际意义：①微小物体特征占比小导致的注意力偏差：微小物体特征仅占很小比例，导致通用通道注意力计算（通常在整个特征图上进行）容易受到低频背景干扰，无法准确识别和突出微小物体特征通道。
                ②通道贡献不均且缺乏针对性关注的问题：每个通道对微小物体表示的贡献不同，但传统方法无法通过动态分配权重来强调富含微小物体特征的通道，从而使微小物体特征响应较弱，难以精确检测。
        实现方式：从高频特征 Fi 中提取的全局空间统计 → 使用分组卷积生成通道权重 → 强化高频通道、抑制低频背景通道。
"""

class HighFreqDCTChannelAttention(nn.Module):
    def __init__(self, channels, pool_size, freq_ratio):
        super().__init__()

        self.channels = channels
        self.ph, self.pw = pool_size
        self.freq_ratio = freq_ratio

        # 分组 1×1 卷积 —— 降维 + 建模通道关系
        self.reduce_conv = nn.Conv2d(channels, channels, 1, groups=32, bias=False)
        self.excite_conv = nn.Conv2d(channels, channels, 1, groups=32, bias=False)

        self.relu = nn.ReLU()

    def _build_freq_mask(self, h, w, ratio):
        """构造频域高频掩码：抑制低频，保留高频"""
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        mask = torch.ones((h, w), requires_grad=False)
        mask[:h0, :w0] = 0
        return mask

    def forward(self, x):
        n, c, h, w = x.size()

        # —— Step1: DCT 得到频域图 —— #
        freq_map = DCT.dct_2d(x, norm='ortho')

        # —— Step2: 构造高频掩码 —— #
        freq_mask = self._build_freq_mask(h, w, self.freq_ratio).to(x.device)
        freq_mask = freq_mask.view(1, 1, h, w).expand_as(freq_map)

        # —— Step3: 高频增强 —— # 离散余弦变换
        masked_freq = freq_map * freq_mask
        spatial_highfreq = DCT.idct_2d(masked_freq, norm='ortho')

        # —— Step4: 双池化统计 —— #
        max_pool = F.adaptive_max_pool2d(spatial_highfreq, (self.ph, self.pw))
        avg_pool = F.adaptive_avg_pool2d(spatial_highfreq, (self.ph, self.pw))

        max_pool = torch.sum(self.relu(max_pool), dim=[2, 3]).view(n, c, 1, 1)
        avg_pool = torch.sum(self.relu(avg_pool), dim=[2, 3]).view(n, c, 1, 1)

        # —— Step5: 通道注意力 —— #
        channel_desc = self.reduce_conv(max_pool) + self.reduce_conv(avg_pool)
        channel_weight = torch.sigmoid(self.excite_conv(channel_desc))

        return x * channel_weight

if __name__ == "__main__":
    # 构造一个示例输入张量，形状为 [批大小=1, 通道数=32, 高=50, 宽=50]
    x = torch.randn(1, 32, 50, 50)
    # 设置高频保留比例：0.25 表示在频域中左上角 25%×25% 的低频区域被抑制
    ratio = (0.25, 0.25)
    # 通道交互模块中池化后的空间尺寸，这里将特征压缩到 8×8 再做统计
    patch = (8, 8)
    model = HighFreqDCTChannelAttention(channels=32, pool_size=patch, freq_ratio=ratio)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")