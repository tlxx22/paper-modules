import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialChannelFeatureModulator(nn.Module):
    def __init__(self, in_channels):
        super(SpatialChannelFeatureModulator, self).__init__()
        # 空间注意力分支
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        
        # 通道注意力分支
        self.channel_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 输出卷积层
        self.final_conv_spatial = nn.Conv2d(in_channels, in_channels, kernel_size=1)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.final_conv_channel = nn.Conv2d(in_channels, in_channels, kernel_size=1)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

    def forward(self, Fin):
        # --- 空间注意力分支 ---
        # 对通道维度求最大和平均 (dim=1 是单个维度，旧版torch.max也支持)
        max_pool = torch.max(Fin, dim=1, keepdim=True)[0]
        mean_pool = torch.mean(Fin, dim=1, keepdim=True)
        concat = torch.cat([max_pool, mean_pool], dim=1)
        Ws = torch.sigmoid(self.spatial_conv(concat))
        spatial_out = Ws * Fin
        
        # --- 通道注意力分支 (修复部分) ---
        Fd = F.relu(self.channel_conv1(Fin))
        Fd = self.channel_conv2(Fd)
        
        # 修复点1: 使用自适应池化对空间维度 (H, W) 进行压缩
        # 自适应池化直接输出 (B, C, 1, 1)，等同于对 dim2 和 dim3 做池化
        max_fd = F.adaptive_max_pool2d(Fd, 1)
        avg_fd = F.adaptive_avg_pool2d(Fd, 1)
        
        Wc = torch.sigmoid(max_fd + avg_fd)
        channel_out = Wc * Fin
        
        # --- 输出融合 ---
        Fout = self.final_conv_spatial(spatial_out) + self.final_conv_channel(channel_out)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        return Fout

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 16, 64, 64).to(device)
    model = SpatialChannelFeatureModulator(16).to(device)

    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")