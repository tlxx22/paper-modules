import torch
import torch.nn as nn
import torch.fft
import math

class TFFA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TFFA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. 小波分支：DoG和墨西哥帽小波
        self.dog_conv = nn.Conv2d(in_channels, out_channels//2, kernel_size=kernel_size, padding=kernel_size//2)
        self.mexican_conv = nn.Conv2d(in_channels, out_channels//2, kernel_size=kernel_size, padding=kernel_size//2)
        self.wavelet_norm = nn.BatchNorm2d(out_channels)
        
        # 2. 傅里叶分支：频域特征提取
        self.fourier_conv = nn.Conv2d(in_channels*2, out_channels, kernel_size=1)  # 实部+虚部
        self.fourier_norm = nn.BatchNorm2d(out_channels)
        
        # 3. 空间分支：逐点卷积
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.spatial_norm = nn.BatchNorm2d(out_channels)
        
        # 注意力融合门控
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # -------------------------- 小波分支 --------------------------
        # DoG卷积（高斯差分近似）
        dog_out = self.dog_conv(x)
        # 墨西哥帽卷积（二阶导数近似）
        mexican_out = self.mexican_conv(x)
        wavelet_out = torch.cat([dog_out, mexican_out], dim=1)  # 拼接两个小波特征
        wavelet_out = self.wavelet_norm(wavelet_out)
        wavelet_out = self.act(wavelet_out)
        
        # -------------------------- 傅里叶分支 --------------------------
        # 傅里叶变换（实部+虚部）
        fft = torch.fft.fft2(x)
        fft_real = fft.real
        fft_imag = fft.imag
        fourier_feat = torch.cat([fft_real, fft_imag], dim=1)  # B×(2C)×H×W
        fourier_out = self.fourier_conv(fourier_feat)
        fourier_out = self.fourier_norm(fourier_out)
        fourier_out = self.act(fourier_out)
        
        # -------------------------- 空间分支 --------------------------
        spatial_out = self.spatial_conv(x)
        spatial_out = self.spatial_norm(spatial_out)
        spatial_out = self.act(spatial_out)
        
        # -------------------------- 注意力融合 --------------------------
        # 拼接三分支特征
        concat_feat = torch.cat([wavelet_out, fourier_out, spatial_out], dim=1)  # B×(3C)×H×W
        attention_weights = self.attention(concat_feat)  # B×C×H×W
        # 加权融合
        fused = wavelet_out * attention_weights + fourier_out * attention_weights + spatial_out * attention_weights
        fused = self.final_conv(fused)
        
        return fused
    



# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建一个随机输入特征图 
    input_tensor = torch.randn(2, 32, 256, 256).to(device)    # (batch_size, channels, height, width)
    tffa = TFFA(32, 32).to(device)
    print(tffa)
    output_tensor = tffa(input_tensor)
    
    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")