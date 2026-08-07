import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarWaveletTransform(nn.Module):
    # 小波变换模块，用于特征分解为四个子带（LL, LH, HL, HH）
    def __init__(self):
        super().__init__()
        # 定义Haar小波分解核（固定权重，不参与训练）
        ll_kernel = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 4.0  # 低通
        lh_kernel = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 4.0  # 水平高通
        hl_kernel = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 4.0  # 垂直高通                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        hh_kernel = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 4.0  # 对角高通
        
        # 堆叠为 (out_channels, in_channels, kernel_size, kernel_size)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        kernel = torch.stack([ll_kernel, lh_kernel, hl_kernel, hh_kernel], dim=0)
        kernel = kernel.unsqueeze(1)  # 形状: (4, 1, 2, 2)
        
        # 注册为buffer，使其可以在forward中访问但不参与梯度计算
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)  # 合并批次和通道维度以独立处理每个通道
        out = F.conv2d(x, self.kernel, stride=2, padding=0)  # 步幅2实现下采样
        out = out.view(B, 4 * C, H // 2, W // 2)  # 恢复批次维度，通道数变为4*C                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        return out


class InverseHaarWaveletTransform(nn.Module):
    # 逆Haar小波变换模块，用于将四个子带重构为原始尺寸特征图
    def __init__(self):
        super().__init__()
        # 定义逆Haar小波核（上采样用）
        ll_kernel = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh_kernel = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0
        hl_kernel = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0
        hh_kernel = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        
        kernel = torch.stack([ll_kernel, lh_kernel, hl_kernel, hh_kernel], dim=0)
        kernel = kernel.unsqueeze(1)  # 形状: (4, 1, 2, 2)
        
        # 注册为buffer
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        B, C4, H, W = x.shape
        C = C4 // 4  # 原始通道数
        x = x.view(B * C, 4, H, W)  # 按通道拆分四个子带
        
        # --- 修改位置 ---
        # groups=1: 将输入的4个通道（子带）与4个卷积核分别卷积后求和，输出为1个通道
        # 之前 groups=4 会导致输出保持4个通道，从而导致 view 形状不匹配                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        out = F.conv_transpose2d(x, self.kernel, stride=2, padding=0, groups=1)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        # ----------------
        
        out = out.view(B, C, 2 * H, 2 * W)  # 恢复原始尺寸
        return out


class WTConv(nn.Module):
    # 小波变换卷积模块（WTConv）：结合小波变换与卷积操作
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        self.wt = HaarWaveletTransform()
        self.conv = nn.Conv2d(
            in_channels * 4,  # 输入为4个子带拼接
            in_channels * 4,  # 输出保持4个子带结构
            kernel_size=kernel_size,
            padding=padding,
            groups=4  # 每个子带独立卷积
        )
        self.iwt = InverseHaarWaveletTransform()

    def forward(self, x):
        x_wt = self.wt(x)  # 小波分解：(B, 4C, H//2, W//2)
        x_conv = self.conv(x_wt)  # 子带卷积：(B, 4C, H//2, W//2)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        x_iwt = self.iwt(x_conv)  # 逆变换重构：(B, C, H, W)
        return x_iwt


class GMWTConvs(nn.Module):
   # 全局多尺度小波变换卷积模块（GMWTConvs）
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 四个卷积层（Conv(4)）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1),                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels//4, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1),                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 小波卷积层
        self.wtconv = WTConv(out_channels)

    def forward(self, x):
        x = self.conv_layers(x)  # 浅层特征提取
        x = self.wtconv(x)  # 小波卷积增强
        return x

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建一个随机输入特征图 
    input_tensor = torch.randn(1, 32, 256, 256).to(device)    # (batch_size, channels, height, width)

    gmwt = GMWTConvs(in_channels=32, out_channels=64).to(device)
    print(gmwt)

    # 前向传播
    output_tensor = gmwt(input_tensor)

    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")