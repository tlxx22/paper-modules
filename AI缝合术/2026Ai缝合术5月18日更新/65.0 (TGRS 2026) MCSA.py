import torch
import torch.nn as nn
import torch.nn.functional as F

class MCSA(nn.Module):
    """多尺度通道空间注意力(Multiscale Channel Spatial Attention)
   UNet-NNI: A Collaborative Inpainting Method Based on Deep-Sea Multibeam Backscatter Intensity Images (TGRS 2026)
    
    核心设计:
    1. 多尺度通道注意力: 使用1×1、2×2、4×4三种尺度的平均池化提取全局上下文                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    2. 多尺度空间注意力: 使用3×3、5×5、7×7三种尺度的卷积提取多尺度空间特征                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    3. 快捷连接(SC): 1×1卷积+BN调整通道数，保证特征维度匹配
    """
    def __init__(self, in_channels, out_channels=None, reduction=16):
        """
        参数:
            in_channels: 输入特征图通道数
            out_channels: 输出特征图通道数，默认为in_channels
            reduction: 通道注意力的压缩比，论文中使用16
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.reduction = reduction
        
        # ====================== 多尺度通道注意力部分 ======================
        # 多尺度平均池化层 (核大小=步长，CA1、CA4、CA16)
        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)  # 1×1池化，步长1                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # 2×2池化，步长2                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.pool16 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)  # 4×4池化，步长4                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        # 共享多层感知机(MLP)，使用1×1卷积实现
        self.mlp = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        
        # ====================== 多尺度空间注意力部分 ======================
        # 多尺度卷积层 (SA31、SA52、SA74)
        self.conv3 = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )  # 3×3卷积，步长1
        self.conv5 = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=5, stride=2, padding=2, bias=False
        )  # 5×5卷积，步长2
        self.conv7 = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=7, stride=4, padding=3, bias=False
        )  # 7×7卷积，步长4
        
        # 空间注意力生成层: 将3C通道的拼接特征降为1通道的注意力图
        self.spatial_conv = nn.Conv2d(3 * in_channels, 1, kernel_size=1, bias=False)
        
        # ====================== 快捷连接(SC)部分 ======================
        # 1×1卷积+BN调整通道数，匹配下一层跳跃连接的维度
        self.sc = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        
    def forward(self, x):
        """
        参数:
            x: 输入特征图，形状为[B, C_in, H, W]
        返回:
            out: 输出特征图，形状为[B, C_out, H, W]
        """
        B, C, H, W = x.size()
        
        # ---------------------- 1. 多尺度通道注意力计算 ----------------------
        # 步骤1: 多尺度平均池化
        pool1 = self.pool1(x)    # [B, C, H, W]
        pool4 = self.pool4(x)    # [B, C, H//2, W//2]
        pool16 = self.pool16(x)  # [B, C, H//4, W//4]
        
        # 步骤2: 对每个尺度的池化结果进行全局平均池化
        gap1 = F.adaptive_avg_pool2d(pool1, 1)    # [B, C, 1, 1]
        gap4 = F.adaptive_avg_pool2d(pool4, 1)    # [B, C, 1, 1]
        gap16 = F.adaptive_avg_pool2d(pool16, 1)  # [B, C, 1, 1]
        
        # 步骤3: 拼接三个尺度的全局特征
        ca = torch.cat([gap1, gap4, gap16], dim=1)  # [B, 3C, 1, 1]
        
        # 步骤4: 共享MLP生成通道注意力权重
        a_c = self.mlp(ca)
        a_c = torch.sigmoid(a_c)  # [B, C, 1, 1]
        
        # 步骤5: 通道注意力加权
        z_prime = x * a_c  # 逐通道相乘，[B, C, H, W]
        
        # ---------------------- 2. 多尺度空间注意力计算 ----------------------
        # 步骤1: 多尺度卷积提取不同感受野的空间特征
        conv3 = self.conv3(z_prime)  # [B, C, H, W]
        conv5 = self.conv5(z_prime)  # [B, C, H//2, W//2]
        conv7 = self.conv7(z_prime)  # [B, C, H//4, W//4]
        
        # 步骤2: 将低分辨率特征上采样到原始尺寸
        up5 = F.interpolate(conv5, size=(H, W), mode='bilinear', align_corners=False)  # [B, C, H, W]                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        up7 = F.interpolate(conv7, size=(H, W), mode='bilinear', align_corners=False)  # [B, C, H, W]                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        # 步骤3: 拼接三个尺度的空间特征
        sa = torch.cat([conv3, up5, up7], dim=1)  # [B, 3C, H, W]
        
        # 步骤4: 生成空间注意力权重
        a_s = self.spatial_conv(sa)
        a_s = torch.sigmoid(a_s)  # [B, 1, H, W]
        
        # 步骤5: 空间注意力加权
        f_m = z_prime * a_s  # 逐像素相乘，[B, C, H, W]
        
        # ---------------------- 3. 快捷连接调整通道数 ----------------------
        out = self.sc(f_m)  # [B, C_out, H, W]
        
        return out


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = MCSA(in_channels=64, out_channels=64, reduction=16).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")