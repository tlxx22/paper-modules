import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    """
    功能：并行提取大/小尺度特征，生成尺度权重自适应增强多尺度屋顶特征
    输入：x (B, C, H, W)
    输出：out (B, C, H, W)
    """
    def __init__(self, in_channels):
        super().__init__()
        # -------------------------- 1. 大尺度分支 L-branch --------------------------
        # 论文：两层 3×3 标准卷积，提取大尺度屋顶几何特征
        self.l_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # -------------------------- 2. 小尺度分支 S-branch --------------------------
        # 论文：三层 3×3 标准卷积 + 两层膨胀卷积（dilation=2，kernel=5×5，padding=4）                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        # 作用：扩大感受野同时保留细粒度细节，专注小尺度屋顶检测
        self.s_branch = nn.Sequential(
            # 3层标准3×3卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 2层膨胀卷积（dilation=2，等效5×5感受野，padding=4保证尺寸不变）
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=4, dilation=2),                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=4, dilation=2),                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 拼接双分支特征 → 1×1卷积压缩通道 → 生成尺度感知权重
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # 归一化为0~1权重
        )

    def forward(self, x):
        # 1. 并行提取大/小尺度特征
        feat_l = self.l_branch(x)  # 大尺度特征
        feat_s = self.s_branch(x)  # 小尺度特征

        # 2. 拼接双分支特征
        feat_cat = torch.cat([feat_l, feat_s], dim=1)

        # 3. 生成尺度自适应权重
        scale_weight = self.fusion(feat_cat)

        # 4. 尺度加权 + 残差连接
        out = x * scale_weight + x

        return out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 16, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = SAM(in_channels=16).to(device)
    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")