import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels, kernel_size=1),                                                                                                  # 微信公众号:AI缝合术
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size**2, kernel_size=1)  # [B, k*k, 1, 1]                                                                                               # 微信公众号:AI缝合术
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 每个样本生成一个动态卷积核 [B, k*k, 1, 1] → [B, 1, k, k]                                                                                                      # 微信公众号:AI缝合术
        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)                                                                                        # 微信公众号:AI缝合术                                                  # 微信公众号:AI缝合术
        # 2. 对每个样本取通道平均 [B, 1, H, W]
        x_mean = x.mean(dim=1, keepdim=True)
        # 3. reshape 成 grouped convolution 所需格式
        x_mean = x_mean.view(1, B, H, W)  # → [1, B, H, W]
        kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)  # [B, 1, k, k]                                                                                # 微信公众号:AI缝合术
        # 4. 执行 grouped convolution，每个 kernel 只作用于对应的样本
        att = F.conv2d(
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )
        # 5. reshape 回原格式 + sigmoid
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        # 6. 应用注意力图
        return x * att

if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 64, 64)

    # 初始化
    dsa = DynamicSpatialAttention(in_channels=32)

    # 前向传播测试
    output = dsa(x)

    # 输出结果形状
    print(dsa)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]    