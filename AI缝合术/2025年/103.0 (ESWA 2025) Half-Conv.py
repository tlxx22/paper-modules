import torch
import torch.nn as nn
from torch import Tensor

class HalfConv(nn.Module):
    def __init__(self, dim, n_div=2):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)                                                                              # 微信公众号:AI缝合术

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)                                                                                               # 微信公众号:AI缝合术
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 256, 256)

    # 初始化
    halfconv = HalfConv(dim=32)

    # 前向传播测试
    output = halfconv(x)

    # 输出结果形状
    print(halfconv)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]    