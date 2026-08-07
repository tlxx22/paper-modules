import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolConvDownSampling(nn.Module):
    def __init__(self, ninput, noutput):                                                                                             # 微信公众号:AI缝合术
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)                                                                                             # 微信公众号:AI缝合术
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)                                                                                              # 微信公众号:AI缝合术
        output = self.bn(output)
        return F.relu(output)
    
if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 256, 256)

    # 初始化 PoolConvDownSampling 模块（输入通道和输出通道相同，满足残差连接）                                                                                             # 微信公众号:AI缝合术
    pcdm = PoolConvDownSampling(32,64)

    # 前向传播测试
    output = pcdm(x)

    # 输出结果形状
    print(pcdm)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)       # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]
