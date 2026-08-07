import torch
from torch import nn
import torch.nn.functional as F

class SPR_SA(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)


    def forward(self, x):
        x = self.conv_0(x)
        x1= F.adaptive_avg_pool2d(x, (1, 1))
        print("x1:",x1.size())
        x1 = F.softmax(x1, dim=1)
        x=x1*x
        x = self.act(x)
        x = self.conv_1(x)
        return x
    
if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 256, 256)

    # 初始化 SPR_SA
    sprsa = SPR_SA(32)  

    # 前向传播测试
    output = sprsa(x)

    # 输出结果形状
    print(sprsa)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)       # [B, C, H, W]                                                                                             # 微信公众号:AI缝合术
    print("输出张量形状:", output.shape)  # [B, C, H, W]         