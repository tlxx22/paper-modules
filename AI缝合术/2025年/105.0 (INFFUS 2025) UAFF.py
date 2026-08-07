import torch
import torch.nn as nn

class UAFF(nn.Module):

    def __init__(self, channels, r=4):
        super(UAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),                                                                     # 微信公众号:AI缝合术
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),                                                                     # 微信公众号:AI缝合术
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),                                                                     # 微信公众号:AI缝合术
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),                                                                     # 微信公众号:AI缝合术
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x0, outx, x1):
        xa = x0 + outx + x1 # x = I_out4, outx = x5,  x1 = x4                                                                                                 # 微信公众号:AI缝合术
        #print("Inside AFF xa: ", xa.size())
        xl = self.local_att(xa)
        #print("Inside AFF xl: ", xl.size())
        xg = self.global_att(xa)
        #print("Inside AFF xg: ", xg.size())
        xlg = xl + xg
        #print("Inside AFF xlg: ", xlg.size())
        wei = self.sigmoid(xlg)

        xo = 2 * x0 * wei + 2 * outx * (1 - wei)                                                                                                             # 微信公众号:AI缝合术
        #print("Inside AFF xo: ", wei.size())
        return xo
    
if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x0 = torch.randn(2, 32, 64, 64)
    outx = torch.randn(2, 32, 64, 64)
    x1 = torch.randn(2, 32, 64, 64)

    # 初始化
    aff = UAFF(channels=32)

    # 前向传播测试
    output = aff(x0, outx, x1)

    # 输出结果形状
    print(aff)
    print("\n微信公众号:AI缝合术\n")
    print("输入x0形状  :", x0.shape)      # [B, C, H, W]
    print("输入x1形状  :", x0.shape)      # [B, C, H, W]
    print("输入outx形状:", x0.shape)      # [B, C, H, W]
    print("输出outx形状:", output.shape)  # [B, C, H, W]
