import torch
import torch.nn as nn
import torch.nn.functional as F

class MBRConv1(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(
            #nn.Conv2d(in_channels, out_channels * rep_scale, 1),
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)
        self.conv_out.weight.requires_grad = False

        self.weight1 = nn.Parameter(torch.zeros_like(self.conv_out.weight))
        nn.init.xavier_normal_(self.weight1)

    def forward(self, inp):   
        x1 = self.conv(inp)
        x = torch.cat([x1, self.conv_bn(x1)], 1)
        final_weight = self.conv_out.weight + self.weight1
        out = F.conv2d(x, final_weight, self.conv_out.bias)
        return out 

class HDPA(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(HDPA, self).__init__()
        self.channels = channels
       
        self.globalatt = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.localatt = nn.Sequential( 
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.globalatt(x)
        max_out, _ = torch.max(x1 * x, dim=1, keepdim=True)   
        x2 = self.localatt(max_out)
        x3 = torch.mul(x1, x2) * x
        return x3
   

if __name__ == "__main__":

    batch_size = 2
    height, width = 256, 256    # 输入图像大小
    channels = 32               # 输入通道数
    rep_scale = 4               # 重复缩放因子

    # 创建输入张量：形状为 (B, C, H, W)
    x = torch.randn(batch_size, channels, height, width)

    # 初始化 HDPA 模块
    hdpa = HDPA(channels=channels, rep_scale=rep_scale)

    # 前向传播测试
    output = hdpa(x)

    # 输出结果形状
    print(hdpa)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]
