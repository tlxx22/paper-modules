import torch
from torch import nn

class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__() 
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2
        return out
    
class LearnableBiasnn(nn.Module): 
    def __init__(self, out_chn): # 微信公众号:AI缝合术
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn,1,1]), requires_grad=True)

    def forward(self, x): # 微信公众号:AI缝合术
        out = x + self.bias.expand_as(x)
        return out

# Multi-scale grouped dilated convolution (MSGDC)    
class MSGDC(nn.Module):
    def __init__(self, in_chn, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same'):
        super(MSGDC, self).__init__()
        self.move = LearnableBiasnn(in_chn)
        
        # AI 缝合术注释：源代码使用了自定义的量化卷积QuantizeConv2d来实现二值化, 降低模型的计算需求和存储占用, 此处我们采用常规2D卷积, 用于普通视觉任务效果更好!
        self.cov1 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation1, bias=True)
        self.cov2 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation2, bias=True)
        self.cov3 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation3, bias=True)
        
        # 直接使用 nn.LayerNorm 作为归一化层，去掉 config 配置
        self.norm = nn.LayerNorm(in_chn, eps=1e-6)  # 层归一化（epsilon 可自定义）
        
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn) 
        self.act3 = RPReLU(in_chn)

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.move(x)
        x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1,2)
        x1 = self.act1(x1) # 微信公众号:AI缝合术
        x2 = self.cov2(x).permute(0, 2, 3, 1).flatten(1,2)
        x2 = self.act2(x2) # 微信公众号:AI缝合术
        x3 = self.cov3(x).permute(0, 2, 3, 1).flatten(1,2)
        x3 = self.act3(x3) # 微信公众号:AI缝合术
        x = self.norm(x1+x2+x3)
        return x.permute(0, 2, 1).view(-1, C, H, W).contiguous()

if __name__ == "__main__":   
    # 设置输入张量大小
    batch_size = 1
    input_channels = 32
    height, width = 256, 256  # 输入图像的尺寸为 256*256

    # 创建输入张量
    x = torch.randn(batch_size, input_channels, height, width)

    # 初始化 MSGDC 模块
    msgdc = MSGDC(in_chn=input_channels, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same')
    print(msgdc)
    print("微信公众号:AI缝合术!")

    # 前向传递，获取输出
    output = msgdc(x)

    # 输出形状检查
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
