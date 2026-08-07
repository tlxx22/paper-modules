import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialChannelAttention(nn.Module):
    """
    部分通道注意力
    Args:
        channel (int): 输入和输出的通道数。 
        ratio (float): 分支1的比例，默认为0.5。
        ...
    Returns:
        Tensor: 经过部分通道注意力处理后的张量。
    """
    def __init__(self, channel, ratio=0.5):
        super().__init__()

        self.ratio = ratio
        channel1 = int(channel*ratio)
        channel2 = channel - channel1
        self.branch1 = nn.Conv2d(channel1, channel1, kernel_size=3, stride=1, padding=1)
        self.cfc1 = nn.Conv2d(channel2, channel2, kernel_size=(1,2), bias=False)
        self.bn = nn.BatchNorm2d(channel2)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        x1, x2 = x.split(int(x.shape[1]*0.5), dim=1)
        x1 = self.branch1(x1)
        b, c, h, w = x2.shape
        # style pooling
        mean = x2.reshape(b, c, -1).mean(-1).view(b,c,1,1)
        std = x2.reshape(b, c, -1).std(-1).view(b,c,1,1)
        u = torch.cat([mean, std], dim=-1)
        z = self.cfc1(u)
        z = self.bn(z)
        g = self.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        x2 = x2 * g.expand_as(x2)
        out = torch.cat([x1, x2], dim=1)
        return out


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建一个随机输入特征图 
    input_tensor = torch.randn(2, 32, 256, 256).to(device)    # (batch_size, channels, height, width)
    
    pca = PartialChannelAttention(32).to(device)
    print(pca)
    output_tensor = pca(input_tensor)
    
    
    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
