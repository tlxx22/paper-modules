import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)                                                          # 微信公众号:AI缝合术
        q, k, v = qkv[0], qkv[1], qkv[2]

        key = F.softmax(k, dim=-1)
        query = F.softmax(q, dim=-2)
        context = key.transpose(-2, -1) @ v
        x = (query @ context).reshape(b, h * w, c)

        x = self.proj(x)

        x = x.permute(0, 2, 1).view(b, c, h, w)

        return x
    
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2)                                                          # 微信公众号:AI缝合术
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.relu(x)
        return x

class MSLA(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dw_conv_3x3 = DepthwiseConv(dim // 4, kernel_size=3)
        self.dw_conv_5x5 = DepthwiseConv(dim // 4, kernel_size=5)
        self.dw_conv_7x7 = DepthwiseConv(dim // 4, kernel_size=7)
        self.dw_conv_9x9 = DepthwiseConv(dim // 4, kernel_size=9)

        self.linear_attention = LinearAttention(dim = dim // 4, num_heads = num_heads)                                                          # 微信公众号:AI缝合术

        self.final_conv = nn.Conv2d(dim, dim, 1)

        self.scale_weights = nn.Parameter(torch.ones(4), requires_grad=True)

    def forward(self, input_):
        b, c, h, w = input_.shape
        input_reshaped = input_.view(b, c, h, w)

        split_size = c // 4
        x_3x3 = input_reshaped[:, :split_size, :, :]
        x_5x5 = input_reshaped[:, split_size:2 * split_size, :, :]
        x_7x7 = input_reshaped[:, 2 * split_size:3 * split_size:, :, :]
        x_9x9 = input_reshaped[:, 3 * split_size:, :, :]

        x_3x3 = self.dw_conv_3x3(x_3x3)
        x_5x5 = self.dw_conv_5x5(x_5x5)
        x_7x7 = self.dw_conv_7x7(x_7x7)
        x_9x9 = self.dw_conv_9x9(x_9x9)


        att_3x3 = self.linear_attention(x_3x3)
        att_5x5 = self.linear_attention(x_5x5)
        att_7x7 = self.linear_attention(x_7x7)
        att_9x9 = self.linear_attention(x_9x9)


        processed_input = torch.cat([
            att_3x3 * self.scale_weights[0],
            att_5x5 * self.scale_weights[1],
            att_7x7 * self.scale_weights[2],
            att_9x9 * self.scale_weights[3]
        ], dim=1)

        final_output = self.final_conv(processed_input)

        return final_output

if __name__ == "__main__":

    batch_size = 1
    height, width = 128, 128    # 输入图像大小
    channels = 32               # 输入通道数
    num_heads = 8               # 注意力头数

    # 创建输入张量：形状为 (B, C, H, W)，四维张量
    x = torch.randn(batch_size, channels, height, width)

    # 初始化 MSLA 模块
    msla = MSLA(dim=channels, num_heads=num_heads)

    # 前向传播测试
    output = msla(x)

    # 输出结果
    print(msla)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, n, dim]

