import torch
from torch import nn

class LocalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"

        self.conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)                                                                                            # 微信公众号:AI缝合术
        self.GAP = nn.AvgPool2d(feature_map_size)

    def forward(self, x):
        N, C, H, W = x.shape
        att = self.GAP(x).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att =  att.reshape(N, C, 1, 1)
        return (x * att) + x

class GlobalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"

        self.conv_q = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)                                                                                            # 微信公众号:AI缝合术
        self.conv_k = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)                                                                                            # 微信公众号:AI缝合术
        self.GAP = nn.AvgPool2d(feature_map_size)

    def forward(self, x):
        N, C, H, W = x.shape

        query = key = self.GAP(x).reshape(N, 1, C)
        query = self.conv_q(query).sigmoid()
        key = self.conv_q(key).sigmoid().permute(0, 2, 1)
        query_key = torch.bmm(key, query).reshape(N, -1)
        query_key = query_key.softmax(-1).reshape(N, C, C)

        value = x.permute(0, 2, 3, 1).reshape(N, -1, C)
        att = torch.bmm(value, query_key).permute(0, 2, 1)
        att = att.reshape(N, C, H, W)
        return x * att


class GLCA(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        self.global_attention = GlobalChannelAttention(feature_map_size,kernel_size)                                                                                            # 微信公众号:AI缝合术
        self.local_attention = LocalChannelAttention(feature_map_size,kernel_size)                                                                                            # 微信公众号:AI缝合术


    def forward(self, x):

        input_left, input_right = x.chunk(2,dim=1)
        x1 = self.global_attention(input_left)
        x2 = self.local_attention(input_right)
        output = torch.cat((x1,x2),dim=1)

        return output + x
    
if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 64, 64)

    # 初始化 GLCA
    glca = GLCA(64, 3)  # AI缝合术注释: 传入两个参数, 第一个是特征图大小(默认宽高一致), 第二个是卷积核大小                                                                                            # 微信公众号:AI缝合术

    # 前向传播测试
    output = glca(x)

    # 输出结果形状
    print(glca)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)       # [B, C, H, W]                                                                                             # 微信公众号:AI缝合术                                                                                            # 微信公众号:AI缝合术
    print("输出张量形状:", output.shape)  # [B, C, H, W]         