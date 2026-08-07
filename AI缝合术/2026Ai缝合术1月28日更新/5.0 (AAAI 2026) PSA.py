import torch
import torch.nn as nn
import torch.nn.functional as F

class partial_spatial_attn(nn.Module):
    def __init__(self, dim, n_head, partial=0.5):
        super().__init__()
        self.dim = dim
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv
        self.nhead = n_head
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 1, bias=False)                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.conv_attn = nn.Conv2d(self.dim_untouched, n_head, 1, bias=False)                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.norm = nn.BatchNorm2d(self.dim_untouched)
        self.norm2 = nn.BatchNorm2d(self.dim_conv)
        #self.act2 = nn.GELU()
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x1, x2 = torch.split(x, [self.dim_untouched, self.dim_conv], 1)                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        weight =self.act(self.conv_attn(x1))
        x1 = x1 * weight
        x1 = self.norm(x1)
        #x2 = self.act2(x2)
        x2 = self.norm2(x2)
        x2 = self.conv(x2)
        x = torch.cat((x1, x2), 1)
        return x

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建一个随机输入特征图 
    input_tensor = torch.randn(1, 32, 256, 256).to(device)    # (batch_size, channels, height, width)

    psa = partial_spatial_attn(32, 1).to(device)
    print(psa)
    output_tensor = psa(input_tensor)     
    
    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
