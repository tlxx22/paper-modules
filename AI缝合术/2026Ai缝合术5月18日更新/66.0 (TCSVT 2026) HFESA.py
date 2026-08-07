import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HFESA(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size):
        super(HFESA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dwconv_w = nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 11//2), groups=dim)
        self.dwconv_h = nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(11//2, 0), groups=dim)
        self.dwconv_hw = nn.Conv2d(dim*2, dim*2, 3, padding=3//2, groups=dim*2)
        self.conv1 = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=1//2, bias=bias)


        self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=1//2, bias=bias)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=1//2, bias=bias)

        self.conv4 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                   nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # High-Resolution Space
        high = self.dwconv_hw(torch.cat([self.dwconv_w(x), self.dwconv_h(x)], dim=1))                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        high = self.conv1(high)

        # Low-Resolution Space
        x_down = self.sr(x)
        qkv = self.qkv_dwconv(self.qkv(x_down))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        v = self.conv2(self.max(v)) + self.conv3(self.avg(v))
        v = v * self.conv4(x)

        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        low = (attn @ v)

        low = rearrange(low, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        out = self.project_out(torch.cat([high, low], dim=1))
        return out


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = HFESA(dim=64, num_heads=8, bias=False, window_size=8).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")