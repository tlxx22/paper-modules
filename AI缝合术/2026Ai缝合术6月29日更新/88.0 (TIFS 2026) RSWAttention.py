import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def check_image_size(x, padder_size, mode='reflect'):
    _, _, h, w = x.size()
    if isinstance(padder_size, int):
        padder_size_h = padder_size
        padder_size_w = padder_size
    else:
        padder_size_h, padder_size_w = padder_size
    mod_pad_h = (padder_size_h - h % padder_size_h) % padder_size_h
    mod_pad_w = (padder_size_w - w % padder_size_w) % padder_size_w
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode=mode)
    return x



class RSWAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, padding_mode='zeros'):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size

        self.mlp = nn.Sequential(
            nn.Linear(window_size**2, window_size**2, bias=True),
            nn.GELU(),
        )

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                        h=self.window_size, w=self.window_size)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = out * self.mlp(v)
        out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                        w1=Wx//self.window_size, h=self.window_size, w=self.window_size)

        return out[:, :, :H, :W]

    def forward(self, x):
        qkv = self.qkv_dwconv(self.qkv(x))
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = RSWAttention(dim=64, num_heads=8, bias=True, window_size=8).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")