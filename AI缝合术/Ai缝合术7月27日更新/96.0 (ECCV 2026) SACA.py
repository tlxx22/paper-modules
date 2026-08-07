import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SCP(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        super().__init__()
        self.project = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.spatial = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            bias=bias,
        )

    def forward(self, x):
        return self.spatial(self.project(x))


class SpatialAwareChannelAttention(nn.Module):                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    def __init__(self, dim, num_heads=4, bias=True):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = LayerNorm2d(dim)

        self.q_proj = SCP(dim, dim, bias=bias)
        self.k_proj = SCP(dim, dim, bias=bias)
        self.v_proj = SCP(dim, dim, bias=bias)

        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )

        self.out_proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        identity = x
        x = self.norm(x)

        b, c, h, w = x.shape
        n = h * w

        q = self.q_proj(x).reshape(
            b, self.num_heads, self.head_dim, n
        )
        k = self.k_proj(x).reshape(
            b, self.num_heads, self.head_dim, n
        )
        v = self.v_proj(x).reshape(
            b, self.num_heads, self.head_dim, n
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attention = torch.matmul(
            q,
            k.transpose(-2, -1),
        )
        attention = attention * self.temperature
        attention = attention.softmax(dim=-1)

        out = torch.matmul(attention, v)
        out = out.reshape(b, c, h, w)
        out = self.out_proj(out)

        return identity + out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = SpatialAwareChannelAttention(64, num_heads=8).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")