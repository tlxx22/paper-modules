import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class NeighborhoodAttention(nn.Module):
    def __init__(self, dim, kernel_size=7, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        assert kernel_size % 2 == 1
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, kernel_size * kernel_size))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = F.unfold(k, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        v = F.unfold(v, kernel_size=self.kernel_size, padding=self.kernel_size // 2)

        k = k.view(b, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size, h * w)
        v = v.view(b, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size, h * w)

        attn = (q.unsqueeze(3) * k).sum(dim=2) * self.scale
        attn = attn + self.relative_position_bias.view(1, self.num_heads, self.kernel_size * self.kernel_size, 1)
        attn = attn.softmax(dim=2)

        out = (attn.unsqueeze(2) * v).sum(dim=3)
        out = out.view(b, c, h, w)
        out = self.proj(out)
        return out


class NAL(nn.Module):
    def __init__(self, dim, kernel_size=7, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = NeighborhoodAttention(dim, kernel_size=kernel_size, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        x = self.alpha * x + self.attn(self.norm1(x))
        x = self.beta * x + self.mlp(self.norm2(x))
        return x


class PAB(nn.Module):
    def __init__(self, dim, neighborhood_sizes=(3, 5, 7), num_heads=8, num_nal=6, mlp_ratio=2.0, gate_hidden_ratio=0.25, temperature=1.0):                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        super().__init__()
        self.dim = dim
        self.neighborhood_sizes = neighborhood_sizes
        self.num_groups = len(neighborhood_sizes)
        assert dim % self.num_groups == 0
        self.group_dim = dim // self.num_groups
        assert self.group_dim % num_heads == 0

        self.pre_convs = nn.ModuleList([
            nn.Conv2d(self.group_dim, self.group_dim, kernel_size=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            for _ in range(self.num_groups)
        ])

        self.branches = nn.ModuleList([
            nn.Sequential(*[
                NAL(
                    dim=self.group_dim,
                    kernel_size=ks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio
                )
                for _ in range(num_nal)
            ])
            for ks in neighborhood_sizes
        ])

        gate_hidden_dim = max(self.num_groups, int(dim * gate_hidden_ratio))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.gate = nn.Sequential(
            nn.Conv2d(dim, gate_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(gate_hidden_dim, self.num_groups, kernel_size=1)
        )

        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.proj = nn.Conv2d(self.group_dim, dim, kernel_size=1)

    def forward(self, x):
        groups = torch.chunk(x, self.num_groups, dim=1)
        outputs = []

        for i in range(self.num_groups):
            feat = self.pre_convs[i](groups[i])
            feat = self.branches[i](feat)
            outputs.append(feat)

        y_cat = torch.cat(outputs, dim=1)
        weight = self.gate(y_cat)
        weight = torch.softmax(weight / self.temperature.clamp_min(1e-6), dim=1)

        y_stack = torch.stack(outputs, dim=1)
        y = (weight.unsqueeze(2) * y_stack).sum(dim=1)
        y = self.proj(y)
        return y


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 96, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = PAB(dim=96, neighborhood_sizes=(3, 5, 7), num_heads=8, num_nal=2).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")