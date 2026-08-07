import math
import torch
import torch.nn as nn

class PartialAttention(nn.Module):
    def __init__(self, dim, ratio=0.5, num_heads=8, qkv_bias=False):                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        super().__init__()
        self.d_prime = int(math.ceil(dim * ratio))
        self.num_heads = num_heads
        self.head_dim = self.d_prime // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(self.d_prime)
        self.qkv = nn.Linear(self.d_prime, self.d_prime * 3, bias=qkv_bias)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        self.proj = nn.Linear(self.d_prime, self.d_prime)

    def forward(self, x):
        B, N, C = x.shape

        x1 = x[:, :, :self.d_prime]
        x2 = x[:, :, self.d_prime:]

        norm_x1 = self.norm(x1)
        qkv = self.qkv(norm_x1).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x1_attn = (attn @ v).transpose(1, 2).reshape(B, N, self.d_prime)
        x1_out = x1 + self.proj(x1_attn)

        out = torch.cat([x1_out, x2], dim=-1)
        return out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 256, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = PartialAttention(dim=128, ratio=0.5, num_heads=8).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")