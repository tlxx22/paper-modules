import torch
import torch.nn as nn
import torch.nn.functional as F

class PolyAttn(nn.Module):
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False):
        super().__init__()
        self.head_dim = head_dim
        self.poly = True
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Conv2d(dim, self.attention_dim * 2, 1, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.attention_dim, dim, 1, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = nn.Parameter(
            torch.tensor([-(head_dim ** -0.5) / ((head_dim ** -0.5) - 1)] * self.num_heads).log().view(1, -1, 1, 1))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.q_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=5,
                                stride=1, padding=2, groups=self.attention_dim, bias=False)
        self.k_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=5,
                                stride=1, padding=2, groups=self.attention_dim, bias=False)
        self.v_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=3,
                                stride=1, padding=1, groups=self.attention_dim, bias=False)
        self.final_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=3,
                                    stride=1, padding=1, groups=self.attention_dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, -1, H, W)
        qk, v = qkv.split(self.attention_dim, 1)
        q = self.q_conv(qk).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        k = self.k_conv(qk).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        v = self.v_conv(v).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        if not self.poly:
            x = F.scaled_dot_product_attention(q, k, v,
                                               dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            attn = ((q @ k.transpose(-2, -1)) * self.scale.sigmoid() + 1) ** 4
            attn = F.normalize(attn, p=1, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, H, W, self.attention_dim).permute(0, 3, 1, 2)
        x = self.final_conv(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = PolyAttn(dim=64, head_dim=32, num_heads=None, qkv_bias=False,
                     attn_drop=0., proj_drop=0., proj_bias=False).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")