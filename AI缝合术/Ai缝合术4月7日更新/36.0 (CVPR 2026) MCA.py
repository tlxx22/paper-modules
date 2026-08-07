import torch
import torch.nn as nn
from einops import rearrange


class MCA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MCA, self).__init__()
        self.num_heads = num_heads
        self.temperature_a = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_v = nn.Parameter(torch.ones(num_heads, 1, 1))

        # q: what we want to attend to (spatial information)
        self.q_proj = nn.Conv2d(
            dim, dim, kernel_size=3,
            padding=1, stride=2, padding_mode='reflect', 
            groups=dim, bias=bias
        )
        # k: what we use to calculate attention (channel information)
        self.k_proj = nn.Conv2d(
            dim, dim, kernel_size=3,
            padding=1, stride=2, padding_mode='reflect', 
            bias=bias
        )
        # v: what we use to calculate the output (channel information)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # a: anchor information (reduced spatial information and channel information)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        self.a_proj = nn.Sequential(
            nn.Conv2d(
                dim, dim, kernel_size=3,
                padding=1, stride=2, padding_mode='reflect', 
                groups=dim, bias=bias
            ),
            nn.Conv2d(dim, dim//2, kernel_size=1)
        )
        # output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x) * x
        a = self.a_proj(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)                                                                                                                                                                                                                          # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!                                                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        a = torch.nn.functional.normalize(a, dim=-1)

        # Q - C×(H/s×W/s), K - C×(H/s×W/s), V - C×(H×W), A - C/r×(H/s×W/s) 

        # transposed self-attention with attention map of shape (C×C)
        attn_a = (q @ a.transpose(-2, -1)) * self.temperature_a
        attn_a = attn_a.softmax(dim=-1)

        attn_k = (a @ k.transpose(-2, -1)) * self.temperature_v
        attn_k = attn_k.softmax(dim=-1)
        
        out_v = (attn_k @ v)

        out = (attn_a @ out_v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!

        out = self.project_out(out)
        return out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 64, 128, 128).to(device)

    model = MCA(dim=64, num_heads=8, bias=False).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")