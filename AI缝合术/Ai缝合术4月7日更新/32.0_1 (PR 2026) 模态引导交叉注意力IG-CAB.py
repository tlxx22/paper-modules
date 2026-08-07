import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention_S(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention_S, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,
                                  bias=bias)

        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
                                   bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        fea_0 = x[0]  # 2024/11/1 added by wwc
        fea_1 = x[1]  # 2024/11/1 added by wwc
        b, c, h, w = fea_0.shape

        qk = self.qk_dwconv(self.qk(fea_0))
        q, k = qk.chunk(2, dim=1)

        v = self.v_dwconv(self.v(fea_1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!

        out = self.project_out(out)

        return out
    
# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor1 = torch.randn(1, 64, 32, 32).to(device)
    input_tensor2 = torch.randn(1, 64, 32, 32).to(device)

    model = CrossAttention_S(dim=64, num_heads=8, bias=False).to(device)
    print(model)
    output_tensor = model([input_tensor1, input_tensor2])

    # 打印维度验证
    print("input_tensor1_shape :", input_tensor1.shape)   
    print("input_tensor2_shape :", input_tensor2.shape)
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
