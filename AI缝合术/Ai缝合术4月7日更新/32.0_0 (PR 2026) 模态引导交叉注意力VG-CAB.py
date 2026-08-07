import torch
import torch.nn as nn
from einops import rearrange

class CrossAttention_M(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention_M, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        rgb_fea = x[0]  # 2024/11/1 added by wwc
        ir_fea = x[1]  # 2024/11/1 added by wwc
        b, c, h, w = rgb_fea.shape

        rgb_qkv = self.qkv_dwconv(self.qkv(rgb_fea))
        rgb_q, rgb_k, rgb_v = rgb_qkv.chunk(3, dim=1)

        ir_qkv = self.qkv_dwconv(self.qkv(ir_fea))
        ir_q, ir_k, ir_v = ir_qkv.chunk(3, dim=1)

        rgb_q = rearrange(rgb_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_k = rearrange(rgb_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        rgb_v = rearrange(rgb_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        ir_q = rearrange(ir_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_k = rearrange(ir_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_v = rearrange(ir_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        rgb_q = torch.nn.functional.normalize(rgb_q, dim=-1)
        rgb_k = torch.nn.functional.normalize(rgb_k, dim=-1)

        ir_q = torch.nn.functional.normalize(ir_q, dim=-1)
        ir_k = torch.nn.functional.normalize(ir_k, dim=-1)

        attn_ir = (rgb_q @ ir_k.transpose(-2, -1)) * self.temperature
        attn_rgb = (ir_q @ rgb_k.transpose(-2, -1)) * self.temperature

        attn_ir = attn_ir.softmax(dim=-1)
        attn_rgb = attn_rgb.softmax(dim=-1)

        out_ir = (attn_ir @ ir_v)
        out_rgb = (attn_rgb @ rgb_v)

        out_ir = rearrange(out_ir, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        out_rgb = rearrange(out_rgb, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_ir = self.project_out(out_ir)
        out_rgb = self.project_out(out_rgb)

        return out_rgb + out_ir
    
# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor1 = torch.randn(1, 64, 32, 32).to(device)
    input_tensor2 = torch.randn(1, 64, 32, 32).to(device)

    model = CrossAttention_M(dim=64, num_heads=8, bias=False).to(device)
    print(model)
    output_tensor = model([input_tensor1, input_tensor2])

    # 打印维度验证
    print("input_tensor1_shape :", input_tensor1.shape)   
    print("input_tensor2_shape :", input_tensor2.shape)
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
