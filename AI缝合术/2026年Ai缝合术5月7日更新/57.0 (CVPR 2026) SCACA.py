import torch
import torch.nn as nn
import torch.nn.functional as F

class SCACA(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 自调制深度卷积
        self.self_modulate = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
            nn.Sigmoid()
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # QKV 投影
        self.spatial_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.channel_qkv = nn.Conv2d(dim, dim * 3, 1)

        # 相对位置偏置
        self.relative_pos_bias = nn.Parameter(torch.randn(num_heads, window_size*window_size, window_size*window_size))

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim, 1)
        )

    def window_partition(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H//self.window_size, self.window_size, W//self.window_size, self.window_size)
        x = x.permute(0,2,4,1,3,5).contiguous()       # [B, h_num, w_num, C, win, win]
        return x.view(-1, C, self.window_size, self.window_size)  # [B*N_win, C, win, win]

    def window_reverse(self, x, H, W):
        B = x.shape[0] // ((H//self.window_size)*(W//self.window_size))
        x = x.view(B, H//self.window_size, W//self.window_size, self.dim, self.window_size, self.window_size)
        x = x.permute(0,3,1,4,2,5).contiguous()
        return x.view(B, self.dim, H, W)

    def spatial_abundance_cross_attention(self, x, ref):
        B, C, H, W = x.shape
        N = self.window_size ** 2

        # 窗口切分
        x_win = self.window_partition(x)
        ref_win = self.window_partition(ref)

        # QKV [B*N_win, 3C, win, win]
        qkv = self.spatial_qkv(x_win)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 拆多头 [B*N_win, heads, N, head_dim]
        q = q.flatten(2).permute(0,2,1).view(-1, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.flatten(2).permute(0,2,1).view(-1, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.flatten(2).permute(0,2,1).view(-1, N, self.num_heads, self.head_dim).permute(0,2,1,3)

        # 参考特征维度对齐
        ref_v = ref_win.flatten(2).permute(0,2,1)  # [B*N_win, N, C]
        ref_v = ref_v.view(-1, N, self.num_heads, self.head_dim).permute(0,2,1,3)  # 与v同维度
        v_mod = v * ref_v  

        # 注意力
        attn = (q @ k.transpose(-2,-1)) * self.scale + self.relative_pos_bias
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v_mod).permute(0,2,1,3).contiguous().view(-1, N, C)
        out = out.view(-1, C, self.window_size, self.window_size)
        return self.window_reverse(out, H, W)

    def channel_abundance_cross_attention(self, x, ref):
        B, C, H, W = x.shape
        qkv = self.channel_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 多头：v形状 [B, num_heads, H×W, head_dim]
        q = q.flatten(2).permute(0,2,1).view(B,-1,self.num_heads,self.head_dim).permute(0,2,1,3)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        k = k.flatten(2).permute(0,2,1).view(B,-1,self.num_heads,self.head_dim).permute(0,2,1,3)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        v = v.flatten(2).permute(0,2,1).view(B,-1,self.num_heads,self.head_dim).permute(0,2,1,3)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 参考全局调制
        ref_g = F.adaptive_avg_pool2d(ref, 1).view(B,1,1,C)
        ref_g = ref_g.view(B, self.num_heads, 1, self.head_dim)
        v_mod = v * ref_g

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v_mod).permute(0,2,1,3).contiguous().view(B,C,H,W)
        return out

    def forward(self, x, ref):
        # 自调制
        ref = ref * (1 + self.self_modulate(ref))

        # 空间分支
        x_norm = self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = x + self.ffn(self.spatial_abundance_cross_attention(x_norm, ref))                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 通道分支
        x_norm = self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = x + self.ffn(self.channel_abundance_cross_attention(x_norm, ref))                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        return x

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 16, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    reference_tensor = torch.randn(2, 16, 32, 32).to(device)

    model = SCACA(dim=16, window_size=8, num_heads=4).to(device)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print(model)
    
    output_tensor = model(input_tensor, reference_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("reference_tensor_shape:", reference_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")