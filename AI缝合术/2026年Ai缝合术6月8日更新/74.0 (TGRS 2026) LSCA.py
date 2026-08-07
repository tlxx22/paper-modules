import torch
import torch.nn as nn

class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = in_channels // heads
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).transpose(1, 2)
        
        q = self.q_proj(x_flat).view(B, N, self.heads, self.head_dim).transpose(1, 2)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        k = self.k_proj(x_flat).view(B, N, self.heads, self.head_dim).transpose(1, 2)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        v = self.v_proj(x_flat).view(B, N, self.heads, self.head_dim).transpose(1, 2)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = out + x_flat
        return out.transpose(1, 2).view(B, C, H, W)

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.qkv_proj = nn.Linear(in_channels, 3 * in_channels)
        self.alpha = nn.Parameter(torch.ones(1))
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).transpose(1, 2)
        
        qkv = self.qkv_proj(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q_c = q.transpose(1, 2)
        k_c = k
        attn = (q_c @ k_c) * self.alpha
        attn = attn.softmax(dim=-1)
        
        v_c = v.transpose(1, 2)
        out_attn = (attn @ v_c).transpose(1, 2)
        
        v_spatial = v.transpose(1, 2).view(B, C, H, W)
        out_dw = self.dwconv(v_spatial).view(B, C, N).transpose(1, 2)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        
        out = out_attn + out_dw
        out = self.out_proj(out)
        out = out + x_flat
        return out.transpose(1, 2).view(B, C, H, W)


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 32, 32).to(device)          

    sam = SpatialAttentionModule(in_channels=64, heads=8).to(device)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
    print(sam)
    cam = ChannelAttentionModule(in_channels=64).to(device)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
    print(cam)

    
    sam_out = sam(input_tensor)
    cam_out = cam(sam_out)
    
    print(f"Input shape     : {input_tensor.shape}")
    print(f"SAM output shape: {sam_out.shape}")
    print(f"CAM output shape: {cam_out.shape}")
                                                                                                                                                                                         # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")