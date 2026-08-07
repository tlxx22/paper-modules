import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelMLP(nn.Module):
    def __init__(self, c, expansion=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c*expansion, 1)
        self.fc2 = nn.Conv2d(c*expansion, c, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x
    
class DWT2D_Haar(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        l = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)])
        h = torch.tensor([1/math.sqrt(2),-1/math.sqrt(2)])
        ll = torch.outer(l, l); lh = torch.outer(l, h)
        hl = torch.outer(h, l); hh = torch.outer(h, h)
        k = torch.stack([ll, lh, hl, hh], dim=0)  # (4,2,2)
        weight = torch.zeros((channels*4, 1, 2, 2))
        for c in range(channels):
            weight[c*4+0,0] = ll
            weight[c*4+1,0] = lh
            weight[c*4+2,0] = hl
            weight[c*4+3,0] = hh
        self.register_buffer('weight', weight)
        self.groups = channels

    def forward(self, x):  # x: (N,C,D,H)
        x = F.pad(x, (0, x.shape[-1]%2, 0, x.shape[-2]%2), mode='reflect')
        y = F.conv2d(x, self.weight, stride=2, groups=self.groups)  # (N,4C,D/2,H/2)
        C = x.shape[1]
        LL,LH,HL,HH = torch.chunk(y, 4, dim=1)  # (N,C,D/2,H/2)
        return LL, LH, HL, HH
    
class SR(nn.Module):
    """Spatial Reduction for K/V"""
    def __init__(self, c, sr_ratio=2):
        super().__init__()
        self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio>1 else None
        self.norm = gn(c)
    def forward(self, x):
        if self.sr is None: return x
        return self.norm(self.sr(x))


class CrossSourceMHA(nn.Module):
    
    def __init__(self, c_q, c_k, c_v, c_out, heads=4, sr_ratio=2):
        super().__init__()
        self.h = heads
        self.q = nn.Conv2d(c_q, c_out, 1)
        self.k = nn.Conv2d(c_k, c_out, 1)
        self.v = nn.Conv2d(c_v, c_out, 1)
        self.norm_q = gn(c_q); self.norm_k = gn(c_k); self.norm_v = gn(c_v)
        self.proj = nn.Conv2d(c_out, c_out, 1)
        self.sr_k = SR(c_k, sr_ratio); self.sr_v = SR(c_v, sr_ratio)
        self.scale = (c_out // heads) ** -0.5

    def _reshape(self, x):  # (N,C,H,W)->(N,heads,HW,dim)
        N,C,H,W = x.shape
        x = x.view(N, self.h, C//self.h, H*W)          # (N,h,dim,HW)
        return x.permute(0,1,3,2).contiguous()         # (N,h,HW,dim)

    def forward(self, q_src, k_src, v_src):
        q = self.q(self.norm_q(q_src))
        k = self.k(self.norm_k(self.sr_k(k_src)))
        v = self.v(self.norm_v(self.sr_v(v_src)))

        N, Cq, Hq, Wq = q.shape
        q = self._reshape(q); k = self._reshape(k); v = self._reshape(v)
        attn = (q * self.scale) @ k.transpose(-2, -1)   # (N,h,HWq,HWk)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                  # (N,h,HWq,dim)
        out = out.permute(0,1,3,2).contiguous().view(N, Cq, Hq, Wq)  # (N,Cq,Hq,Wq)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        return self.proj(out)
     
def gn(c):  # GroupNorm
    return nn.GroupNorm(num_groups=min(32, max(1, c//4)), num_channels=c)

class μPCAD(nn.Module):
    """
    Microlocal Polyphase Co-Attentive Decimator 
    """
    def __init__(self, C_in, C_out, heads=4, sr_ratio=2, mlp_ratio=4, drop=0.0):
        super().__init__()
        C_mid = 2*C_in  

        self.proj_pool = nn.Conv2d(C_in, 2*C_in, 1)   # -> split for Max/Avg
        self.proj_wav  = nn.Conv2d(C_in, C_in, 1)     # -> for DWT
        self.dwt = DWT2D_Haar(C_in)

        self.maxpool = nn.MaxPool2d(2,2)
        self.avgpool = nn.AvgPool2d(2,2)

        # [Max, Avg, LL, LH, HL, HH] -> 3x3 -> MLP
        self.fuse_local = nn.Sequential(
            nn.Conv2d(6*C_in, C_mid, 3, padding=1),
            gn(C_mid), nn.GELU(),
            ChannelMLP(C_mid, expansion=mlp_ratio, drop=drop)
        )

        # Q=Max, K=Avg, V=Wavelet
        self.attn = CrossSourceMHA(c_q=C_in, c_k=C_in, c_v=4*C_in,
                                   c_out=C_mid, heads=heads, sr_ratio=sr_ratio)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.attn_mlp = ChannelMLP(C_mid, expansion=mlp_ratio, drop=drop)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))  
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_mid, C_mid//4, 1), nn.GELU(),
            nn.Conv2d(C_mid//4, C_mid, 1), nn.Sigmoid()
        )

        self.blur_w = nn.Conv3d(C_mid, C_mid, kernel_size=(1,1,3),
                                padding=(0,0,1), groups=C_mid, bias=False)
        with torch.no_grad():
            k = torch.tensor([1.,2.,1.]).view(1,1,1,1,3)/4.0
            self.blur_w.weight.copy_(k.repeat(C_mid,1,1,1,1))
        for p in self.blur_w.parameters(): p.requires_grad = False

        self.down_w = nn.Conv3d(C_mid, C_out, kernel_size=(1,1,3),
                                stride=(1,1,2), padding=(0,0,1), bias=True)

    def forward(self, x):  # x: (B,C,D,H,W)
        B,C,D,H,W = x.shape

        x4 = x.permute(0,4,1,2,3).contiguous().view(B*W, C, D, H)  # (BW,C,D,H)

        pool_in = self.proj_pool(x4)                        # (BW,2C,D,H)
        q_in, k_in = torch.chunk(pool_in, 2, dim=1)         # for Max / Avg
        v_in  = self.proj_wav(x4)                           # (BW,C,D,H)

        x_max = self.maxpool(q_in)                          # (BW,C,D/2,H/2)
        x_avg = self.avgpool(k_in)                          # (BW,C,D/2,H/2)
        LL,LH,HL,HH = self.dwt(v_in)                        # (BW,C,D/2,H/2)*4

        local = torch.cat([x_max, x_avg, LL, LH, HL, HH], dim=1)  # (BW,6C,·,·)
        local = self.fuse_local(local)                      # (BW,C_mid,D/2,H/2)

        wav_cat = torch.cat([LL,LH,HL,HH], dim=1)           # (BW,4C,·,·)
        attn = self.attn(q_src=x_max, k_src=x_avg, v_src=wav_cat)  # (BW,C_mid,·,·)
        attn = self.attn_mlp(attn)

        fused = torch.sigmoid(self.alpha)*local + torch.sigmoid(self.beta)*attn \
                + self.gamma * nn.Conv2d(LL.shape[1], local.shape[1], 1, bias=False).to(x.device)(LL)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        fused = fused * self.se(fused)                      # SE

        y = fused.view(B, W, -1, D//2, H//2).permute(0,2,3,4,1).contiguous()  # (B,C_mid,D/2,H/2,W)
        y = self.blur_w(y)                                  
        y = self.down_w(y)                                  # (B,C_out,D/2,H/2,W/2)
        return y


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # b c d h w 注意输入张量维度
    input_tensor = torch.randn(2, 64, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = μPCAD(C_in=64, C_out=64, heads=4, sr_ratio=2, mlp_ratio=4, drop=0.0).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")