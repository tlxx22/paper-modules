import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

# Spectral Banding Self-Attention
class SBSAtt(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SBSAtt, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.factor = 2
        self.idx_dict = {}
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[...,t_pad[0]:hw-t_pad[1]]
        
    def comp2real(self, x):
        b, _, h, w = x.shape
        return torch.cat([x.real, x.imag], 1)
#        return torch.stack([x.real, x.imag], 2).view(b,-1,h,w)
    def real2comp(self, x):
        xr, xi = x.chunk(2, dim=1)
        return torch.complex(xr, xi)

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def get_idx_map(self, h, w):
        l1_u = torch.arange(h//2).view(1,1,-1,1)
        l2_u = torch.arange(w).view(1,1,1,-1)
        half_map_u = l1_u @ l2_u
        l1_d = torch.arange(h - h//2).flip(0).view(1,1,-1,1)
        l2_d = torch.arange(w).view(1,1,1,-1)
        half_map_d = l1_d @ l2_d
        return torch.cat([half_map_u, half_map_d], 2).view(1,1,-1).argsort(-1)
    def get_idx(self, x):
        h, w = x.shape[-2:]
        if (h, w) in self.idx_dict:
            return self.idx_dict[(h, w)]
        idx_map = self.get_idx_map(h, w).to(x.device).detach()
        self.idx_dict[(h, w)] = idx_map
        return idx_map
    def attn(self, qkv):
        h = qkv.shape[2]
        q,k,v = qkv.chunk(3, dim=1)
        
        q, pad_w, idx = self.fft(q)
        q, pad = self.pad(q, self.factor)
        k, pad_w, _ = self.fft(k)
        k, pad = self.pad(k, self.factor)
        v, pad_w, _ = self.fft(v)
        v, pad = self.pad(v, self.factor)
        
        q = rearrange(q, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        k = rearrange(k, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        v = rearrange(v, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (c factor) hw -> b (head c) (factor hw)', head=self.num_heads, factor=self.factor)
        out = self.unpad(out, pad)
        out = self.ifft(out, pad_w, idx, h)
        return out
    def fft(self, x):
        x, pad = self.pad(x, 2)
        x = torch.fft.rfft2(x.float(), norm="ortho")
        x = self.comp2real(x)
        idx = self.get_idx(x)
        b, c = x.shape[:2]
        x = x.contiguous().view(b, c, -1)
        x = torch.gather(x, 2, index=idx.repeat(b,c,1)) # b, 6c, h*(w//2+1)
        return x, pad, idx
    def ifft(self, x, pad, idx, h):
        b, c = x.shape[:2]
        x = torch.scatter(x, 2, idx.repeat(b,c,1), x)
        x = x.view(b, c, h, -1)
        x = self.real2comp(x)
        x = torch.fft.irfft2( x, norm='ortho' )#.abs()
        x = self.unpad(x, pad)
        return x
    def forward(self, x):
        b,c,h,w = x.shape

        attn_map = x

        qkv = self.qkv_dwconv(self.qkv(x))

#        qkv, pad_w, idx = self.fft(qkv)
#        qkv, pad = self.pad(qkv, self.factor)

        attn_map = qkv  
        out = self.attn(qkv) 
        attn_map = out


#        out = self.unpad(out, pad)
#        out = self.ifft(out, pad_w, idx, h)

        out = self.project_out(out)
        attn_map = out
        return out
    
if __name__ == "__main__":
    # 设置参数
    dim = 64
    num_heads = 8
    bias = True

    # 创建 SBSAtt 实例
    sbsatt = SBSAtt(dim=dim, num_heads=num_heads, bias=bias).cuda()
    print(sbsatt)
    print("\n微信公众号: AI缝合术!\n")

    # 输入张量，形状为 [batch_size, channels, height, width]
    x = torch.randn(1, dim, 64, 64).cuda()

    # 前向传播
    output = sbsatt(x).cuda()

    
    # 打印输入和输出的形状
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {output.shape}")
