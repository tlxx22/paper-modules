import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CrossTransAttention(nn.Module):
    def __init__(self, num_heads=4, dim=64):
        super().__init__()
        self.num_heads = num_heads
        bias=True
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim*1, bias=bias)                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, feat_x, feat_y):
        b, c, h, w = feat_x.shape
        
        q = self.q_dwconv(self.q(feat_x))
        kv = self.kv_dwconv(self.kv(feat_y))
        k,v = kv.chunk(2, dim=1)

        # (B, C, H, W) -> (B, head, head_dim, HW)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class MutCrossAttention(nn.Module):
    """Mutual cross attention: HS->MS and MS->HS."""
    def __init__(self, num_heads=4, dim=64):
        super().__init__()
        self.mca = CrossTransAttention(num_heads=num_heads,dim=dim)

    def forward(self, feat_hs, feat_ms):
        feat_h2m = self.mca(feat_x=feat_hs, feat_y=feat_ms)
        feat_m2h = self.mca(feat_x=feat_ms, feat_y=feat_hs)
        return feat_h2m, feat_m2h

class MGCM(nn.Module):
    def __init__(self, dim, bias):
        super(MGCM, self).__init__()
        self.in_x = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.in_y = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)

        pool_sizes = [8, 4, 2]
        self.pools_x = nn.ModuleList([nn.AvgPool2d(k, k) for k in pool_sizes])
        self.pools_y = nn.ModuleList([nn.AvgPool2d(k, k) for k in pool_sizes])
        self.convs_x = nn.ModuleList([nn.Conv2d(dim, dim, 3, 1, 1, bias=bias) for _ in pool_sizes])                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.convs_y = nn.ModuleList([nn.Conv2d(dim, dim, 3, 1, 1, bias=bias) for _ in pool_sizes])                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.attns = nn.ModuleList([MutCrossAttention(num_heads=4, dim=dim) for _ in pool_sizes])

        self.relu = nn.GELU()
        self.sum_x = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.sum_y = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)

    def forward(self, x, y):
        x_size = x.size()
        res_x = self.in_x(x)
        res_y = self.in_y(y)
        for i in range(len(self.pools_x)):
            if i == 0:
                x_, y_ = self.attns[i](self.convs_x[i](self.pools_x[i](x)), self.convs_y[i](self.pools_y[i](y)))                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            else:
                x_, y_ = self.attns[i](self.convs_x[i](self.pools_x[i](x)+x_up), self.convs_y[i](self.pools_y[i](y)+y_up))                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            res_x = torch.add(res_x, F.interpolate(x_, x_size[2:], mode='bilinear', align_corners=True))
            res_y = torch.add(res_y, F.interpolate(y_, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_x) - 1:
                x_up = F.interpolate(x_, scale_factor=2, mode='bilinear', align_corners=True)
                y_up = F.interpolate(y_, scale_factor=2, mode='bilinear', align_corners=True)
        out_x = x + self.sum_x(self.relu(res_x))
        out_y = y + self.sum_y(self.relu(res_y))

        return out_x, out_y

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor_fm = torch.randn(1, 64, 32, 32).to(device)
    input_tensor_fh = torch.randn(1, 64, 32, 32).to(device)

    mgcm =  MGCM(dim=64, bias=True).to(device)
    print(mgcm)

    output_tensor_fm, output_tensor_fh = mgcm(input_tensor_fm, input_tensor_fh)

    # 打印输入输出形状
    print("Input fm shape:", input_tensor_fm.shape)
    print("Input fh shape:", input_tensor_fh.shape)
    print("Output fm shape:", output_tensor_fm.shape)  
    print("Output fh shape:", output_tensor_fh.shape)  
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")