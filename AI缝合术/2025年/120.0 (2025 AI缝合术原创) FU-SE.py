import torch
import torch.nn as nn
import torch.nn.functional as F

class FUSEChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8, lf_ratio=0.125, eps=1e-6):                                                                                              # 微信公众号:AI缝合术
        super().__init__()
        self.C = channels
        self.r = max(1, reduction)
        self.lf_ratio = lf_ratio
        self.eps = eps

        # 用可学习的 per-channel 权重 (C x 4) 和 bias (C,)
        self.per_channel_weight = nn.Parameter(torch.full((channels, 4), 0.25))                                                                                              # 微信公众号:AI缝合术
        self.per_channel_bias = nn.Parameter(torch.zeros(channels))

        # 低秩跨通道混合（类似 SE 的 MLP）
        hidden = max(1, channels // self.r)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True)
        )

        # 温度门控参数（根据全局不确定性调节）
        self.tau_proj = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.tau_proj.weight, 1.0)
        nn.init.constant_(self.tau_proj.bias, 0.0)

    def _spatial_stats(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=(2, 3))          # [B, C]
        var = (x - mean[:, :, None, None]).pow(2).mean(dim=(2, 3))
        std = (var + self.eps).sqrt()     # [B, C]
        return mean, std

    def _freq_stats(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        Xf = torch.fft.rfft2(x, norm='ortho')  # [B,C,H,W//2+1] complex                                                                                              # 微信公众号:AI缝合术
        mag = Xf.abs()                          # [B,C,H,W//2+1] real                                                                                              # 微信公众号:AI缝合术

        k = max(3, int(min(H, W) * self.lf_ratio))
        kh = min(H, k)
        kw = min(W // 2 + 1, k)

        lfe = mag[:, :, :kh, :kw].mean(dim=(2, 3))  # [B, C]
        all_e = mag.mean(dim=(2, 3))                # [B, C]
        hfe = torch.clamp(all_e - lfe, min=0.0)     # [B, C]
        return lfe, hfe

    def forward(self, x):
        # 输入 x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1) 多源挤压
        mean, std = self._spatial_stats(x)   # [B, C], [B, C]
        lfe, hfe = self._freq_stats(x)       # [B, C], [B, C]

        # stats: [B, C, 4]
        stats = torch.stack([mean, std, lfe, hfe], dim=-1)

        # 2) 逐通道融合：用 per-channel 权重 (C,4)
        #  stats: [B, C, 4]
        #  weight: [C, 4] -> unsqueeze -> [1, C, 4] 广播到 [B, C, 4]
        s_seed = (stats * self.per_channel_weight.unsqueeze(0)).sum(dim=-1) \
                 + self.per_channel_bias.unsqueeze(0)   # [B, C]                                                                                              # 微信公众号:AI缝合术

        # 3) 低秩跨通道混合（MLP）
        #    mlp expects input [B, channels]
        logits = self.mlp(s_seed)  # [B, C]

        # 4) 温度门控（不确定性 proxy）
        g_uncert = std.mean(dim=1, keepdim=True)       # [B,1]
        tau = F.softplus(self.tau_proj(g_uncert)) + 1e-3  # [B,1]                                                                                              # 微信公众号:AI缝合术
        gates = torch.sigmoid(logits / tau)            # [B, C]

        # 5) 重加权
        y = x * gates[:, :, None, None]                # [B, C, H, W]                                                                                              # 微信公众号:AI缝合术
        return y

if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 64, 64)  # 例如 batch=1, 通道=32, 高=64, 宽=64

    # 初始化 FUSEChannelAttention 模块
    fuse = FUSEChannelAttention(channels=32)

    # 前向传播测试
    output = fuse(x)

    # 输出结果形状
    print(fuse)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]                                                                                              # 微信公众号:AI缝合术
    print("输出张量形状:", output.shape)  # [B, C, H, W]                                                                                              # 微信公众号:AI缝合术