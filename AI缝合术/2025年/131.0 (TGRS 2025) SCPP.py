import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(B, C)  # (B, C)                                                                                                                                                                     # 微信公众号:AI缝合术
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.sig(s).view(B, C, 1, 1)
        return x * s


class ScaleAwareModule(nn.Module):
    """
    多尺度扩张卷积 + SE + 1x1 映射 + 像素级 softmax 权重融合
    输入: (B,C,H,W)
    输出: fused (B,C,H,W)
    """
    def __init__(self, channels, dilation_rates=(1, 3, 5), se_reduction=16):                                                                                                                                                                     # 微信公众号:AI缝合术
        super().__init__()
        self.channels = channels
        self.dil_rates = dilation_rates

        # 为每个分支构建 dilated conv -> BN -> ReLU -> SE -> 1x1 映射
        self.branches = nn.ModuleList()
        for d in dilation_rates:
            branch = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),                                                                                                                                                                     # 微信公众号:AI缝合术
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                SEBlock(channels, reduction=se_reduction),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # 映射回 channels                                                                                                                                                                     # 微信公众号:AI缝合术
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

        # 通过 1x1 生成 3 个 per-pixel attention logits -> softmax( dim=1 over 3 branches )                                                                                                                                                                     # 微信公众号:AI缝合术
        self.attn_conv = nn.Conv2d(channels, len(dilation_rates), kernel_size=1, bias=True)                                                                                                                                                                     # 微信公众号:AI缝合术

    def forward(self, x):
        # x: (B,C,H,W)
        outs = []
        for br in self.branches:
            outs.append(br(x))  # 每个 (B,C,H,W)
        # 逐像素求和（先求和再产生注意力）
        sum_feats = outs[0]
        for o in outs[1:]:
            sum_feats = sum_feats + o  # (B,C,H,W)

        logits = self.attn_conv(sum_feats)  # (B,3,H,W)
        attn = F.softmax(logits, dim=1)     # (B,3,H,W), 在 branch dim 上 softmax                                                                                                                                                                     # 微信公众号:AI缝合术

        # 将 attn 分配到每个分支并加权融合
        fused = 0
        for i, o in enumerate(outs):
            a = attn[:, i:i+1, :, :]   # (B,1,H,W)
            fused = fused + o * a     # 广播到通道维后加权                                                                                                                                                                     # 微信公众号:AI缝合术
        return fused, outs, attn


class SCPP(nn.Module):
    """
    Scale-aware Pyramid Pooling (SCPP)
    输入: (B,C,H,W)
    输出: (B, out_channels, H, W)
    """
    def __init__(self, in_channels, out_channels=None, se_reduction=16, dilation_rates=(1,3,5)):                                                                                                                                                                     # 微信公众号:AI缝合术
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        # 标准 1x1 分支（捕捉局部细节）
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),                                                                                                                                                                     # 微信公众号:AI缝合术
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 尺度感知模块
        self.scale_module = ScaleAwareModule(in_channels, dilation_rates=dilation_rates, se_reduction=se_reduction)                                                                                                                                                                     # 微信公众号:AI缝合术

        # 全局池化分支：GAP -> FC -> ReLU -> expand to HxW
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=True),
            nn.ReLU(inplace=True)
        )

        # 最终融合后的 1x1 conv 输出
        # 拼接 channels: scale_fused(C) + conv1x1(C) + global(C) = 3C
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, self.out_channels, kernel_size=1, bias=False),                                                                                                                                                                     # 微信公众号:AI缝合术
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 1x1 分支
        local = self.conv1x1(x)          # (B,C,H,W)

        # scale-aware module -> fused scale feature
        scale_fused, branches, attn = self.scale_module(x)  # (B,C,H,W)                                                                                                                                                                     # 微信公众号:AI缝合术

        # global branch
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)         # (B,C)
        g = self.global_fc(gap)                             # (B,C)
        g = g.view(B, C, 1, 1).expand(-1, -1, H, W)         # (B,C,H,W)

        # concat and project
        cat = torch.cat([scale_fused, local, g], dim=1)     # (B,3C,H,W)
        out = self.out_conv(cat)                            # (B, out_channels, H, W)
        return out, {"scale_fused": scale_fused, "local": local, "global": g, "attn": attn}                                                                                                                                                                     # 微信公众号:AI缝合术


# ------------张量测试---------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, H, W = 1, 32, 128, 128
    x = torch.randn(B, C, H, W, device=device)

    scpp = SCPP(in_channels=C, out_channels=C, se_reduction=16, dilation_rates=(1,3,5)).to(device)                                                                                                                                                                     # 微信公众号:AI缝合术
    out, meta = scpp(x)

    print(scpp)
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
    print("Input :", x.shape)
    print("Output:", out.shape)

