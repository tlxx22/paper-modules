import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/abs/2512.05494
    论文题目：Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation（AAAI 2026）
    中文题目：基于结构感知的解码：融合方向、频域‑空间与结构注意力的医学图像分割（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV12KQRBtEDB/
    自适应交叉融合注意力（Adaptive Cross-Fusion Attention，ACFA）
        实际意义：①Transformer解码器对关键区域的响应能力不足：传统Transformer在医学图像中难以精准聚焦重要结构（如器官边界、肿瘤区域），导致重要特征响应较弱。
                ②连续性和结构方向建模能力弱：基于Transformer的自注意力虽能捕捉全局依赖，但对结构方向（如水平/垂直/平面一致性）和精细空间连续性建模不足，容易丢失方向性信息。
        实现方式：通过“通道-空间门控 + 三方向可学习引导 + 深度卷积融合”，精准强化关键区域响应与结构方向连续性。
"""

# =========================
# LayerNorm for 2D feature map
# 输入: [B, C, H, W]
# 在通道维上做 LayerNorm
# =========================
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x

# =========================
# 深度卷积块
# =========================
class DWConv2d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            bias=False
        )

    def forward(self, x):
        return self.dwconv(x)

# =========================
# 1D 深度卷积块
# 输入: [B, C, L]
# =========================
class DWConv1d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dwconv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            bias=False
        )

    def forward(self, x):
        return self.dwconv(x)


# =========================
# Channel Gate
# 这里采用论文公式对应的 avg + max 形式
# 再通过 sigmoid 得到通道注意力
# =========================
class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat = F.adaptive_avg_pool2d(x, 1)
        max_feat = F.adaptive_max_pool2d(x, 1)

        attn = self.mlp(avg_feat) + self.mlp(max_feat)
        attn = self.sigmoid(attn)
        return x * attn

# =========================
# Spatial Gate
# 采用 CBAM 风格的空间门控：
# 先做通道均值池化和最大池化，再7x7卷积
# =========================
class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(spatial_feat))
        return x * attn

# =========================
# ACFA: Adaptive Cross-Fusion Attention
# =========================
class ACFA(nn.Module):
    def __init__(self, channels, height, width, reduction=16):
        """
        Args:
            channels: 输入通道数，要求能被4整除
            height: 该层特征图固定高度 H
            width: 该层特征图固定宽度 W
            reduction: channel gate 的压缩率
        """
        super().__init__()
        assert channels % 4 == 0, "channels 必须能被 4 整除"

        self.channels = channels
        self.height = height
        self.width = width
        self.branch_channels = channels // 4

        # ---- 1) Channel Gate + Spatial Gate ----
        self.channel_gate = ChannelGate(channels, reduction=reduction)
        self.spatial_gate = SpatialGate(kernel_size=7)

        # ---- 2) 三个方向的可学习参数 ----
        # 对应论文中的 TensorHW, TensorH, TensorW
        self.tensor_hw = nn.Parameter(torch.rand(1, self.branch_channels, height, width))
        self.tensor_h = nn.Parameter(torch.rand(1, self.branch_channels, height, 1))
        self.tensor_w = nn.Parameter(torch.rand(1, self.branch_channels, 1, width))

        # ---- 3) HW 方向分支 ----
        self.hw_pwconv = nn.Conv2d(self.branch_channels, self.branch_channels, kernel_size=1, bias=False)
        self.hw_dwconv = nn.Conv2d(
            self.branch_channels, self.branch_channels,
            kernel_size=3, padding=1, groups=self.branch_channels, bias=False
        )

        # ---- 4) H 方向分支 ----
        self.h_pwconv1d = nn.Conv1d(self.branch_channels, self.branch_channels, kernel_size=1, bias=False)
        self.h_dwconv1d = nn.Conv1d(
            self.branch_channels, self.branch_channels,
            kernel_size=3, padding=1, groups=self.branch_channels, bias=False
        )

        # ---- 5) W 方向分支 ----
        self.w_pwconv1d = nn.Conv1d(self.branch_channels, self.branch_channels, kernel_size=1, bias=False)
        self.w_dwconv1d = nn.Conv1d(
            self.branch_channels, self.branch_channels,
            kernel_size=3, padding=1, groups=self.branch_channels, bias=False
        )

        # ---- 6) 第4个通用分支 ----
        self.branch4_dwconv = nn.Conv2d(
            self.branch_channels, self.branch_channels,
            kernel_size=3, padding=1, groups=self.branch_channels, bias=False
        )
        self.branch4_pwconv = nn.Conv2d(self.branch_channels, self.branch_channels, kernel_size=1, bias=False)

        self.act = nn.GELU()

        # ---- 7) 拼接后的归一化与融合 ----
        self.norm = LayerNorm2d(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f"输入通道应为 {self.channels}，但得到 {C}"
        assert H == self.height and W == self.width, \
            f"该实现使用固定方向参数，输入尺寸必须为 ({self.height}, {self.width})，但得到 ({H}, {W})"

        # ------------------------------------------------
        # Step 1: 通道门控
        # 对应论文公式 (1)
        # ------------------------------------------------
        x_cg = self.channel_gate(x)

        # ------------------------------------------------
        # Step 2: 空间门控
        # 对应论文公式 (2)
        # ------------------------------------------------
        x_sg = self.spatial_gate(x_cg)

        # ------------------------------------------------
        # Step 3: 沿通道分成4个子特征
        # 对应论文中的四个子集
        # ------------------------------------------------
        x1, x2, x3, x4 = torch.chunk(x_sg, 4, dim=1)

        # ------------------------------------------------
        # Step 4: HW 方向分支
        # 对应公式 (6)
        # 用输入子特征 x1 与方向参数 tensor_hw 做逐元素调制
        # ------------------------------------------------
        hw_weight = self.hw_dwconv(self.act(self.hw_pwconv(self.tensor_hw)))
        out_hw = x1 * hw_weight

        # ------------------------------------------------
        # Step 5: H 方向分支
        # 对应公式 (7)
        # tensor_h: [1, C/4, H, 1]
        # 转成 1D 卷积输入 [1, C/4, H]
        # 再广播回 [1, C/4, H, 1]
        # ------------------------------------------------
        h_weight = self.tensor_h.squeeze(-1)                     # [1, C/4, H]
        h_weight = self.h_dwconv1d(self.act(self.h_pwconv1d(h_weight)))
        h_weight = h_weight.unsqueeze(-1)                        # [1, C/4, H, 1]
        out_h = x2 * h_weight

        # ------------------------------------------------
        # Step 6: W 方向分支
        # 对应公式 (8)
        # tensor_w: [1, C/4, 1, W]
        # 转成 1D 卷积输入 [1, C/4, W]
        # 再广播回 [1, C/4, 1, W]
        # ------------------------------------------------
        w_weight = self.tensor_w.squeeze(-2)                     # [1, C/4, W]
        w_weight = self.w_dwconv1d(self.act(self.w_pwconv1d(w_weight)))
        w_weight = w_weight.unsqueeze(-2)                        # [1, C/4, 1, W]
        out_w = x3 * w_weight

        # ------------------------------------------------
        # Step 7: 第4个通用上下文分支
        # 对应公式 (9)
        # ------------------------------------------------
        out_4 = self.branch4_pwconv(self.act(self.branch4_dwconv(x4)))

        # ------------------------------------------------
        # Step 8: 四分支拼接 + LN + Conv fusion
        # ------------------------------------------------
        out = torch.cat([out_hw, out_h, out_w, out_4], dim=1)
        out = self.norm(out)
        out = self.fuse(out)

        return out



if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = ACFA(channels=32, height=50, width=50)
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")