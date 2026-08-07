import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_
from natten.functional import na2d_qk, na2d_av

FUSED = True
try:
    from natten.functional import na2d
except ImportError:
    FUSED = False
    print("natten 0.17 not installed, using dummy implementation")

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(                                                          # 微信公众号:AI缝合术
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class DeformableNeighborhoodAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        offset_range_factor=1.0,
        stride=1,
        use_pe=True,
        dwc_pe=True,
        no_off=False,
        fixed_pe=False,
        is_causal: bool = False,
        rel_pos_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):

        super().__init__()
        n_head_channels = dim // num_heads
        n_groups = num_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = num_heads
        self.nc = n_head_channels * num_heads
        self.n_groups = num_heads
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = kernel_size
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels,
                      kk, stride, pad_size, groups=self.n_group_channels),
            LayerNorm2d(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2 * self.kernel_size[0] - 1),
                    (2 * self.kernel_size[1] - 1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.rpe_table = nn.Conv2d(
            self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key,
                           dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key,
                           dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(
            B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(
            B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(
            q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)                                                          # 微信公众号:AI缝合术
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg

        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor(
                [1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)                                                          # 微信公众号:AI缝合术
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(
                x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(
                3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups,
                                self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, H, W)

        residual_lepe = self.rpe_table(q)

        if self.rpb is not None or not FUSED:
            q = einops.rearrange(q, 'b (g c) h w -> b g h w c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)                                                          # 微信公众号:AI缝合术
            k = einops.rearrange(self.proj_k(x_sampled), 'b (g c) h w -> b g h w c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)                                                          # 微信公众号:AI缝合术
            v = einops.rearrange(self.proj_v(x_sampled), 'b (g c) h w -> b g h w c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)                                                          # 微信公众号:AI缝合术

            q = q*self.scale
            attn = na2d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = na2d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            out = einops.rearrange(out, 'b g h w c -> b (g c) h w')

        else:
            q = einops.rearrange(q, 'b (g c) h w -> b h w g c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            k = einops.rearrange(self.proj_k(x_sampled), 'b (g c) h w -> b h w g c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            v = einops.rearrange(self.proj_v(x_sampled), 'b (g c) h w -> b h w g c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            out = na2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        y = self.proj_drop(self.proj_out(out))

        return y


# 在linxu环境下运行
if __name__ == "__main__":

    # 超参数设置
    batch_size = 1
    height, width = 128, 128    # 输入图像大小
    channels = 64               # 输入通道数，需能被 num_heads 整除
    num_heads = 8               # 注意力头数
    kernel_size = 7
    dilation = 1

    # 创建输入张量：形状为 (B, C, H, W)
    x = torch.randn(batch_size, channels, height, width)

    # 初始化 DeformableNeighborhoodAttention 模块
    attn = DeformableNeighborhoodAttention(
        dim=channels,
        num_heads=num_heads,
        kernel_size=kernel_size,
        dilation=dilation,
        offset_range_factor=1.0,
        stride=1,
        use_pe=True,
        dwc_pe=True,
        no_off=False,
        fixed_pe=False,
        is_causal=False,
        rel_pos_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    )

    # 前向传播测试
    with torch.no_grad():
        output = attn(x)

    # 输出结果
    print(attn)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]