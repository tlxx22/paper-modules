
import math
import torch
import torch.nn as nn
from typing import Literal
from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    """
    MLP block with pre-layernorm, GELU activation, and dropout.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """
    Global multi-head self-attention block with optional projection.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = (
            dim_head * heads
        )  # the total dimension used inside the multi-head attention. When concatenating all heads, the combined dimension is dim_head × heads
        project_out = not (
            heads == 1 and dim_head == dim
        )  # if we're using just 1 head and its dimension equals dim, then we can skip the final linear projection.

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)  # Applies LN over the last dimension.

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        Expected input shape: [B, L, C]
        """
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # chunk splits into 3 chuncks along the last dimension, this gives Q, K, V
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LocalAttention2D(nn.Module):
    """
    Windowed/local attention for 2D grids using unfold & fold.
    """

    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride  # kernel_size
        self.dim = dim
        padding = 0

        self.norm = nn.LayerNorm(dim)

        self.Attention = AttentionBlock(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )

        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = rearrange(
            x, "B H W C -> B C H W"
        )  # Rearrange to [B, C, H, W] for unfolding

        # unfold into local 2D patches
        patches = self.unfold(x)  # [B, C*K*K, L] where W is the number of patches

        patches = rearrange(
            patches,
            "B (C K1 K2) L -> (B L) (K1 K2) C",
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        patches = self.norm(patches)

        # Intra-Window self.attention
        out = self.Attention(patches)  # [B*L, K*K, C]

        # Reshape back to [B, C*K*K, L]
        out = rearrange(
            out,
            "(B L) (K1 K2) C -> B (C K1 K2) L",
            B=B,
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        # Fold back to [B, C, H, W] with overlap
        fold = nn.Fold(
            output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride
        )
        out = fold(out)

        # Normalize overlapping regions
        norm = self.unfold(torch.ones((B, 1, H, W), device=x.device))  # [B, K*K, L]
        norm = fold(norm)  # [B, 1, H, W]
        out = out / norm

        # Reshape to [B, H, W, C]
        out = rearrange(out, "B C H W -> B H W C")

        return out


class Multipole_Attention2D(nn.Module):
    """
    Hierarchical local attention across multiple scales with down/up-sampling.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        local_attention_kernel_size,
        local_attention_stride,
        downsampling: Literal["avg_pool", "conv"],
        upsampling: Literal["avg_pool", "conv"],
        sampling_rate,
        heads,
        dim_head,
        dropout,
        channel_scale,
    ):
        super().__init__()

        self.levels = int(math.log(image_size, sampling_rate))  # math.log(x, base)

        channels_conv = [in_channels * (channel_scale**i) for i in range(self.levels)]

        # A shared local attention layer for all levels
        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=channels_conv[0],
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )

        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

        if upsampling == "avg_pool":
            current = image_size

            for _ in range(self.levels):
                assert (
                    current % sampling_rate == 0
                ), f"Image size not divisible by sampling_rate size at level {_}: current={current}, sampling_ratel={sampling_rate}"
                current = current // sampling_rate

            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )

        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        # x: [B, H, W, C], returns the same shape
        # Level 0
        x_in = x

        x_out = []
        x_out.append(self.Attention(x_in))

        # Levels from 1 to L
        for l in range(1, self.levels):
            x_in = self.down(x_in)
            x_out_down = self.Attention(x_in)
            x_out.append(x_out_down)

        res = x_out.pop()
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        return res


class Multipole_TransformerBlock(nn.Module):
    """
    Transformer block stacking multiple Multipole_Attention2D + FeedForward layers.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        kernel_size,
        local_attention_stride,
        downsampling,
        upsampling,
        sampling_rate,
        dim,
        depth,
        heads,
        dim_head,
        att_dropout,
        channel_scale,
        mlp_dim,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Multipole_Attention2D(                                                                                               # 微信公众号:AI缝合术
                            image_size,
                            in_channels,
                            kernel_size,
                            local_attention_stride,                                                                                               # 微信公众号:AI缝合术
                            downsampling,
                            upsampling,
                            sampling_rate,
                            heads,
                            dim_head,
                            att_dropout,
                            channel_scale,
                        ),
                        FeedForward(dim, mlp_dim),                                                                                               # 微信公众号:AI缝合术
                    ]
                )
            )

    def forward(self, x):
        """
        Expected input shape: [B, H, W, C]                                                                                               # 微信公众号:AI缝合术
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MANO(nn.Module):
    """
    Multipole Attention Neural Operator for 2D simulation data.                                                                                               # 微信公众号:AI缝合术
    """

    def __init__(
        self,
        device,
        image_size,
        dim,
        depth,
        heads,
        dim_head,
        att_dropout,
        channel_scale,
        mlp_dim,
        channels,
        emb_dropout,
        local_attention_span,
        local_attention_stride,
        att_sampling: Literal["avg_pool", "conv"],                                                                                               # 微信公众号:AI缝合术
        att_sampling_rate,
    ):
        super().__init__()
        self.in_channels = channels

        self.linear_p = nn.Linear(channels, dim)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Multipole_TransformerBlock(                                                                                               # 微信公众号:AI缝合术
            image_size,
            dim,
            local_attention_span,
            local_attention_stride,
            att_sampling,
            att_sampling,
            att_sampling_rate,
            dim,
            depth,
            heads,
            dim_head,
            att_dropout,
            channel_scale,
            mlp_dim,
        )

        # self.linear_q = nn.Linear(dim, dim // 2)
        # self.output_layer = nn.Linear(dim // 2, 1)

        self.linear_q = nn.Linear(dim, dim)
        self.output_layer = nn.Linear(dim, self.in_channels)                                                                                               # 微信公众号:AI缝合术

        self.activation = nn.Tanh()

        self.to(device)

    def forward(self, x):
        """
        Expected input shape: [B, C, H, W]
        """
        x = rearrange(x, 'B C H W -> B H W C') 
        x = self.linear_p(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.linear_q(x)
        x = self.activation(x)
        x = self.output_layer(x)
        x = rearrange(x, 'B H W C -> B C H W')  # 恢复到原始张量格式                                                                                               # 微信公众号:AI缝合术
        return x

    
if __name__ == "__main__":

    # 输入张量：形状为 [B, C, H, W]
    x = torch.randn(1, 32, 64, 64) 

    # 初始化 MANO 模型
    mano = MANO(
        device="cpu",
        image_size=64,
        dim=128,
        depth=8,
        heads=4,
        dim_head=32,
        att_dropout=0.1,
        channel_scale=2,
        mlp_dim=128,
        channels=32, 
        emb_dropout=0.1,
        local_attention_span=2,
        local_attention_stride=1,
        att_sampling="conv",
        att_sampling_rate=4,
        )


    # 前向传播测试
    output = mano(x)

    # 输出结果形状
    print(mano)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]                                                                                               # 微信公众号:AI缝合术
    print("输出张量形状:", output.shape)  # [B, H, W]                                                                                               # 微信公众号:AI缝合术
