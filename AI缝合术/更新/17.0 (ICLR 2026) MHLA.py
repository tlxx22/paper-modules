import torch
from torch import nn
from einops import rearrange
import numpy as np
import math

class BlockDistanceConv(nn.Module):
    """
    A 1x1 convolution layer with weights based on spatial distances between blocks.
    """
    def __init__(
        self, num_patches_per_side=16, patch_group_size=16, transform="linear", local_thres=1.5, exp_sigma=3                                                                                                                             # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
    ):
        super().__init__()

        self.num_patches_per_side = num_patches_per_side
        self.patch_group_size = patch_group_size
        self.transform = transform
        self.local_thres = local_thres
        self.exp_sigma = exp_sigma

        # Calculate number of blocks per side
        patches_per_block_side = int(np.sqrt(patch_group_size))
        self.blocks_per_side = num_patches_per_side // patches_per_block_side
        self.total_blocks = self.blocks_per_side**2

        # Create distance matrix
        distance_matrix = self._compute_block_distances()

        # Apply transformation
        weight_matrix = self._apply_transform(distance_matrix)

        # Create 1x1 conv layer
        self.conv = nn.Conv2d(
            in_channels=self.total_blocks,
            out_channels=self.total_blocks,
            kernel_size=1,
            bias=False,
        )

        # Set the weights as fixed (no gradient)
        with torch.no_grad():
            self.conv.weight.data = weight_matrix.unsqueeze(-1).unsqueeze(-1)

        # Freeze the weights
        self.conv.weight.requires_grad = False

    def _compute_block_distances(self):
        """Compute Euclidean distances between all block centers."""
        block_centers = []
        for i in range(self.blocks_per_side):
            for j in range(self.blocks_per_side):
                center_x = i + 0.5
                center_y = j + 0.5
                block_centers.append([center_x, center_y])

        block_centers = torch.tensor(block_centers, dtype=torch.float32)

        # Compute pairwise distances
        distance_matrix = torch.zeros(self.total_blocks, self.total_blocks)
        for i in range(self.total_blocks):
            for j in range(self.total_blocks):
                dist = torch.norm(block_centers[i] - block_centers[j], p=2)
                distance_matrix[i, j] = dist

        return distance_matrix

    def _apply_transform(self, distance_matrix):
        """Apply transformation function to distance matrix."""
        if self.transform == "linear":
            max_dist = distance_matrix.max()
            mat = 1.0 - (distance_matrix / max_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "cos":
            max_dist = distance_matrix.max()
            normalized_dist = distance_matrix / max_dist * math.pi / 4
            mat = torch.cos(normalized_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "exp":
            sigma = distance_matrix.max() / 3
            mat = torch.exp(-distance_matrix / self.exp_sigma)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "gaussian":
            sigma = distance_matrix.max() / 3
            return torch.exp(-(distance_matrix**2) / (2 * sigma**2))
        
        elif self.transform == "local":
            thres = getattr(self, "local_thres", 1.5)
            mat = (distance_matrix <= thres).float()
            mat = mat / mat.sum(dim=0, keepdim=True)
            return mat

        else:
            raise ValueError(f"Unknown transform: {self.transform}")

    def forward(self, x):
        """
        Forward pass through the distance-based convolution.
        Args:
            x: Input tensor of shape (B, total_blocks, H, W)
        Returns:
            Output tensor of shape (B, total_blocks, H, W)
        """
        return self.conv(x)

    def get_weight_matrix(self):
        """Return the weight matrix for inspection."""
        return self.conv.weight.data.squeeze(-1).squeeze(-1)

class MHLA(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=None,
        dropout=0.1,
        fixed_weight_value=None,
        qk_norm=False,
        transform="linear",
        **kwargs,
    ):
        super(MHLA, self).__init__()

        if dim_head is None:
            dim_head = dim // heads
        inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_dim = dim_head
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.qkv_bias = kwargs.get("qkv_bias", False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=self.qkv_bias)

        self.q_norm = nn.RMSNorm(dim_head) if qk_norm else nn.Identity()  # 修正norm维度（应为head_dim）                                                                                                                             # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.k_norm = nn.RMSNorm(dim_head) if qk_norm else nn.Identity()
        
        self.lepe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.block_size = kwargs.get("block_size", 49)
        self.block_len = int(math.sqrt(self.block_size))
        self.embed_len = kwargs.get("embed_len", 196)
        self.num_pieces = self.embed_len // self.block_size
        self.pieces_len = int(math.sqrt(self.num_pieces))
        
        local_thres = kwargs.get("local_thres", 1.5)
        exp_sigma = kwargs.get("exp_sigma", 3)

        self.piece_attn = BlockDistanceConv(
            num_patches_per_side=int(math.sqrt(self.embed_len)),
            patch_group_size=self.block_size,
            transform=transform,
            local_thres=local_thres,
            exp_sigma=exp_sigma,
        )

        self.eps = kwargs.get("eps", 1e-6)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))                                                                                                                             # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

        if fixed_weight_value is not None:
            self._init_weights_with_fixed_value(fixed_weight_value)

    def _init_weights_with_fixed_value(self, value):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)
        nn.init.constant_(self.to_qkv.weight, value)
        for module in self.to_out:
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, value)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def init_to_value(model, value=1.0):
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)
        return model

    # 移除torch.compile装饰器
    def _process_qkv_impl(self, q, k, v, B, N, H, D):
        q = self.q_norm(q)  # [B, H, N, D]
        k = self.k_norm(k)  # [B, H, N, D]

        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        q, k, v = map(
            lambda t: rearrange(t, "b n w (h d) -> (b h) n w d", h=H, d=D),                                                                                                                             # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            (q, k, v)
        )
        k = k.transpose(-2, -1) 
        return q, k, v

    # 移除torch.compile装饰器
    def _mlp_lepe(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        try:
            lepe = self.lepe(rearrange(
                v, 'b (h w) (p1 p2) d -> b d (h p1) (w p2)',                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
                h=self.pieces_len, w=self.pieces_len, 
                p1=self.block_len, p2=self.block_len
            ))
            lepe = rearrange(
                lepe, 'b d (h p1) (w p2) -> b (h w) (p1 p2) d', 
                h=self.pieces_len, w=self.pieces_len, 
                p1=self.block_len, p2=self.block_len
            )
        except Exception as e:
            lepe = self.lepe(rearrange(v, 'b n w d -> b d n w'))
            lepe = rearrange(lepe, 'b d n w -> b n w d')
        return q, k, v, lepe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        B, N, W, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q, k, v, lepe = self._mlp_lepe(x)
        q, k, v = self._process_qkv_impl(q, k, v, B, N, H, D)

        kv = torch.matmul(k, v)  # [B*H, num_pieces, D, D]
        kv = self.piece_attn(kv)  # [B*H, num_pieces, D, D]

        k_sum = k.sum(dim=-1, keepdim=True)  # [B*H, num_pieces, D, 1]
        normalizer = self.piece_attn(torch.matmul(q, k_sum)) + self.eps  # [B*H, num_pieces, block_size, 1]                                                                                                                             # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

        out = torch.matmul(q, kv) / normalizer  # [B*H, num_pieces, block_size, D]
        out = rearrange(out, "(b h) n w d -> b n w (h d)", b=B, h=self.num_heads)
        out = out + lepe

        return self.to_out(out)

# 使用示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输入张量形状：MHLA期望 (B, N, W, C) 非 (B, C, H, W)
    # 其中：
    # - B: batch_size=2
    # - N: num_pieces=4 (对应embed_len=196, block_size=49 → 196/49=4)
    # - W: block_size=49 (每个block的元素数)
    # - C: dim=32 (通道数/特征维度)
    input_tensor = torch.randn(2, 4, 49, 32).to(device)
    
    # 初始化MHLA
    mhla = MHLA(
        dim=32, 
        heads=8,  
        qkv_bias=False,
        block_size=49,
        embed_len=196,
        transform="linear"
    ).to(device)
    
    print(mhla)
    output_tensor = mhla(input_tensor)
    
    # 打印输入输出形状
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")