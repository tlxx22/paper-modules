import torch
import torch.nn as nn
from torch import Tensor

# RMSNorm Transformer常用标准实现
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# block_attn_res核心函数
def block_attn_res(blocks: list[Tensor], partial_block: Tensor, proj: nn.Linear, norm: RMSNorm) -> Tensor:                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
    """
    Inter-block attention: attend over block reps + partial sum.
    blocks:
        N tensors of shape [B, T, D]: completed block representations for each previous block
    partial_block:
        [B, T, D]:  intra-block partial sum (b_n^i)
    """
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h


class BlockAttentionTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_number: int,
        block_size: int = 4,
        mlp_ratio: int = 4
    ):
        super().__init__()
        self.dim = dim
        self.layer_number = layer_number
        self.block_size = block_size

        # block attention投影与归一化层
        self.attn_res_proj = nn.Linear(dim, 1, bias=False)
        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_proj = nn.Linear(dim, 1, bias=False)
        self.mlp_res_norm = RMSNorm(dim)

        # 自注意力
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

        # MLP模块
        self.mlp_norm = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, blocks: list[Tensor], hidden_states: Tensor) -> tuple[list[Tensor], Tensor]:                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        partial_block = hidden_states
        # apply block attnres before attn
        # blocks already include token embedding
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        # if reaches block boundary, start new block
        # block_size counts ATTN + MLP; each transformer layer has 2
        if self.layer_number % (self.block_size // 2) == 0:
            blocks.append(partial_block)
            partial_block = None

        # self-attention layer
        attn_out = self.attn(self.attn_norm(h), self.attn_norm(h), self.attn_norm(h))[0]
        partial_block = partial_block + attn_out if partial_block is not None else attn_out                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

        # apply block attnres before MLP
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

        # MLP layer
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


if __name__ == "__main__":

    batch_size = 2       # B: 批次大小
    seq_len = 10         # T: 序列长度
    hidden_dim = 64      # D: 特征维度
    num_heads = 8        # 注意力头数（需被hidden_dim整除）
    block_size = 4       # 块大小（匹配原代码注释逻辑）
    layer_number = 2     # 当前层编号（可修改测试边界触发逻辑）

    # 初始化模型层
    model_layer = BlockAttentionTransformerLayer(
        dim=hidden_dim,
        num_heads=num_heads,
        layer_number=layer_number,
        block_size=block_size
    )

    print(model_layer)

    # 构造测试输入张量
    # 历史完成的blocks：2个历史块，每个形状为 [B, T, D]
    blocks = [
        torch.randn(batch_size, seq_len, hidden_dim),
        torch.randn(batch_size, seq_len, hidden_dim)
    ]
    # 当前层输入hidden_states，形状为 [B, T, D]
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

    # ========== 前向传播 ==========
    model_layer.eval()
    with torch.no_grad():
        output_blocks, output_partial = model_layer(blocks, hidden_states)

    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output_partial.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")