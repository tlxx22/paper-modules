import torch
from torch import nn
import torch.nn.functional as F

# 代码整理:微信公众号:AI缝合术
class RelationAwareAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_relation_heads,
        dropout=0.1,
        batch_first=True
    ):
        super().__init__()
        assert num_relation_heads > 0, "num_relation_heads must be greater than 0"
        assert num_relation_heads <= num_heads, "num_relation_heads cannot exceed num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_relation_heads = num_relation_heads
        self.num_normal_heads = num_heads - num_relation_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query,
        key,
        value,
        relation_weights=None,
        attn_mask=None,
        need_weights=False,
        skip_relation=False
    ):
        batch_size, num_queries = query.shape[:2]

        # project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # reshape to multi-head format
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # apply mask (if any)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        # according to skip_relation decide whether to apply relation_weights
        if not skip_relation and relation_weights is not None:
            if self.num_relation_heads == self.num_heads:
                # all heads are relation heads
                attn_weights = torch.exp(attn_weights) * relation_weights
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                # mix normal heads and relation heads
                # apply relation_weights to relation heads
                relation_attn = torch.exp(attn_weights[:, -self.num_relation_heads:]) * relation_weights
                relation_attn = relation_attn / (relation_attn.sum(dim=-1, keepdim=True) + 1e-6)
                # 代码整理:微信公众号:AI缝合术
                # normal heads使用普通的softmax
                normal_attn = F.softmax(attn_weights[:, :self.num_normal_heads], dim=-1)

                # combine two attention weights
                attn_weights = torch.cat([normal_attn, relation_attn], dim=1)
        else:
            # all heads use normal softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            # 代码整理:微信公众号:AI缝合术
        attn_weights = self.dropout(attn_weights)

        # calculate output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)
        output = self.norm(output)

        return output, None

if __name__ == "__main__":
    batch_size = 4
    seq_length = 10
    embed_dim = 64
    num_heads = 8
    num_relation_heads = 2
    dropout = 0.1

    # 创建随机输入张量 (query, key, value)
    query = torch.randn(batch_size, seq_length, embed_dim)
    key = torch.randn(batch_size, seq_length, embed_dim)
    value = torch.randn(batch_size, seq_length, embed_dim)

    # 创建 RelationAwareAttention 实例
    relation_attention = RelationAwareAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_relation_heads=num_relation_heads,
        dropout=dropout
    )
    print(relation_attention)

    # 假设没有额外的 mask 或 relation_weights
    attn_mask = None
    relation_weights = None

    # 前向传播
    output, _ = relation_attention(
        query,
        key,
        value,
        relation_weights=relation_weights,
        attn_mask=attn_mask
    )

    # 打印输入和输出的形状
    print(f"Input shapes: query={query.shape}, key={key.shape}, value={value.shape}")
    print(f"Output shape: {output.shape}")