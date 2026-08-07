import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def pure_infsa_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    rho: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute pure InfSA attention scores.
    Builds the single-hop attention matrix:
        A = ρ · Norm(ReLU(QK^T / d))
    where Norm(·) is Frobenius normalization ensuring ||A||_F < 1 for
    guaranteed convergence of the infinite series across layers.
    Args:
        q: Query tensor of shape ``(B, H, N, D)`` or ``(B*H, N, D)``.
        k: Key tensor of shape ``(B, H, N, D)`` or ``(B*H, N, D)``.
        rho: Decay parameter in (0, 1). Controls the spectral radius of A.                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        eps: Small constant for numerical stability.
    Returns:
        Attention score matrix of shape ``(B, H, N, N)`` or ``(B*H, N, N)``.
    Example::
        >>> q = torch.randn(2, 8, 197, 64)  # (B, H, N, D)
        >>> k = torch.randn(2, 8, 197, 64)
        >>> scores = pure_infsa_scores(q, k, rho=0.9)
        >>> scores.shape
        torch.Size([2, 8, 197, 197])
    """
    D = q.shape[-1]
    scale = math.sqrt(1.0 / D)
    q_scaled = q * scale
    k_scaled = k * scale

    # Affinity matrix A = ReLU(QK^T / d)
    A = torch.matmul(q_scaled, k_scaled.transpose(-2, -1))  # (..., N, N)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
    A = torch.relu(A)

    # Frobenius normalization over the last two dims
    # Flatten last two dims for norm computation, then reshape
    orig_shape = A.shape
    A_flat = A.flatten(-2)  # (..., N*N)
    frob_norm = torch.norm(A_flat, p=2, dim=-1, keepdim=True)  # (..., 1)
    A_flat = A_flat / (frob_norm + eps)
    A = A_flat.view(orig_shape)

    return rho * A


def linear_infsa_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    rho: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute linear InfSA attention scores via eigenvector approximation.                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
    Approximates the principal eigenvector of the implicit attention matrix
    A = ReLU(QK^T) using a soft query pooling + single matrix-vector product,
    yielding O(N·D) complexity instead of O(N²).
    The algorithm:
        1. Compute energy-based query summary: q̄ = Σ_t w_t · q_t
        2. Compute attention: a = normalize(ReLU(q̄ · K^T / d))
        3. Return: ρ · a  (shape: per-token importance scores)
    The returned scores are per-token importance weights (not a full N×N
    matrix), which are used to compute a weighted sum of values.
    Args:
        q: Query tensor of shape ``(B, H, N, D)`` or ``(B*H, N, D)``.
        k: Key tensor of shape ``(B, H, N, D)`` or ``(B*H, N, D)``.
        rho: Decay parameter in (0, 1).
        eps: Small constant for numerical stability.
    Returns:
        Attention importance vector of shape ``(..., N, 1)``, broadcastable
        for element-wise multiplication with values.
    Example::
        >>> q = torch.randn(2, 8, 197, 64)  # (B, H, N, D)
        >>> k = torch.randn(2, 8, 197, 64)
        >>> scores = linear_infsa_scores(q, k, rho=0.9)
        >>> scores.shape
        torch.Size([2, 8, 197, 1])
    """
    D = q.shape[-1]
    scale = math.sqrt(1.0 / D)
    q_scaled = q * scale
    k_scaled = k * scale

    # Step 1: Soft energy-based query pooling → q̄
    energy = torch.relu(q_scaled.norm(p=2, dim=-1))  # (..., N)
    energy_sum = energy.sum(dim=-1, keepdim=True) + eps  # (..., 1)
    weights = energy / energy_sum  # (..., N)
    q_bar = torch.einsum("...n,...nd->...d", weights, q_scaled)  # (..., D)

    # Step 2: Compute attention scores a = normalize(ReLU(q̄ · K^T))
    a = torch.relu(torch.einsum("...d,...nd->...n", q_bar, k_scaled))  # (..., N)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
    a = a / (a.sum(dim=-1, keepdim=True) + eps)

    # Step 3: Apply decay and return as column vector for broadcasting
    return (rho * a).unsqueeze(-1)  # (..., N, 1)

def infsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variant: str = "pure_infsa",
    rho: float = 0.95,
    dropout_p: float = 0.0,
    training: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute InfSA attention output from pre-projected Q, K, V.
    This is the main functional API. It computes attention scores using the
    specified InfSA variant and applies them to values.
    For ``pure_infsa``:
        output = A @ V  where A = ρ · Norm(ReLU(QK^T / d))
    For ``linear_infsa``:
        output = broadcast(scores) * V  (element-wise, then summed)
        where scores are per-token importance weights.
    Args:
        q: Query tensor ``(B, H, N, D)`` or ``(B*H, N, D)``.
        k: Key tensor ``(B, H, N, D)`` or ``(B*H, N, D)``.
        v: Value tensor ``(B, H, N, D)`` or ``(B*H, N, D)``.
        variant: Which InfSA variant: ``"pure_infsa"`` or ``"linear_infsa"``.
        rho: Decay parameter in (0, 1).
        dropout_p: Dropout probability on attention scores (training only).
        training: Whether the model is in training mode.
        eps: Numerical stability constant.
    Returns:
        Attention output tensor, same shape as ``v``.
    Example::
        >>> q = torch.randn(2, 8, 197, 64)
        >>> k = torch.randn(2, 8, 197, 64)
        >>> v = torch.randn(2, 8, 197, 64)
        >>> out = infsa_attention(q, k, v, variant="pure_infsa", rho=0.9)
        >>> out.shape
        torch.Size([2, 8, 197, 64])
    """
    if variant == "pure_infsa":
        scores = pure_infsa_scores(q, k, rho=rho, eps=eps)  # (..., N, N)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        if dropout_p > 0.0 and training:
            scores = F.dropout(scores, p=dropout_p)
        return torch.matmul(scores, v)

    elif variant == "linear_infsa":
        scores = linear_infsa_scores(q, k, rho=rho, eps=eps)  # (..., N, 1)
        if dropout_p > 0.0 and training:
            scores = F.dropout(scores, p=dropout_p)
        # Weighted sum: each token gets the globally-pooled representation
        pooled = (scores * v).sum(dim=-2, keepdim=True)  # (..., 1, D)
        N = v.shape[-2]
        return pooled.expand_as(v)  # (..., N, D) — broadcast to all positions

    else:
        raise ValueError(
            f"Unknown InfSA variant: '{variant}'. "
            f"Choose from: 'pure_infsa', 'linear_infsa'."
        )


class InfSAAttention(nn.Module):
    """Multi-Head Infinite Self-Attention module.
    Drop-in replacement for ``nn.MultiheadAttention`` that uses InfSA
    instead of softmax attention.
    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of parallel attention heads.
        variant: InfSA variant — ``"pure_infsa"`` or ``"linear_infsa"``.
        dropout: Dropout probability on attention scores. Default: 0.0.
        bias: If True, add bias to input/output projections. Default: True.
        batch_first: If True, input/output shape is ``(B, N, E)``.
            If False, shape is ``(N, B, E)``. Default: True.
        rho_init: Initial value for the learnable ρ parameter. Default: 0.95.
            Internally stored as a logit: ρ = sigmoid(rho_logit).
        rho_trainable: If True, ρ is a learnable parameter. Default: True.
    Shape:
        - query: ``(B, N, E)`` if batch_first else ``(N, B, E)``
        - key: ``(B, S, E)`` if batch_first else ``(S, B, E)``
        - value: ``(B, S, E)`` if batch_first else ``(S, B, E)``
        - Output: ``(B, N, E)`` if batch_first else ``(N, B, E)``
    Example::
        >>> attn = InfSAAttention(embed_dim=512, num_heads=8, variant="pure_infsa")
        >>> x = torch.randn(4, 197, 512)  # (B, N, E)
        >>> out, scores = attn(x, x, x)
        >>> out.shape
        torch.Size([4, 197, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        variant: str = "pure_infsa",
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        rho_init: float = 0.95,
        rho_trainable: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__()
        if variant not in ("pure_infsa", "linear_infsa"):
            raise ValueError(
                f"Unknown variant '{variant}'. Choose 'pure_infsa' or 'linear_infsa'."
            )
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be > 0, got {embed_dim} and {num_heads}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.variant = variant
        self.dropout = dropout
        self.batch_first = batch_first

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Trainable rho via logit reparameterization
        rho_logit = math.log(rho_init / (1.0 - rho_init))
        if rho_trainable:
            self.rho_logit = nn.Parameter(torch.tensor(rho_logit))
        else:
            self.register_buffer("rho_logit", torch.tensor(rho_logit))

        self._reset_parameters()

    @property
    def rho(self) -> float:
        """Current ρ value (sigmoid of the stored logit)."""
        return torch.sigmoid(self.rho_logit).item()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: Optional[Tensor] = None,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.
        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            key_padding_mask: Not used by InfSA (accepted for API compatibility).
            need_weights: If True, return attention scores.
            attn_mask: Not used by InfSA (accepted for API compatibility).
            average_attn_weights: If True, average weights across heads.
            is_causal: Not used by InfSA (accepted for API compatibility).
        Returns:
            Tuple of (output, attention_weights). Weights are None if
            need_weights is False.
        """
        # HuggingFace compatibility: accept hidden_states as self-attention input
        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is not None and query is None:
            query = key = value = hidden_states
        # 常规自注意力：仅传query时，key/value默认等于query
        if query is not None:
            if key is None:
                key = query
            if value is None:
                value = key

        # 安全校验：确保query/key/value不为None
        if query is None or key is None or value is None:
            raise ValueError("query, key, value must not be None for InfSA attention")

        # Handle seq-first format
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, N, _ = query.shape
        S = key.shape[1]
        H = self.num_heads
        D = self.head_dim

        # Project Q, K, V
        q = self.q_proj(query).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(key).view(B, S, H, D).transpose(1, 2)    # (B, H, S, D)
        v = self.v_proj(value).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)

        # Get rho from learnable parameter
        rho_val = torch.sigmoid(self.rho_logit).item()

        # 保存注意力权重（如果需要返回）
        attn_weights = None
        # Compute InfSA attention
        output = infsa_attention(
            q, k, v,
            variant=self.variant,
            rho=rho_val,
            dropout_p=self.dropout if self.training else 0.0,
            training=self.training,
        )  # (B, H, N, D)

        # Reshape back: (B, H, N, D) → (B, N, E)
        output = output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

        # Output projection
        output = self.out_proj(output)

        # Handle seq-first format
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(4, 197, 512).to(device)
    # model = InfSAAttention(embed_dim=512, num_heads=8, variant="pure_infsa").to(device)
    model = InfSAAttention(embed_dim=512, num_heads=8, variant="linear_infsa").to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")