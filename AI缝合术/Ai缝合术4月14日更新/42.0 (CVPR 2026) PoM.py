import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import os

"""Triton-accelerated causal-mask kernel for PoM (Polynomial Mixer).

For the causal mask, output position m aggregates context positions 0..m:

  out[b, m, d] = (1/(m+1)) * sum_{n=0}^{m} poly(act(x[b, n, d]))

where poly(h) = sum_{k=0}^{K-1} coeff[d, k] * h^(k+1).

This is O(N) in compute and memory (same as the no-mask case) but produces
(B, N, D) instead of (B, 1, D), enabling autoregressive sequence mixing.

Forward kernel
--------------
Grid: (B, ceil(D/BLOCK_D))
Each program streams n = 0..N-1, accumulates a running polynomial sum `acc`,
and writes `acc / (n+1)` to out[b, n, :] at each step.
Coefficients are preloaded into registers once per program; X is streamed
with evict_first to avoid L2 pollution.

Backward kernel
---------------
Grid: (B, ceil(D/BLOCK_D))
Gradient of the loss w.r.t. x[b, n0, d]:
  grad_x[b, n0, d] = suffix_w[b, n0, d] * d_poly/d_h * d_act/d_x

where suffix_w[b, n, d] = sum_{m=n}^{N-1} go[b, m, d] / (m+1) is the
suffix-weighted sum of upstream gradients.

The kernel iterates n = N-1 .. 0 (via i = 0..N-1, n = N-1-i):
  - Accumulates suffix_w in reverse (one load of go per step).
  - Loads x[b, n, d], computes activation, d_act, grad_h.
  - Writes grad_x[b, n, d] = suffix_w * grad_h * d_act.
  - Accumulates partial grad_coeff sums a[k] += suffix_w * h^(k+1).
After the loop, atomic_add flushes a[k] into GC (reducing across B).

Exposed API
-----------
TRITON_CAUSAL_AVAILABLE : bool
poly_agg_causal_triton(x, coeff, k) -> Tensor  (B, N, D)
"""

try:
    import triton
    import triton.language as tl
    TRITON_CAUSAL_AVAILABLE = not os.environ.get("POM_DISABLE_TRITON", "")
except ImportError:
    TRITON_CAUSAL_AVAILABLE = False


if TRITON_CAUSAL_AVAILABLE:

    # -------------------------------------------------------------------------
    # BLOCK_D heuristics – same policy as the no-mask kernel.
    # -------------------------------------------------------------------------

    def _fwd_block_d(D: int) -> int:
        return min(256, 1 << (D - 1).bit_length())

    def _bwd_block_d(D: int) -> int:
        return min(128, 1 << (D - 1).bit_length())

    # -------------------------------------------------------------------------
    # Forward kernel
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_causal_fwd(
        X_ptr,          # (B, N, D) – input, any dtype
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        O_ptr,          # (B, N, D) – output, fp32
        N,              # sequence length  (runtime)
        D,              # feature dim      (runtime)
        stride_xb,      # X.stride(0)
        stride_xn,      # X.stride(1)
        stride_ob,      # O.stride(0)
        stride_on,      # O.stride(1)
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload coeff[d_off, 0..K-1] into registers for the entire N loop.
        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for n in range(N):
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # pom_activation: clamp(leaky_relu(x, 0.01), -0.1, 6.0)
            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            poly = tl.zeros((BLOCK_D,), dtype=tl.float32)
            hp   = h
            if K > 0:
                poly += c0 * hp
            if K > 1:
                hp *= h; poly += c1 * hp
            if K > 2:
                hp *= h; poly += c2 * hp
            if K > 3:
                hp *= h; poly += c3 * hp
            if K > 4:
                hp *= h; poly += c4 * hp
            if K > 5:
                hp *= h; poly += c5 * hp
            if K > 6:
                hp *= h; poly += c6 * hp
            if K > 7:
                hp *= h; poly += c7 * hp
            if K > 8:
                hp *= h; poly += c8 * hp
            if K > 9:
                hp *= h; poly += c9 * hp
            if K > 10:
                hp *= h; poly += c10 * hp
            if K > 11:
                hp *= h; poly += c11 * hp
            if K > 12:
                hp *= h; poly += c12 * hp
            if K > 13:
                hp *= h; poly += c13 * hp
            if K > 14:
                hp *= h; poly += c14 * hp
            if K > 15:
                hp *= h; poly += c15 * hp

            acc += poly
            # Causal mean: divide running sum by (n+1) — number of tokens seen so far.
            tl.store(
                O_ptr + b * stride_ob + n * stride_on + d_off,
                acc / (n + 1),
                mask=dmask,
            )

    # -------------------------------------------------------------------------
    # Backward kernel
    #
    # Iterates n = N-1 .. 0 (via i = 0 .. N-1, n = N-1-i) to accumulate the
    # suffix-weighted upstream gradient:
    #   suffix_w[n] = sum_{m=n}^{N-1} go[b, m, d] / (m+1)
    #
    # At each step:
    #   grad_x[b, n, d] = suffix_w[n] * grad_h(h[n]) * d_act(x[n])
    #   a[k]           += suffix_w[n] * h[n]^(k+1)   (→ grad_coeff[d, k])
    #
    # Power sharing: same hp / hph trick as the no-mask backward.
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_causal_bwd(
        GO_ptr,         # (B, N, D) – upstream gradient, fp32
        X_ptr,          # (B, N, D) – saved input
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        GX_ptr,         # (B, N, D) – grad w.r.t. X, fp32
        GC_ptr,         # (D, K)    – grad w.r.t. coeff, fp32 (zero-init, atomic)
        B, N, D,
        stride_xb, stride_xn,
        stride_gob, stride_gon,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload coefficients (constant over the N loop).
        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        # grad_coeff accumulators (reduced across N, then atomic-added across B).
        a0  = tl.zeros((BLOCK_D,), tl.float32)
        a1  = tl.zeros((BLOCK_D,), tl.float32)
        a2  = tl.zeros((BLOCK_D,), tl.float32)
        a3  = tl.zeros((BLOCK_D,), tl.float32)
        a4  = tl.zeros((BLOCK_D,), tl.float32)
        a5  = tl.zeros((BLOCK_D,), tl.float32)
        a6  = tl.zeros((BLOCK_D,), tl.float32)
        a7  = tl.zeros((BLOCK_D,), tl.float32)
        a8  = tl.zeros((BLOCK_D,), tl.float32)
        a9  = tl.zeros((BLOCK_D,), tl.float32)
        a10 = tl.zeros((BLOCK_D,), tl.float32)
        a11 = tl.zeros((BLOCK_D,), tl.float32)
        a12 = tl.zeros((BLOCK_D,), tl.float32)
        a13 = tl.zeros((BLOCK_D,), tl.float32)
        a14 = tl.zeros((BLOCK_D,), tl.float32)
        a15 = tl.zeros((BLOCK_D,), tl.float32)

        # Running suffix-weighted upstream gradient (accumulated in reverse).
        suffix_w = tl.zeros((BLOCK_D,), tl.float32)

        for i in range(N):
            n = N - 1 - i   # reverse: n goes N-1, N-2, ..., 0

            # Accumulate suffix_w += go[b, n, d] / (n+1)
            go_n = tl.load(
                GO_ptr + b * stride_gob + n * stride_gon + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            suffix_w += go_n / (n + 1)

            # Load x[b, n, d] for activation / derivative computation.
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            # Derivative of pom_activation:
            #   1    for  0 <= x <= 6
            #   0.01 for -10 <= x < 0
            #   0    otherwise (clamped)
            d_act = tl.where(
                (x >= 0.0) & (x <= 6.0), 1.0,
                tl.where((x < 0.0) & (x >= -10.0), 0.01, 0.0),
            )

            # Power-sharing loop – hp advances h^0 → h^1 → ... → h^(K-1).
            # At step k:
            #   grad_h  uses hp (= h^k)           → polynomial derivative
            #   a[k]    uses hph (= h^(k+1))       → coeff gradient
            # The same hph product serves both.
            grad_h = tl.zeros((BLOCK_D,), tl.float32)
            hp     = tl.full((BLOCK_D,), 1.0, tl.float32)   # h^0

            if K > 0:
                hph = hp * h
                grad_h += c0 * hp;          a0  += suffix_w * hph;  hp = hph
            if K > 1:
                hph = hp * h
                grad_h += c1 * 2.0 * hp;   a1  += suffix_w * hph;  hp = hph
            if K > 2:
                hph = hp * h
                grad_h += c2 * 3.0 * hp;   a2  += suffix_w * hph;  hp = hph
            if K > 3:
                hph = hp * h
                grad_h += c3 * 4.0 * hp;   a3  += suffix_w * hph;  hp = hph
            if K > 4:
                hph = hp * h
                grad_h += c4 * 5.0 * hp;   a4  += suffix_w * hph;  hp = hph
            if K > 5:
                hph = hp * h
                grad_h += c5 * 6.0 * hp;   a5  += suffix_w * hph;  hp = hph
            if K > 6:
                hph = hp * h
                grad_h += c6 * 7.0 * hp;   a6  += suffix_w * hph;  hp = hph
            if K > 7:
                hph = hp * h
                grad_h += c7 * 8.0 * hp;   a7  += suffix_w * hph;  hp = hph
            if K > 8:
                hph = hp * h
                grad_h += c8 * 9.0 * hp;   a8  += suffix_w * hph;  hp = hph
            if K > 9:
                hph = hp * h
                grad_h += c9 * 10.0 * hp;  a9  += suffix_w * hph;  hp = hph
            if K > 10:
                hph = hp * h
                grad_h += c10 * 11.0 * hp; a10 += suffix_w * hph;  hp = hph
            if K > 11:
                hph = hp * h
                grad_h += c11 * 12.0 * hp; a11 += suffix_w * hph;  hp = hph
            if K > 12:
                hph = hp * h
                grad_h += c12 * 13.0 * hp; a12 += suffix_w * hph;  hp = hph
            if K > 13:
                hph = hp * h
                grad_h += c13 * 14.0 * hp; a13 += suffix_w * hph;  hp = hph
            if K > 14:
                hph = hp * h
                grad_h += c14 * 15.0 * hp; a14 += suffix_w * hph;  hp = hph
            if K > 15:
                hph = hp * h
                grad_h += c15 * 16.0 * hp; a15 += suffix_w * hph;  hp = hph

            tl.store(
                GX_ptr + b * stride_xb + n * stride_xn + d_off,
                suffix_w * grad_h * d_act,
                mask=dmask,
            )

        # Atomic-add partial grad_coeff sums into GC (reduces across B).
        # GC is zero-initialised by the Python wrapper before this kernel runs.
        if K > 0:  tl.atomic_add(GC_ptr + d_off * K + 0,  a0,  mask=dmask)
        if K > 1:  tl.atomic_add(GC_ptr + d_off * K + 1,  a1,  mask=dmask)
        if K > 2:  tl.atomic_add(GC_ptr + d_off * K + 2,  a2,  mask=dmask)
        if K > 3:  tl.atomic_add(GC_ptr + d_off * K + 3,  a3,  mask=dmask)
        if K > 4:  tl.atomic_add(GC_ptr + d_off * K + 4,  a4,  mask=dmask)
        if K > 5:  tl.atomic_add(GC_ptr + d_off * K + 5,  a5,  mask=dmask)
        if K > 6:  tl.atomic_add(GC_ptr + d_off * K + 6,  a6,  mask=dmask)
        if K > 7:  tl.atomic_add(GC_ptr + d_off * K + 7,  a7,  mask=dmask)
        if K > 8:  tl.atomic_add(GC_ptr + d_off * K + 8,  a8,  mask=dmask)
        if K > 9:  tl.atomic_add(GC_ptr + d_off * K + 9,  a9,  mask=dmask)
        if K > 10: tl.atomic_add(GC_ptr + d_off * K + 10, a10, mask=dmask)
        if K > 11: tl.atomic_add(GC_ptr + d_off * K + 11, a11, mask=dmask)
        if K > 12: tl.atomic_add(GC_ptr + d_off * K + 12, a12, mask=dmask)
        if K > 13: tl.atomic_add(GC_ptr + d_off * K + 13, a13, mask=dmask)
        if K > 14: tl.atomic_add(GC_ptr + d_off * K + 14, a14, mask=dmask)
        if K > 15: tl.atomic_add(GC_ptr + d_off * K + 15, a15, mask=dmask)

    # -------------------------------------------------------------------------
    # autograd.Function wrapper
    # -------------------------------------------------------------------------

    class _PolyAggCausal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, coeff: torch.Tensor, k: int):
            B, N, D = x.shape
            coeff_c = coeff.float().contiguous()
            out     = torch.empty(B, N, D, dtype=torch.float32, device=x.device)

            BLOCK_D = _fwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_causal_fwd[grid](
                x, coeff_c, out,
                N, D,
                x.stride(0), x.stride(1),
                out.stride(0), out.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            ctx.save_for_backward(x, coeff_c)
            ctx.k = k
            return out.to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x_saved, coeff = ctx.saved_tensors
            k              = ctx.k
            x       = x_saved.contiguous()
            B, N, D = x.shape

            go = grad_out.float().contiguous()  # (B, N, D)

            grad_x_buf = torch.empty(B, N, D, dtype=torch.float32, device=x.device)
            # GC must be zero-initialised: kernel accumulates via atomic_add.
            grad_c = torch.zeros(D, k, dtype=torch.float32, device=x.device)

            BLOCK_D = _bwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_causal_bwd[grid](
                go, x, coeff, grad_x_buf, grad_c,
                B, N, D,
                x.stride(0), x.stride(1),
                go.stride(0), go.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            return grad_x_buf.to(x_saved.dtype), grad_c.to(coeff.dtype), None

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def poly_agg_causal_triton(
        x: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Fused causal polynomial aggregation.

        For each position m, computes the polynomial mean over tokens 0..m:
          out[b, m, d] = (1/(m+1)) * sum_{n=0}^{m} poly(act(x[b, n, d]))

        Args:
            x     : (B, N, D) input tensor
            coeff : (D, K)    polynomial coefficients
            k     : polynomial degree (≤ 16)

        Returns:
            (B, N, D) causal-mean-aggregated polynomial features
        """
        if k > 16:
            raise NotImplementedError(
                f"Triton causal kernel supports k ≤ 16 (got {k}). "
                "Extend the c0..c15 / a0..a15 pattern or use the PyTorch fallback."
            )
        return _PolyAggCausal.apply(x, coeff, k)
    
"""Triton-accelerated kernels for PoM (Polynomial Mixer).

The hot path in polynomial_aggregation_ (no-mask branch) is:
  1. pom_activation  (element-wise)
  2. polynomial weighted sum  producing (B, N, D)
  3. mean over N  producing (B, 1, D)

PyTorch materialises a (B, N, D, K) intermediate before the mean.
The kernels here fuse all three steps into a single pass over the
input, eliminating the intermediate and halving memory bandwidth.

Forward kernel optimisations
-----------------------------
- coeff[d, k] is preloaded once per (b, d_blk) program and held in
  registers for the entire N loop.
- X is streamed with eviction_policy="evict_first" to avoid polluting L2.
- cache_modifier=".ca" on coeff loads provides an L1 fallback on spill.
- K is tl.constexpr: the polynomial loop is fully unrolled.
- BLOCK_D is chosen by a Python heuristic (no autotuning, no file I/O).

Backward kernel optimisations
-------------------------------
- A single fused kernel computes both grad_x and grad_coeff in one pass
  over X, so X is read only once (vs. twice with separate kernels).
- go (upstream gradient) and coeff are preloaded into registers.
- The hp accumulator is shared: the same power vector serves both the
  grad_h computation (for grad_x) and the coeff accumulator.
- grad_coeff is accumulated per-batch-element and written to the output
  buffer via tl.atomic_add, avoiding a separate reduction pass.
- grad_x is stored as fp32; cast to input dtype in Python wrapper.

Exposed API
-----------
TRITON_AVAILABLE : bool
poly_agg_mean_triton(x, coeff, k) -> Tensor  (B, 1, D)
"""

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = not os.environ.get("POM_DISABLE_TRITON", "")
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    # ---------------------------------------------------------------------------
    # BLOCK_D heuristics — computed in Python, no autotuning, no file I/O.
    #
    # Forward: up to 256 (streaming kernel, light register pressure).
    # Backward: up to 128 (heavier: coeff + grad_coeff accumulators in regs).
    # Both: next power of 2 >= D, clamped to the respective cap.
    # ---------------------------------------------------------------------------

    def _fwd_block_d(D: int) -> int:
        return min(256, 1 << (D - 1).bit_length())

    def _bwd_block_d(D: int) -> int:
        return min(128, 1 << (D - 1).bit_length())

    # ---------------------------------------------------------------------------
    # Forward kernel
    # ---------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_mean_fwd(
        X_ptr,          # (B, N, D) – input, any dtype
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        O_ptr,          # (B, D)    – output, fp32
        N,              # sequence length (runtime)
        D,              # feature dim    (runtime)
        stride_xb,      # X.stride(0) = N*D
        stride_xn,      # X.stride(1) = D
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload coeff[d_off, 0..K-1] once per program; held in registers.
        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for n in range(N):
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            poly = tl.zeros((BLOCK_D,), dtype=tl.float32)
            hp   = h
            if K > 0:
                poly += c0 * hp
            if K > 1:
                hp *= h; poly += c1 * hp
            if K > 2:
                hp *= h; poly += c2 * hp
            if K > 3:
                hp *= h; poly += c3 * hp
            if K > 4:
                hp *= h; poly += c4 * hp
            if K > 5:
                hp *= h; poly += c5 * hp
            if K > 6:
                hp *= h; poly += c6 * hp
            if K > 7:
                hp *= h; poly += c7 * hp
            if K > 8:
                hp *= h; poly += c8 * hp
            if K > 9:
                hp *= h; poly += c9 * hp
            if K > 10:
                hp *= h; poly += c10 * hp
            if K > 11:
                hp *= h; poly += c11 * hp
            if K > 12:
                hp *= h; poly += c12 * hp
            if K > 13:
                hp *= h; poly += c13 * hp
            if K > 14:
                hp *= h; poly += c14 * hp
            if K > 15:
                hp *= h; poly += c15 * hp

            acc += poly

        tl.store(O_ptr + b * D + d_off, acc / N, mask=dmask)

    # ---------------------------------------------------------------------------
    # Fused backward kernel – grad_x and grad_coeff in one X pass
    #
    # Grid: (B, ceil(D/BLOCK_D))
    #   Each program handles one batch element and one D tile.
    #   After the N loop it atomic_adds its partial grad_coeff sums into GC,
    #   reducing across B without a separate reduction kernel.
    #
    # Power sharing:
    #   hp advances as h^0 → h^1 → ... → h^(K-1).
    #   At step k:
    #     grad_h update uses hp  (= h^k)           → coeff derivative
    #     acc[k]  update uses hp * h (= h^(k+1))   → coeff gradient
    #   The same hp * h product serves both, so no extra multiply.
    # ---------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_mean_bwd(
        GO_ptr,         # (B, D)    – upstream gradient, fp32
        X_ptr,          # (B, N, D) – saved input
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        GX_ptr,         # (B, N, D) – grad w.r.t. X, fp32
        GC_ptr,         # (D, K)    – grad w.r.t. coeff, fp32 (zero-init, atomic)
        B, N, D,
        stride_xb, stride_xn,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # --- Preload: constant over the N loop ---
        go    = tl.load(GO_ptr + b * D + d_off, mask=dmask, other=0.0).to(tl.float32)
        inv_N = 1.0 / N

        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        # --- grad_coeff accumulators (one per degree, reduced across N) ---
        a0  = tl.zeros((BLOCK_D,), tl.float32)
        a1  = tl.zeros((BLOCK_D,), tl.float32)
        a2  = tl.zeros((BLOCK_D,), tl.float32)
        a3  = tl.zeros((BLOCK_D,), tl.float32)
        a4  = tl.zeros((BLOCK_D,), tl.float32)
        a5  = tl.zeros((BLOCK_D,), tl.float32)
        a6  = tl.zeros((BLOCK_D,), tl.float32)
        a7  = tl.zeros((BLOCK_D,), tl.float32)
        a8  = tl.zeros((BLOCK_D,), tl.float32)
        a9  = tl.zeros((BLOCK_D,), tl.float32)
        a10 = tl.zeros((BLOCK_D,), tl.float32)
        a11 = tl.zeros((BLOCK_D,), tl.float32)
        a12 = tl.zeros((BLOCK_D,), tl.float32)
        a13 = tl.zeros((BLOCK_D,), tl.float32)
        a14 = tl.zeros((BLOCK_D,), tl.float32)
        a15 = tl.zeros((BLOCK_D,), tl.float32)

        for n in range(N):
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            d_act = tl.where(
                (x >= 0.0) & (x <= 6.0), 1.0,
                tl.where((x < 0.0) & (x >= -10.0), 0.01, 0.0),
            )

            # hp advances as h^0, h^1, ..., h^(K-1).
            # At step k: grad_h uses hp (= h^k); a[k] uses hp*h (= h^(k+1)).
            # hph = hp * h is computed once and serves both, then hp is advanced.
            grad_h = tl.zeros((BLOCK_D,), tl.float32)
            hp     = tl.full((BLOCK_D,), 1.0, tl.float32)   # h^0

            if K > 0:
                hph = hp * h
                grad_h += c0 * hp;   a0 += go * hph;   hp = hph
            if K > 1:
                hph = hp * h
                grad_h += c1 * 2.0 * hp;   a1 += go * hph;   hp = hph
            if K > 2:
                hph = hp * h
                grad_h += c2 * 3.0 * hp;   a2 += go * hph;   hp = hph
            if K > 3:
                hph = hp * h
                grad_h += c3 * 4.0 * hp;   a3 += go * hph;   hp = hph
            if K > 4:
                hph = hp * h
                grad_h += c4 * 5.0 * hp;   a4 += go * hph;   hp = hph
            if K > 5:
                hph = hp * h
                grad_h += c5 * 6.0 * hp;   a5 += go * hph;   hp = hph
            if K > 6:
                hph = hp * h
                grad_h += c6 * 7.0 * hp;   a6 += go * hph;   hp = hph
            if K > 7:
                hph = hp * h
                grad_h += c7 * 8.0 * hp;   a7 += go * hph;   hp = hph
            if K > 8:
                hph = hp * h
                grad_h += c8 * 9.0 * hp;   a8 += go * hph;   hp = hph
            if K > 9:
                hph = hp * h
                grad_h += c9 * 10.0 * hp;  a9 += go * hph;   hp = hph
            if K > 10:
                hph = hp * h
                grad_h += c10 * 11.0 * hp; a10 += go * hph;  hp = hph
            if K > 11:
                hph = hp * h
                grad_h += c11 * 12.0 * hp; a11 += go * hph;  hp = hph
            if K > 12:
                hph = hp * h
                grad_h += c12 * 13.0 * hp; a12 += go * hph;  hp = hph
            if K > 13:
                hph = hp * h
                grad_h += c13 * 14.0 * hp; a13 += go * hph;  hp = hph
            if K > 14:
                hph = hp * h
                grad_h += c14 * 15.0 * hp; a14 += go * hph;  hp = hph
            if K > 15:
                hph = hp * h
                grad_h += c15 * 16.0 * hp; a15 += go * hph;  hp = hph

            tl.store(
                GX_ptr + b * stride_xb + n * stride_xn + d_off,
                go * inv_N * grad_h * d_act,
                mask=dmask,
            )

        # Atomic-add partial grad_coeff sums into GC (reduces across B).
        # GC is zero-initialised by the Python wrapper before this kernel runs.
        if K > 0:  tl.atomic_add(GC_ptr + d_off * K + 0,  a0  * inv_N, mask=dmask)
        if K > 1:  tl.atomic_add(GC_ptr + d_off * K + 1,  a1  * inv_N, mask=dmask)
        if K > 2:  tl.atomic_add(GC_ptr + d_off * K + 2,  a2  * inv_N, mask=dmask)
        if K > 3:  tl.atomic_add(GC_ptr + d_off * K + 3,  a3  * inv_N, mask=dmask)
        if K > 4:  tl.atomic_add(GC_ptr + d_off * K + 4,  a4  * inv_N, mask=dmask)
        if K > 5:  tl.atomic_add(GC_ptr + d_off * K + 5,  a5  * inv_N, mask=dmask)
        if K > 6:  tl.atomic_add(GC_ptr + d_off * K + 6,  a6  * inv_N, mask=dmask)
        if K > 7:  tl.atomic_add(GC_ptr + d_off * K + 7,  a7  * inv_N, mask=dmask)
        if K > 8:  tl.atomic_add(GC_ptr + d_off * K + 8,  a8  * inv_N, mask=dmask)
        if K > 9:  tl.atomic_add(GC_ptr + d_off * K + 9,  a9  * inv_N, mask=dmask)
        if K > 10: tl.atomic_add(GC_ptr + d_off * K + 10, a10 * inv_N, mask=dmask)
        if K > 11: tl.atomic_add(GC_ptr + d_off * K + 11, a11 * inv_N, mask=dmask)
        if K > 12: tl.atomic_add(GC_ptr + d_off * K + 12, a12 * inv_N, mask=dmask)
        if K > 13: tl.atomic_add(GC_ptr + d_off * K + 13, a13 * inv_N, mask=dmask)
        if K > 14: tl.atomic_add(GC_ptr + d_off * K + 14, a14 * inv_N, mask=dmask)
        if K > 15: tl.atomic_add(GC_ptr + d_off * K + 15, a15 * inv_N, mask=dmask)

    # ---------------------------------------------------------------------------
    # autograd.Function wrapper
    # ---------------------------------------------------------------------------

    class _PolyAggMean(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, coeff: torch.Tensor, k: int):
            B, N, D = x.shape
            # Do NOT call x.contiguous().  The kernel uses stride_xb / stride_xn,
            # so it correctly reads a non-contiguous slice such as
            # qc_proj_out[..., :po_dim] without an extra copy.
            coeff_c = coeff.float().contiguous()
            out     = torch.empty(B, D, dtype=torch.float32, device=x.device)

            BLOCK_D = _fwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_mean_fwd[grid](
                x, coeff_c, out,
                N, D,
                x.stride(0), x.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            ctx.save_for_backward(x, coeff_c)
            ctx.k = k
            return out.unsqueeze(1).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x_saved, coeff = ctx.saved_tensors
            k              = ctx.k
            # Make contiguous for backward: the bwd kernel uses the same pointer
            # layout for both reading X and writing GX, so a single stride set
            # is needed.  The extra copy here is acceptable (bwd is not the
            # bottleneck this optimisation targets).
            x       = x_saved.contiguous()
            B, N, D = x.shape

            go = grad_out.squeeze(1).float().contiguous()   # (B, D)

            # fp32 buffers; cast grad_x to input dtype after the kernel.
            grad_x_buf = torch.empty(B, N, D, dtype=torch.float32, device=x.device)
            # GC must be zero-initialised: the kernel accumulates via atomic_add.
            grad_c = torch.zeros(D, k, dtype=torch.float32, device=x.device)

            BLOCK_D = _bwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_mean_bwd[grid](
                go, x, coeff, grad_x_buf, grad_c,
                B, N, D,
                x.stride(0), x.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            return grad_x_buf.to(x_saved.dtype), grad_c.to(coeff.dtype), None

    # ---------------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------------

    def poly_agg_mean_triton(
        x: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Fused polynomial aggregation + mean (no mask).

        Args:
            x     : (B, N, D) input tensor
            coeff : (D, K)    polynomial coefficients
            k     : polynomial degree (≤ 16)

        Returns:
            (B, 1, D) mean-aggregated polynomial features
        """
        if k > 16:
            raise NotImplementedError(
                f"Triton kernel supports k ≤ 16 (got {k}). "
                "Extend the c0..c15 / a0..a15 pattern or use the PyTorch fallback."
            )
        return _PolyAggMean.apply(x, coeff, k)

"""Triton-accelerated 1-D mask kernel for PoM (Polynomial Mixer).

For a (B, N) float mask, output position 0 aggregates the masked tokens:

  out[b, d] = sum_{n: mask[b,n]!=0} mask[b,n] * poly(act(x[b,n,d]))
              ---------------------------------------------------
                          sum_{n} mask[b,n]

which is a weighted mean (or binary mean when mask values are 0/1).
Output shape: (B, 1, D) — same as the no-mask case.

This fuses activation + polynomial expansion + masked sum + normalisation
into a single GPU pass, eliminating the (B, N, D, K) intermediate that the
PyTorch fallback materialises.

Forward kernel
--------------
Grid: (B, ceil(D/BLOCK_D))
Each program accumulates a scalar `cnt` and a (BLOCK_D,) `acc` in one loop
over N.  The mask value m[b, n] is a scalar load; it gates both `acc` and
`cnt`.

Backward kernel
---------------
Grid: (B, ceil(D/BLOCK_D))
Gradients:

  grad_x[b, n, d] = go[b,d] * (m[b,n] / cnt[b]) * d_poly/d_h * d_act/d_x
  grad_coeff[d,k]  = sum_{b,n} go[b,d] * (m[b,n] / cnt[b]) * h[b,n,d]^(k+1)

Implementation uses two sequential loops over N within the same program:
  Loop 1 (mask-only): compute cnt[b] from N scalar mask loads.
  Loop 2 (mask + x):  compute grad_x and accumulate grad_coeff.
The first loop is very cheap (N scalar loads vs the N*BLOCK_D x-loads of
loop 2) and avoids the complexity of saving cnt across forward/backward.

The `go_eff = go / cnt` vector is precomputed once after loop 1 and reused
in loop 2, enabling the same power-sharing pattern as the other kernels.

Exposed API
-----------
TRITON_MASKED_AVAILABLE : bool
poly_agg_masked_triton(x, mask, coeff, k) -> Tensor  (B, 1, D)
"""

try:
    import triton
    import triton.language as tl
    TRITON_MASKED_AVAILABLE = not os.environ.get("POM_DISABLE_TRITON", "")
except ImportError:
    TRITON_MASKED_AVAILABLE = False


if TRITON_MASKED_AVAILABLE:

    # -------------------------------------------------------------------------
    # BLOCK_D heuristics – identical policy to the other kernels.
    # -------------------------------------------------------------------------

    def _fwd_block_d(D: int) -> int:
        return min(256, 1 << (D - 1).bit_length())

    def _bwd_block_d(D: int) -> int:
        return min(128, 1 << (D - 1).bit_length())

    # -------------------------------------------------------------------------
    # Forward kernel
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_masked_fwd(
        X_ptr,          # (B, N, D) – input, any dtype
        M_ptr,          # (B, N)    – mask, any dtype (treated as float)
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        O_ptr,          # (B, D)    – output, fp32
        N,              # sequence length  (runtime)
        D,              # feature dim      (runtime)
        stride_xb,      # X.stride(0)
        stride_xn,      # X.stride(1)
        stride_mb,      # M.stride(0)
        stride_mn,      # M.stride(1)
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload coeff[d_off, 0..K-1] into registers once.
        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        cnt = 0.0   # scalar: sum of mask weights

        for n in range(N):
            # Scalar mask load — same value for all d in this block.
            m_val = tl.load(M_ptr + b * stride_mb + n * stride_mn).to(tl.float32)

            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # pom_activation
            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            poly = tl.zeros((BLOCK_D,), dtype=tl.float32)
            hp   = h
            if K > 0:
                poly += c0 * hp
            if K > 1:
                hp *= h; poly += c1 * hp
            if K > 2:
                hp *= h; poly += c2 * hp
            if K > 3:
                hp *= h; poly += c3 * hp
            if K > 4:
                hp *= h; poly += c4 * hp
            if K > 5:
                hp *= h; poly += c5 * hp
            if K > 6:
                hp *= h; poly += c6 * hp
            if K > 7:
                hp *= h; poly += c7 * hp
            if K > 8:
                hp *= h; poly += c8 * hp
            if K > 9:
                hp *= h; poly += c9 * hp
            if K > 10:
                hp *= h; poly += c10 * hp
            if K > 11:
                hp *= h; poly += c11 * hp
            if K > 12:
                hp *= h; poly += c12 * hp
            if K > 13:
                hp *= h; poly += c13 * hp
            if K > 14:
                hp *= h; poly += c14 * hp
            if K > 15:
                hp *= h; poly += c15 * hp

            acc += m_val * poly
            cnt += m_val

        # Safe division: all-masked-out rows produce zero output.
        inv_cnt = tl.where(cnt > 0.0, 1.0 / cnt, 0.0)
        tl.store(O_ptr + b * D + d_off, acc * inv_cnt, mask=dmask)

    # -------------------------------------------------------------------------
    # Backward kernel
    #
    # Two-pass design within a single program:
    #   Pass 1: scan M to accumulate cnt (N scalar loads — very cheap).
    #   Pass 2: process X and M together, using the precomputed inv_cnt.
    #
    # go_eff = go / cnt  is computed once; grad_x and a[k] both scale by it.
    # This matches the power-sharing structure of the other backward kernels.
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_masked_bwd(
        GO_ptr,         # (B, D)    – upstream gradient, fp32
        X_ptr,          # (B, N, D) – saved input
        M_ptr,          # (B, N)    – mask
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        GX_ptr,         # (B, N, D) – grad w.r.t. X, fp32
        GC_ptr,         # (D, K)    – grad w.r.t. coeff, fp32 (zero-init, atomic)
        B, N, D,
        stride_xb, stride_xn,
        stride_mb, stride_mn,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload go[b, d] and coefficients (constant over both N loops).
        go    = tl.load(GO_ptr + b * D + d_off, mask=dmask, other=0.0).to(tl.float32)

        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        # ---- Pass 1: count masked tokens (N scalar loads) --------------------
        cnt = 0.0
        for n in range(N):
            m_val = tl.load(M_ptr + b * stride_mb + n * stride_mn).to(tl.float32)
            cnt += m_val

        inv_cnt  = tl.where(cnt > 0.0, 1.0 / cnt, 0.0)
        # Effective upstream gradient: go / cnt  (scaled once, reused per step)
        go_eff   = go * inv_cnt      # (BLOCK_D,)

        # ---- grad_coeff accumulators -----------------------------------------
        a0  = tl.zeros((BLOCK_D,), tl.float32)
        a1  = tl.zeros((BLOCK_D,), tl.float32)
        a2  = tl.zeros((BLOCK_D,), tl.float32)
        a3  = tl.zeros((BLOCK_D,), tl.float32)
        a4  = tl.zeros((BLOCK_D,), tl.float32)
        a5  = tl.zeros((BLOCK_D,), tl.float32)
        a6  = tl.zeros((BLOCK_D,), tl.float32)
        a7  = tl.zeros((BLOCK_D,), tl.float32)
        a8  = tl.zeros((BLOCK_D,), tl.float32)
        a9  = tl.zeros((BLOCK_D,), tl.float32)
        a10 = tl.zeros((BLOCK_D,), tl.float32)
        a11 = tl.zeros((BLOCK_D,), tl.float32)
        a12 = tl.zeros((BLOCK_D,), tl.float32)
        a13 = tl.zeros((BLOCK_D,), tl.float32)
        a14 = tl.zeros((BLOCK_D,), tl.float32)
        a15 = tl.zeros((BLOCK_D,), tl.float32)

        # ---- Pass 2: compute grad_x and accumulate grad_coeff ----------------
        for n in range(N):
            m_val = tl.load(M_ptr + b * stride_mb + n * stride_mn).to(tl.float32)

            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            d_act = tl.where(
                (x >= 0.0) & (x <= 6.0), 1.0,
                tl.where((x < 0.0) & (x >= -10.0), 0.01, 0.0),
            )

            # Power-sharing: same pattern as the other backward kernels.
            # go_eff_m = m_val * go_eff  scales both grad_x and coeff accum.
            go_eff_m = m_val * go_eff   # (BLOCK_D,)
                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            grad_h = tl.zeros((BLOCK_D,), tl.float32)
            hp     = tl.full((BLOCK_D,), 1.0, tl.float32)   # h^0

            if K > 0:
                hph = hp * h
                grad_h += c0 * hp;          a0  += go_eff_m * hph;  hp = hph
            if K > 1:
                hph = hp * h
                grad_h += c1 * 2.0 * hp;   a1  += go_eff_m * hph;  hp = hph
            if K > 2:
                hph = hp * h
                grad_h += c2 * 3.0 * hp;   a2  += go_eff_m * hph;  hp = hph
            if K > 3:
                hph = hp * h
                grad_h += c3 * 4.0 * hp;   a3  += go_eff_m * hph;  hp = hph
            if K > 4:
                hph = hp * h
                grad_h += c4 * 5.0 * hp;   a4  += go_eff_m * hph;  hp = hph
            if K > 5:
                hph = hp * h
                grad_h += c5 * 6.0 * hp;   a5  += go_eff_m * hph;  hp = hph
            if K > 6:
                hph = hp * h
                grad_h += c6 * 7.0 * hp;   a6  += go_eff_m * hph;  hp = hph
            if K > 7:
                hph = hp * h
                grad_h += c7 * 8.0 * hp;   a7  += go_eff_m * hph;  hp = hph
            if K > 8:
                hph = hp * h
                grad_h += c8 * 9.0 * hp;   a8  += go_eff_m * hph;  hp = hph
            if K > 9:
                hph = hp * h
                grad_h += c9 * 10.0 * hp;  a9  += go_eff_m * hph;  hp = hph
            if K > 10:
                hph = hp * h
                grad_h += c10 * 11.0 * hp; a10 += go_eff_m * hph;  hp = hph
            if K > 11:
                hph = hp * h
                grad_h += c11 * 12.0 * hp; a11 += go_eff_m * hph;  hp = hph
            if K > 12:
                hph = hp * h
                grad_h += c12 * 13.0 * hp; a12 += go_eff_m * hph;  hp = hph
            if K > 13:
                hph = hp * h
                grad_h += c13 * 14.0 * hp; a13 += go_eff_m * hph;  hp = hph
            if K > 14:
                hph = hp * h                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                grad_h += c14 * 15.0 * hp; a14 += go_eff_m * hph;  hp = hph
            if K > 15:
                hph = hp * h
                grad_h += c15 * 16.0 * hp; a15 += go_eff_m * hph;  hp = hph

            tl.store(
                GX_ptr + b * stride_xb + n * stride_xn + d_off,
                go_eff_m * grad_h * d_act,
                mask=dmask,
            )

        # Atomic-add partial grad_coeff sums into GC (reduces across B).
        if K > 0:  tl.atomic_add(GC_ptr + d_off * K + 0,  a0,  mask=dmask)
        if K > 1:  tl.atomic_add(GC_ptr + d_off * K + 1,  a1,  mask=dmask)
        if K > 2:  tl.atomic_add(GC_ptr + d_off * K + 2,  a2,  mask=dmask)
        if K > 3:  tl.atomic_add(GC_ptr + d_off * K + 3,  a3,  mask=dmask)
        if K > 4:  tl.atomic_add(GC_ptr + d_off * K + 4,  a4,  mask=dmask)
        if K > 5:  tl.atomic_add(GC_ptr + d_off * K + 5,  a5,  mask=dmask)
        if K > 6:  tl.atomic_add(GC_ptr + d_off * K + 6,  a6,  mask=dmask)
        if K > 7:  tl.atomic_add(GC_ptr + d_off * K + 7,  a7,  mask=dmask)
        if K > 8:  tl.atomic_add(GC_ptr + d_off * K + 8,  a8,  mask=dmask)
        if K > 9:  tl.atomic_add(GC_ptr + d_off * K + 9,  a9,  mask=dmask)
        if K > 10: tl.atomic_add(GC_ptr + d_off * K + 10, a10, mask=dmask)
        if K > 11: tl.atomic_add(GC_ptr + d_off * K + 11, a11, mask=dmask)
        if K > 12: tl.atomic_add(GC_ptr + d_off * K + 12, a12, mask=dmask)
        if K > 13: tl.atomic_add(GC_ptr + d_off * K + 13, a13, mask=dmask)
        if K > 14: tl.atomic_add(GC_ptr + d_off * K + 14, a14, mask=dmask)
        if K > 15: tl.atomic_add(GC_ptr + d_off * K + 15, a15, mask=dmask)

    # -------------------------------------------------------------------------
    # autograd.Function wrapper
    # -------------------------------------------------------------------------

    class _PolyAggMasked(torch.autograd.Function):                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        @staticmethod
        def forward(ctx, x: torch.Tensor, mask: torch.Tensor,
                    coeff: torch.Tensor, k: int):
            B, N, D = x.shape
            coeff_c = coeff.float().contiguous()
            mask_c  = mask.float().contiguous()
            out     = torch.empty(B, D, dtype=torch.float32, device=x.device)

            BLOCK_D = _fwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_masked_fwd[grid](
                x, mask_c, coeff_c, out,
                N, D,
                x.stride(0), x.stride(1),
                mask_c.stride(0), mask_c.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            ctx.save_for_backward(x, mask_c, coeff_c)
            ctx.k = k
            return out.unsqueeze(1).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x_saved, mask_c, coeff = ctx.saved_tensors
            k = ctx.k
            x       = x_saved.contiguous()
            B, N, D = x.shape

            go = grad_out.squeeze(1).float().contiguous()   # (B, D)

            grad_x_buf = torch.empty(B, N, D, dtype=torch.float32, device=x.device)
            grad_c     = torch.zeros(D, k, dtype=torch.float32, device=x.device)

            BLOCK_D = _bwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_masked_bwd[grid](
                go, x, mask_c, coeff, grad_x_buf, grad_c,
                B, N, D,
                x.stride(0), x.stride(1),
                mask_c.stride(0), mask_c.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            return grad_x_buf.to(x_saved.dtype), None, grad_c.to(coeff.dtype), None
                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def poly_agg_masked_triton(
        x: torch.Tensor,
        mask: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Fused masked polynomial aggregation.

        Computes the mask-weighted polynomial mean over the sequence:
          out[b, 0, d] = (sum_n mask[b,n] * poly(act(x[b,n,d])))
                         / (sum_n mask[b,n])

        Args:
            x     : (B, N, D) input tensor
            mask  : (B, N)    float mask (0/1 or continuous weights)
            coeff : (D, K)    polynomial coefficients
            k     : polynomial degree (≤ 16)

        Returns:
            (B, 1, D) mask-weighted-mean aggregated polynomial features
        """
        if k > 16:
            raise NotImplementedError(
                f"Triton masked kernel supports k ≤ 16 (got {k}). "
                "Extend the c0..c15 / a0..a15 pattern or use the PyTorch fallback."
            )                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        return _PolyAggMasked.apply(x, mask, coeff, k)

# =============================================================================
# Core activation
# =============================================================================

def pom_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.leaky_relu(x, 0.01, inplace=False), min=-0.1, max=6)


# =============================================================================
# Masking and Aggregation
# =============================================================================

def mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over seq_len. mask: (B, N) → output: (B, 1, D)."""
    m = mask.unsqueeze(-1)
    return (h * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)


def full_mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-attention masked mean. mask: (B, M, N) → output: (B, M, D)."""
    mask = mask.to(h.dtype)
    h = torch.einsum('bnd,bmn->bmd', h, mask)
    return h / mask.sum(dim=2, keepdim=True)


# =============================================================================
# Polynomial Aggregation and Selection
# =============================================================================

def polynomial_aggregation_(
    x: torch.Tensor,
    coeff: torch.Tensor,
    k: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Polynomial aggregation over the sequence dimension.

    Args:
        x:     (B, N, D) input
        coeff: (D, K) polynomial coefficients
        k:     polynomial degree
        mask:  None, "causal", (B, N) tensor for masked mean, or (B, M, N) for cross-attention

    Returns:
        (B, 1, D) for mask=None or 2-D mask; (B, N, D) for "causal"; (B, M, D) for 3-D mask
    """
    # Fused path: no-mask CUDA → single Triton kernel (no (B,N,D,K) intermediate)
    if mask is None and TRITON_AVAILABLE and x.is_cuda:
        return poly_agg_mean_triton(x, coeff, k)
                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    # Fused path: causal CUDA → single Triton kernel (no N×N mask materialised)
    if mask == "causal" and TRITON_CAUSAL_AVAILABLE and x.is_cuda:
        return poly_agg_causal_triton(x, coeff, k)

    # Fused path: 1-D mask CUDA → single Triton kernel (no (B,N,D,K) intermediate)
    if (isinstance(mask, torch.Tensor) and mask.dim() == 2
            and TRITON_MASKED_AVAILABLE and x.is_cuda):
        return poly_agg_masked_triton(x, mask, coeff, k)

    # PyTorch fallback: compute polynomial powers iteratively to avoid h**i overhead
    h = pom_activation(x).unsqueeze(-1)  # (B, N, D, 1)
    hp, powers = h, [h]
    for _ in range(k - 1):
        hp = hp * h
        powers.append(hp)
    h = (torch.cat(powers, dim=-1) * coeff).sum(-1)  # (B, N, D)

    if mask is None:
        return h.mean(dim=1, keepdim=True)
    if mask == "causal":
        B, N, _ = h.shape
        causal_mask = torch.tril(torch.ones(N, N, device=h.device, dtype=h.dtype))
        return full_mask_mixer(h, causal_mask.unsqueeze(0).expand(B, -1, -1))
    if mask.dim() == 2:
        return mask_mixer(h, mask.to(h.device))
    if mask.dim() == 3:
        return full_mask_mixer(h, mask.to(h.device))
    raise ValueError(f'Unsupported mask: expected None, "causal", or a 2/3-D tensor.')


def polynomial_selection_(s: torch.Tensor, h: torch.Tensor, n_sel_heads: int) -> torch.Tensor:
    """Gated selection of aggregated polynomial features.

    Args:
        s: (B, T, D) gating signal (n_sel_heads=1) or (B, T, n_sel_heads)
        h: (B, G, D) aggregated context  (G is 1 or T)

    Returns:
        (B, max(G,T), D)
    """
    b, g, dh = h.shape
    t = s.shape[1]
    assert g == 1 or t == 1 or g == t, f"incompatible shapes: g={g} t={t}"
    if n_sel_heads <= 1:
        # n_sel_heads=0 or 1: s has the same channel dim as h → element-wise gate
        return (s * h).view(b, max(g, t), dh)
    # Multi-head: s is (B, T, n_sel_heads); broadcast over head_dim                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    s = s.unsqueeze(-1)
    h = h.view(b, g, n_sel_heads, dh // n_sel_heads)
    return (s * h).view(b, max(g, t), dh)


def pom(
    xq: torch.Tensor,
    xc: torch.Tensor,
    coeff: torch.Tensor,
    k: int,
    n_sel_heads: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Polynomial Mixer: aggregate context polynomially, gate with query."""
    h = polynomial_aggregation_(xc, coeff, k, mask)
    return polynomial_selection_(xq, h, n_sel_heads)


# =============================================================================
# PoM Module
# =============================================================================

class PoM(nn.Module):
    """Polynomial Mixer (PoM) — linear-complexity alternative to self-attention.

    Aggregates context tokens via a polynomial expansion and weighted mean,
    then gates the result with a learned selection signal from the query.
                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    Args:
        dim:         Input/output feature dimension.
        degree:      Polynomial degree (number of powers to include).
        expand:      Channel expansion factor for the polynomial projection.
        n_groups:    Groups for the polynomial projection (>1 → grouped Conv1d).
        n_sel_heads: Selection heads (1 → scalar gating; >1 → multi-head gating).
        bias:        Add bias to linear projections.
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        expand: int,
        n_groups: int,
        n_sel_heads: int,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.order = degree
        self.order_expand = expand
        self.n_groups = n_groups
        self.n_sel_heads = n_sel_heads
        assert dim % n_groups == 0, "dim must be divisible by n_groups"
        assert n_sel_heads <= 1 or dim * expand % n_sel_heads == 0, \
            "dim * expand must be divisible by n_sel_heads"
        self.head_dim = dim * expand // max(n_sel_heads, 1)

        self._po_dim = expand * dim
        self._se_dim = n_sel_heads if n_sel_heads > 1 else expand * dim                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        if n_groups > 1:
            # Grouped projection must stay as Conv1d; keep se_proj separate.
            self.po_proj = nn.Conv1d(dim, expand * dim, kernel_size=1, bias=bias, groups=n_groups)
            self.se_proj = nn.Linear(dim, self._se_dim, bias=True)
        else:
            # Fuse po_proj and se_proj into a single GEMM (qc_proj) with a
            # standalone bias for the selection branch (se_bias).
            self.qc_proj = nn.Linear(dim, self._po_dim + self._se_dim, bias=False)
            self.se_bias = nn.Parameter(torch.zeros(self._se_dim))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        self.po_coeff = nn.Parameter(torch.randn(dim * expand, degree).clamp(-0.001, 0.001))
        self.ag_proj = nn.Linear(expand * dim, dim, bias=bias)

    def _get_h_s(
        self, xq: torch.Tensor, xc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h, s): projected context and hardsigmoid gating signal.

        For n_groups == 1 and self-mixing (xq is xc), a single fused GEMM
        produces both h and s, halving the projection cost.
        """
        if self.n_groups > 1:
            h = self.po_proj(xc.transpose(1, 2)).transpose(1, 2)
            s = F.hardsigmoid(self.se_proj(xq), inplace=True)
        elif xq is xc:
            out = self.qc_proj(xq)
            h = out[..., :self._po_dim]
            s = F.hardsigmoid(out[..., self._po_dim:] + self.se_bias, inplace=True)
        else:
            w = self.qc_proj.weight
            h = F.linear(xc, w[:self._po_dim])
            s = F.hardsigmoid(
                F.linear(xq, w[self._po_dim:]) + self.se_bias, inplace=True
            )
        return h, s

    def forward(
        self,
        xq: torch.Tensor,
        xc: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            xq:   (B, T, D) query tokens
            xc:   (B, N, D) context tokens; if None, self-mixing is performed                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            mask: optional attention mask
        """
        if xc is None:
            xc = xq
        h, s = self._get_h_s(xq, xc)
        sh = pom(s, h, self.po_coeff, self.order, self.n_sel_heads, mask)
        return self.ag_proj(sh)

    def state_forward(
        self,
        xq: torch.Tensor,
        xc: Optional[torch.Tensor] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Incremental forward with running weighted-mean state.

        Args:
            xq:    (B, T, D) query tokens
            xc:    (B, N, D) context tokens; if None, self-mixing is performed
            state: {'h': running mean tensor, 'n': token count} or None

        Returns:
            (output, new_state)
        """
        if xc is None:
            xc = xq
        h_raw, s = self._get_h_s(xq, xc)
        h_current = polynomial_aggregation_(h_raw, self.po_coeff, self.order)
        n_current = h_current.shape[1]

        if state is not None:
            n_past = state['n']
            h = (n_past * state['h'] + n_current * h_current) / (n_past + n_current)
        else:
            h, n_past = h_current, 0

        sh = polynomial_selection_(s, h, self.n_sel_heads)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        return self.ag_proj(sh), {'h': h, 'n': n_past + n_current}

    @torch.no_grad()
    def ar_forward(
        self,
        xq: torch.Tensor,
        state: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Autoregressive forward (no gradient tracking)."""
        B, T, D = xq.shape
        h, s = self._get_h_s(xq, xq)
        sh = polynomial_selection_(s, h, self.n_sel_heads)
        new_state = {'max_len': state['max_len'], 'h': h, 'n': state['n'] + T}                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        return self.ag_proj(sh), new_state

    def reset(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state['h'] = 0.
        state['n'] = 0
        return state
    



# 使用示例
if __name__ == "__main__":

    dim = 64         # 特征维度
    degree = 2       # 多项式阶数
    expand = 2       # 通道扩展因子
    n_groups = 1     # 分组数
    n_sel_heads = 1  # 选择头数
    batch_size = 2   # Batch size
    seq_len_q = 10   # Query 序列长度
    seq_len_kv = 15  # Context 序列长度（交叉混合时使用）

    model = PoM(dim, degree, expand, n_groups, n_sel_heads)
    print(model)
    model.eval()  # 切换到评估模式

    # 2. 测试场景 1: Self-Mixing (xc = None)
    print("=" * 50)
    print("Testing Self-Mixing (xc = None)")
    print("=" * 50)
    xq_self = torch.randn(batch_size, seq_len_q, dim)
    print(f"Input xq shape: {xq_self.shape}")
    
    with torch.no_grad():
        output_self = model(xq_self)
    
    print(f"Output shape:   {output_self.shape}\n")

    # 3. 测试场景 2: Cross-Mixing (xc 为独立 Context)
    print("=" * 50)
    print("Testing Cross-Mixing (xc is provided)")
    print("=" * 50)
    xq_cross = torch.randn(batch_size, seq_len_q, dim)
    xc_cross = torch.randn(batch_size, seq_len_kv, dim)
    print(f"Input xq shape: {xq_cross.shape}")
    print(f"Input xc shape: {xc_cross.shape}")
    
    with torch.no_grad():
        output_cross = model(xq_cross, xc_cross)
    
    print(f"Output shape:   {output_cross.shape}\n")

    # 4. 测试场景 3: 带 Mask 的 Cross-Mixing
    print("=" * 50)
    print("Testing Cross-Mixing with Mask")
    print("=" * 50)
    mask = torch.ones(batch_size, seq_len_q, seq_len_kv)  # 任意形状的 Mask
    print(f"Input xq shape: {xq_cross.shape}")
    print(f"Input xc shape: {xc_cross.shape}")
    print(f"Mask shape:     {mask.shape}")
    
    with torch.no_grad():
        output_masked = model(xq_cross, xc_cross, mask)
    
    print(f"Output shape:   {output_masked.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")