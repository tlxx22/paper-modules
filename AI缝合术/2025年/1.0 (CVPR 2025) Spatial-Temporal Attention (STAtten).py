import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# 论文题目：Spiking Transformer with Spatial-Temporal Attention
# 中文题目：具有时空注意的脉冲Transformer
# 论文链接：https://arxiv.org/pdf/2409.19764
# 官方github：https://github.com/Intelligent-Computing-Lab-Yale/STAtten
# 所属机构：耶鲁大学电气工程系
# 代码整理：微信公众号：AI缝合术

class dvs_pooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)
    
class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mode="direct_xor",
        dvs=False,
        layer=0,
        attention_mode="T_STAtten",
        chunk_size=2,
        spike_mode="lif"
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.attention_mode = attention_mode
        if dvs:
            self.pool = dvs_pooling()
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch")

        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        self.talking_heads_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch")

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)
        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch")

        self.mode = mode
        self.layer = layer
        self.chunk_size = chunk_size

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        head_dim = C // self.num_heads
        identity = x
        N = H * W
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()
        if self.dvs:
            x_pool = self.pool(x)

        x_for_qkv = x.flatten(0, 1)

        # Q
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        if self.dvs:
            q_conv_out = self.pool(q_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = (q_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous())

        # K
        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (k_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous())

        # V
        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (v_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous())
        # Shape: (T B head N C//head)

        ###### Attention #####
        if self.attention_mode == "STAtten":
            if self.dvs:
                scaling_factor = 1 / (H*H*self.chunk_size)
            else:
                scaling_factor = 1 / H

            # Vectorized Attention
            num_chunks = T // self.chunk_size
            # Reshape q, k, v to process all chunks at once: (num_chunks, B, num_heads, chunk_size, N, head_dim)
            q_chunks = q.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            k_chunks = k.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
            v_chunks = v.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)

            # Merge chunk_size and N dimensions: (num_chunks, B, num_heads, chunk_size * N, head_dim)
            q_chunks = q_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            k_chunks = k_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
            v_chunks = v_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)

            # Compute attention for all chunks simultaneously
            attn = torch.matmul(k_chunks.transpose(-2, -1),
                                v_chunks) * scaling_factor  # (num_chunks, B, num_heads, head_dim, head_dim)
            out = torch.matmul(q_chunks, attn)  # (num_chunks, B, num_heads, chunk_size * N, head_dim)

            # Reshape back to separate temporal and spatial dimensions
            out = out.reshape(num_chunks, B, self.num_heads, self.chunk_size, N, head_dim).permute(0, 3, 1, 2, 4, 5)
            # Flatten chunks back to T: (T, B, num_heads, N, head_dim)
            output = out.reshape(T, B, self.num_heads, N, head_dim)

            x = output.transpose(4,3).reshape(T, B, C, N).contiguous() # (T, B, head, C//h, N)
            x = self.attn_lif(x).reshape(T, B, C, H, W)
            if self.dvs:
                x = x.mul(x_pool)
                x = x + x_pool

            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_after_qkv"] = x

            x = (
                self.proj_bn(self.proj_conv(x.flatten(0, 1)))
                .reshape(T, B, C, H, W)
                .contiguous()
            )

        """Spike-driven Transformer"""
        if self.attention_mode == "SDT":
            kv = k.mul(v)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
            if self.dvs:
                kv = self.pool(kv)
            kv = kv.sum(dim=-2, keepdim=True)
            kv = self.talking_heads_lif(kv)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
            x = q.mul(kv)
            if self.dvs:
                x = self.pool(x)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

            x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
            x = (
                self.proj_bn(self.proj_conv(x.flatten(0, 1)))
                .reshape(T, B, C, H, W)
                .contiguous()
            )
        assert self.attention_mode not in ["STAtten, SDT"] 

        x = x + identity
        # return x, v, hook
        return x

if __name__ == "__main__":
    
    # 设置测试参数
    T = 4  # 时间步长
    B = 1  # 批次大小
    C = 32  # 通道数
    H = 256  # 高度
    W = 256  # 宽度
    
    # 创建一个随机输入张量，形状为 (T, B, C, H, W)
    x = torch.randn(T, B, C, H, W).cuda()  # 假设使用 GPU 运行

    # 初始化 MS_SSA_Conv 模块
    dim = C  # 输入的通道数等于输出的通道数
    model = MS_SSA_Conv(dim=dim).cuda()  # 假设使用 GPU
    print(model)

    # 运行模型前向传播
    output = model(x)

    print("\n微信公众号: AI缝合术!\n")

    # 打印输出的形状，检查是否符合预期
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
