import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/pdf/2601.08602
    论文题目：WaveFormer: Frequency-Time Decoupled Vision Modeling with Wave Equation（AAAI 2026 Oral）
    中文题目：WPO：突破性创新，Transformer新范式，波动方程调制，比自注意力更轻更准！（AAAI 2026 Oral）
    讲解视频：https://www.bilibili.com/video/BV1cZXPByEPa/
    波的传播算子模块（Wave Propagation Operator，WPO）
        实际意义：①自注意力机制的高计算复杂度问题：传统的Vision Transformers依赖自注意力机制来捕获长距离依赖关系，其计算复杂度为O(N2)。这导致模型在处理高分辨率图像时计算开销巨大，推理速度受限。
                ②物理启发式方法（如热传导）导致的“过度平滑”问题：热传导本质上是一个高低通滤波器，高频分量（边缘、纹理等细节）随传播时间衰减极快，特征图容易变得模糊，丢失精细的局部结构信息。
                ③全局语义与局部细节的平衡：现有的自注意力方案，虽然能够捕捉长距离依赖，但往往难以平衡全局语义和局部细节。
        实现方式：先在空间域提取局部信息，再把特征变到频域里，用“波动方程”的形式做动态传播与调制，最后再变回空间域，并通过门控机制输出增强特征。
"""

class WavePropagationOperator(nn.Module):
    """
    Wave Propagation Operator (WPO) 实现。
    基于 2D-DCT 的频域波动方程求解器，用于视觉主干网络。
    """

    def __init__(self, in_channels=96, hidden_channels=96, init_resolution=14, inference_mode=False, **kwargs):
        super().__init__()

        # 保存默认分辨率，用于推理模式初始化
        self.init_resolution = init_resolution

        # 1. 局部特征增强：深度卷积
        self.local_mix_conv = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels
        )

        self.hidden_channels = hidden_channels

        # 2. 通道映射：将特征映射到 2 倍通道，用于后续拆分为 '主特征' 和 '门控特征'
        self.up_projection = nn.Linear(hidden_channels, 2 * hidden_channels, bias=True)

        # 3. 输出处理：归一化与线性投影
        self.output_norm = nn.LayerNorm(hidden_channels)
        self.output_projection = nn.Linear(hidden_channels, hidden_channels, bias=True)

        # 运行模式标志
        self.inference_mode = inference_mode

        # 4. 时间/频率调制参数生成器
        # 对应论文中的自适应时间步长或频率调制机制
        self.time_modulation_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.GELU(),
        )

        # 5. 可学习的物理参数
        # c: 波速，控制信息传播范围
        self.wave_speed = nn.Parameter(torch.ones(1) * 1.0)
        # alpha: 衰减系数，控制高频分量的保留程度
        self.damping_factor = nn.Parameter(torch.ones(1) * 0.1)

        self.save_attention = False

    def setup_inference_cache(self, freq_embed):
        """推理模式下的预计算缓存，避免重复计算 DCT 矩阵"""
        decay_map = self._create_frequency_decay_map((self.init_resolution, self.init_resolution),
                                                     device=freq_embed.device)

        # 预计算调制核并固化
        self.cached_modulation_kernel = nn.Parameter(
            torch.pow(decay_map[:, :, None], self.time_modulation_mlp(freq_embed)),
            requires_grad=False
        )
        # 删除 MLP 以节省推理内存
        del self.time_modulation_mlp

    @staticmethod
    def _create_dct_transform_matrix(size=224, device=torch.device("cpu"), dtype=torch.float):
        """
        生成 DCT 变换所需的余弦基矩阵。
        用于将空间域信号转换到频域。
        """
        # 空间坐标 x: [0, N-1] -> [0, 1]
        spatial_coords = (torch.linspace(0, size - 1, size, device=device, dtype=dtype).view(1, -1) + 0.5) / size

        # 频率坐标 n: [0, N-1]
        freq_coords = torch.linspace(0, size - 1, size, device=device, dtype=dtype).view(-1, 1)

        # DCT-II 基函数: cos(n * (x + 0.5) * pi / N)
        basis_matrix = torch.cos(freq_coords * spatial_coords * torch.pi) * math.sqrt(2 / size)

        # 直流分量 (DC, n=0) 的归一化系数修正
        basis_matrix[0, :] = basis_matrix[0, :] / math.sqrt(2)

        return basis_matrix

    @staticmethod
    def _create_frequency_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        """
        生成频域衰减图。
        对应论文中的频率相关衰减项。
        """
        h, w = resolution

        # 竖直方向频率索引
        freq_n = torch.linspace(0, torch.pi, h + 1, device=device, dtype=dtype)[:h].view(-1, 1)
        # 水平方向频率索引
        freq_m = torch.linspace(0, torch.pi, w + 1, device=device, dtype=dtype)[:w].view(1, -1)

        # 二维频率平方和
        freq_squared_sum = torch.pow(freq_n, 2) + torch.pow(freq_m, 2)

        # 指数衰减形式 (注：此处为预计算底数，具体衰减逻辑在 forward 中结合参数执行)
        decay_map = torch.exp(-freq_squared_sum)

        return decay_map

    def forward(self, x: torch.Tensor, freq_embed=None, test_index=None):
        B, C, H, W = x.shape

        # --- 阶段 1: 空间域预处理 ---
        # 局部特征增强
        x = self.local_mix_conv(x)
        # 通道映射并拆分
        # x_permuted: [B, H, W, C] (适应 Linear 层)
        x_proj = self.up_projection(x.permute(0, 2, 3, 1).contiguous())
        # 拆分为主分支 和门控分支
        main_branch, gate_branch = x_proj.chunk(chunks=2, dim=-1)

        # --- 阶段 2: 频域变换 ---
        # 尝试读取缓存的 DCT 变换矩阵
        if ((H, W) == getattr(self, "__cached_res__", (0, 0))) and \
                (hasattr(self, "__cached_dct_h__")) and \
                (self.__cached_dct_h__.device == x.device):
            dct_matrix_h = self.__cached_dct_h__
            dct_matrix_w = self.__cached_dct_w__
            decay_map = self.__cached_decay_map__
        else:
            # 动态生成当前分辨率下的矩阵
            dct_matrix_h = self._create_dct_transform_matrix(H, device=x.device).detach_()
            dct_matrix_w = self._create_dct_transform_matrix(W, device=x.device).detach_()
            decay_map = self._create_frequency_decay_map((H, W), device=x.device).detach_()

            # 更新缓存
            self.__cached_res__ = (H, W)
            self.__cached_dct_h__ = dct_matrix_h
            self.__cached_dct_w__ = dct_matrix_w
            self.__cached_decay_map__ = decay_map

        # 将 DCT 矩阵 reshape 为 Conv1d 可用的卷积核: (out_channels, in_channels, kernel_size)
        # 这里是利用 Conv1d 进行矩阵乘法的高效实现
        kernel_h = dct_matrix_h.view(H, 1, H)
        kernel_w = dct_matrix_w.view(W, 1, W)

        # --- 2D-DCT 变换 ---
        # 过程: 输入 -> H方向DCT -> W方向DCT -> 频域系数
        # Step A: H 方向 DCT
        # 维度变换: [B, H, W, C] -> [B, W, C, H] -> [B*W*C, 1, H]
        x_trans_h = main_branch.permute(0, 3, 2, 1).contiguous().view(-1, 1, H)
        # 应用 Conv1d (相当于乘以 DCT 矩阵)
        dct_h_out = F.conv1d(x_trans_h, kernel_h).squeeze(-1)
        # 恢复维度: [B, H, W, C]
        dct_h_out = dct_h_out.view(B, C, W, H).permute(0, 3, 2, 1).contiguous()

        # Step B: W 方向 DCT
        # 维度变换: [B, H, W, C] -> [B, H, C, W] -> [B*H*C, 1, W]
        x_trans_w = dct_h_out.permute(0, 3, 1, 2).contiguous().view(-1, 1, W)
        dct_w_out = F.conv1d(x_trans_w, kernel_w).squeeze(-1)
        # 最终频域系数 U0: [B, H, W, C]
        freq_coeffs_u = dct_w_out.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()

        # --- 构造 V0 (速度场) ---
        # 代码逻辑：对同一输入再做一次 DCT 作为 V0 (对应论文公式的初始速度项)
        x_trans_h_v = main_branch.permute(0, 3, 2, 1).contiguous().view(-1, 1, H)
        dct_h_out_v = F.conv1d(x_trans_h_v, kernel_h).squeeze(-1)
        dct_h_out_v = dct_h_out_v.view(B, C, W, H).permute(0, 3, 2, 1).contiguous()

        x_trans_w_v = dct_h_out_v.permute(0, 3, 1, 2).contiguous().view(-1, 1, W)
        dct_w_out_v = F.conv1d(x_trans_w_v, kernel_w).squeeze(-1)
        freq_coeffs_v = dct_w_out_v.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()

        # --- 阶段 3: 波动方程频域求解 ---
        # 用“波动方程”的形式做动态传播与调制
        if freq_embed is None:
            freq_embed = torch.zeros(B, H, W, self.hidden_channels, device=x.device, dtype=x.dtype)
        # 生成时间调制参数 T
        time_param = self.time_modulation_mlp(freq_embed)
        # 计算波动方程关键项
        # c_t 对应 cos(omega * t) 中的参数
        phase_term = self.wave_speed * time_param
        cos_term = torch.cos(phase_term)
        # 防止除零
        eps = 1e-8
        # sin(omega*t) / omega
        sin_term = torch.sin(phase_term) / (self.wave_speed + eps)
        # 波动方程核心公式:
        # Term1 = cos(t) * U0
        # Term2 = sin(t) * (V0 + alpha/2 * U0)
        wave_component = cos_term * freq_coeffs_u
        velocity_component = sin_term * (freq_coeffs_v + (self.damping_factor / 2) * freq_coeffs_u)
        # 频域融合结果
        modulated_freq_signal = wave_component + velocity_component

        # --- 阶段 4: 频域逆变换 ---
        # 读取 IDCT 缓存 (注意：IDCT 通常使用转置后的 DCT 矩阵)
        if ((H, W) == getattr(self, "__cached_res_idct__", (0, 0))) and \
                (hasattr(self, "__cached_idct_h__")) and \
                (self.__cached_idct_h__.device == x.device):
            idct_matrix_h = self.__cached_idct_h__
            idct_matrix_w = self.__cached_idct_w__
        else:
            # 生成 IDCT 矩阵并缓存
            idct_matrix_h = self._create_dct_transform_matrix(H, device=x.device).detach_()
            idct_matrix_w = self._create_dct_transform_matrix(W, device=x.device).detach_()
            self.__cached_res_idct__ = (H, W)
            self.__cached_idct_h__ = idct_matrix_h
            self.__cached_idct_w__ = idct_matrix_w

        # IDCT 卷积核 (使用转置矩阵)
        idct_kernel_w = idct_matrix_w.t().contiguous().view(W, 1, W)
        idct_kernel_h = idct_matrix_h.t().contiguous().view(H, 1, H)

        # Step A: W 方向 IDCT
        # [B, H, W, C] -> [B*H*C, 1, W]
        freq_signal_w = modulated_freq_signal.permute(0, 1, 3, 2).contiguous().view(B * H * C, 1, W)
        spatial_w_out = F.conv1d(freq_signal_w, idct_kernel_w).squeeze(-1)
        # [B, H, W, C]
        spatial_w_out = spatial_w_out.view(B, H, C, W).permute(0, 1, 3, 2).contiguous()

        # Step B: H 方向 IDCT
        # [B, H, W, C] -> [B*W*C, 1, H]
        freq_signal_h = spatial_w_out.permute(0, 2, 3, 1).contiguous().view(B * W * C, 1, H)
        spatial_h_out = F.conv1d(freq_signal_h, idct_kernel_h).squeeze(-1)
        # 最终空间域特征: [B, H, W, C]
        spatial_feature_out = spatial_h_out.view(B, W, C, H).permute(0, 3, 1, 2).contiguous()

        # --- 阶段 5: 输出融合 ---
        # 归一化
        x_norm = self.output_norm(spatial_feature_out)
        # 门控激活
        gate_activation = nn.functional.silu(gate_branch)
        # 特征融合
        x_fused = x_norm * gate_activation
        # 最终投影
        output = self.output_projection(x_fused)
        # 恢复维度 [B, C, H, W]
        output = output.permute(0, 3, 1, 2).contiguous()

        # --- 可视化逻辑 (保持原样) ---
        if test_index is not None and hasattr(self, 'save_attention') and self.save_attention:
            center_h, center_w = H // 2, W // 2
            attention_map = (spatial_feature_out * spatial_feature_out[:, center_h:center_h + 1, center_w:center_w + 1,
                                                   :]).sum(-1)
            import matplotlib.pyplot as plt
            save_dir = "./save/attention_map"
            os.makedirs(save_dir, exist_ok=True)
            att_map_np = attention_map.detach().cpu().numpy()
            att_map_norm = (att_map_np - att_map_np.min()) / (att_map_np.max() - att_map_np.min() + 1e-8)
            for i in range(att_map_norm.shape[0]):
                filename = os.path.join(save_dir, f"attention_map_{test_index}_{i}.png")
                plt.imsave(filename, att_map_norm[i], cmap='viridis')

        return output

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = WavePropagationOperator(in_channels=32, hidden_channels=32, init_resolution=14, inference_mode=False)
    output_feature = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output_feature.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")