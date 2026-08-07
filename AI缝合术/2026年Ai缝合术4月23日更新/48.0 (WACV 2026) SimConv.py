import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def lcm(a, b):
    """计算两个数的最小公倍数 (LCM)"""
    return a * b // math.gcd(a, b)


def lcm_multiple(numbers):
    """计算多个数的最小公倍数"""
    current_lcm = numbers[0]
    for num in numbers[1:]:
        current_lcm = lcm(current_lcm, num)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    return current_lcm


class SimConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        candidate_kernels: list = [1, 3, 5],  # 候选卷积核尺寸 ( Hybrid 方案)
        R_theta: float = 0.1,    # 特征相关性阈值 
        M_theta: float = 0.4,    # 核选择阈值 
        E_theta: float = 0.15    # 相似度校验阈值 
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.candidate_kernels = candidate_kernels                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.R_theta = R_theta
        self.M_theta = M_theta
        self.E_theta = E_theta

        # ---------- 1. 初始化窗口尺寸  ----------
        # 窗口尺寸为候选核边长的 LCM，限制最大为 5 以避免计算量爆炸
        self.window_size = lcm_multiple(candidate_kernels)
        self.window_size = min(self.window_size, 5)

        # ---------- 2. 特征维度投影 (用于相似度计算) ----------
        # 当 in_channels != out_channels 时，将输入投影到输出通道
        self.proj_I = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

        # ---------- 3. 候选卷积核 (论文图4: Conv1 & Conditional Kernel) ----------
        self.conv_k1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=candidate_kernels[0], padding=candidate_kernels[0]//2                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        )  # 基准核 K1 (1x1)
        self.conv_k2 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=candidate_kernels[1], padding=candidate_kernels[1]//2                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        )  # 大核 K2 (3x3)
        self.conv_k3 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=candidate_kernels[2], padding=candidate_kernels[2]//2
        )  # 小核 K3 (5x5)

        # ---------- 4. 局部-全局权重卷积 ----------
        self.conv_weight = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        """
        计算通道维度的余弦相似度 
        x, y: (B, C, H, W)
        return: (B, 1, H, W)
        """
        dot = torch.sum(x * y, dim=1, keepdim=True)  # 点积 (B,1,H,W)
        norm_x = torch.norm(x, dim=1, keepdim=True)   # x 范数 (B,1,H,W)
        norm_y = torch.norm(y, dim=1, keepdim=True)   # y 范数 (B,1,H,W)
        return dot / (norm_x * norm_y + 1e-8)          # 余弦相似度

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        B, C, H, W = I.shape

        # ==============================================================================
        # 步骤 1: 计算空间特征差分 
        # ==============================================================================
        # x 方向差分 (左右相邻像素差)
        R_x = F.pad(I[:, :, :, 1:] - I[:, :, :, :-1], (1, 0, 0, 0))  # (B,C,H,W)
        # y 方向差分 (上下相邻像素差)
        R_y = F.pad(I[:, :, 1:, :] - I[:, :, :-1, :], (0, 0, 1, 0))  # (B,C,H,W)

        # ==============================================================================
        # 步骤 2: 生成相关性布尔矩阵 M (阈值 R_theta=0.1)
        # ==============================================================================
        M_x = (torch.abs(R_x) < self.R_theta).float()  # x 方向相关性掩码                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        M_y = (torch.abs(R_y) < self.R_theta).float()  # y 方向相关性掩码
        M = (M_x + M_y) / 2  # 融合两个方向的相关性 (B,C,H,W)

        # ==============================================================================
        # 步骤 3: 动态选择卷积核 (阈值 M_theta=0.4)
        # ==============================================================================
        # 对通道维度取平均，得到空间维度的相关性比例
        M_spatial = torch.mean(M, dim=1, keepdim=True)  # (B,1,H,W)
        # 平均池化计算局部窗口内 1 的比例
        proportion = F.avg_pool2d(
            M_spatial, 
            kernel_size=self.window_size, 
            stride=1, 
            padding=self.window_size//2
        )  # (B,1,H,W)
        # 生成核选择掩码: >M_theta 选大核 (K2), 否则选小核 (K3)
        mask = (proportion > self.M_theta).float()  # (B,1,H,W)

        # ==============================================================================
        # 步骤 4: 双分支特征提取 ( feat1 & feat2)
        # ==============================================================================
        feat1 = self.conv_k1(I)          # 基准分支 (K1=1x1)
        feat2_k2 = self.conv_k2(I)       # 大核分支 (K2=3x3)
        feat2_k3 = self.conv_k3(I)       # 小核分支 (K3=5x5)
        # 根据 mask 动态选择 feat2
        feat2 = mask * feat2_k2 + (1 - mask) * feat2_k3  # (B,out_channels,H,W)

        # ==============================================================================
        # 步骤 5: 余弦相似度校验 (阈值 E_theta=0.15)
        # ==============================================================================
        I_proj = self.proj_I(I)  # 投影输入到输出通道用于相似度计算                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        E1 = self._cosine_similarity(feat1, I_proj)  # feat1 与输入的相似度
        E2 = self._cosine_similarity(feat2, I_proj)  # feat2 与输入的相似度
        D = torch.abs(E1 - E2)  # 相似度差值 (B,1,H,W)

        # 根据 D 选择最优特征 (论文 Eq.10)
        selected_feat = torch.where(
            D > self.E_theta, feat1,          # 差值大，保留基准 feat1
            torch.where(
                D < self.E_theta, feat2,      # 差值小，保留动态 feat2
                (feat1 + feat2) / 2            # 中间区域，两者融合
            )
        )

        # ==============================================================================
        # 步骤 6: 计算局部-全局权重并输出
        # ==============================================================================
        Weight = self.conv_weight(I)  # 全局权重 (B,out_channels,H,W)
        output = Weight + selected_feat  # 最终输出 (B,out_channels,H,W)

        return output


# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == "__main__":
    # 模拟输入: Batch=2, 通道=64, 尺寸=32x32
    in_channels = 64
    out_channels = 64
    input_tensor = torch.randn(2, in_channels, 32, 32)

    # 初始化 SimConv
    simconv = SimConv(
        in_channels=in_channels,
        out_channels=out_channels,
        candidate_kernels=[1, 3, 9],  # Hybrid 方案
        R_theta=0.1,
        M_theta=0.4,
        E_theta=0.15
    )


    # 打印模型结构
    print(simconv)

    # 前向传播
    output_tensor = simconv(input_tensor)

    # 打印输入输出尺寸
    print(f"Input shape:  {input_tensor.shape}")   # (2, 64, 32, 32)
    print(f"Output shape: {output_tensor.shape}")  # (2, 128, 32, 32)
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")