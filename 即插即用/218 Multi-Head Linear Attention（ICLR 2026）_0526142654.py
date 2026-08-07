import torch
import torch.nn as nn
import numpy as np  # 添加了这行来导入 numpy
from einops import rearrange
import math

"""
    论文地址：https://arxiv.org/pdf/2601.08602
    论文题目：MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head（ICLR 2026）
    中文题目：MHLA：通过Token 级多头机制恢复线性注意力的表达能力（ICLR 2026）
    讲解视频：https://www.bilibili.com/video/BV18ZXqBXEyC/
    多头线性注意力（Multi-Head Linear Attention，MHLA）
        实际意义：①全局上下文崩溃问题：传统的线性注意力方法会使用一个共享的全局信息来计算所有token的关系，这样当序列变长时，模型就会失去对不同token的关注，导致性能下降。
                ②特征表现能力不足问题：因为所有的查询Q都使用相同的全局信息，模型不能灵活地根据每个查询来选择关注的内容，导致模型表达能力不足，无法有效处理长序列任务。
        实现方式：通过在 Token 维度分块并动态混合局部上下文，在保持线性计算效率的同时恢复注意力表达能力。
"""

class BlockDistanceConv(nn.Module):
    def __init__(
            self, num_patches_per_side=16, patch_group_size=16, distance_transform="linear", local_threshold=1.5, exp_decay_param=3
    ):
        super().__init__()

        # 每一边patch的数量（例如16表示16×16的patch）
        self.num_patches_per_side = num_patches_per_side

        # 每个block包含的patch数量
        self.patch_group_size = patch_group_size

        # 距离变换函数类型（如：线性、余弦等）
        self.distance_transform = distance_transform

        # 局部连接阈值，用于局部关系建模
        self.local_threshold = local_threshold

        # 指数衰减参数，用于计算距离衰减
        self.exp_decay_param = exp_decay_param

        # 计算每个block边长patch数量（例如16 patch -> 4x4）
        patches_per_block_side = int(np.sqrt(patch_group_size))

        # 每边block数量
        self.blocks_per_side = num_patches_per_side // patches_per_block_side
        # block总数量
        self.total_blocks = self.blocks_per_side ** 2
        # 计算block之间的距离矩阵
        distance_matrix = self._compute_block_distances()
        # 根据距离矩阵应用变换得到权重矩阵
        weight_matrix = self._apply_transform(distance_matrix)

        # 创建1×1卷积层
        self.conv = nn.Conv2d(
            in_channels=self.total_blocks,
            out_channels=self.total_blocks,
            kernel_size=1,
            bias=False,
        )

        # 不计算梯度的情况下设置卷积权重
        with torch.no_grad():
            self.conv.weight.data = weight_matrix.unsqueeze(-1).unsqueeze(-1)

    def _compute_block_distances(self):
        # 存储所有block的中心坐标
        block_centers = []

        # 遍历所有block，计算其中心坐标
        for i in range(self.blocks_per_side):
            for j in range(self.blocks_per_side):
                center_x = i + 0.5
                center_y = j + 0.5
                block_centers.append([center_x, center_y])

        # 转换为tensor格式
        block_centers = torch.tensor(block_centers, dtype=torch.float32)

        # 初始化距离矩阵
        distance_matrix = torch.zeros(self.total_blocks, self.total_blocks)

        # 计算所有block之间的欧氏距离
        for i in range(self.total_blocks):
            for j in range(self.total_blocks):
                dist = torch.norm(block_centers[i] - block_centers[j], p=2)
                distance_matrix[i, j] = dist

        return distance_matrix

    def _apply_transform(self, distance_matrix):
        # 根据距离矩阵应用不同的变换
        if self.distance_transform == "linear":
            max_dist = distance_matrix.max()
            mat = 1.0 - (distance_matrix / max_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.distance_transform == "cos":
            max_dist = distance_matrix.max()
            normalized_dist = distance_matrix / max_dist * math.pi / 4
            mat = torch.cos(normalized_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.distance_transform == "exp":
            mat = torch.exp(-distance_matrix / self.exp_decay_param)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.distance_transform == "gaussian":
            sigma = distance_matrix.max() / 3
            return torch.exp(-(distance_matrix ** 2) / (2 * sigma ** 2))

        elif self.distance_transform == "local":
            mat = (distance_matrix <= self.local_threshold).float()
            mat = mat / mat.sum(dim=0, keepdim=True)
            return mat

        else:
            raise ValueError(f"Unknown transform: {self.distance_transform}")

    def forward(self, x):
        # 使用卷积处理输入特征
        return self.conv(x)

    def get_weight_matrix(self):
        # 返回当前卷积层的权重矩阵
        return self.conv.weight.data.squeeze(-1).squeeze(-1)


class MHLA_Normed_Torch(nn.Module):

    def __init__(
            self,
            input_dim,
            num_heads=8,
            head_dim=None,
            dropout=0.1,
            fixed_weight_value=None,
            qk_norm=False,
            distance_transform="cos",
            **kwargs,
    ):
        super(MHLA_Normed_Torch, self).__init__()

        # 每个头的维度
        if head_dim is None:
            head_dim = input_dim // num_heads

        # 内部维度
        inner_dim = head_dim * num_heads

        # 头的数量
        self.num_heads = num_heads

        # 每个头的维度
        self.head_dim = head_dim

        # 注意力缩放因子
        self.scale = head_dim ** -0.5

        # 输入归一化层
        self.norm = nn.LayerNorm(input_dim)

        # 是否使用偏置
        is_bias = kwargs["qkv_bias"] if "qkv_bias" in kwargs else False

        # QKV的线性投影
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=is_bias)

        # 是否对Q和K做归一化，改为使用 LayerNorm 代替 RMSNorm
        self.q_norm = nn.LayerNorm(input_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(input_dim) if qk_norm else nn.Identity()

        # depthwise卷积用于位置编码
        self.position_encoding = nn.Conv2d(input_dim, input_dim, 5, 1, 2, groups=input_dim)

        # window大小
        self.window_size = kwargs["window_size"] if "window_size" in kwargs else 49

        # window的边长
        self.window_len = int(self.window_size ** 0.5)

        # 嵌入长度
        self.embed_len = kwargs["embed_len"] if "embed_len" in kwargs else 196

        # piece的数量
        self.num_pieces = self.embed_len // self.window_size

        # piece的边长
        self.pieces_len = int(self.num_pieces ** 0.5)

        # 局部连接阈值
        local_threshold = kwargs.get("local_threshold", 1.5)

        # 指数衰减参数
        exp_decay_param = kwargs.get("exp_decay_param", 3)

        # 创建BlockDistanceConv模块
        self.piece_attention = BlockDistanceConv(
            num_patches_per_side=int(self.embed_len ** 0.5),
            patch_group_size=self.window_size,
            distance_transform=distance_transform,
            local_threshold=local_threshold,
            exp_decay_param=exp_decay_param,
        )

        # 数值稳定项
        self.eps = kwargs.get("eps", 1e-6)

        # 输出层
        self.to_output = nn.Sequential(nn.Linear(inner_dim, input_dim), nn.Dropout(dropout))

        # 如果指定固定权重初始化
        if fixed_weight_value is not None:
            self._init_weights_with_fixed_value(fixed_weight_value)

    def _init_weights_with_fixed_value(self, value):
        # 初始化模型权重为固定值
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)

        nn.init.constant_(self.to_qkv.weight, value)
        for module in self.to_output:
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, value)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def init_to_value(model, value=1.0):
        # 将模型的所有权重初始化为指定值
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)
        return model

    def _process_qkv_impl(self, q, k, v, B, N, H, D):
        # 对Q和K进行归一化处理
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 对K和Q做ReLU激活，并添加稳定项
        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        # 重排维度
        q, k, v = map(
            lambda t: rearrange(
                t, "b n w (h d) -> (b h) n w d", h=H, d=D
            ),
            (q, k, v)
        )

        # 交换维度用于矩阵乘法
        k = k.transpose(-2, -1)

        return q, k, v

    def _mlp_position_encoding(self, x):
        # 生成QKV并计算位置编码
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # 计算位置编码
        position_encoding = self.position_encoding(
            rearrange(
                v,
                'b (h w) (p1 p2) d -> b d (h p1) (w p2)',
                h=self.pieces_len,
                w=self.pieces_len,
                p1=self.window_len,
                p2=self.window_len
            )
        )

        # 还原位置编码的形状
        position_encoding = rearrange(
            position_encoding,
            'b d (h p1) (w p2) -> b (h w) (p1 p2) d',
            h=self.pieces_len,
            w=self.pieces_len,
            p1=self.window_len,
            p2=self.window_len
        )

        return q, k, v, position_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入进行LayerNorm
        x = self.norm(x)

        # 获取输入形状
        B, N, W, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # 计算QKV和位置编码（这段代码 可以单独摘出 作为一个小创新1）
        q, k, v, position_encoding = self._mlp_position_encoding(x)

        # 处理QKV（使用新的激活函数 进行处理）
        q, k, v = self._process_qkv_impl(q, k, v, B, N, H, D)

        # 计算K和V
        kv = torch.matmul(k, v)
        # 通过BlockDistanceConv建模空间关系（Token 维度分块与混合）
        kv = self.piece_attention(kv)

        # 计算K的求和
        k_sum = k.sum(dim=-1, keepdim=True)

        # 归一化因子
        normalizer = self.piece_attention(torch.matmul(q, k_sum)) + self.eps

        # 计算线性注意力输出
        out = torch.matmul(q, kv) / normalizer

        # 恢复多头维度
        out = rearrange(out, "(b h) n w d -> b n w (h d)", b=B, h=self.num_heads)

        # 加入位置编码
        out = out + position_encoding

        # 输出投影
        return self.to_output(out)

if __name__ == "__main__":
    # 构造输入：假设输入特征维度为(1, 16, 16, 64)，这里的16, 16是图像的高和宽，64是通道数
    input_tensor = torch.randn(1, 16, 16, 64)
    # 创建MHLA_Normed_Torch模型，设置相应的参数
    model = MHLA_Normed_Torch(
        input_dim=64,         # 输入特征维度
        num_heads=8,          # 注意力头数量
        embed_len=256,        # 嵌入长度
        window_size=16,       # 窗口大小
        qkv_bias=False,       # 是否使用偏置
        dropout=0.1,          # Dropout比例
        qk_norm=True,         # 是否对QK进行归一化
    )
    output_tensor = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")