import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

def index_reverse(index):
    """
    反转索引函数。给定一个索引张量，返回反转后的索引。
    参数:
        index: 一个形状为 [B, HW] 的张量，表示索引。
    返回:
        index_r: 反转后的索引张量，形状与输入相同。
    """
    index_r = torch.zeros_like(index)  # 创建一个与输入索引相同形状的全零张量
    ind = torch.arange(0, index.shape[-1]).to(index.device)  # 生成从0到HW的索引
    for i in range(index.shape[0]):  # 对每个批次进行遍历
        index_r[i, index[i, :]] = ind  # 根据索引赋值
    return index_r

def semantic_neighbor(x, index):
    """
    基于索引重新排列张量，类似于图像卷积中对邻居的操作。
    参数:
        x: 输入张量，形状为 [B, N, C]。
        index: 索引张量，形状为 [B, N]，表示如何重新排列输入张量。
    返回:
        shuffled_x: 按照索引重新排列后的张量，形状与输入相同。
    """
    dim = index.dim()  # 获取索引张量的维度
    assert x.shape[:dim] == index.shape, "x ({:}) 和 index ({:}) 的形状不匹配".format(x.shape, index.shape)  # 确保输入和索引形状一致

    # 根据输入张量和索引的维度关系调整索引的维度
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)  # 扩展索引以匹配输入张量的形状

    # 按照索引重新排列输入张量
    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x

class Selective_Scan(nn.Module):
    """
    Selective Scan 模块，执行选择性扫描操作。
    """
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        """
        初始化 Selective_Scan 模块的各个参数。
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  # 输入模型的维度
        self.d_state = d_state  # 状态维度
        self.expand = expand  # 扩展比例
        self.d_inner = int(self.expand * self.d_model)  # 内部维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 动态排名

        # 输入投影层
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # 投影权重
        del self.x_proj  # 删除原始投影层

        # 初始化 dt 投影层
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # 投影权重
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # 投影偏置
        del self.dt_projs  # 删除原始投影层

        # 初始化 A_log 和 D 参数
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # A_log 初始化
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # D 初始化
        self.selective_scan = selective_scan_fn  # 选择性扫描函数

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        """
        初始化 dt 投影层，用于保持初始化时的方差。
        """
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # 初始化 dt 投影权重
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化偏置，使得 softplus 函数的输出在 [dt_min, dt_max] 范围内
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # 计算 softplus 的反函数
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)  # 复制到偏置
        dt_proj.bias._no_reinit = True  # 标记该偏置不需要重新初始化

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        """
        初始化 A_log，用于实现 S4D 的特殊初始化。
        """
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # 将 A 转为对数形式
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)  # 扩展 A_log
            if merge:
                A_log = A_log.flatten(0, 1)  # 合并维度
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True  # 标记不需要权重衰减
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        """
        初始化 D 参数，用于实现跳跃连接（skip connection）。
        """
        D = torch.ones(d_inner, device=device)  # 初始化为全 1
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)  # 扩展 D
            if merge:
                D = D.flatten(0, 1)  # 合并维度
        D = nn.Parameter(D)  # 保持为浮点类型
        D._no_weight_decay = True  # 标记不需要权重衰减
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        """
        核心前向传播函数，执行选择性扫描操作。
        """
        B, L, C = x.shape  # 获取批量大小、序列长度和输入通道数
        K = 1  # mambairV2 只需要 1 次扫描
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # 调整维度为 B, 1, C, L

        # 计算投影后的值
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # 计算 dt 投影
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # 处理为 (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) + prompt  # 加入 prompt（提示）
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # dt 投影偏置

        # 执行选择性扫描操作
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float  # 检查输出类型

        return out_y[:, 0]  # 返回第一维的输出

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        """
        前向传播函数，调用核心前向传播并返回结果。
        """
        b, l, c = prompt.shape  # 获取 prompt 的形状
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)  # 转置并调整形状
        y = self.forward_core(x, prompt)  # 调用核心前向传播
        y = y.permute(0, 2, 1).contiguous()  # 调整输出形状
        return y  # 返回结果


class ASSM(nn.Module):
    """
    注意力状态空间模块（Attentive State Space Module，ASSM）用于处理输入并执行选择性扫描操作, 实现非因果建模。
    """
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim  # 输入维度
        self.input_resolution = input_resolution  # 输入分辨率
        self.num_tokens = num_tokens  # 令牌数量
        self.inner_rank = inner_rank  # 内部秩

        # Mamba 参数
        self.expand = mlp_ratio  # 扩展比率
        hidden = int(self.dim * self.expand)  # 隐藏层维度
        self.d_state = d_state  # 状态维度
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)  # 输出归一化层
        self.act = nn.SiLU()  # 激活函数
        self.out_proj = nn.Linear(hidden, dim, bias=True)  # 输出投影

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),  # 输入投影层
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),  # CPE 层
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # 令牌嵌入层
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        """
        前向传播函数，处理输入并执行选择性扫描操作。
        """
        B, n, C = x.shape
        H, W = x_size  # 高度和宽度

        # 生成全局嵌入
        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        # 预测路由并使用 Gumbel-softmax 采样
        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        # 获取最大类别的索引
        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        # 输入通过卷积进行投影
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))  # 加入 CPE 调制
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b, n, c

        # 使用选择性扫描处理语义邻域
        semantic_x = semantic_neighbor(x, x_sort_indices)  # SGN-unfold
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))  # 输出层
        x = semantic_neighbor(y, x_sort_indices_reverse)  # SGN-fold

        return x  # 返回最终的处理结果

if __name__ == '__main__':
    # 设置输入参数
    B, H, W = 1, 224, 224  # batch_size, 高度和宽度
    dim = 32  # 输入通道数 (dim)，即输入张量的通道数
    d_state = 32  # 状态维度 (d_state)，在模型中作为内部状态的维度
    input_resolution = (H, W)  # 输入分辨率 (input_resolution)，模型接受的输入图像的空间尺寸 (H, W)
    num_tokens = 32  # token 数量 (num_tokens)，表示嵌入层中的 token 数量
    inner_rank = 32  # 内部秩, 与低秩分解等操作相关
    mlp_ratio = 2.  # MLP 比例, 用于定义 MLP 层的宽度比例，通常用于控制隐藏层的规模

    # 创建输入张量
    x = torch.randn(B, H * W, dim).cuda()  # 输入形状为 (B, HW, dim)
    token = nn.Embedding(num_tokens, dim).cuda()  # 假设token是一个嵌入层
    x_size = (H, W)

    # 创建ASSM模块
    model = ASSM(dim=dim, d_state=d_state, input_resolution=input_resolution, num_tokens=num_tokens, inner_rank=inner_rank, mlp_ratio=mlp_ratio).cuda()

    # 打印模型结构
    print(model)
    print("\n微信公众号: AI缝合术!\n")

    # 前向传播
    output = model(x, x_size, token)
    
    # 打印输入和输出的形状
    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")