import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

"""
    论文地址：https://arxiv.org/pdf/2602.21917
    论文题目：Scan Clusters, Not Pixels: A Cluster-Centric Paradigm for Efficient Ultra-high-definition Image Restoration（CVPR 2026）
    中文题目：扫描聚类，而非像素：面向高效超高清图像修复的聚类中心范式（CVPR 2026）
    讲解视频：https://www.bilibili.com/video/BV1fVoBBjEyS/
    聚类中心扫描模块（Cluster-Centric Scanning Module，CCSM）
        实际意义：①“逐像素扫描”带来的计算与显存瓶颈：Mamba方法虽可避免Transformer二次复杂度问题，但仍以像素为基本扫描单位。对于4K图像来说，像素数量极其庞大，逐像素扫描会带来很高显存和计算负担，难以在消费级GPU上进行全分辨率推理。
                ②现有扫描方式与自然图像“语义冗余”不匹配的问题：自然图像存在大量语义相近、特征相似的区域。现有逐像素处理机制没有利用图像“特征聚合”和“区域一致性”的统计规律，造成冗余计算。
                ③全局依赖建模效率低的问题：图像任务需要较强的长程依赖建模能力，但如果直接在所有像素上做全局建模，代价高。
        实现方式：通过“特征聚合 → 中心精炼 → Mamba全局建模 → 相似性扩散”四步，将全像素扫描转化为仅扫描 n 个语义中心的簇中心扫描，实现UHD图像高效全局建模。
"""

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    selective_scan_fn = None


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    # 对第一组特征在最后一个维度上做 L2 归一化，便于后续计算余弦相似度
    x1 = F.normalize(x1, dim=-1)
    # 对第二组特征在最后一个维度上做 L2 归一化，便于后续计算余弦相似度
    x2 = F.normalize(x2, dim=-1)
    # 通过矩阵乘法计算两组特征之间的两两余弦相似度
    sim = torch.matmul(x1, x2.transpose(-2, -1))
    # 返回相似度矩阵，形状为 [B, M, N]
    return sim


class ClusterCentricScanningModule(nn.Module):
    def __init__(
        self,
        d_model=32,
        proposal_hw=2,
        fold_hw=1,
        heads=1,
        d_state=8,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        # 调用父类初始化函数
        super().__init__()

        # 统一保存设备和数据类型参数，后面创建层时会直接复用
        factory_kwargs = {"device": device, "dtype": dtype}

        # 输入通道维度，也就是模块输入特征的通道数
        self.d_model = d_model
        # 聚类中心生成时的空间尺寸，proposal_hw=2 表示最终生成 2×2=4 个中心
        self.proposal_hw = proposal_hw
        # 是否先把特征图切块处理，fold_hw>1 时可降低大图上的计算量
        self.fold_hw = fold_hw
        # 多头数，用来把通道分到多个 head 上分别建模
        self.heads = heads
        # 状态空间模型内部的状态维度
        self.d_state = d_state
        # 前面局部卷积的卷积核大小
        self.d_conv = d_conv
        # 通道扩展倍率，类似很多轻量模块里先升维再处理
        self.expand = expand

        # 计算每个 head 内部实际使用的通道数
        self.d_inner = int(self.expand * self.d_model) // self.heads

        # 自动确定 dt 的低秩维度；若用户手动指定，则直接使用指定值
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入线性映射，把输入映射成两部分：x 分支和 z 门控分支
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 深度可分离卷积中的 depthwise 部分，用于进行局部空间建模
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # 使用 SiLU 激活函数增强非线性表达能力
        self.act = nn.SiLU()

        # 这里先定义一个线性层，用于把中心序列映射到 dt、B、C 三类 selective scan 所需参数
        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
        )

        # 把上面线性层的权重取出并堆叠成参数张量，便于后续用 einsum 直接计算
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))

        # 原始线性层对象删除，只保留权重参数
        del self.x_proj

        # 对 dt、B、C 组成的序列再做一维深度卷积，增强序列局部上下文建模能力
        self.x_conv = nn.Conv1d(
            in_channels=self.dt_rank + self.d_state * 2,
            out_channels=self.dt_rank + self.d_state * 2,
            kernel_size=7,
            padding=3,
            groups=self.dt_rank + self.d_state * 2,
        )

        # 初始化 dt 投影层，用于把低秩 dt 参数映射到内部通道维度
        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )

        # 提取 dt 投影层的权重，后续也会通过 einsum 调用
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))

        # 提取 dt 投影层的偏置，这个偏置在 selective scan 中会作为 delta_bias 使用
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))

        # 删除原始投影层对象，只保留参数
        del self.dt_projs

        # 初始化状态空间模型中的 A 参数的对数形式
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)

        # 初始化状态空间模型中的 D 参数
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        # 对 selective scan 输出做 LayerNorm
        self.out_norm = nn.LayerNorm(self.d_inner)

        # 输出线性层，把内部通道映射回原始 d_model 通道数
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 如果设置了 dropout，则启用 dropout，否则为 None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # 相似度分支，把输入映射到多头空间，用来计算像素与中心之间的相似度
        self.f = nn.Conv2d(self.d_inner, self.d_inner * self.heads, kernel_size=1)

        # 值分支，把输入映射到多头空间，用来参与中心聚合和结果扩散
        self.v = nn.Conv2d(self.d_inner, self.d_inner * self.heads, kernel_size=1)

        # 多头处理结束后，再用 1×1 卷积把通道融合回去
        self.proj = nn.Conv2d(self.d_inner * self.heads, self.d_inner, kernel_size=1)

        # 可学习缩放系数，用于调节相似度值的强弱
        self.sim_alpha = nn.Parameter(torch.ones(1))

        # 可学习偏置项，用于平移相似度分布
        self.sim_beta = nn.Parameter(torch.zeros(1))

        # 用自适应平均池化生成固定数量的聚类中心
        # 当 proposal_hw=2 时，会得到 2×2 共 4 个中心
        self.centers_proposal = nn.AdaptiveAvgPool2d((self.proposal_hw, self.proposal_hw))

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        # 创建一个线性层，把低秩 dt 表示映射到 d_inner 维
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # 根据低秩维度计算初始化标准差
        dt_init_std = dt_rank ** -0.5 * dt_scale

        # 如果选择 constant 初始化，则把权重初始化为常数
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        # 如果选择 random 初始化，则在给定范围内均匀采样初始化
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        # 若给了不支持的初始化方式，则直接报错
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        # 在对数空间中采样 dt，再映射回指数空间，以保证 dt 落在指定范围内
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        # 将 dt 变换到更适合 selective scan 使用的偏置形式
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        # 不记录梯度地把这个值拷贝到 bias 中
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        # 标记该 bias 不要被默认重初始化逻辑覆盖
        dt_proj.bias._no_reinit = True

        # 返回初始化后的 dt 投影层
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # 构造从 1 到 d_state 的基础状态序列，并复制到每个内部通道
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()

        # 取对数，得到 A 的对数参数形式
        A_log = torch.log(A)

        # 如果需要多份复制，则按 copies 维度复制
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            # 如果 merge=True，则把复制维和通道维合并
            if merge:
                A_log = A_log.flatten(0, 1)

        # 转成可学习参数
        A_log = nn.Parameter(A_log)

        # 标记这个参数不参与权重衰减
        A_log._no_weight_decay = True

        # 返回 A 的对数参数
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # 初始化 D 为全 1，表示状态空间模型中的直连项
        D = torch.ones(d_inner, device=device)

        # 如果需要多份复制，则按 copies 维度复制
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            # 如果 merge=True，则把复制维和通道维合并
            if merge:
                D = D.flatten(0, 1)

        # 转成可学习参数
        D = nn.Parameter(D)

        # 标记这个参数不参与权重衰减
        D._no_weight_decay = True

        # 返回 D 参数
        return D

    def selective_scan_fallback(self, xs, dts, As, Bs_, Cs_, Ds, delta_bias):
        # 当环境里没有安装 mamba_ssm 时，这里提供一个简化替代版本
        # 注意：这只是为了代码能跑通，不是论文里的严格 selective scan 实现
        y = xs + dts.mean(dim=1, keepdim=True)
        y = y + Bs_.mean(dim=2)
        y = y + Cs_.mean(dim=2)
        y = y + Ds.view(1, -1, 1)
        return y

    def forward_core(self, x):
        # 先通过值分支生成 value 特征，后续用于中心聚合和再分发
        value = self.v(x)
        # 再通过相似度分支生成用于相似度计算的特征
        x = self.f(x)

        # 按多头形式重排，把通道拆成多个 head 分别处理
        x = rearrange(x, "b (e c) h w -> (b e) c h w", e=self.heads)
        # 值分支也同步按多头形式重排
        value = rearrange(value, "b (e c) h w -> (b e) c h w", e=self.heads)

        # 如果设置了 fold_hw>1，则把大特征图切分成多个小块，【减少单次计算量】
        if self.fold_hw > 1:
            # 保存当前张量形状
            b0, c0, h0, w0 = x.shape

            # 检查高和宽能否被 fold_hw 整除，否则无法均匀切块
            assert h0 % self.fold_hw == 0 and w0 % self.fold_hw == 0, \
                f"特征图尺寸 ({h0}, {w0}) 必须能被 fold_hw={self.fold_hw} 整除"

            # 把相似度分支特征按空间切成多个块
            x = rearrange(
                x,
                "b c (f1 h) (f2 w) -> (b f1 f2) c h w",
                f1=self.fold_hw,
                f2=self.fold_hw
            )

            # 把值分支特征按同样方式切块
            value = rearrange(
                value,
                "b c (f1 h) (f2 w) -> (b f1 f2) c h w",
                f1=self.fold_hw,
                f2=self.fold_hw
            )

        # 获取当前特征图尺寸
        b, c, h, w = x.shape

        # 【自适应平均池化生成聚类中心】，形状为 [B, C, proposal_hw, proposal_hw]
        # 对应论文中的“中心学习/中心表示”思想
        centers = self.centers_proposal(x)
        # 对值分支也做同样的池化，得到值特征对应的中心，把它展平为 [B, M, C]，其中 M=proposal_hw×proposal_hw
        value_centers = rearrange(
            self.centers_proposal(value),
            "b c h w -> b (h w) c"
        )

        # 重新获取中心张量的形状
        b, c, ch, cw = centers.shape

        # 【计算中心与所有像素之间的余弦相似度】[相似度]
        # 先把中心展平成 [B, M, C]，把像素展平成 [B, N, C]
        # 再通过 pairwise_cos_sim 得到 [B, M, N]
        # 最后通过可学习缩放和偏置后接 sigmoid，把相似度映射到更稳定的范围
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        # 对每个像素，只保留它最相似的那个中心
        # 这一步相当于做了一次“硬分配”
        _, sim_max_idx = sim.max(dim=1, keepdim=True)
        # 构造一个全 0 掩码
        mask = torch.zeros_like(sim)
        # 在最相似中心的位置填 1
        mask.scatter_(1, sim_max_idx, 1.0)
        # 【利用掩码机制实现只保留最高相似度的中心】
        sim = sim * mask

        # 把 value 特征展平成 [B, N, C]，便于和相似度矩阵做加权聚合
        value_flat = rearrange(value, "b c h w -> b (h w) c")

        # Feature Aggregating：
        # 把所有像素的 value 特征根据相似度加权聚合到各个中心上
        # 再加上原始 value_centers，并做归一化
        # 输出形状为 [B, M, C]
        out = (
            (value_flat.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers
        ) / (sim.sum(dim=-1, keepdim=True) + 1.0)

        # 取出中心序列形状
        B, L, C = out.shape

        # 这里只保留一个扫描方向/组，因此 K=1
        K = 1
        # 把中心序列转成 selective scan 需要的 [B, C, L] 形式
        xs = rearrange(out, "b l c -> b c l")
        # 再扩展出 K 这个维度，整理成 [B, K, D, L] 的形式
        xs = torch.stack([xs], dim=1).view(B, 1, -1, L)
        # 通过 einsum 计算 x_proj 映射，得到 dt、B、C 三类参数的联合表示
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.view(B, K, -1, L),
            self.x_proj_weight
        )
        # 对联合表示做一维深度卷积，增强序列局部相关性
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)
        # 把联合表示拆成 dts、Bs_、Cs_ 三部分
        dts, Bs_, Cs_ = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2
        )

        # 把低秩的 dt 参数投影到 d_inner 维
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(B, K, -1, L),
            self.dt_projs_weight
        )

        # 整理 xs 为 selective scan 需要的 float 类型和形状
        xs = xs.float().view(B, -1, L)
        # 整理 dts 为 selective scan 需要的 float 类型和形状
        dts = dts.contiguous().float().view(B, -1, L)
        # 整理 Bs_ 为 selective scan 需要的 float 类型和形状
        Bs_ = Bs_.float().view(B, K, -1, L)
        # 整理 Cs_ 为 selective scan 需要的 float 类型和形状
        Cs_ = Cs_.float().view(B, K, -1, L)
        # 取出 D 参数
        Ds = self.Ds.float().view(-1)
        # 根据 A_logs 恢复 A 参数，并取负指数形式
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        # 取出 dt 投影偏置
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # 如果已经安装 mamba_ssm，则调用官方 selective_scan_fn
        if HAS_MAMBA:
            out_y = selective_scan_fn(
                xs,
                dts,
                As,
                Bs_,
                Cs_,
                Ds,
                z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, K, -1, L)
        # 否则调用 fallback 简化版本
        else:
            out_y = self.selective_scan_fallback(
                xs, dts, As, Bs_, Cs_, Ds, dt_projs_bias
            ).view(B, K, -1, L)

        # 把 selective scan 的输出重新变回 [B, M, C]
        out = rearrange(out_y[:, 0], "b c l -> b l c")

        # 【相似性扩散】
        # 按照像素与中心的相似度重新扩散回每个像素位置,得到每个像素的全局上下文表示
        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
        # 把像素序列重新恢复成二维特征图 [B, C, H, W]
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        # 如果前面做过切块，这里要把各块重新拼回原图
        if self.fold_hw > 1:
            out = rearrange(
                out,
                "(b f1 f2) c h w -> b c (f1 h) (f2 w)",
                f1=self.fold_hw,
                f2=self.fold_hw
            )

        # 把多头结果重新合并到通道维
        out = rearrange(out, "(b e) c h w -> b (e c) h w", e=self.heads)

        # 用 1×1 卷积做一次通道融合，得到最终核心输出
        out = self.proj(out)

        return out

    def forward(self, x):
        # 把输入从 [B, C, H, W] 转成 [B, H, W, C]
        # 因为后面的线性层 nn.Linear 默认作用在最后一个维度
        x = rearrange(x, "b c h w -> b h w c")

        # 记录当前输入的 batch、高、宽、通道
        B, H, W, C = x.shape

        # 输入投影，把输入映射到更高维的内部空间
        xz = self.in_proj(x)

        # 沿最后一个维度一分为二
        # x 是主分支特征
        # z 是门控分支特征，后面用来做调制
        x, z = xz.chunk(2, dim=-1)

        # 把主分支特征转回卷积格式 [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()

        # 先做一次局部卷积增强，提取局部邻域信息
        x = self.act(self.conv2d(x))

        # 进入 CCSM 主体：
        # 先中心聚合，再 selective scan，再中心扩散
        y = self.forward_core(x)

        # 把输出重新调整成 [B, H, W, C]
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # 对输出做归一化，稳定训练
        y = self.out_norm(y)
        # 用 z 分支做门控调制
        # 对应论文公式里 F_out = F_f · SiLU(MLP(F_in)) 的思想
        y = y * F.silu(z)

        # 通过线性层把内部表示映射回原始通道维度
        out = self.out_proj(y)
        # 再调整回标准图像张量格式 [B, C, H, W]
        out = rearrange(out, "b h w c -> b c h w")
        # 如果启用了 dropout，则做一次随机失活
        if self.dropout is not None:
            out = self.dropout(out)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = ClusterCentricScanningModule(d_model=32)
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")