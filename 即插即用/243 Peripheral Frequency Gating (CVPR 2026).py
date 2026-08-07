import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2026/html/Cai_PFGNet_A_Fully_Convolutional_Frequency-Guided_Peripheral_Gating_Network_for_Efficient_CVPR_2026_paper.html
    论文题目：PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning (CVPR 2026)
    中文题目：PFGNet：面向高效时空预测学习的全卷积频域引导外围门控网络(CVPR 2026)
    讲解视频：https://www.bilibili.com/video/BV198EQ6EEt3/
    外围频率门控模块（Peripheral Frequency Gating ,PFG）
        实际意义：①固定卷积感受野难以适应复杂运动的问题：不同空间位置需要的上下文范围并不一样，但普通卷积无法按像素动态调整感受野。例如：平滑背景区域不需要很大感受野，否则会引入冗余信息；目标边界区域需要关注局部结构变化。
                ②简单大卷积核“看得远但不够精准”的问题：卷积核被证明可以扩大感受野、增强长距离依赖能力。但论文指出，大卷积核会引入有用结构信息、冗余低频背景。
                ③低频背景冗余和高频噪声干扰的问题：模型容易受到两类无效信息干扰：第一类是低频背景冗余，这些区域占据大量像素，但对未来运动预测帮助有限。第二类是高频噪声干扰，局部噪声、不稳定纹理，影响模型对真实运动结构的判断。
        实现方式：通过“局部频率线索提取—像素级尺度门控—多尺度大核外围响应—可学习中心抑制”的流程，自适应增强边缘、纹理，同时抑制背景冗余与噪声干扰。

"""

class DropPath(nn.Module):
    # 定义 DropPath 模块，用于实现随机深度，训练时随机丢弃残差分支，推理时保持不变
    def __init__(self, drop_prob: float = 0.0) -> None:
        # 调用父类 nn.Module 的初始化方法
        super().__init__()

        # 将传入的丢弃概率转换为 float 类型，避免后续计算出现类型问题
        self.drop_prob = float(drop_prob)

    # 定义前向传播函数，输入 x 的形状通常为 [B, C, H, W]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果丢弃概率为 0，或者当前不是训练模式，则直接返回输入，相当于不使用 DropPath
        if self.drop_prob == 0.0 or not self.training:
            return x

        # keep_prob 表示残差分支被保留下来的概率
        keep_prob = 1.0 - self.drop_prob

        # 构造随机张量的形状，只在 batch 维度上随机，对通道和空间维度共享同一个保留状态
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        # 生成随机张量，数值范围为 [keep_prob, keep_prob + 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

        # 对随机张量向下取整，得到 0 或 1 的二值掩码
        random_tensor.floor_()

        # 对保留的分支进行除以 keep_prob 的缩放，保证训练时输出期望不变
        return x.div(keep_prob) * random_tensor


class GRN(nn.Module):
    # 定义 Global Response Normalization，全局响应归一化模块，用于增强通道间响应稳定性
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        # 调用父类 nn.Module 的初始化方法
        super().__init__()

        # 定义可学习缩放参数 gamma，形状为 [1, C, 1, 1]，可对每个通道单独缩放
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

        # 定义可学习偏置参数 beta，形状为 [1, C, 1, 1]，可对每个通道单独平移
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # 定义一个很小的数，防止除法时分母为 0
        self.eps = eps

    # 定义前向传播函数，输入 x 的形状为 [B, C, H, W]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在空间维度 H 和 W 上计算 L2 范数，得到每个通道的全局响应强度
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)

        # 用全局响应强度对特征进行归一化，得到相对响应特征
        nx = x / (gx + self.eps)

        # 将原始特征、缩放后的归一化特征、偏置项相加，形成残差式归一化输出
        return x + self.gamma * nx + self.beta


class PFGA(nn.Module):
    # 定义 Peripheral-Frequency Guided Aggregation，即频率引导的外周响应聚合模块
    class Branch(nn.Module):
        # 定义单个尺度分支，每个分支对应一个大核尺度，例如 9、15 或 31
        def __init__(self, channels: int, kernel_size: int, center_suppress: bool = True) -> None:
            # 调用父类 nn.Module 的初始化方法
            super().__init__()

            # 如果卷积核大小不是奇数，则无法进行对称 padding，因此直接报错
            if kernel_size % 2 == 0:
                raise ValueError("kernel_size must be odd.")

            # 保存是否启用中心抑制机制
            self.center_suppress = center_suppress

            # 定义横向深度卷积，卷积核为 1×k，只在宽度方向扩大感受野
            self.dw_h = nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=channels,
                bias=False,
            )

            # 定义纵向深度卷积，卷积核为 k×1，只在高度方向扩大感受野
            self.dw_v = nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
                groups=channels,
                bias=False,
            )

            # 如果启用中心抑制，则额外构建一个 3×3 深度卷积分支
            if center_suppress:
                # 定义中心区域卷积，用于提取局部中心响应
                self.dw_center = nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    groups=channels,
                    bias=False,
                )

                # 定义可学习中心抑制系数 beta，每个通道一个系数
                self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

            # 如果不启用中心抑制，则不构建中心卷积分支
            else:
                # 将中心卷积分支设置为空
                self.dw_center = None

                # 注册一个空参数 beta，使模块结构保持一致
                self.register_parameter("beta", None)

        # 定义单尺度分支的前向传播
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 先经过横向 1×k 深度卷积，再经过纵向 k×1 深度卷积，近似实现 k×k 大核卷积
            peripheral = self.dw_v(self.dw_h(x))

            # 如果启用中心抑制，则从外周响应中减去中心响应
            if self.center_suppress:
                # 使用 3×3 深度卷积提取中心局部响应
                center = self.dw_center(x)

                # 使用 tanh 将 beta 限制到 [-1, 1]，再对中心响应进行可学习抑制或增强
                peripheral = peripheral - torch.tanh(self.beta) * center

            # 返回当前尺度下的外周响应结果
            return peripheral

    # 定义 PFGA 模块的初始化函数
    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...] = (9, 15, 31),
        center_suppress: bool = True,
        use_grn: bool = False,
    ) -> None:
        # 调用父类 nn.Module 的初始化方法
        super().__init__()

        # 保存所有大核尺度，论文中常用的尺度为 9、15、31
        self.kernel_sizes = kernel_sizes

        # 为每一个大核尺度构建一个外周响应分支
        self.branches = nn.ModuleList(
            [PFGA.Branch(channels, k, center_suppress=center_suppress) for k in kernel_sizes]
        )

        # 构造 Sobel-X 卷积核，用于检测水平方向梯度变化
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # 构造 Sobel-Y 卷积核，用于检测垂直方向梯度变化
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # 构造 Laplacian 卷积核，用于检测局部二阶变化，也就是边缘和突变区域
        laplace = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # 将 Sobel-X 注册为 buffer，使其随模型移动到 GPU，但不作为可学习参数更新
        self.register_buffer("sobel_x", sobel_x, persistent=False)

        # 将 Sobel-Y 注册为 buffer，使其固定用于频率特征提取
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        # 将 Laplacian 注册为 buffer，使其固定用于局部高频响应提取
        self.register_buffer("laplace", laplace, persistent=False)

        # 用 1×1 卷积把 3 通道频率描述图映射为 K 个尺度的门控 logits
        self.gate_head = nn.Conv2d(3, len(kernel_sizes), kernel_size=1, bias=True)

        # 如果 use_grn=True，则在输出端使用 GRN，否则直接使用恒等映射
        self.grn = GRN(channels) if use_grn else nn.Identity()

    # 定义深度卷积形式的固定滤波操作
    def _depthwise_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        # 获取输入特征的通道数 C
        channels = x.shape[1]

        # 将单通道固定卷积核复制 C 份，使每个通道都使用同一个滤波器
        weight = kernel.repeat(channels, 1, 1, 1)

        # 使用 groups=channels 实现逐通道卷积，不混合不同通道的信息
        return F.conv2d(x, weight, padding=1, groups=channels)

    # 定义频率描述图提取函数
    def _frequency_maps(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 Sobel-X 提取水平方向梯度响应
        gx = self._depthwise_filter(x, self.sobel_x)

        # 使用 Sobel-Y 提取垂直方向梯度响应
        gy = self._depthwise_filter(x, self.sobel_y)

        # 使用 Laplacian 提取局部二阶变化响应
        lap = self._depthwise_filter(x, self.laplace)

        # 计算梯度幅值，用于描述局部边缘强度
        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

        # 计算 3×3 邻域内的局部均值
        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # 计算 3×3 邻域内平方特征的局部均值
        mean2 = F.avg_pool2d(x * x, kernel_size=3, stride=1, padding=1)

        # 根据 Var(X)=E(X²)-E(X)² 计算局部方差，并将负数截断为 0
        var = torch.clamp(mean2 - mean * mean, min=0.0)

        # 对梯度幅值在通道维度求平均，得到单通道梯度频率图
        f_grad = grad_mag.mean(dim=1, keepdim=True)

        # 对 Laplacian 绝对值在通道维度求平均，得到单通道曲率频率图
        f_lap = lap.abs().mean(dim=1, keepdim=True)

        # 对局部方差在通道维度求平均，得到单通道纹理变化频率图
        f_var = var.mean(dim=1, keepdim=True)

        # 将三种频率提示拼接为 [B, 3, H, W] 的频率描述图
        return torch.cat([f_grad, f_lap, f_var], dim=1)

    # 定义 PFGA 模块的前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分别计算每一个大核尺度分支的外周响应
        peripheral_responses = [branch(x) for branch in self.branches]

        # 从输入特征中提取梯度、Laplacian 和局部方差组成的频率描述图
        freq = self._frequency_maps(x)
        # 使用 1×1 卷积将频率描述图转换为不同尺度的门控 logits
        logits = self.gate_head(freq)
        # 在尺度维度上做 softmax，使每个像素位置的所有尺度权重之和为 1
        alpha = torch.softmax(logits, dim=1)

        # 创建一个和单个外周响应同形状的零张量，用于累加多尺度结果
        y = torch.zeros_like(peripheral_responses[0])
        # 遍历每个尺度的响应，并使用对应的门控权重进行加权融合
        for idx, response in enumerate(peripheral_responses):
            # alpha[:, idx:idx+1] 的形状为 [B, 1, H, W]，可广播到所有通道
            y = y + response * alpha[:, idx: idx + 1]

        # 对融合后的外周响应进行可选 GRN 处理，然后返回结果
        return self.grn(y)


class PFGBlock(nn.Module):
    # 定义完整的 PFG Block，输入和输出形状均为 [B, C, H, W]
    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...] = (9, 15, 31),
        mlp_ratio: float = 4.0,
        dw_kernel: int = 3,
        groups_pw: int = 1,
        layerscale_init: float = 1e-6,
        drop: float = 0.0,
        drop_path: float = 0.0,
        center_suppress: bool = True,
    ) -> None:
        # 调用父类 nn.Module 的初始化方法
        super().__init__()

        # 保存输入和输出通道数，PFGBlock 不改变通道数
        self.channels = channels

        # 定义空间混合前的归一化层，GroupNorm 对 batch size 不敏感，适合小 batch 训练
        self.norm_dw = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)

        # 定义通道混合前的归一化层
        self.norm_pw = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)

        # 构建频率引导的多尺度外周响应聚合模块
        self.token_mixer = PFGA(
            channels,
            kernel_sizes=kernel_sizes,
            center_suppress=center_suppress,
            use_grn=False,
        )

        # 定义空间混合分支后的 GRN 归一化模块
        self.grn_dw = GRN(channels)

        # 定义通道混合分支后的 GRN 归一化模块
        self.grn_pw = GRN(channels)

        # 根据 mlp_ratio 计算通道扩展后的隐藏层通道数，默认扩展为 4 倍
        hidden_channels = max(channels, int(channels * mlp_ratio))

        # 使用 1×1 卷积将通道从 C 扩展到 2E，后续会分成 u 和 v 两部分
        self.pw_in = nn.Conv2d(
            channels,
            2 * hidden_channels,
            kernel_size=1,
            bias=True,
            groups=groups_pw,
        )

        # 对 v 分支使用 depthwise 卷积，引入局部空间上下文
        self.dw_v = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=dw_kernel,
            padding=dw_kernel // 2,
            groups=hidden_channels,
            bias=False,
        )

        # 使用 1×1 卷积将隐藏通道数从 E 压回原始通道数 C
        self.pw_out = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=True,
            groups=groups_pw,
        )

        # 定义 GELU 激活函数，用于空间混合分支
        self.act = nn.GELU()

        # 定义空间混合分支的 LayerScale 参数，初始值很小，使训练初期更稳定
        self.gamma_dw = nn.Parameter(torch.ones(channels) * layerscale_init)

        # 定义通道混合分支的 LayerScale 参数，初始值同样很小
        self.gamma_pw = nn.Parameter(torch.ones(channels) * layerscale_init)

        # 如果 drop 大于 0，则使用 Dropout，否则使用恒等映射
        self.dropout_dw = nn.Dropout(drop) if drop > 0 else nn.Identity()

        # 如果 drop 大于 0，则在通道混合分支后使用 Dropout，否则不做处理
        self.dropout_pw = nn.Dropout(drop) if drop > 0 else nn.Identity()

        # 如果 drop_path 大于 0，则启用随机深度，否则使用恒等映射
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # 调用参数初始化函数，对卷积和归一化层进行初始化
        self._init_parameters()

    # 定义模型参数初始化函数
    def _init_parameters(self) -> None:
        # 遍历当前模块中的所有子模块
        for module in self.modules():
            # 如果当前子模块是二维卷积层，则初始化卷积参数
            if isinstance(module, nn.Conv2d):
                # 使用 Kaiming 正态初始化卷积权重，适合搭配 ReLU/GELU 类激活函数
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

                # 如果卷积层存在 bias，则将 bias 初始化为 0
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # 如果当前子模块是 GroupNorm，则初始化归一化参数
            elif isinstance(module, nn.GroupNorm):
                # 将归一化缩放参数初始化为 1
                nn.init.ones_(module.weight)

                # 将归一化偏置参数初始化为 0
                nn.init.zeros_(module.bias)

    # 定义完整 PFGBlock 的前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入特征做归一化，为后续频率引导空间混合做准备
        y = self.norm_dw(x)
        # 使用 PFGA 进行频率引导的多尺度外周响应建模
        y = self.token_mixer(y)

        # 对空间混合结果使用 GELU 激活，增强非线性表达能力
        y = self.act(y)
        # 对空间混合结果使用 GRN，稳定全局响应分布
        y = self.grn_dw(y)
        # 对空间混合结果进行 Dropout 或恒等映射
        y = self.dropout_dw(y)

        # 使用 LayerScale 缩放空间混合分支，并通过残差连接加回原输入
        x = x + self.drop_path(y * self.gamma_dw.view(1, self.channels, 1, 1))
        # 对更新后的特征做归一化，为后续通道混合做准备
        z = self.norm_pw(x)
        # 使用 1×1 卷积扩展通道，并沿通道维度均分为 u 和 v 两个分支
        u, v = torch.chunk(self.pw_in(z), chunks=2, dim=1)

        # 对 v 分支使用 depthwise 卷积，引入局部空间信息
        v = self.dw_v(v)
        # 使用 SiLU(u) 作为门控信号，对 v 分支进行逐元素调制，形成 GLU 风格通道混合
        z = F.silu(u) * v
        # 使用 1×1 卷积将通道数从 hidden_channels 压回 channels
        z = self.pw_out(z)
        # 对通道混合后的特征使用 GRN，增强训练稳定性
        z = self.grn_pw(z)
        # 对通道混合结果进行 Dropout 或恒等映射
        z = self.dropout_pw(z)

        # 使用 LayerScale 缩放通道混合分支，并通过残差连接加回主干特征
        x = x + self.drop_path(z * self.gamma_pw.view(1, self.channels, 1, 1))
        return x


if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    block = PFGBlock(channels=32)
    y = block(x)
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")