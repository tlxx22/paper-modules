import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/download/38042/42004
    论文题目：Gradient as Conditions: Rethinking HOG for All-in-one Image Restoration（AAAI 2026）
    中文题目：梯度作为条件：为一体化图像恢复重新思考方向梯度直方图（HOG）的应用（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1Y79sBxE1v/
    动态 HOG 感知自注意力（Dynamic HOG-Aware Self-Attention，DHOGSA）
        实际意义：①普通注意力“看得远”，但“不懂退化类型”问题：传统自注意力能建立长距离关系，但更像“看哪些位置彼此相关”，却不太清楚这些相关性是不是和噪声（雨、雪、雾）有关。
                ②“有用结构”和“干扰纹理”混在一起问题：既要保留真实边缘、纹理和结构，又要去掉退化干扰。如果分不清哪些梯度是图像本身结构，哪些是退化造成的伪结构，就容易把噪声当细节保留下来。
        实现方式：先用 HOG（方向梯度直方图）对特征进行像素级+Patch级排序，再计算双分支注意力，实现退化自适应动态长距离建模，实现对退化特异性、多尺度长距离空间依赖的精准捕获。
                HOG 描述图像局部边缘和纹理结构的传统特征方法，不直接看“这是什么物体”，而是看图像变化剧烈程度，以及不同方向梯度有多强。
"""

class Attention_DHOGSA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        bias=False,
        ifBox=True,
        patch_size=8,
        clip_limit=1.0,
        n_bins=9
    ):
        super(Attention_DHOGSA, self).__init__()

        # factor 用于后续 token 重排时的分组因子，这里直接设为注意力头数
        self.factor = num_heads

        # ifBox 是一个控制标志，用来决定 reshape_attn 中采用哪一种 token 重排方式
        self.ifBox = ifBox

        # 多头注意力的头数
        self.num_heads = num_heads

        # patch_size 表示做 patch 划分时，每个 patch 的边长
        self.patch_size = patch_size

        # n_bins 表示 HOG 方向直方图的分桶数量
        self.n_bins = n_bins

        # temperature 是每个注意力头独立的可学习缩放参数
        # 作用是调节注意力分数的“温度”，影响 softmax 前分数的尖锐程度
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkv 是一个 1x1 卷积，用来把输入特征映射为 5 路特征
        # 分别对应 q1、k1、q2、k2、v
        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)

        # qkv_dwconv 是一个深度可分离卷积
        # 用来在生成 q1、k1、q2、k2、v 之后，再补充局部空间信息
        self.qkv_dwconv = nn.Conv2d(
            dim * 5,
            dim * 5,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 5,
            bias=bias
        )

        # project_out 是输出投影层
        # 用于将最后融合后的特征重新映射回原始通道维度
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # bin_proj 用于将 patch 级别的 HOG 统计结果映射为特征图
        # 输入通道数是 HOG 的方向桶数 n_bins
        # 输出通道数是输入通道的一半 dim // 2
        self.bin_proj = nn.Conv2d(n_bins, dim // 2, kernel_size=1, bias=bias)

        # 构建 Sobel x 方向卷积核
        # 它主要用于提取图像或特征图在水平方向上的梯度
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).reshape(1, 1, 3, 3)

        # 构建 Sobel y 方向卷积核
        # 它主要用于提取图像或特征图在垂直方向上的梯度
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32
        ).reshape(1, 1, 3, 3)

        # register_buffer 表示把 sobel_x 注册成模型缓冲区
        # 它不会被当成可学习参数，但会跟随模型一起保存和迁移设备
        # repeat(dim, 1, 1, 1) 表示为每个通道复制一份 Sobel 卷积核
        self.register_buffer("sobel_x", sobel_x.repeat(dim, 1, 1, 1))

        # 同理，为每个通道注册 sobel_y 卷积核
        self.register_buffer("sobel_y", sobel_y.repeat(dim, 1, 1, 1))

    def pad(self, x, factor):
        # 这个函数的作用是：
        # 把最后一个维度补零到 factor 的整数倍，方便后续 reshape 和分组计算

        # 获取最后一个维度的长度
        n = x.shape[-1]

        # 如果 n 已经是 factor 的整数倍，就不用补
        # 否则补到最近的 factor 整数倍
        pad_right = 0 if n % factor == 0 else (n // factor + 1) * factor - n

        # 在最后一个维度右侧补零
        x = F.pad(x, (0, pad_right), mode='constant', value=0)

        # 返回补零后的张量，以及补零信息，后面恢复时要用
        return x, (0, pad_right)

    def unpad(self, x, pad_info):
        # 这个函数的作用是：
        # 把 pad 函数补上的多余位置去掉，恢复原始长度

        # 取出右侧补零的数量
        _, pad_right = pad_info

        # 如果没有补零，直接返回
        if pad_right == 0:
            return x

        # 去掉最后一个维度右侧补上的 pad_right 个元素
        return x[:, :, :-pad_right]

    def softmax_1(self, x, dim=-1):
        # 这是原始代码里使用的一种“非标准 softmax”
        # 与标准 softmax 最大区别在于分母后面额外加了 1
        # 这种写法可能是为了数值更平稳，但不属于常规 softmax 定义

        # 对输入做指数映射
        logit = x.exp()

        # 按指定维度归一化，并在分母加 1 防止结果过大
        logit = logit / (logit.sum(dim, keepdim=True) + 1)

        # 返回归一化后的结果
        return logit

    def reshape_attn(self, q, k, v, ifBox):
        # 这个函数的核心作用是：
        # 先把 q、k、v 在 token 维度上重新排布，再做多头注意力计算

        # b 是 batch size，c 是通道数
        b, c = q.shape[:2]

        # 对 q、k、v 的最后一个维度做补零
        # 这样后续可以按 factor 整除并重排
        q, pad_info = self.pad(q, self.factor)
        k, _ = self.pad(k, self.factor)
        v, _ = self.pad(v, self.factor)

        # 计算重排后的 hw 长度
        # 因为最后一个维度会被 factor 划分
        hw = q.shape[-1] // self.factor

        # ifBox=True 时，采用一种重排方式
        # 原始形状: [B, C, N]
        # 重排后: [B, head, c_per_head * factor, hw]
        if ifBox:
            q = rearrange(
                q,
                'b (head c) (factor hw) -> b head (c factor) hw',
                factor=self.factor,
                hw=hw,
                head=self.num_heads
            )
            k = rearrange(
                k,
                'b (head c) (factor hw) -> b head (c factor) hw',
                factor=self.factor,
                hw=hw,
                head=self.num_heads
            )
            v = rearrange(
                v,
                'b (head c) (factor hw) -> b head (c factor) hw',
                factor=self.factor,
                hw=hw,
                head=self.num_heads
            )
        else:
            # ifBox=False 时，采用另一种重排方式
            q = rearrange(
                q,
                'b (head c) (hw factor) -> b head (c factor) hw',
                factor=self.factor,
                hw=hw,
                head=self.num_heads
            )
            k = rearrange(
                k,
                'b (head c) (hw factor) -> b head (c factor) hw',
                factor=self.factor,
                hw=hw,
                head=self.num_heads
            )
            v = rearrange(
                v,
                'b (head c) (hw factor) -> b head (c factor) hw',
                factor=self.factor,
                hw=hw,
                head=self.num_heads
            )

        # 对 q 和 k 做归一化
        # 常见目的是稳定点积范围，使注意力计算更平滑
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 计算注意力分数
        # q @ k.transpose(-2, -1) 表示做矩阵乘法
        # 再乘以可学习温度参数 self.temperature
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 使用自定义 softmax 做归一化
        attn = self.softmax_1(attn, dim=-1)

        # 用注意力权重加权 v，得到输出
        out = attn @ v

        # 将输出从多头重排回原始 token 排布形式
        if ifBox:
            out = rearrange(
                out,
                'b head (c factor) hw -> b (head c) (factor hw)',
                factor=self.factor,
                hw=hw,
                b=b,
                head=self.num_heads
            )
        else:
            out = rearrange(
                out,
                'b head (c factor) hw -> b (head c) (hw factor)',
                factor=self.factor,
                hw=hw,
                b=b,
                head=self.num_heads
            )

        # 去掉前面为了整除而补上的零
        out = self.unpad(out, pad_info)

        # 返回注意力计算后的结果
        return out

    def split_into_patches(self, x):
        # 这个函数的作用是：
        # 把输入特征图按照 patch_size 划分成多个 patch

        # 获取输入形状
        b, c, h, w = x.shape

        # 计算高度方向需要补多少，保证能被 patch_size 整除
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size

        # 计算宽度方向需要补多少，保证能被 patch_size 整除
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        # 如果高度或宽度不能整除 patch_size，就进行补零
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 使用 rearrange 把特征图拆成 patch 序列
        # 输出形状为 [B, N_patch, C, patch_area]
        patches = rearrange(
            x,
            'b c (h p1) (w p2) -> b (h w) c (p1 p2)',
            p1=self.patch_size,
            p2=self.patch_size
        )

        # 计算划分后高度方向上 patch 的数量
        n_h = (h + pad_h) // self.patch_size

        # 计算划分后宽度方向上 patch 的数量
        n_w = (w + pad_w) // self.patch_size

        # 把恢复原图所需的信息都保存下来
        shape_info = (b, c, h, w, pad_h, pad_w, n_h, n_w)

        # 返回 patch 序列和形状信息
        return patches, shape_info

    def merge_patches(self, patches, shape_info):
        # 这个函数的作用是：
        # 把 patch 序列重新还原为特征图

        # 读取之前保存的形状信息
        b, c, h, w, pad_h, pad_w, n_h, n_w = shape_info

        # 使用 rearrange 把 patch 拼回二维特征图
        x = rearrange(
            patches,
            'b (h w) c (p1 p2) -> b c (h p1) (w p2)',
            h=n_h,
            w=n_w,
            p1=self.patch_size,
            p2=self.patch_size
        )

        # 如果之前做过补零，这里把多余的区域裁掉，恢复原始大小
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]

        # 返回恢复后的特征图
        return x

    def apply_hog_to_patch(self, x_half):
        # 这个函数的作用是：
        # 对输入特征图的前半部分通道做 patch 级别的 HOG 引导排序

        # 获取输入形状
        b, c, h, w = x_half.shape

        # 用 Sobel x 核提取水平方向梯度
        gx = F.conv2d(x_half, self.sobel_x[:c], padding=1, groups=c)

        # 用 Sobel y 核提取垂直方向梯度
        gy = F.conv2d(x_half, self.sobel_y[:c], padding=1, groups=c)

        # 根据 gx 和 gy 计算梯度幅值
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        # 根据 gx 和 gy 计算梯度方向，范围是 [-pi, pi]
        orientation = torch.atan2(gy, gx)

        # 把连续方向值映射到离散的 HOG bin 中
        # 最终每个像素都会被分配到一个方向桶
        orientation_bin = ((orientation + torch.pi) / (2 * torch.pi) * self.n_bins).long() % self.n_bins

        # 将原始特征 x_half 划分为 patch
        patches_x, shape_info = self.split_into_patches(x_half)

        # 将梯度幅值图也划分为 patch
        patches_mag, _ = self.split_into_patches(magnitude)

        # 将方向 bin 图划分为 patch
        patches_ori, _ = self.split_into_patches(orientation_bin.float())

        # 读取 patch 形状
        b, n_patches, c, patch_pixels = patches_x.shape

        # sort_values 用于存储后续排序依据
        sort_values = torch.zeros_like(patches_x)

        # hog_features 用于存储每个 patch 的 HOG 统计结果
        # 形状为 [B, N_patch, n_bins]
        hog_features = torch.zeros(b, n_patches, self.n_bins, device=x_half.device)

        # 遍历每一个 HOG 方向桶
        for i in range(self.n_bins):
            # 构造当前方向桶的掩码
            bin_mask = (patches_ori == i).float()

            # 只保留当前方向桶对应位置的幅值
            bin_magnitude = patches_mag * bin_mask

            # 用不同方向桶的编号作为加权系数，累加得到排序依据
            sort_values += bin_magnitude * (i + 1)

            # 对每个 patch，在当前方向桶上统计平均响应
            # dim=[-1, -2] 表示同时对 patch 内像素和通道求均值
            hog_features[..., i] = bin_magnitude.mean(dim=[-1, -2])

        # 对每个 patch 的 HOG 统计向量做归一化
        hog_features = hog_features / (hog_features.sum(dim=-1, keepdim=True) + 1e-8)

        # 根据 sort_values 对 patch 内部像素位置排序
        # 这里先在通道维上求和，得到每个 patch 内每个像素的综合排序值
        # 然后沿最后一个维度排序，得到排序索引
        _, sort_indices = sort_values.sum(dim=2, keepdim=True).expand_as(patches_x).sort(dim=-1)

        # 根据排序索引重新排列 patch 内部像素
        patches_x_sorted = torch.gather(patches_x, -1, sort_indices)

        # 把排序后的 patch 序列重新恢复成特征图
        x_half_processed = self.merge_patches(patches_x_sorted, shape_info)

        # 返回处理后的前半通道特征、排序索引、HOG 特征以及形状信息
        return x_half_processed, sort_indices, hog_features, shape_info

    def forward(self, x):
        # 获取输入形状
        b, c, h, w = x.shape
        # 要求输入通道数必须是偶数
        # 因为后续会把通道一分为二，前一半做 HOG 引导处理，后一半保持原始分支
        assert c % 2 == 0, "输入通道数 dim 必须为偶数，因为模块内部会将通道一分为二。"
        # half_c 表示一半的通道数
        half_c = c // 2

        # 第一步：取前半部分通道
        x_half = x[:, :half_c, :, :]
        # 对前半通道执行 patch 级 HOG 引导排序
        # 得到处理后的特征、patch 内排序索引、HOG 统计特征和形状信息【先用 HOG（方向梯度直方图）对特征进行像素级+Patch级排序】
        x_half_processed, idx_patch, hog_features, shape_info = self.apply_hog_to_patch(x_half)

        # 第二步：把 patch 级别的 HOG 统计特征恢复为空间图
        # 先读取 patch 数量
        _, n_patches, _ = hog_features.shape
        # 读取 patch 网格的高度和宽度
        n_h = shape_info[-2]
        n_w = shape_info[-1]
        # 将 [B, N_patch, n_bins] 变成 [B, n_bins, n_h, n_w]
        # 这样就变成了 patch 网格上的“方向统计图”
        hog_map = rearrange(
            hog_features,
            'b (nh nw) bins -> b bins nh nw',
            nh=n_h,
            nw=n_w
        ).contiguous()
        # 使用 1x1 卷积把 HOG 统计图映射成半通道特征图
        hog_map = self.bin_proj(hog_map)
        # 再插值上采样到原始空间分辨率
        hog_map = F.interpolate(hog_map, size=(h, w), mode='bilinear', align_corners=False)

        # 把处理后的前半通道与原始后半通道拼接
        # 其中前半通道额外加上 HOG 引导生成的空间特征图【累加】
        x = torch.cat((x_half_processed + hog_map, x[:, half_c:, :, :]), dim=1)

        # 第三步：生成 q1、k1、q2、k2、v【双分支注意力】
        # 先经过 1x1 卷积，再经过深度卷积
        qkv = self.qkv_dwconv(self.qkv(x))
        # 按通道维分成 5 路
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        # 第四步：基于 v 的梯度信息做 token 重排
        # 对 v 提取水平方向梯度
        gx = F.conv2d(v, self.sobel_x[:c], padding=1, groups=c)
        # 对 v 提取垂直方向梯度
        gy = F.conv2d(v, self.sobel_y[:c], padding=1, groups=c)

        # 计算梯度幅值，并展平成 [B, C, H*W]
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).view(b, c, -1)
        # 计算梯度方向，并展平成 [B, C, H*W]
        orientation = torch.atan2(gy, gx).view(b, c, -1)
        # 把方向值从 [-pi, pi] 归一化到 [0, 1]
        orientation_norm = (orientation + torch.pi) / (2 * torch.pi)
        # 用方向归一化值去加权幅值，构造排序依据
        weighted_magnitude = magnitude * orientation_norm
        # 在通道维求和，得到每个 token 的综合排序分数
        # 然后按 token 维排序，得到索引 idx
        _, idx = weighted_magnitude.sum(dim=1).sort(dim=-1)

        # 把 idx 扩展到每个通道都可用
        idx = idx.unsqueeze(1).expand(b, c, -1)

        # 根据排序索引重排 v
        v_sorted = torch.gather(v.view(b, c, -1), dim=2, index=idx)
        # 根据同样的索引重排 q1
        q1_sorted = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        # 根据同样的索引重排 k1
        k1_sorted = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        # 根据同样的索引重排 q2
        q2_sorted = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        # 根据同样的索引重排 k2
        k2_sorted = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        # 第五步：分别做两路重排注意力
        # 第一路采用 ifBox=True 的重排方式
        out1 = self.reshape_attn(q1_sorted, k1_sorted, v_sorted, True)
        # 第二路采用 ifBox=False 的重排方式
        out2 = self.reshape_attn(q2_sorted, k2_sorted, v_sorted, False)

        # 第六步：把重排后的结果恢复回原始 token 顺序
        # 先创建与 out1 相同形状的全零张量
        out1_recover = torch.zeros_like(out1)
        # 先创建与 out2 相同形状的全零张量
        out2_recover = torch.zeros_like(out2)

        # 使用 scatter_ 按照 idx 把 out1 放回原始 token 位置
        out1_recover.scatter_(2, idx, out1)
        # 使用 scatter_ 按照 idx 把 out2 放回原始 token 位置
        out2_recover.scatter_(2, idx, out2)

        # 把恢复后的 token 序列重新变回特征图
        out1_recover = out1_recover.view(b, c, h, w)
        out2_recover = out2_recover.view(b, c, h, w)

        # 两路输出做逐元素相乘，实现双分支融合
        out = out1_recover * out2_recover

        # 再通过 1x1 卷积做输出映射
        out = self.project_out(out)

        # 第七步：恢复前半通道在 patch 内部原本的像素顺序
        # 先取出前半通道输出
        out_replace = out[:, :half_c, :, :]
        # 再切成 patch
        patches_out, shape_info = self.split_into_patches(out_replace)
        # 创建一个同形状全零张量，用于恢复 patch 内原始顺序
        out_restore = torch.zeros_like(patches_out)
        # 按照前面保存的 idx_patch，把排序后的 patch 内像素放回原始位置
        out_restore.scatter_(-1, idx_patch, patches_out)
        # 将恢复顺序后的 patch 再拼回特征图
        out_replace = self.merge_patches(out_restore, shape_info)
        # 用恢复后的前半通道替换原输出中的前半通道
        out[:, :half_c, :, :] = out_replace

        return out

if __name__ == "__main__":
    x = torch.randn(2, 32, 128, 128)
    model = Attention_DHOGSA(
        dim=32,
        num_heads=8,
        bias=False,
        patch_size=8,
        n_bins=9
    )
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")