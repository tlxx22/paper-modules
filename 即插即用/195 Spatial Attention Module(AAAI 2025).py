import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F

""" 
    论文地址：https://ojs.aaai.org/index.php/AAAI/article/view/32687/34842
    论文题目：Efficient Attention-Sharing Information Distillation Transformer for Lightweight Single Image Super-Resolution (AAAI 2025) 
    中文题目：用于轻量化单图像超分辨率的高效注意力共享信息蒸馏 Transformer (AAAI 2025)
    讲解视频：https://www.bilibili.com/video/BV1cBUGBdE4H/
    基于分块机制的空间注意力（Spatial Attention Module,SCA）
        实际意义：①局部与长程依赖难以兼顾的问题：CNN仅能捕捉局部像素关联，而Transformer自注意力机制全局建模，但计算成本极高，无法同时兼顾局部细节与长程语义。
                ②传统窗口注意力的感受野受限：使用局部窗口注意力仅局限在窗口内，跨窗口信息交互受到限制。
        实现方式：输入特征分块+生成 Q、K、V+注意力矩阵+局部与全局像素关联。
"""

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps                                                       # 记录 epsilon 以便反向使用
        N, C, H, W = x.size()                                               # 获取输入张量的维度
        mu = x.mean(1, keepdim=True)                                        # 计算通道维度的均值
        var = (x - mu).pow(2).mean(1, keepdim=True)                         # 计算通道维度的方差
        y = (x - mu) / (var + eps).sqrt()                                   # 标准化处理
        ctx.save_for_backward(y, var, weight)                               # 储存反向传播需要的数据
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)             # 应用可学习的缩放与偏移
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps                                                        # 取回 epsilon
        N, C, H, W = grad_output.size()                                      # 取回输出梯度维度
        y, var, weight = ctx.saved_variables                                 # 获取前向储存的变量
        g = grad_output * weight.view(1, C, 1, 1)                            # 梯度乘以缩放系数
        mean_g = g.mean(dim=1, keepdim=True)                                 # 通道方向求均值
        mean_gy = (g * y).mean(dim=1, keepdim=True)                          # y 加权的均值
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)         # 计算输入梯度
        return (
            gx,                                                              # 输入梯度
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),             # 权重梯度
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),                   # 偏置梯度
            None                                                             # eps 无梯度
        )


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))  # 可学习缩放因子
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))   # 可学习偏置
        self.eps = eps                                                        # 防止除零的 epsilon

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)   # 调用自定义 LN


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.):
        super().__init__()
        hidden_features = int(dim * mult)                               # 计算隐藏特征维度
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)     # 输入通道扩展并分成两份
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, 3, 1, 1,
                                groups=hidden_features*2, bias=bias)             # 深度卷积保持空间建模
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)        # 输出通道压回原维度

    def forward(self, x):
        x = self.project_in(x)                                            # 执行通道扩展
        x1, x2 = self.dwconv(x).chunk(2, dim=1)                           # 深度卷积后按通道一分为二
        x = F.gelu(x1) * x2                                               # 门控机制：激活×控制信号
        x = self.project_out(x)                                           # 投影回原维度
        return x

class SelfAttentionPE(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0., window_size=7, with_pe=True):
        super().__init__()
        assert (dim % dim_head) == 0                                      # 确保可整除以形成多头
        self.heads = dim // dim_head                                      # 计算注意力头数
        self.scale = dim_head ** -0.5                                     # 查询向量缩放
        self.with_pe = with_pe                                               # 是否使用相对位置编码

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)                # 线性映射生成 Q K V

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),                                           # 归一化注意力权重
            nn.Dropout(dropout)                                           # 防止过拟合
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),                              # 合并多头后的线性投影
            nn.Dropout(dropout)
        )

        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)   # 相对位置编码表

            pos = torch.arange(window_size)                                # 坐标构建
            grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))    # 生成二维网格
            grid = rearrange(grid, 'c i j -> (i j) c')                     # 展平成序列
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')   # 计算相对位置差
            rel_pos += window_size - 1                                     # 偏移到正索引区间
            rel_pos_indices = (rel_pos * torch.tensor([2*window_size-1, 1])).sum(dim=-1)  # 映射为索引
            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)   # 注册为常驻缓冲

    def forward(self, x):
        batch, height, width, w1, w2, _, device, h = *x.shape, x.device, self.heads     # 提取输入维度信息
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')                          # 展平窗口内 token
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)                                      # 生成 Q K V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d)-> b h n d', h=h), (q, k, v))    # 分成多头
        q = q * self.scale                                                              # 缩放查询向量
        sim = einsum('b h i d, b h j d -> b h i j', q, k)                               # 计算注意力相似度矩阵

        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)                              # 取相对位置偏置
            sim = sim + rearrange(bias, 'i j h -> h i j')                               # 添加偏置

        attn = self.attend(sim)                                                         # softmax 注意力权重
        out = einsum('b h i j, b h j d -> b h i d', attn, v)                            # 加权求和
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=w1, w2=w2)            # 合并窗口 token
        out = self.to_out(out)                                                          # 输出线性映射
        return rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)            # 恢复空间形状


class AttentionAM(nn.Module):
    def __init__(self, channel_num=64, window_size=8, with_pe=True, dropout=0.0):
        super(AttentionAM, self).__init__()
        self.w = window_size                                                       # 保存窗口尺寸
        self.norm = nn.LayerNorm(channel_num)                                      # 对窗口 token 做 LN
        self.attn = SelfAttentionPE(channel_num, channel_num, dropout, window_size, with_pe)  # 局部自注意力
        self.cnorm = LayerNorm2d(channel_num)                                      # 空间层归一化
        self.gfn = Gated_Conv_FeedForward(channel_num, dropout=dropout)           # 卷积前馈模块

    def forward(self, x):
        x_ = rearrange(x, 'b d (x w1) (y w2)-> b x y w1 w2 d', w1=self.w, w2=self.w)  # 划分为窗口 token
        x = self.attn(self.norm(x_))                                                 # 执行窗口注意力
        x = rearrange(x + x_, 'b x y w1 w2 d -> b d (x w1) (y w2)')                  # 残差并恢复空间结构

        x = self.gfn(self.cnorm(x)) + x                                               # 卷积 FFN 并加残差
        return x

if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 256, 256)
    model = AttentionAM(channel_num=32, window_size=8, with_pe=True)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
