import  torch
from    torch import nn, einsum
from    einops import rearrange
import  torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()

        hidden_features = int(dim*mult)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SelfAttentionA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        with_pe = True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = True

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )
        
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)


    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        # project for queries, keys, values
        q, k,v = self.to_qkv(x).chunk(3, dim = -1)
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))
        # scale
        q = q * self.scale
        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)
        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width), attn
    
class SLA_A(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias = True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(SLA_A, self).__init__()
        self.w= 8
        self.norm = nn.LayerNorm(channel_num) # prenormresidual
        self.attn = SelfAttentionA(dim = channel_num, dim_head = channel_num, dropout = dropout, window_size = self.w, with_pe=with_pe)
        self.cnorm = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim = channel_num, dropout = dropout)
        
    def forward(self, x):
        x_ = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1 = self.w, w2 = self.w)
        x, a = self.attn(self.norm(x_))
        x = rearrange(x+x_, 'b x y w1 w2 d -> b d (x w1) (y w2)')
        x = self.gfn(self.cnorm(x))+x
        return x, a

if __name__ == "__main__":

    batch_size = 1
    channels = 32
    height = 256
    width = 256

    # 生成随机输入张量
    x = torch.randn(batch_size, channels, height, width)

    # 实例化 SLA_A, 为了保证输入张量形状为四维, 此处采用SLA_A, 包含了空间注意力和其他卷积操作
    model = SLA_A(channel_num=channels, window_size=8, with_pe=True)
    print(model)
    print("微信公众号:AI缝合术!")

    # 进行前向传播
    output, attention_map = model(x)

    # 打印输入和输出的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")
    print("可重复利用注意力，也可删去此操作.")
    print(f"Attention map shape: {attention_map.shape}")