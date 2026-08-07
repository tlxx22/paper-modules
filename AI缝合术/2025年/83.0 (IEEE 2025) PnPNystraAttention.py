import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def closest_square_factors(N):
    """
    Return a pair (r, c) with r*c = N and |r-c| minimized.
    """
    s = int(math.isqrt(N))   # floor of sqrt(N)
    for i in range(s, 0, -1):
        if N % i == 0:
            return i, N // i
        
class ExpLinearAfterThreshold(nn.Module): # 微信公众号:AI缝合术
    def __init__(self, max_val=0.0):
        super(ExpLinearAfterThreshold, self).__init__()
        self.max_val = max_val

    def forward(self, x):
        return torch.exp(x.clamp(max=self.max_val)) + torch.relu(x - self.max_val)

class SepConv2d(torch.nn.Module):
    def __init__(self,  # 微信公众号:AI缝合术
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        # 微信公众号:AI缝合术
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x
        
######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v    

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)  # 微信公众号:AI缝合术
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

class PnPNystraAttention(nn.Module):
    def __init__(self, num_landmarks, iters, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):  # 微信公众号:AI缝合术

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
            
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':  # 微信公众号:AI缝合术
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:  # 微信公众号:AI缝合术
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.activ = ExpLinearAfterThreshold()

        self.num_landmarks = num_landmarks
        self.iters = iters
    
    def moore_penrose_iter_pinv(self, x, iters = 6):
        device = x.device

        abs_x = torch.abs(x)
        col = abs_x.sum(dim = -1)
        row = abs_x.sum(dim = -2)
        z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

        I = torch.eye(x.shape[-1], device = device)
        I = rearrange(I, 'i j -> () i j')

        for _ in range(iters):
            xz = x @ z
            z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

        return z

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        
        num_landmarks = self.num_landmarks
        iters = self.iters
        window_size = self.win_size[0]

        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale

        h, w = closest_square_factors(num_landmarks)

        q_m = F.adaptive_avg_pool2d(q.reshape(B_*self.num_heads, window_size, window_size, self.dim//self.num_heads).permute(0, 3, 1, 2), output_size = (h, w)).permute(0, 2, 3, 1).reshape(B_, self.num_heads, num_landmarks, self.dim//self.num_heads) # 微信公众号:AI缝合术
        k_m = F.adaptive_avg_pool2d(k.reshape(B_*self.num_heads, window_size, window_size, self.dim//self.num_heads).permute(0, 3, 1, 2), output_size = (h, w)).permute(0, 2, 3, 1).reshape(B_, self.num_heads, num_landmarks, self.dim//self.num_heads) # 微信公众号:AI缝合术

        temp = self.activ(q_m @ k_m.transpose(-2, -1))

        pseudo_inv = self.moore_penrose_iter_pinv(temp, iters)

        prod = (self.activ(q @ k_m.transpose(-2, -1)) @ pseudo_inv) @ (self.activ(q_m @ k.transpose(-2,-1)) @ torch.cat([v, torch.ones_like(v[..., :1])], dim=-1))

        x = (prod[..., :-1] / (prod[..., -1].unsqueeze(-1) + 1e-12)).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == "__main__":
    # 模拟输入参数
    batch_size = 1
    num_landmarks = 16   # 设置 landmark 数量
    num_heads = 8        # 设置注意力头数
    dim = 64             # 通道数，通常为一个较大的值
    win_size = (32, 32)  # 窗口大小
    iters = 6            # Moore-Penrose 迭代次数

    # 创建输入张量：假设输入图像为 batch_size x dim x height x width 的格式
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, height * width, dim)  # 输入的 shape 为 [B, N, C]

    # 实例化模型
    model = PnPNystraAttention(
        num_landmarks=num_landmarks,
        iters=iters,
        dim=dim,
        win_size=win_size,
        num_heads=num_heads,
        token_projection='linear',  # 可选：'linear' 或 'conv'
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1
    )

    # 设备配置：GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # 前向传播
    output = model(input_tensor)

    # 打印模型结构和输入输出形状
    print(model)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", input_tensor.shape)  # [B, N, C] => [1, 1024, 64]
    print("输出张量形状:", output.shape)       # [B, N, C] => [1, 1024, 64]
