import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Spatial_Attention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(Spatial_Attention,self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Spectral_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
       
        super(Spectral_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
    
class PG_Spectral_Attention(nn.Module):
    def __init__(self, dim, compress_ratio, num_heads, prompt_len, bias):
        super(PG_Spectral_Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim//compress_ratio) ** -0.5

        self.linear_down = nn.Linear(dim, dim//compress_ratio,bias=bias)#compress_ratio=8，prompt_len=128
        self.linear_up = nn.Linear(dim//compress_ratio, dim, bias=bias)
        self.linear_prompt = nn.Linear(dim, prompt_len, bias=bias)
        self.prompt_param = nn.Parameter(torch.rand(1,1,prompt_len,dim//compress_ratio))#1,1,128,8

        self.q = nn.Linear(dim//compress_ratio, dim//compress_ratio, bias=bias)
        self.kv = nn.Linear(dim//compress_ratio, dim*2//compress_ratio, bias=bias)
        self.proj = nn.Linear(dim//compress_ratio, dim//compress_ratio)

    def forward(self, x_kv):
        shourtcut = x_kv
        B_, N, C = x_kv.shape#B,64,64
        x_kv = x_kv.mean(dim=1).unsqueeze(1)#B,64,64->B,1,64  
        prompt_weights = F.softmax(self.linear_prompt(x_kv),dim=-1)
        x_kv = self.linear_down(x_kv) #B,1,8
        
        spectral_prompt = prompt_weights.unsqueeze(-1) * self.prompt_param.repeat(B_,1,1,1)#B,1,128,8
        spectral_prompt = torch.sum(spectral_prompt,dim=2)#B,1,8

        q = self.q(spectral_prompt)#B,1,8  
        kv = self.kv(x_kv)
        k,v = kv.chunk(2, dim=2)#B,1,8      

        attn_weights = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)

        out = (attn_weights @ v.transpose(-2, -1))#B,8,1
        out = out.transpose(-2, -1).contiguous()#B,1,8
        out = self.proj(out)#B,1,8
        out = self.linear_up(out)#B,1,64
        out = out*shourtcut #B,64,64

        return out

class PGSSA(nn.Module):
    def __init__(self, dim, num_heads, input_resolution=[64,64],window_size=8,shift_size=0,drop_path=0.0,
                 mlp_ratio=4., compress_ratio=8, prompt_len=128, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,act_layer=nn.GELU,bias=False):
        super(PGSSA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.global_spectral_attn = Spectral_Attention(dim, num_heads, bias)
        self.local_spectral_attn = PG_Spectral_Attention(dim, compress_ratio, num_heads, prompt_len, bias)
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.input_resolution = input_resolution

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = Spatial_Attention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
    
    def forward(self, x, text_prompt=None):
        B, C, H, W = x.shape
        
        # Global Spectral Attention
        x2 = self.global_spectral_attn(x)  # global spectral attention

        # Local Spectral Attention
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if self.input_resolution == [H,W]:                        
            sa_attns = self.attn(x_windows, mask=self.attn_mask) 
        else:
            sa_attns = self.attn(x_windows, mask=self.calculate_mask([H,W]).to(x.device))

        x1 = self.local_spectral_attn(sa_attns)
        x1 = x1.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(x1, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x
    
        x1 = x1.view(B, C, H, W)
        x2 = x2.view(B, C, H, W)

        # Final output
        out = x1 + x2
        return out

if __name__ == '__main__':

    # 定义输入张量的尺寸
    B = 1  # batch size
    C = 32  # channel数
    H = 256  # height
    W = 256  # width

    # 创建一个随机张量作为输入
    x = torch.randn(B, C, H, W)

    # 创建PGSSTB模块的实例
    dim = C
    num_heads = 8
    compress_ratio = 8
    prompt_len = 128
    bias = False

    model = PGSSA(dim=dim, num_heads=num_heads, compress_ratio=compress_ratio, prompt_len=prompt_len, bias=bias)
    print(model)
    print("微信公众号: AI缝合术!")

    # 将输入传递给模型
    output = model(x)

    # 打印输出的形状
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {output.shape}')