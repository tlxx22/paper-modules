import  torch
from    torch import nn, einsum
from    einops import rearrange
import  torch.nn.functional as F

class Channel_Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        bias=False, 
        dropout = 0.,
        window_size = 7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)
        out = self.project_out(out)
        return out

if __name__ == "__main__":
    # 设定测试参数
    batch_size = 1  # 批大小
    channels = 32   # 通道数
    height = 256    # 特征图高度
    width = 256     # 特征图宽度
    heads = 8       # 注意力头数
    window_size = 8 # 窗口大小
    
    # 创建测试输入张量
    x = torch.randn(batch_size, channels, height, width)
    
    # 初始化通道注意力模块
    channel_attn = Channel_Attention(dim=channels, heads=heads, window_size=window_size)
    print(channel_attn)
    print("微信公众号: AI缝合术!")
    
    # 前向传播
    output = channel_attn(x)
    
    # 打印输出形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)