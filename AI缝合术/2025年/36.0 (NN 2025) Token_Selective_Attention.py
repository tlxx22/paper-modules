import torch
from torch import nn
from einops import rearrange

# Token Selective Attention
class Token_Selective_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, k, group_num):
        super(Token_Selective_Attention, self).__init__()
        self.num_heads = num_heads
        self.k = k
        self.group_num = group_num
        self.dim_group = dim // group_num
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(self.group_num, self.group_num * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(self.group_num * 3, self.group_num * 3, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                  groups=self.group_num * 3, bias=bias)  # 331
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b,self.group_num,c//self.group_num,h,w)
        b, t, c, h, w = x.shape  # 2,4,32,8,8

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = rearrange(q, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        k = rearrange(k, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, _, N = q.shape  # N=hw

        mask = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)

        attn = (q.transpose(-2, -1) @ k) * self.temperature  # [b, hw, hw]

        index = torch.topk(attn, k=int(N * self.k), dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # [b, c, hw]

        out = rearrange(out, 'b head c (h w t) -> b t (head c) h w', head=self.num_heads, h=h, w=w)

        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)

        return out
    
if __name__ == '__main__':
    # 参数设置
    batch_size = 1               # 批量大小
    dim = 64                     # 输入通道数
    height, width = 32, 32       # 输入图像的高度和宽度

    # 创建随机输入张量，形状为 (batch_size, dim, height, width)
    x = torch.randn(batch_size, dim, height, width).cuda()

    # 创建 Token_Selective_Attention 模型
    model = Token_Selective_Attention(dim=dim, num_heads=8, bias=False, k=0.8, group_num=4).cuda()

    # 打印模型结构
    print(model)
    print("微信公众号: AI缝合术!")

    # 进行前向传播，得到输出
    output = model(x)
    
    # 打印输入和输出的形状
    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")
 