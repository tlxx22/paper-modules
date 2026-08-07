import torch
import torch.nn as nn 
from types import SimpleNamespace

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

class SCSA(nn.Module):
    
    def __init__(self, in_planes):
        super(SCSA, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def SCA(self, content, style, content_sem, style_sem, map_32, map_64):
        F = self.f(mean_variance_norm(content_sem))
        G = self.g(mean_variance_norm(style_sem))
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        max_neg_value = -torch.finfo(S.dtype).max
        if F.size()[1]==1024:
            map = map_32
            map = map.repeat(b, 1, 1)
            S.masked_fill_(map<0.5, max_neg_value)
        if F.size()[1]==4096:
            map = map_64
            map = map.repeat(b, 1, 1)
            S.masked_fill_(map<0.5, max_neg_value)
        S = self.sm(S)
        H = self.h(style)
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        return O
    
    def SSA(self, content, style, content_sem, style_sem, map_32, map_64):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        max_neg_value = -torch.finfo(S.dtype).max
        if F.size()[1]==1024:
            map = map_32
            map = map.repeat(b, 1, 1)
            S.masked_fill_(map<0.5, max_neg_value)
        if F.size()[1]==4096:
            map = map_64
            map = map.repeat(b, 1, 1)
            S.masked_fill_(map<0.5, max_neg_value)
        max_indices = torch.argmax(S, dim=2, keepdim=True)
        B = torch.full_like(S, max_neg_value)
        B.scatter_(2, max_indices, S.gather(2, max_indices))
        S = B        
        S = self.sm(S)
        H = self.h(style)
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        return O

    def forward(self, content, style, content_sem, style_sem, map_32, map_64, args):
        x_SCA = self.SCA(content, style, content_sem, style_sem, map_32, map_64)
        x_SSA = self.SSA(content, style, content_sem, style_sem, map_32, map_64)
        x = args.t1*x_SCA+args.t2*x_SSA + content
        return x

if __name__ == "__main__":
    # 模拟输入参数
    batch_size = 1
    channels = 64
    height = 32
    width = 32
    map_32_size = height * width  # 32×32=1024
    map_64_size = (height * 2) * (width * 2)  # 64×64=4096

    # 创建输入张量：内容图、风格图、语义图等
    content = torch.randn(batch_size, channels, height, width)
    style = torch.randn(batch_size, channels, height, width)
    content_sem = torch.randn(batch_size, channels, height, width)
    style_sem = torch.randn(batch_size, channels, height, width)

    # 创建遮罩图 map_32 和 map_64（值域为0~1，float类型）
    map_32 = torch.ones(1, map_32_size, map_32_size)
    map_64 = torch.ones(1, map_64_size, map_64_size)

    # 创建参数对象 args，其中 t1 和 t2 是融合权重
    args = SimpleNamespace(t1=0.6, t2=0.4)

    # 实例化模型
    model = SCSA(in_planes=channels)

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content = content.to(device)
    style = style.to(device)
    content_sem = content_sem.to(device)
    style_sem = style_sem.to(device)
    map_32 = map_32.to(device)
    map_64 = map_64.to(device)
    model = model.to(device)

    # 执行前向传播
    output = model(content, style, content_sem, style_sem, map_32, map_64, args)

    # 打印模型结构（可选）
    print(model)
    print("微信公众号:AI缝合术")

    # 打印输入输出形状
    print("输入 content 形状:", content.shape)
    print("输入 style 形状:", style.shape)
    print("输出 shape:", output.shape)
