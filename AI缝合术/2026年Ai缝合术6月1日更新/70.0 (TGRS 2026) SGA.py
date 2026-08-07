import torch
import torch.nn as nn
import torch.nn.functional as F

class CMConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4, bias=False):
        super(CMConv, self).__init__()
        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation, groups=groups * dilation_set, bias=bias)
        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation, groups=groups * dilation_set, bias=bias)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)

        # Gradient masking hook / 梯度掩码 Hook
        def backward_hook(grad):
            out = grad.clone()
            out[self.mask.bool()] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).byte().cuda() 
        _in_channels = in_ch // (groups * dilation_set)
        _out_channels = out_ch // (groups * dilation_set)
        
        # Generate mask / 生成掩码
        for i in range(dilation_set):
            for j in range(groups):
                self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
                self.mask[((i + dilation_set // 2) % dilation_set + j * groups) * _out_channels: ((i + dilation_set // 2) % dilation_set + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        self.conv.weight.data[self.mask.bool()] = 0
        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):
        # Channel splitting and merging / 通道拆分与合并
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift


class SGAM_Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SGAM_Conv_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(out_ch)
        self.ReLU = nn.ReLU(inplace=False)
        self.conv2 = CMConv(out_ch, out_ch, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        return self.ReLU(x + self.BN2(self.conv2(self.ReLU(self.BN1(x)))))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    
class GaoSi_core(nn.Module):
    def __init__(self, in_ch):
        super(GaoSi_core, self).__init__()

    def forward(self, M, A):
        _, _, h, w = A.size()
        q = M.mean(dim=[2, 3], keepdim=True) # Spatial mean / 空间均值
        k = A 
        square = (k - q).pow(2) # Variance calculation / 方差计算
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + 1e-8) + 0.5
        att_weight = nn.Sigmoid()(att_score)
        return att_weight * A

class SGA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SGA, self).__init__()
        native_ch = out_ch // 2
        self.SGAM_conv = nn.Conv2d(in_ch, native_ch, kernel_size=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.BN1 = nn.BatchNorm2d(native_ch)
        self.ReLU = nn.ReLU(inplace=True)
        self.GaoSi = GaoSi_core(native_ch)
        self.conv_finally = SGAM_Conv_Block(out_ch, out_ch)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, F1, F2):
        A1 = self.SGAM_conv(F1)
        A2 = self.SGAM_conv(F2)
        A1_wave = self.ReLU(self.BN1(A1))
        A2_wave = self.ReLU(self.BN1(A2))

        M = (A1_wave + A2_wave) * 0.5 # Mutual feature / 交互特征
        A1_hat = self.GaoSi(M, A1)
        A2_hat = self.GaoSi(M, A2)
        result = torch.cat([A1_hat * self.beta + A1, A2_hat * self.beta + A2], dim=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        return self.conv_finally(result)

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor_f1 = torch.randn(2, 64, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    input_tensor_f2 = torch.randn(2, 64, 32, 32).to(device)

    model = SGA(64, 64).to(device)
    print(model)
    output_tensor = model(input_tensor_f1, input_tensor_f2)

    # 打印维度验证
    print("input_tensor_f1_shape :", input_tensor_f1.shape)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!\n
    print("input_tensor_f2_shape :", input_tensor_f2.shape)
    print("output_tensor_shape   :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")