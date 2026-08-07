import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class simam_imt(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_imt, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        return y


class Down_wt(nn.Module):
    def __init__(self, in_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Conv2d(in_ch, in_ch*2, kernel_size=1, stride=1)

    def forward(self, x):#1 32 256 256
        size = x.shape[2:]
        yL, yH = self.wt(x)

        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x =y_HL+ y_LH + y_HH#1 96 128 128

        # 使用最邻近插值上采样
        x = F.interpolate(x, size=size, mode='nearest')#1 96 256 256
        x = self.conv_bn_relu(x)
        return x
    
class WDE(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dwconv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        self.dwconv3x3_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!

        self.conv_0 = Down_wt(in_ch=dim)
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(dim*2, dim, 1, 1, 0)

    def forward(self, x):#1 32 256 256
        v = self.dwconv3x3(x)
        attn = self.dwconv3x3_2(x)
        attn = self.conv_0(attn)
        attn = self.act(attn)#1 96 256 256
        attn = self.conv_1(attn)#1 32 256 256
        attn = torch.tanh(attn)
        res = attn.mul(v)
        return res


class CAA(nn.Module):
    def __init__(self, dim=36, scale=8):
        super(CAA, self).__init__()
        self.down_scale = scale

        self.conv1_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.conv1_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!

        self.gelu = nn.GELU()


        self.WDE = WDE(dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.simam_imt = simam_imt()

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.conv1_0(f).chunk(2, dim=1)  # 1 32 256 256
        e_t = self.simam_imt(x)
        e_l = self.dwconv(F.adaptive_avg_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_caa = F.interpolate(self.gelu(self.conv1_1(e_l * self.alpha)), size=(h, w), mode='nearest') +  e_t * self.belt                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        x_l = x * x_caa
        y_d = self.WDE(y)
        out = x_l + y_d
        out = self.ca(out) * out
        out = self.conv1_2(out)

        return out


class PSAA(nn.Module):
    def __init__(self, in_dim, out_dim, scale):
        super(PSAA, self).__init__()
        self.LG_att = CAA(in_dim, scale)

        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_f = x + self.LG_att(x)

        out = self.Conv3(x_f) + self.Conv1(x_f)
        return out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = PSAA(32, 32, 8).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")