import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation  
    """Pad to 'same' shape outputs."""  
    if d > 1:   
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    if p is None:  
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

  
class Conv(nn.Module): 
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    # default_act = nn.SiLU()  # default activation
    default_act = nn.GELU()  # note act
    # default_act = nn.ReLU()
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""   
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)    
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()                                                                                                                                                                                              # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    def forward(self, x): 
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x): 
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x)) 

class CKConv(nn.Module):
    def __init__(self, c1, c2, kk=[3, 5, 7], s=1):
        super().__init__()

        if not isinstance(kk, list) or not all(ki in [3, 5, 7, 9] for ki in kk):
            raise ValueError("k must be a list containing 3, 5, and/or 7")

        self.kk = kk
        self.c1 = c1
        self.c2 = c2
        self.s = s

        self.branches = nn.ModuleDict()


        for ki in kk:

            self.branches[f'k{ki}_body'] = Conv(c2, c2//2, (3, 3), s=1, g=c2//2)
            self.branches[f'k{ki}_head_h'] = Conv(c2, c2//2, (1, ki), s=s, p=(0, (ki - 1) // 2), g=c2//2)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            self.branches[f'k{ki}_head_v'] = Conv(c2//2, c2//2, (ki, 1), s=s, p=((ki - 1) // 2, 0), g=c2//2)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            self.branches[f'k{ki}_conv2'] = nn.Conv2d(c2//2, c2, 1, groups=c2//2)

        self.conv_fuse = nn.Conv2d(len(kk) * c2, c2, 1, groups=16)   # note 1

    def forward(self, x):

        outputs = []

        for ki in self.kk:
            y = self.branches[f'k{ki}_head_h'](x)
            y = self.branches[f'k{ki}_head_v'](y)
            ys = self.branches[f'k{ki}_body'](x)
            out = ys + y
            out = self.branches[f'k{ki}_conv2'](out)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out = self.conv_fuse(out)

        return out
    

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 256, 256).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = CKConv(32, 32, kk=[3, 5, 7]).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")