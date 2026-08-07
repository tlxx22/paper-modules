import torch
import torch.nn as nn
import torch.nn.functional as F

class TMConv(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__(in_channels, out_channels, kernel_size=k,
                         stride=1, padding=k//2, dilation=1, groups=1, bias=True)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        mask = torch.ones_like(self.weight)
        tri2d = torch.triu(torch.ones(k, k, dtype=mask.dtype, device=mask.device))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        with torch.no_grad():
            mask *= tri2d
        self.register_buffer("mask", mask, persistent=True)

        self.weight.register_hook(lambda g: g * self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, stride=1, padding=self.padding)
    

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 256, 256).to(device)

    model = TMConv(32, 32, 3).to(device)

    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")