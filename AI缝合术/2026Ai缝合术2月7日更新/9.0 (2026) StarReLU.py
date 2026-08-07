import torch
import torch.nn as nn

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
    
# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建一个随机输入特征图 
    input_tensor = torch.randn(1, 3, 3, 3).to(device)    # (batch_size, channels, height, width)
    model = StarReLU().to(device)
    print(model)
    output_tensor = model(input_tensor)
    
    # 打印输入输出形状
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理!\n")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print(f"Input: {input_tensor}")
    print(f"Output: {output_tensor}")
    