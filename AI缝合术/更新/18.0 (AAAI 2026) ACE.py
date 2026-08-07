import torch
import torch.nn as nn


class ACE(nn.Module):
    """
    Downsample shallow features to the target number of channels 
    using a combination of depthwise and pointwise convolutions.  
    The number of channels is doubled at each step until reaching out_channels.
    """

    def __init__(self, in_channels, out_channels, down_times=1):
        super(ACE, self).__init__()
        layers = []

        mid_channels = [in_channels * (2 ** i) for i in range(down_times + 1)]                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

        if mid_channels[-1] != out_channels:
            mid_channels[-1] = out_channels

        for i in range(down_times):
            inch = mid_channels[i]
            outch = mid_channels[i + 1]
            
            layers.append(nn.Sequential(
                nn.Conv2d(inch, inch, kernel_size=3, stride=2, padding=1, groups=inch, bias=False),                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
                nn.BatchNorm2d(inch),
                nn.ReLU(inplace=True),
                nn.Conv2d(inch, outch, kernel_size=1, bias=False),
                nn.BatchNorm2d(outch),
                nn.ReLU(inplace=True)
            ))

        self.blocks = nn.Sequential(*layers)

        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x):
        x = self.blocks(x)
        x = self.out_proj(x)
        return x

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    model = ACE(3, 64, down_times=1).to(device)

    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")