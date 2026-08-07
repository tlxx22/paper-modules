import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(ch_in, ch_out, kernel_size, bias=True):
    return nn.Conv2d(
        ch_in, ch_out, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MHCB(nn.Module):
    def __init__(self, ch_in, ch_out, bias=True, activation=nn.ReLU(inplace=True)):
        super(MHCB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = default_conv(ch_in=ch_in, ch_out=ch_in, kernel_size=kernel_size_1,  bias=bias)
        self.conv_3_2 = default_conv(ch_in=ch_out, ch_out=ch_out, kernel_size=kernel_size_1,  bias=bias)
        self.conv_5_1 = default_conv(ch_in=ch_in, ch_out=ch_in, kernel_size=kernel_size_2,  bias=bias)
        self.conv_5_2 = default_conv(ch_in=ch_out, ch_out=ch_out, kernel_size=kernel_size_2,  bias=bias)
        self.confusion_3 = nn.Conv2d(ch_in * 3, ch_out, 1, padding=0, bias=True)
        self.confusion_5 = nn.Conv2d(ch_in * 3, ch_out, 1, padding=0, bias=True)
        self.confusion_bottle = nn.Conv2d(ch_in * 3 + ch_out * 2, ch_out, 1, padding=0, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x):
        input_1 = x  # [1, 3, 256, 256]
        output_3_1 = self.activation(self.conv_3_1(input_1))  # [1, 3, 256, 256]
        output_3_1 += x
        output_5_1 = self.activation(self.conv_5_1(input_1))  # [1, 3, 256, 256]
        output_5_1 += x
        input_2 = torch.cat([input_1, output_3_1, output_5_1], 1)  # [1, 9, 256, 256]

        input_2_3 = self.confusion_3(input_2)
        input_2_5 = self.confusion_5(input_2)

        output_3_2 = self.activation(self.conv_3_2(input_2_3))
        output_5_2 = self.activation(self.conv_5_2(input_2_5))
        input_3 = torch.cat([input_1, output_3_1, output_5_1, output_3_2, output_5_2], 1)
        output = self.confusion_bottle(input_3)
        # output += x
        return output  # [1, 3, 256, 256]

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建一个随机输入特征图 
    input_tensor = torch.randn(1, 32, 256, 256).to(device)    # (batch_size, channels, height, width)
    
    mhcb=MHCB(ch_in=32, ch_out=32).to(device)

    print(mhcb)
    
    # 前向传播
    output_tensor = mhcb(input_tensor)
    
    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
