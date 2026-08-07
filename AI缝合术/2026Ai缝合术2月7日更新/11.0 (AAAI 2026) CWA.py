import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    if kernel_size == (1, 3, 3):
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=(0, 1, 1), bias=False, dilation=dilation_rate)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=False, dilation=dilation_rate)
    
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size, stride, padding=padding)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size, padding=padding, 
                             dilation_rate=dilation_rate)
        self.bn2 = nn.InstanceNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(planes),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        residue = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residue)
        out = self.relu(out)
        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)  # (b, c)
        y_max = self.max_pool(x).view(b, c)  # (b, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1, 1)

        return self.sigmoid(y_avg+y_max)  # (b, c, 1, 1, 1)
    

class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)

        return x * y
    

class CWCABlock(nn.Module):
    def __init__(self, channel):
        super(CWCABlock, self).__init__()

        c = channel
        # conv接收拼接后的2c通道，最后转回c通道
        self.conv = nn.Sequential(
            BasicBlock(2*c, c, 3, 1, 1),  # 输入2c→输出c，适配拼接后的通道数
            BasicBlock(c, c, 3, 1, 1)     # 保持c通道
        )
        # weight_conv输入通道为c（conv输出的通道数）
        self.weight_conv = nn.Sequential(
            nn.Conv3d(c, 2, 3, 1, 1),     # 输入c→输出2（两个权重图）
            nn.Softmax(dim=1)
        )
        # 通道注意力接收2c通道（拼接后的con_fm+decon_fm）
        self.channel_attention = ChannelAttention(2*c)
        self.se = SELayer(2*c)

    def forward(self, con_fm, decon_fm):
        # 拼接后通道数：2*c（如16+16=32）
        concat_fm = torch.cat([con_fm, decon_fm], dim=1)
        
        # 1. 特征融合：2c→c通道
        x = self.conv(concat_fm)          # 输出：(b, c, h, w, t)
        
        # 2. 生成权重图：c→2通道（对应con_fm和decon_fm的权重）
        weight_map = self.weight_conv(x)  # 输出：(b, 2, h, w, t)
        
        # 3. 权重加权原始特征
        weighted_con = con_fm * weight_map[:, 0, ...].unsqueeze(1)  # (b, c, h, w, t)
        weighted_decon = decon_fm * weight_map[:, 1, ...].unsqueeze(1)  # (b, c, h, w, t)
        concat = torch.cat([weighted_con, weighted_decon], dim=1)    # 输出：(b, 2c, h, w, t)
        
        # 4. 通道注意力加权
        channel_wise = self.channel_attention(concat)               # 输出：(b, 2c, 1, 1, 1)
        output = concat * channel_wise

        # 5. Squeeze-and-Excitation加权
        output = self.se(output)                    # 输出：(b, 2c, h, w, t)

        return output

# 使用示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟输入：(batch, channel, d, h, w)
    input_tensor1 = torch.randn(1, 16, 128, 128, 128).to(device)
    input_tensor2 = torch.randn(1, 16, 128, 128, 128).to(device)

    model = CWCABlock(16).to(device)

    print(model)
    output_tensor = model(input_tensor1, input_tensor2)

    # 打印维度验证
    print("input_tensor1_shape  :", input_tensor1.shape)   # (1, 16, 128, 128, 128)
    print("input_tensor2_shape  :", input_tensor2.shape)   # (1, 16, 128, 128, 128)
    print("output_tensor_shape  :", output_tensor.shape)  # (1, 32, 128, 128, 128)
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")