import torch
import torch.nn as nn

class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)
    
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class MSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1, 3, 5], stride=1):
        super().__init__()
        # 为每个尺度创建独立的卷积分支（包含卷积+激活）
        self.branches = nn.ModuleList()
        for ks in kernel_size:
            padding = ks // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=stride, padding=padding),                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
                    nn.BatchNorm2d(out_channels),  # 可选：添加BN层稳定训练
                    nn.ReLU(inplace=True)
                )
            )
        
        self.fusion = nn.Conv2d(len(kernel_size) * out_channels, out_channels, kernel_size=1)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)
        x = self.fusion(x)
        return x

class SAMC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1,3,5]):
        super().__init__()

       
        self.cab = CAB(out_channels)
        self.sab = SAB()
        self.msconv = MSConv(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.cab(x) * x
        x = self.sab(x) * x
        x = self.msconv(x)
        return x


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    model = SAMC(3, 3).to(device)

    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")