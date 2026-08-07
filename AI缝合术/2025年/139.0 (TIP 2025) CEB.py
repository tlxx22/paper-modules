import torch
import torch.nn as nn
import torch.nn.functional as F

class CEB(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0),                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0),                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x): 
        B, C, H, W = x.shape
        x = self.conv(x)
        skip = x
        x1, x2 = torch.split(x, C // 2, dim=1)
        avg_out = self.mlp1(self.avg_pool(x1))
        max_out = self.mlp2(self.max_pool(x2))
        y1 = self.sigmoid1(avg_out)
        y2 = self.sigmoid2(max_out)
        z = torch.cat((x1 * y1, x2 * y2), dim=1)
        perm = torch.randperm(C)
        z = z[:, perm, :, :]
        z = z + skip
        return z


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建一个随机输入特征图 
    input_tensor = torch.randn(1, 32, 256, 256).to(device)    # (batch_size, channels, height, width)                                                                                                                               # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
    
    ceb = CEB(num_feat=32).to(device)
    print(ceb)
    # 前向传播
    output_tensor = ceb(input_tensor)

    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
