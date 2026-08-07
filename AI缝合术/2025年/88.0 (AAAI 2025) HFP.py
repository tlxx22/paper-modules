import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT

class DctSpatialInteraction(nn.Module):
    def __init__(self,
                in_channels,
                ratio,
                isdct = True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)]
        )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho') 
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
         # 微信公众号:AI缝合术
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight
    
class DctChannelInteraction(nn.Module):
    def __init__(self,
                in_channels, 
                patch,
                ratio,
                isdct=True
                ):
        super().__init__()
        # 微信公众号:AI缝合术
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.channel2x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct: # true when in p1&p2 # false when in p3&p4 # 微信公众号:AI缝合术
            amaxp = F.adaptive_max_pool2d(x,  output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x,  output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp)) # 2025 03 15 szc  # 微信公众号:AI缝合术
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features  # 微信公众号:AI缝合术
        dct_ = DCT.idct_2d(dct, norm='ortho') 

        amaxp = F.adaptive_max_pool2d(dct_,  output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_,  output_size=(self.h, self.w))       
        amaxp = torch.sum(self.relu(amaxp), dim=[2,3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2,3]).view(n, c, 1, 1)

        # channel = torch.cat([self.channel1x1(aavgp), self.channel1x1(amaxp)], dim = 1) # TODO: The values of aavgp and amaxp appear to be on different scales. Add is a better choice instead of concate. # 微信公众号:AI缝合术
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp) # 2025 03 15 szc  # 微信公众号:AI缝合术
        return x * torch.sigmoid(self.channel2x1(channel))
        
    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight    

class HFP(nn.Module):
    def __init__(self, 
                in_channels,
                ratio,
                patch = (8,8),
                isdct = True
                ):
        super().__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct = isdct) 
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct = isdct)
        self.out =  nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels)]
            )
    def forward(self, x):
        spatial = self.spatial(x) # output of spatial path
        channel = self.channel(x) # output of channel path
        return self.out(spatial + channel)
    

if __name__ == "__main__":

    # 输入配置
    batch_size = 1
    channels = 32
    height = 256
    width = 256

    # 构造输入张量 [B, C, H, W]
    x = torch.randn(batch_size, channels, height, width)

    # 实例化模型
    ratio = (0.25, 0.25)   # 高频保留比例
    patch = (8, 8)         # 用于通道交互的池化尺度
    model = HFP(in_channels=channels, ratio=ratio, patch=patch, isdct=False)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = model.to(device)

    # 前向传播
    output = model(x)

    # 输出模型结构与形状信息
    print(model)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]
