import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DWTInverse


class ConvDWT(nn.Module):  # DWT: (B,C,H,W) -> (B,4C,H/2,W/2), no parameters are learnable
    def __init__(self, wave='haar', mode='zero'):
        super(ConvDWT, self).__init__()
        # one-level DWT
        self.dwt_forward = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, x):
        # input size: x (B, C, H, W)
        with torch.cuda.amp.autocast(enabled=False):
            if x.dtype != torch.float32:
                x = x.float()
            Yl, Yh = self.dwt_forward(x)
        b, c, h, w = x.shape
        # Yl (B, C, H/2, W/2) for low-frequency LL
        # List Yh for high-frequency from each level of DWT
        # Yh[0] (B, C, 3, H/2, W/2) for high-frequency LH,HL,HH

        Yh = Yh[0].transpose(1, 2).reshape(Yh[0].shape[0], -1, Yh[0].shape[3], Yh[0].shape[4])

        # output size: output (B, 4C, H/2, W/2)
        output = torch.cat((Yl, Yh), dim=1)
        output = F.interpolate(output, size=(h // 2, w // 2), mode='bilinear', align_corners=False)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        return output

class ConvIDWT(nn.Module):  # IDWT
    def __init__(self, wave='haar', mode='zero'):
        super(ConvIDWT, self).__init__()
        self.dwt_inverse = DWTInverse(wave=wave, mode=mode)

    def forward(self, low_freqs, high_freqs):
        # low_freqs: (B, C, H/2, W/2)
        # high_freqs: (B, 3C, H/2, W/2)
        B, C, H, W = low_freqs.shape

        high_freqs = high_freqs.reshape(B, C, 3, H, W)

        with torch.cuda.amp.autocast(enabled=False):
            reconstruction = self.dwt_inverse((low_freqs, [high_freqs.float()]))
        reconstruction = F.interpolate(reconstruction, size=(2 * H, 2 * W), mode='bilinear', align_corners=False)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!

        return reconstruction
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bn_before_sigmoid=False):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.bn_before_sigmoid = bn_before_sigmoid
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        if bn_before_sigmoid:
            self.bn = nn.BatchNorm2d(1)
            self.bn.bias.data.fill_(0)
            self.bn.bias.requires_grad = False
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        if self.bn_before_sigmoid:
            x = self.bn(x)
        return self.sigmoid(x)
    
class LearnableGaussianFilterBank(nn.Module):
    def __init__(self, kernel_size, num_filters, num_channels):
        super(LearnableGaussianFilterBank, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.C = num_channels
        self.padding = kernel_size // 2  # Padding size to maintain input size

        # Create learnable parameters for sigmas
        self.sigmas = nn.ParameterList([nn.Parameter(torch.tensor([1.0])) for _ in range(num_filters)])

    def forward(self, x):
        # Apply Gaussian filters using convolution
        weights = [self._gaussian_kernel(self.kernel_size, sigma).repeat(self.C, 1, 1, 1) for sigma in self.sigmas]
        filtered_outputs = [F.conv2d(F.pad(x, (self.padding,self.padding,self.padding,self.padding), mode='replicate')                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
                                     , weight.to(x.device), groups=self.C) for weight in weights]
        return torch.cat(filtered_outputs, dim=1)

    def _gaussian_kernel(self, kernel_size, sigma):
        # Create a 2D Gaussian kernel with learnable sigma as tensors
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[:, :, i, j] = torch.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))

        return kernel / kernel.sum() # normalization
    
# LFP Module:
class LFP(nn.Module): # Low-frequency Guided Feature Purification
    def __init__(self, in_channels, wave='haar', mode='symmetric', with_gauss=True, gauss_gate=0.5):
        super(LFP, self).__init__()
        self.dwt = ConvDWT(wave=wave, mode=mode)
        self.idwt = ConvIDWT(wave=wave, mode=mode)
        self.with_gauss = with_gauss
        self.gauss_gate = gauss_gate

        self.attention = SpatialAttention()
        if self.with_gauss:
            self.gaussian_filter = LearnableGaussianFilterBank(kernel_size=3, num_filters=1, num_channels=3 * in_channels)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!

    def forward(self, x):
        B, C, H, W = x.shape
        dwt_out = self.dwt(x)  # (B, 4C, H/2, W/2)

        LL = dwt_out[:, :C, :, :]
        Yh = dwt_out[:, C:, :, :]

        # low-frequency guided modulation of high-frequency
        att = self.attention(LL)  # (B, 1, H/2, W/2)
        Yh = Yh * att

        if self.with_gauss: # Gaussian Filter for high-frequency
            Yh_blurred = self.gaussian_filter(Yh)
            mask = (Yh.abs() < self.gauss_gate).float()
            Yh = Yh * (1 - mask) + Yh_blurred * mask

        x_rec = self.idwt(LL, Yh) # (B, C, H, W)
        return x_rec
    

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 32, 256, 256).to(device)
    model = LFP(in_channels=32).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")