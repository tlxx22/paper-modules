import torch
from torch import nn

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()                                                                                                                                # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    
    return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16):                                                                                                                                # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        if self.in_planes < ratio:
            ratio = self.in_planes
        self.reduced_channels = self.in_planes // ratio
        if self.out_planes == None:
            self.out_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.activation = nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(in_planes, self.reduced_channels, 1, bias=False)
                        
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        max_pool_out= self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel size must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),                                                                                                                                # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU6(inplace=True)
            )
            for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        # For example, you could concatenate or add them; here, we just return the list
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):

    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1,3,5]):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        self.ex_c = int(self.in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.ex_c),
            nn.ReLU(inplace=True)
        )        
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(self.ex_c, self.kernel_sizes, self.stride,dw_parallel=dw_parallel)                                                                                                                                # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

        if self.add == True:
            self.combined_channels = self.ex_c*1
        else:
            self.combined_channels = self.ex_c*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False), # 
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False) 

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add == True:
            dout = 0
            for dwout in dwconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_c))                                                                                                                                # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x+out
        else:
            return out


class   MKIRA(nn.Module):

    def __init__(self, in_c, out_c, stride=1, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1,3,5]):                                                                                                                                # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        super(MKIRA, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.ca = ChannelAttention(in_c)
        self.sa = SpatialAttention()
        self.block1 = MultiKernelInvertedResidualBlock(in_c, out_c, stride, expansion_factor, dw_parallel, add, kernel_sizes)                                                                                                                                 # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        
    def forward(self, x):
        ca_out = self.ca(x)*x
        sa_out = self.sa(ca_out)*ca_out
        block1_out = self.block1(sa_out)
        return block1_out
    

# ------------张量测试---------------
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(1, 32, 128, 128, device=device)

    mkira = MKIRA(in_c=32, out_c=32).to(device)                                                                                                                                                                     # 微信公众号:AI缝合术
    out = mkira(x)

    print(mkira)

    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
    print("Input :", x.shape)
    print("Output:", out.shape)
