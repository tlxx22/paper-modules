import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.fft_norm = fft_norm
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        if torch.__version__ > '1.7.1':
            x = x.to(torch.float32)
            batch = x.shape[0]
            # (batch, c, h, w/2+1, 2)
            fft_dim = (-2, -1)
            ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.relu(self.bn(ffted.to(torch.float32)))
            ffted = ffted.to(torch.float32)

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])

            ifft_shape_slice = x.shape[-2:]
            output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        else:
            batch, c, h, w = x.size()
            r_size = x.size()

            # (batch, c, h, w/2+1, 2)
            ffted = torch.rfft(x, signal_ndim=2, normalized=True)
            # (batch, c, 2, h, w/2+1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.relu(self.bn(ffted))

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

            output = torch.irfft(ffted, signal_ndim=2,
                                 signal_sizes=r_size[2:], normalized=True)

        return output


class SeparableFourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3):
        # bn_layer not used
        super(SeparableFourierUnit, self).__init__()
        self.groups = groups
        row_out_channels = out_channels // 2
        col_out_channels = out_channels - row_out_channels
        self.row_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=row_out_channels * 2,
                                        kernel_size=(kernel_size, 1),
                                        # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.col_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=col_out_channels * 2,
                                        kernel_size=(kernel_size, 1),
                                        # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.row_bn = torch.nn.BatchNorm2d(row_out_channels * 2)
        self.col_bn = torch.nn.BatchNorm2d(col_out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def process_branch(self, x, conv, bn):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft(x, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.relu(bn(conv(ffted)))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfft(ffted, s=x.shape[-1:], norm="ortho")
        return output

    def forward(self, x):
        rowwise = self.process_branch(x, self.row_conv, self.row_bn)
        colwise = self.process_branch(x.permute(0, 1, 3, 2), self.col_conv, self.col_bn).permute(0, 1, 3, 2)
        out = torch.cat((rowwise, colwise), dim=1)
        return out


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, separable_fu=False, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l.to(torch.float32)))
        x_g = self.act_g(self.bn_g(x_g.to(torch.float32)))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, activation_layer=nn.ReLU):
        super(FFCResnetBlock, self).__init__()

        self.ffc1 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=dilation, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=activation_layer, enable_lfu=False)

        self.ffc2 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=1, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=activation_layer, enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output

        return output
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class LowFrequencyProcessing(nn.Module):
    def __init__(self, nf=64, num_blocks=6, input_channels=3):
        """
        Unified Stage --- Combines Fourier Reconstruction and Spatial-Texture Reconstruction.
        """
        super(LowFrequencyProcessing, self).__init__()

        # Initial feature extraction
        self.initial_conv = nn.Conv2d(input_channels, nf, kernel_size=1, stride=1, padding=0)
        # FFT-based feature extraction blocks (First Stage)
        self.fft_blocks = nn.ModuleList([FFT_Process(nf) for _ in range(6)])
        # Dual-branch blocks (Second Stage)
        self.ffc_blocks = nn.ModuleList([FFCResnetBlock(nf) for _ in range(num_blocks)])
        self.multi_blocks = nn.ModuleList([MultiConvBlock(nf) for _ in range(num_blocks)])
        self.fusion_block = ChannelAttentionFusion(nf)
        # Downsampling layers for feature fusion
        self.concat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0),
                SEBlock(nf)
            ) for _ in range(3)
        ])
        # Reconstruction trunk
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, 1)

        # Upsample convolution layers
        self.upconv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, fr, y_map):
        # Initial feature extraction
        xori = x
        x0 = self.initial_conv(x)
        # FFT-based feature extraction with skip connections
        x, fr, y_map = self.fft_blocks[0](x0, fr, y_map)
        x1, fr, y_map = self.fft_blocks[1](x, fr, y_map)
        x2, fr, y_map = self.fft_blocks[2](x1, fr, y_map)

        # Downsample and fuse features
        x3_input = torch.cat((x2, x1), dim=1)
        x3_input = self.concat_layers[0](x3_input)
        x3, fr, y_map = self.fft_blocks[3](x3_input, fr, y_map)

        x4_input = torch.cat((x3, x), dim=1)
        x4_input = self.concat_layers[1](x4_input)
        x4, fr, y_map = self.fft_blocks[4](x4_input, fr, y_map)

        x5_input = torch.cat((x4, x0), dim=1)
        x5_input = self.concat_layers[1](x5_input)
        x5, fr, y_map = self.fft_blocks[5](x5_input, fr, y_map)

        fft_features = x5
        multi_features = x5
        for ffc_block, multi_block in zip(self.ffc_blocks, self.multi_blocks):
            fft_features = ffc_block(fft_features)
            multi_features = multi_block(multi_features)
            # Fuse features using Channel Attention Fusion
        fused_features = self.fusion_block(fft_features, multi_features)
        out_noise = self.upconv_last(fused_features) + xori
        return out_noise

class FFT_Process(nn.Module):
    def __init__(self, nf):
        super(FFT_Process, self).__init__()
        # Preprocessing for frequency domain
        self.nf = nf
        self.freq_preprocess = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.feature_fusion = nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.process_amp = self._make_process_block(nf)
        self.process_pha = self._make_process_block(nf)
        self.process_fr = self._make_process_block(nf)
        self.process_map = self._make_process_block(nf)
        self.process_sigmoid_amp = FrequencyFusion(nf)
        self.process_amp_post = self._make_process_block(nf)
        self.process_pha_post = self._make_process_block_pha(nf)

    def _make_process_block(self, nf):
        return nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        )

    def _make_process_block_pha(self, nf):
        return nn.Sequential(
            nn.Conv2d(nf*2, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, fr, y_map):
        _, _, H, W = x.shape
        # Frequency domain processing
        x_freq = torch.fft.rfft2(self.freq_preprocess(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process_amp(mag)
        pha = self.process_pha(pha)

        # Process infrared features and Cross-modality interaction
        fr = self.process_fr(fr)
        pha = torch.cat([pha, fr],dim=1)

        # Process brightness attention map
        mag = self.process_sigmoid_amp(mag, y_map)
        pha = self.process_pha_post(pha)
        mag = self.process_amp_post(mag)

        # Reconstruct frequency domain features
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out_ff = x_out + x
        return x_out_ff, fr, y_map

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch,int(in_channel/(r**2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:,out_channel:out_channel * 2, :, :] / 2
    x3 = x[:,out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:,out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
    
class DownFRG(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWT()  # 小波下采样

    def forward(self, x):
        # 使用小波变换进行下采样
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)
        return x_LL, (x_HL, x_LH, x_HH)

class UpFRG(nn.Module):
    def __init__(self):
        super().__init__()
        self.iwt = IWT()  # 小波上采样

    def forward(self, x_LL, x_H):
        # 使用逆小波变换进行上采样
        x_HL, x_LH, x_HH = x_H
        x = self.iwt(torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1))
        return x
    
from functools import partial
import pywt

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters = self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters = self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x
    
class DeformableDilatedConv(nn.Module):
    """可变形卷积 + 空洞卷积模块 + 小波卷积 + 残差"""
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DeformableDilatedConv, self).__init__()
        self.dilated_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            dilation=2, padding=2
        )
        self.dilated_conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=4, padding=4)

        self.wtconv = WTConv2d(in_channels, out_channels, kernel_size=5, wt_levels=3)
        self.relu = nn.ReLU()
        self.fuseconv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x1 = self.relu(self.dilated_conv(x))
        x2 = self.relu(self.dilated_conv3(x))
        x3 = self.wtconv(x)
        x = self.fuseconv(torch.cat([x1,x2,x3],dim=1))
        return x + residual

class SpatialAugmentModel(nn.Module):
    def __init__(self, nf=16):
        super(SpatialAugmentModel, self).__init__()
        # 输入卷积：3 通道 -> nf 通道
        self.input_conv = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=True)
        # 两个 WTConv 模块
        self.spatialaugment1 = DeformableDilatedConv(in_channels=nf, out_channels=nf, dilation=2)
        self.spatialaugment2 = DeformableDilatedConv(in_channels=nf, out_channels=nf, dilation=2)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        # 输出卷积：nf 通道 -> 3 通道
        self.output_conv = nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # 输入卷积
        res = x
        x = self.input_conv(x)
        x = self.spatialaugment1(x)
        x = self.spatialaugment2(x)
        # 输出卷积
        x = self.output_conv(x)
        return x + res

class WTConvAttentionModel(nn.Module):
    def __init__(self, nf=16):
        super(WTConvAttentionModel, self).__init__()
        # 输入卷积：3 通道 -> nf 通道
        self.input_conv = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=True)
        # 两个 WTConv 模块
        self.wtconv1 = WTConv2d(nf, nf, kernel_size=5, wt_levels=3)
        self.wtconv2 = WTConv2d(nf, nf, kernel_size=5, wt_levels=3)
        self.output_conv = nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # 输入卷积
        res = x
        x = self.input_conv(x)
        # 两个 WTConv 操作
        x = self.wtconv1(x)
        x = self.wtconv2(x)
        x = self.output_conv(x)
        return x + res


class MultiConvBlock(nn.Module):
    def __init__(self, dim, num_heads=4, expand_ratio=2):
        super(MultiConvBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Channel reduction layer
        self.conv_reduction = nn.Conv2d(dim, dim // 4, kernel_size=1, stride=1, bias=True)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

        # Multi-scale convolution layers
        self.local_convs = nn.ModuleList([
            nn.Conv2d(
                dim // 4, dim // 4,
                kernel_size=(3 + i * 2),
                padding=(1 + i),
                stride=1,
                groups=dim // 4  # Grouped convolution
            ) for i in range(num_heads)
        ])

        # Feature fusion layer
        self.conv_fusion = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.se_block = SEBlock(dim)

    def forward(self, x):
        # Channel reduction
        x_reduced = self.leakyrelu(self.conv_reduction(x))

        # Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.local_convs:
            x_scale = self.leakyrelu(conv(x_reduced))  # Apply multi-scale convolution                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            x_scale = x_scale * torch.sigmoid(x_reduced)  # Element-wise modulation
            multi_scale_features.append(x_scale)

        # Concatenate multi-scale features
        x_concat = torch.cat(multi_scale_features, dim=1)

        # Feature fusion and residual connection
        x_fused = self.conv_fusion(x_concat)
        x_fused = self.se_block(x_fused)
        return x + x_fused  # Residual connection


class FrequencyFusion(nn.Module):
    def __init__(self, channels):
        super(FrequencyFusion, self).__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

        # 最终调制
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, D, G):
        # 拼接特征
        cat_feature = torch.cat([D, G], dim=1)
        # 通道注意力权重
        channel_weight = self.channel_attention(cat_feature)
        # 加权特征
        weighted_D = D * channel_weight
        # 特征融合
        fused_feature = self.fusion_conv(cat_feature)
        # 生成门控权重
        gate_weight = self.gate(fused_feature)
        # 最终输出
        output = weighted_D + G * gate_weight

        return output

class ChannelAttentionFusion(nn.Module):
    def __init__(self, nf):
        """
        Channel Attention Fusion module for combining fft_features and multi_features.

        Args:
            nf (int): Number of feature channels.
        """
        super(ChannelAttentionFusion, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = nn.Sequential(
            nn.Conv2d(nf * 2, nf // 4, 1, bias=False),  # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // 4, nf * 2, 1, bias=False),  # Restore channels
            nn.Sigmoid()
        )

    def forward(self, fft_features, multi_features):
        # Concatenate features along the channel dimension
        combined_features = torch.cat([fft_features, multi_features], dim=1)

        # Generate attention weights
        attention_weights = self.fc(self.global_avg_pool(combined_features))

        # Split attention weights for fft_features and multi_features
        fft_weight, multi_weight = torch.split(attention_weights, fft_features.size(1), dim=1)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

        # Apply attention weights
        fused_features = fft_weight * fft_features + multi_weight * multi_features
        return fused_features


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale + x
    


class DFGFLow(nn.Module):
    def __init__(self, nf=16, numblocks = 6):
        super(DFGFLow, self).__init__()
        self.s_nf = nf
        self.processblock = LowFrequencyProcessing(nf=self.s_nf, num_blocks=numblocks, input_channels=3)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        # Initial convolution layers for Fourier features
        self.conv_first_fr = nn.Conv2d(3, self.s_nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_first_map = nn.Conv2d(3, self.s_nf, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, x_light):
        x_light_fre = torch.fft.rfft2(x_light, norm='backward')
        x_light_mag = torch.abs(x_light_fre)
        x_light_pha = torch.angle(x_light_fre)
        x_light_mag = self.conv_first_fr(x_light_mag)
        x_light_pha = self.conv_first_map(x_light_pha)

        # 幅值增强
        x_amplitude = self.processblock(x, x_light_pha, x_light_mag)
        x_amplitude = torch.fft.irfft2(x_amplitude, s=x.shape[-2:], norm='backward')

        return x_amplitude

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor_x1 = torch.randn(1, 3, 64, 64).to(device)
    input_tensor_x2 = torch.randn(1, 3, 64, 64).to(device)

    model = DFGFLow(nf=16, numblocks=6).to(device)
    
    print(model)
    output_tensor = model(input_tensor_x1, input_tensor_x2)

    # 打印维度验证
    print("input_tensor_shape_x1  :", input_tensor_x1.shape)   
    print("input_tensor_shape_x2  :", input_tensor_x2.shape)
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")