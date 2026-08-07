# 注意！
# 所缺文件从https://github.com/Chernobyllight/SaMam/tree/main/ARCHI/SAVSSM链接下载
import torch.nn as nn

from utils.SS2D_Decoder import SS2D
from common.SAIN import SRAdaIN
from common.SCM import SCM

from utils.archi_utils import PatchEmbed, PatchUnEmbed


class SAVSSM(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 64,
            d_state: int = 16,
            expand: float = 2.,
            representation_dim: int = 64,
            mamba_from_trion=1,
            zero_init=0,
            **kwargs,
    ):
        super().__init__()
        if mamba_from_trion:
            print('inference by mamba_ssm, quick!')
        else:
            print('inference by pure torch, slow!')

        self.SAIN1 = SRAdaIN(in_channels=hidden_dim, representation_dim=representation_dim,zero_init=zero_init)
        self.SSM = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,
                        representation_dim=representation_dim,mamba_from_trion=mamba_from_trion,zero_init=zero_init,**kwargs)

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

        self.ca = SCM(representation_dim=representation_dim, channels_out=hidden_dim, reduction=4, zero_init=zero_init)

    def forward(self, input, representation):
        # x [B,HW,C]
        B, C, H, W = input.size()

        # branch 1
        x = self.SAIN1(input, representation) # x: B, C, H, W
        x = self.patch_embed(x)  # B,L,C
        # B, L, C = input.shape
        x = x.view(B, H, W, C).contiguous()  # x -> [B,H,W,C]
        x = self.SSM(x, representation)
        x = x.view(B, -1, C).contiguous()
        x = self.patch_unembed(x, C, H, W)

        # branch 2
        x1 = self.ca([input, representation])

        return x + x1




if __name__ == '__main__':
    print('------微信公众号：AI缝合术------')
    print()
    import torch

    expend = 2.
    representation_channel = 32
    use_mamba_ssm = 1 # if you don't install mamba_ssm, set it to 0.

    img_feature_channel = 64
    img_feature_height = 96
    img_feature_width = 96
    img_feature = torch.randn((1, img_feature_channel, img_feature_height, img_feature_width)).cuda()  # (1,C,H,W)
    style_representation = torch.randn((1, representation_channel, 3, 3)).cuda()
    print('input shape:', img_feature.shape)

    SAVSSM_sample = SAVSSM(hidden_dim=img_feature_channel,expand=expend,representation_dim=representation_channel,mamba_from_trion=use_mamba_ssm).cuda()
    output = SAVSSM_sample.forward(img_feature, style_representation)
    print('output shape:', output.shape)




