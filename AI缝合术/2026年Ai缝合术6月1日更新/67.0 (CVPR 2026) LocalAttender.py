import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def residual_auto(x1, x2):
    s1 = x1.shape[1]
    s2 = x2.shape[1]
    if s1 == s2:
        return x1 + x2
    elif s1 > s2: # truncation
        s = x2.shape[1]
        xr = x1[:,:s,:,:]
        return xr + x2
    else: # padding
        s = x1.shape[1]
        pad = torch.zeros_like(x2)
        pad[:,:s,:,:] = x1
        return pad + x2
    
class LocalAttender(nn.Module):
    def __init__(self, in_channels, num_connected=5, conv_res=True, low_mem=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_connected = num_connected
        # select offset set based on num_connected
        if self.num_connected == 5:
            self.offsets = [(-1,0),(0,-1),(0,0),(0,1),(1,0)]
            self.pn = 1 # padding number
        elif self.num_connected == 9:
            self.offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
            self.pn = 1
        elif self.num_connected == 13:
            self.offsets = [(-2,0),(-1,-1),(-1,0),(-1,1),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-1),(1,0),(1,1),(2,0)]                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            self.pn = 2
        elif self.num_connected == 17:
            self.offsets = [(-2,-2),(-2,0),(-2,2),(-1,-1),(-1,0),(-1,1),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-1),(1,0),(1,1),(2,-2),(2,0),(2,2)]
            self.pn = 2
        elif self.num_connected == 25:
            self.offsets = [(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-2),(1,-1),(1,0),(1,1),(1,2),(2,-2),(2,-1),(2,0),(2,1),(2,2)]                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            self.pn = 2
        else:
            print('ERROR: LocalAttender invalid num_connected')
            exit(-1)
        # attender map maker
        self.conv1 = nn.Conv2d(in_channels, num_connected, kernel_size=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.conv_res = conv_res
        # padding
        self.pad = nn.ReplicationPad2d(self.pn)
        # low mem mode
        self.low_mem = low_mem


    # visualize the offset map for the given setting
    def show_offsets(self):
        k = 3 + (2*self.pn)
        b = self.pn + 1
        vis = np.zeros([k,k])
        for off in self.offsets:
            h, w = off
            vis[b+h, b+w] = 1
        print(vis)


    def make_offsets(self, x):
        H = x.shape[2]
        W = x.shape[3]
        # [B, C, H, W]
        x = self.pad(x)
        # [B, C, H+(2*PN), W+(2*PN)]
        x_all = []
        for off in self.offsets:
            off_0, off_1 = off
            x_off = x[:, :, self.pn+off_0:self.pn+off_0+H, self.pn+off_1:self.pn+off_1+W]                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            # [B, C, H, W]
            x_all.append(x_off)
        x = torch.stack(x_all, dim=2)
        # [B, C, D, H, W]
        return x


    def forward(self, guide, value):
        # value shape: [B, C, H, W]
        x = value
        B = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        # create attender map with shape: [B, D, H_out, W_out]
        att = self.conv1(guide)
        if self.conv_res:
            att = residual_auto(guide, att)
        D = att.shape[1]
        H_out = att.shape[2]
        W_out = att.shape[3]

        # identify integer scaling factor
        I = att.shape[2] // x.shape[2]
        assert I * x.shape[2] == att.shape[2]
        assert I * x.shape[3] == att.shape[3]

        # handle value map
        x = self.make_offsets(x) # [B, C, H, W] -> [B, C, D, H, W]
        x = torch.unsqueeze(x, dim=4) # -> [B, C, D, H, 1, W]
        x = torch.unsqueeze(x, dim=6) # -> [B, C, D, H, 1, W, 1]
        
        # handle attender map
        att = F.softmax(att, dim=1) # [B, D, H_out, W_out]
        att = torch.reshape(att, [B, D, H, I, W, I]) # -> [B, D, H, I, W, I]
        att= torch.unsqueeze(att, dim=1) # -> [B, 1, D, H, I, W, I]

        # pool features
        if not self.low_mem:
            # normal mode - parallel pooling
            res = x * att # -> [B, C, D, H, I, W, I]
            res = torch.sum(res, dim=2) # -> [B, C, H, I, W, I]
            res = res.reshape([B, C, H_out, W_out]) # -> [B, C, H_out, W_out]
        else:
            # low-mem mode - sequential pooling
            res = torch.zeros([B,C,H_out,W_out]).to(x.device)
            for d in range(self.num_connected):
                x_d = x[:,:,d,:,:,:,:]
                att_d = att[:,:,d,:,:,:,:]
                res_d = x_d * att_d
                res_d = res_d.reshape([B, C, H_out, W_out])
                res += res_d
        return res

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_guide_tensor = torch.randn(2, 64, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    inpit_value_tensor = torch.randn(2, 64, 32, 32).to(device)

    model = LocalAttender(64, num_connected=5, conv_res=True, low_mem=False).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    print(model)

    output_tensor = model(input_guide_tensor, inpit_value_tensor)

    # 打印维度验证
    print("input_guide_tensor  :", input_guide_tensor.shape)
    print("inpit_value_tensor  :", inpit_value_tensor.shape)
    print("output_tensor       :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")