import torch
import torch.nn as nn
import torch.nn.functional as F

# Dynamic block downsampling module
class DBDM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.offset_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.offset_conv2 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1)
        
        self.block_conv = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
    def forward(self, x):
        B, C, H, W = x.shape
        h, w = H // 2, W // 2
        
        offset = F.relu(self.offset_conv1(x))
        offset = self.offset_conv2(offset)
        offset = offset.view(B, 4, 2, h, w).permute(0, 1, 3, 4, 2)
        
        base_grids = self.generate_base_grid(H, W, x.device)
        base_grids = base_grids.unsqueeze(0)
        
        grid = base_grids + offset
        grid = torch.clamp(grid, -1, 1)
        
        blocks = []
        for k in range(4):
            block_grid = grid[:, k, :, :, :]
            block = F.grid_sample(x, block_grid, align_corners=True)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            blocks.append(block)
        
        feats = []
        for block in blocks:
            feat = self.block_conv(block)
            feats.append(feat)
        fusefeat = torch.cat(feats, dim=1)
        
        residual = self.residual_conv(x)
        
        out = fusefeat + residual
        out = self.final_conv(out)
        
        return out
    
    def generate_base_grid(self, H, W, device):
        h, w = H // 2, W // 2
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                              torch.linspace(-1, 1, w, device=device),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                              indexing='ij')
        base_grid = torch.stack([x, y], dim=-1)
        
        block_offsets = torch.tensor([
            [-0.5, -0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5]
        ], device=device)
        
        base_grid = base_grid.unsqueeze(0)
        block_offsets = block_offsets.view(4, 1, 1, 2)
        base_grids = base_grid * 0.5 + block_offsets
        
        return base_grids

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = DBDM(64, 128).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")