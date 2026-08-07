import torch
import torch.nn as nn
import torch.nn.functional as F

class PKSModule(nn.Module):
    """
    [PKINet-v2] Poly-Kernel Scope (PKS) Module
    Branches:
    1. Axial Dense: 1x19 + 19x1 (Series) -> RF 19 [Global Backbone]
    2. Sparse: 7x7, d=3 -> RF 19 [Wide Context]
    3. Sparse: 5x5, d=3 -> RF 13 [Medium Transition]
    4. Sparse: 3x3, d=3 -> RF 7  [New: Sub-Medium]
    5. Dense:  3x3, d=1 -> RF 3  [Micro Texture]
    """
    def __init__(self, dim, branch_scale=1.0):
        super().__init__()
        self.dim = dim
        self.branch_scale = branch_scale
        
        # Max Kernel Size for Fusion (Determined by RF of Branch 1 & 2)
        # Branch 1: 19x19
        # Branch 2: 7 + (7-1)*(3-1) = 7 + 12 = 19
        self.max_k = 19 

        # [Head] Pre-process Conv (5x5) - Kept from V5 structure
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        # [Tail] 1x1 Mixing
        self.conv1 = nn.Conv2d(dim, dim, 1)

        # --- Branch 1: Axial Dense (1x19 + 19x1) ---
        k_axial = 19
        self.branch1_axial = nn.Sequential(
            nn.Conv2d(dim, dim, (1, k_axial), stride=1, padding=(0, k_axial//2), groups=dim, bias=False),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.Conv2d(dim, dim, (k_axial, 1), stride=1, padding=(k_axial//2, 0), groups=dim, bias=False),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(dim)
        )

        # --- Branch 2: Sparse (7x7, d=3) ---
        k_b2, d_b2 = 7, 3
        pad_b2 = (k_b2 - 1) * d_b2 // 2
        self.branch2_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, k_b2, stride=1, padding=pad_b2, dilation=d_b2, groups=dim, bias=False),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                nn.BatchNorm2d(dim)
        )

        # --- Branch 3: Sparse (5x5, d=3) ---
        k_b3, d_b3 = 5, 3
        pad_b3 = (k_b3 - 1) * d_b3 // 2
        self.branch3_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, k_b3, stride=1, padding=pad_b3, dilation=d_b3, groups=dim, bias=False),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                nn.BatchNorm2d(dim)
        )

        # --- Branch 4: Sparse (3x3, d=3) [NEW] ---
        k_b4, d_b4 = 3, 3
        pad_b4 = (k_b4 - 1) * d_b4 // 2
        self.branch4_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, k_b4, stride=1, padding=pad_b4, dilation=d_b4, groups=dim, bias=False),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                nn.BatchNorm2d(dim)
        )

        # --- Branch 5: Dense (3x3, d=1) ---
        self.branch5_dense = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim)
        )

    def forward(self, x):

        # Training Mode
        x_feat = self.conv0(x)
        
        # Accumulate all 5 branches
        # 1. Axial 19x19
        attn = self.branch1_axial(x_feat)
        # 2. Sparse 7x7 (d=3)
        attn = attn + self.branch2_sparse(x_feat)
        # 3. Sparse 5x5 (d=3)
        attn = attn + self.branch3_sparse(x_feat)
        # 4. Sparse 3x3 (d=3)
        attn = attn + self.branch4_sparse(x_feat)
        # 5. Dense 3x3
        attn = attn + self.branch5_dense(x_feat)
        attn = attn * self.branch_scale
        
        # Tail Proj
        attn = self.conv1(attn)
            
        return x * attn
    
# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = PKSModule(dim=64).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")