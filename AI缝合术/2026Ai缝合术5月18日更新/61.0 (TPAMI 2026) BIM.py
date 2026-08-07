import torch
import torch.nn as nn
import torch.nn.functional as F

class BIM(nn.Module):
    def __init__(self, dim):
        super(BIM, self).__init__()
        
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.dwconv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        self.proj_2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.dwconv_9x9 = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        hidden_dim = int(1.2 * dim) 
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.ho_dwconv_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.ho_dwconv_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.ho_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        
        x1 = self.dwconv_3x3(self.proj_1(x))
        x2 = self.dwconv_9x9(self.proj_2(x))
        
        x_cat = torch.cat([x1, x2], dim=1)
        x_flat = x_cat.view(b, 2 * c, -1)
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        sim_map = torch.bmm(x_norm, x_norm.transpose(1, 2))                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        attn = self.mlp(sim_map)
        attn = attn.view(b, 2 * c, 1, 1)
        a1, a2 = torch.split(attn, c, dim=1)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        x1_hat = x1 * a1
        x2_hat = x2 * a2
        z1 = x1_hat * x2_hat
        z2 = self.ho_dwconv_1(z1) * x2_hat
        z3 = self.ho_dwconv_2(z2) * x1_hat
        y = self.ho_proj(z3)
        
        return y

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 16, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = BIM(16).to(device)                                                                                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, A
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")