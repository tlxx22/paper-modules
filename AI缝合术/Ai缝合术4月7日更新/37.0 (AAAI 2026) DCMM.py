import torch
import torch.nn as nn
import torch.nn.functional as F

class DCMM(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.cp_bias_scale = nn.Parameter(torch.tensor(10,  dtype=torch.float32))                                                                                                                                                                                             # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        self.n_clusters = 100

        # CP-DCMM group assignment projection from query
        self.group_proj = nn.Linear(self.head_dim, self.n_clusters)

        # Degree scalar projection from query
        self.theta_proj = nn.Linear(self.head_dim, 1)

        # Affinity matrix B: [H, K, K]
        self.affinity_B = nn.Parameter(torch.randn(self.num_heads, self.n_clusters, self.n_clusters)  )                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!

    def forward(self, x, cp_dcmm = 1):
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        _, h, _, _ = q.shape
        _, _, n, _ = v.shape

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        
        z_constraint = torch.tensor(0.0, device=x.device)
        B_constraint = torch.tensor(0.0, device=x.device)
        theta_constraint = torch.tensor(0.0, device=x.device)
        cp_bias = torch.zeros_like(attn).to(device=x.device)

        if cp_dcmm:
            attn_ori = attn.clone()
            # Group assignment z from Q: [B, H, N, K]
            z_logits = self.group_proj(q)  # [B, H, N, K]
            z = F.softmax(z_logits, dim=-1)

            # z_i B z_j^T: compute structural bias
            B_aff = F.sigmoid(self.affinity_B).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, K, K]                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
            cp_bias_ori = torch.einsum("bhik,bhkl,bhjl->bhij", z, B_aff, z).tanh()

            # Degree correction from Q
            theta = F.relu(self.theta_proj(q).squeeze(-1))  # [B, H, N] non-negative
            theta_i = theta.unsqueeze(-1)           # [B, H, N, 1]
            theta_j = theta.unsqueeze(-2)           # [B, H, 1, N]
            degree_term = theta_i * theta_j         # [B, H, N, N]

            # CP-DCMM bias
            #cp_bias = self.cp_bias_scale * cp_bias_ori * degree_term

            # CP-DCMM bias
            cp_bias = self.cp_bias_scale *  degree_term * cp_bias_ori 

            # Add to original attention
            attn = attn + cp_bias   # add CP bias

            z_constraint = - (z * torch.log(z + 1e-8)).sum(dim=-1).mean()                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 1024, 64).to(device)

    model = DCMM(dim=64).to(device)

    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")