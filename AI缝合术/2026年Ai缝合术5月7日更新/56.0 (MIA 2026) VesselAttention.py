import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VesselAttentionModule(nn.Module):
    def __init__(self, in_channels, sigma=[1.0, 2.0, 3.0]):
        super(VesselAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.sigma = sigma
        self.num_scales = len(sigma)
        
        # 可学习的多尺度权重
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        # 1x1x1卷积调整通道
        self.conv = nn.Conv3d(1, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _create_gaussian_kernel(self, sigma, kernel_size=5):
        # 生成1维高斯核
        kernel_1d = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        kernel_1d = torch.exp(-0.5 * (kernel_1d ** 2) / (sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # 修复1：3D高斯核构造 (einsum公式修正，删除错误转义字符)
        kernel_3d = torch.einsum('i,j,k->ijk', kernel_1d, kernel_1d, kernel_1d)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # [1,1,k,k,k]
        
        # 计算3D二阶导数 (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
        d2_kernel = []
        np_kernel = kernel_3d[0,0].cpu().numpy()
        # 修复2：正确计算3D Hessian分量
        derivs = [
            np.gradient(np.gradient(np_kernel, axis=0), axis=0),  # xx
            np.gradient(np.gradient(np_kernel, axis=1), axis=1),  # yy
            np.gradient(np.gradient(np_kernel, axis=2), axis=2),  # zz
            np.gradient(np.gradient(np_kernel, axis=0), axis=1),  # xy
            np.gradient(np.gradient(np_kernel, axis=0), axis=2),  # xz
            np.gradient(np.gradient(np_kernel, axis=1), axis=2),  # yz
        ]
        for d in derivs:
            d2_kernel.append(torch.from_numpy(d).unsqueeze(0).unsqueeze(0).float())                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        return torch.cat(d2_kernel, dim=0)  # [6,1,k,k,k]

    def forward(self, x):
        B, C, D, H, W = x.shape
        frangi_maps = []
        
        for s in self.sigma:
            # 获取高斯二阶导数核
            kernel = self._create_gaussian_kernel(s).to(x.device)
            
            # 修复3：卷积计算Hessian分量 (分组卷积保证通道匹配)
            hessian_components = F.conv3d(x, kernel, padding=2, groups=C)
            
            # 拆分6个Hessian分量
            Hxx, Hyy, Hzz, Hxy, Hxz, Hyz = torch.split(hessian_components, 1, dim=1)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            
            # 修复4：正确构造3D Hessian矩阵
            hessian = torch.zeros(B, D, H, W, 3, 3, device=x.device)
            hessian[..., 0, 0] = Hxx.squeeze(1)
            hessian[..., 1, 1] = Hyy.squeeze(1)
            hessian[..., 2, 2] = Hzz.squeeze(1)
            hessian[..., 0, 1] = hessian[..., 1, 0] = Hxy.squeeze(1)
            hessian[..., 0, 2] = hessian[..., 2, 0] = Hxz.squeeze(1)
            hessian[..., 1, 2] = hessian[..., 2, 1] = Hyz.squeeze(1)
            
            # 计算特征值
            eigenvalues = torch.linalg.eigvalsh(hessian)  # 实对称矩阵用eigvalsh更稳定
            abs_eig = eigenvalues.abs()
            sorted_eig, _ = torch.sort(abs_eig, dim=-1, descending=True)  # 降序排列
            
            # 修复5：修正语法错误 & 数值稳定性
            lambda1, lambda2, lambda3 = sorted_eig[..., 0], sorted_eig[..., 1], sorted_eig[..., 2]                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            lambda3 = lambda3.clamp(min=1e-8)
            Rb = lambda2 / lambda3
            S = torch.sqrt(lambda1**2 + lambda2**2 + lambda3**2).clamp(min=1e-8)
            
            # Frangi响应公式
            frangi_response = torch.exp(-(Rb**2)/(2*0.5**2)) * (1 - torch.exp(-(S**2)/(2*2**2)))                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            frangi_maps.append(frangi_response.unsqueeze(1))
        
        # 多尺度融合
        frangi_maps = torch.cat(frangi_maps, dim=1)  # [B, S, D, H, W]
        scale_weights = F.softmax(self.scale_weights, dim=0).view(1, self.num_scales, 1, 1, 1)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        attention_map = (frangi_maps * scale_weights).sum(dim=1, keepdim=True)  # [B,1,D,H,W]
        
        # 生成注意力权重
        attention = self.sigmoid(self.conv(attention_map))
        # 残差连接输出
        return x * attention + x

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 1, 10, 10, 10).to(device)

    model = VesselAttentionModule(in_channels=1, sigma=[1.0, 2.0, 3.0]).to(device)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")