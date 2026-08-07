import torch
import torch.nn as nn
import torch.nn.functional as F

def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos
    
class WindowAttention(nn.Module):
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        # Curvature guidance components
        self.curvature_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        # Raw attention logits
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B*nW, heads, N, N]

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn_logits = attn_logits + relative_position_bias.unsqueeze(0)

        N = attn_logits.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn_logits = attn_logits.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_logits = attn_logits.view(-1, self.num_heads, N, N)

        # Compute curvature
        x2d = qkv[0].transpose(1, 2).view(B, C, H, W)
        curvature = self.curvature_conv(x2d).mean(dim=1, keepdim=True).view(B, -1)  # [B, H*W]
        curvature = F.layer_norm(curvature, curvature.shape[1:])
        curvature_map = curvature.view(B, 1, H, W)
        curvature_patches = img2windows(curvature_map, self.H_sp, self.W_sp)
        curvature_win = curvature_patches.view(-1, self.H_sp * self.W_sp)
        curvature_win = curvature_win.unsqueeze(1).unsqueeze(-1)
        
        # Modulate attention logits with curvature
        modulated_logits = attn_logits * (1 + self.alpha * curvature_win)

        # Final blended attention scores
        attn_std = F.softmax(attn_logits, dim=-1, dtype=attn_logits.dtype)
        attn_cga = F.softmax(modulated_logits, dim=-1, dtype=attn_logits.dtype)
        attn = self.beta * attn_std + (1 - self.beta) * attn_cga
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C
        
        return x
    
class LCGA(nn.Module):
    """
    Local Curvature-Guided Attention (LCGA)
    The implementation builds on CAT code https://github.com/zhengchen1999/CAT/blob/main/basicsr/archs/cat_arch.py                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
    """
    def __init__(self, dim, num_heads,
                 split_size=[2,4], shift_size=[1,2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., idx=0, reso=64, rs_id=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.idx = idx
        self.rs_id = rs_id
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                WindowAttention(
                    dim//2, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                for i in range(self.branch_num)])

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)

            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None

            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for H-Shift
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1) # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for V-Shift
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1) # nW, sw[1], sw[0], 1                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        # v without partition
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)

        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
            qkv = qkv.view(3, B, _H, _W, C)
            # H-Shift
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C//2)
            # V-Shift
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C//2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))

            else:
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
            x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # H-Rwin
            x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)

        # mix
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + lcm

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 1024, 64).to(device)

    model = LCGA(dim=64, num_heads=8).to(device)

    print(model)
    
    output_tensor = model(input_tensor, 32, 32) # H, W

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")