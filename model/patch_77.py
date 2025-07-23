import random
import math
import torch
import torch.nn as nn

class FeatureBlocker(nn.Module):
    def __init__(self, grid_size=7):
        super().__init__()
        self.grid_size = grid_size

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            g = self.grid_size
            h_block = H // g
            w_block = W // g
            if H % g != 0 or W % g != 0:
                raise ValueError(f"Input size {(H, W)} must be divisible by grid size {g}")
            x_unfold = x.unfold(2, h_block, h_block).unfold(4, w_block, w_block)
            x_unfold = x_unfold.permute(0, 1, 2, 4, 3, 5).contiguous()
            x_reshaped = x_unfold.view(B, C, g, g, h_block * w_block)
            x_out = x_reshaped.permute(0, 1, 4, 2, 3).contiguous()
            x_out = x_out.view(B, C * h_block * w_block, g, g)
            
            if C * h_block * w_block > 256:
                d_sample = random.sample(list(range(C * h_block * w_block)), 256)
                x_out = x_out[:,d_sample,:,:]
        
        if len(x.shape) == 3:
            B, L, C = x.shape
            g = self.grid_size
            H = W = int(math.sqrt(L))
            if H * W != L:
                raise ValueError(f"Input length {L} must be a perfect square (got sqrt(L)={H:.1f})")
            if H % g != 0 or W % g != 0:
                raise ValueError(f"Input dimensions {H}x{W} must be divisible by grid_size {g}")
            h_block, w_block = H // g, W // g
            block_area = h_block * w_block
            x_spatial = x.permute(0, 2, 1).view(B, C, H, W)
            x_unfold = x_spatial.unfold(2, h_block, h_block).unfold(3, w_block, w_block)
            x_unfold = x_unfold.contiguous().view(B, C, g, g, block_area)
            x_out = x_unfold.permute(0, 2, 3, 1, 4).contiguous()  # [B, g, g, C, block_area]
            x_out = x_out.view(B, g*g, C * block_area)
            
            if C * block_area > 256:
                d_sample = random.sample(list(range(C * block_area)), 256)
                x_out = x_out[:, :, d_sample]
                
        return x_out


# 示例用法
if __name__ == "__main__":
    # 测试用例
    B, H, W, C = 2, 14, 14, 64
    g = 7

    # 原始空间特征输入
    x_spatial = torch.randn(B, C, H, W)  # [2, 64, 14, 14]
    blocker = FeatureBlocker(g)
    out_spatial = blocker(x_spatial)  # [2, 64*4, 7, 7]

    # 展平后的特征输入
    x_flat = x_spatial.permute(0, 2, 3, 1).view(B, H * W, C)  # [2, 196, 64]
    out_flat = blocker(x_flat)  # [2, 49, 256]

    print(out_flat.shape)  # torch.Size([2, 49, 256])

