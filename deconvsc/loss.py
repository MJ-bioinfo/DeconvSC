import torch
import torch.nn.functional as F

def compute_gene_correlation_loss(x_real, x_recon):

    # 1. 为了数值稳定性，添加微小噪声防止标准差为0
    epsilon = 1e-8
    
    # 2. 归一化 (Z-score per gene)
    real_mean = x_real.mean(dim=0, keepdim=True)
    recon_mean = x_recon.mean(dim=0, keepdim=True)
    real_centered = x_real - real_mean
    recon_centered = x_recon - recon_mean
    
    # 计算标准差
    real_std = x_real.std(dim=0, keepdim=True) + epsilon
    recon_std = x_recon.std(dim=0, keepdim=True) + epsilon
    
    # 3. 计算相关性矩阵 (Pearson Correlation Matrix) [Genes, Genes]
    n = x_real.size(0)
    if n <= 1: 
        return torch.tensor(0.0, device=x_real.device) # Batch太小无法计算
    
    real_corr = (real_centered.t() @ real_centered) / (n - 1)
    real_corr = real_corr / (real_std.t() @ real_std)
    
    recon_corr = (recon_centered.t() @ recon_centered) / (n - 1)
    recon_corr = recon_corr / (recon_std.t() @ recon_std)
    
    # 4. 计算两个矩阵的 MSE Loss
    loss = F.mse_loss(real_corr, recon_corr)
    
    return loss