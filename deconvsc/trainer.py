import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from .loss import compute_gene_correlation_loss
from .utils import set_seed, get_cell_type_indices

def train_vae(vae_model, X_tensor, labels, real_bulk_log, used_device, batch_size, 
              core_indices_tensor, epoch_num, learning_rate, hidden_list, 
              mid_hidden_size, num_cell_types, breed_2_list, color_map, output_dir, seed=18):

    set_seed(seed)
    cell_type_indices = get_cell_type_indices(labels, num_cell_types)
    cell_type_indices = [torch.as_tensor(idx, device=used_device, dtype=torch.long) for idx in cell_type_indices]
    
    dataset = TensorDataset(X_tensor, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

    vae = vae_model.to(used_device)
    vae = vae.to(torch.bfloat16)
    core_indices_tensor = core_indices_tensor.to(used_device)
    
    criterion = nn.MSELoss()
    optimizer = AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Weights configuration
    target_beta = 0.05
    lambda_recon = 1.0
    initial_lambda_corr = 20.0
    
    best_vae = None
    min_loss = float('inf')
    epoch_final = 0
    dtype_preference = torch.bfloat16 

    pbar = tqdm(range(epoch_num), desc="Training")
    for epoch in pbar:
        set_seed(seed + epoch)
        vae.train()
        
        lambda_corr = initial_lambda_corr if epoch < epoch_num * 0.5 else initial_lambda_corr * (1.0 - ((epoch - epoch_num * 0.5) / (epoch_num * 0.5))) + 1.0
        beta = target_beta * (epoch / int(epoch_num * 0.2)) if epoch < int(epoch_num * 0.2) else target_beta
        train_loss_epoch = 0
        
        for batch_idx, (cell_features, label_indices) in enumerate(dataloader):
            cell_features = cell_features.to(used_device, dtype=torch.float32, non_blocking=True)
            label_one_hot = F.one_hot(label_indices, num_cell_types).float().to(used_device, non_blocking=True)
            
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                with torch.autocast(device_type='cuda', dtype=dtype_preference):
                    sc_recon, sc_enc_mu, sc_enc_logvar = vae(cell_features, label_one_hot, core_indices_tensor)
                    
                    loss_recon = criterion(sc_recon, cell_features)
                    loss_kld = -0.5 * torch.mean(torch.sum(1 + sc_enc_logvar - sc_enc_mu.pow(2) - sc_enc_logvar.exp(), dim=1))
                    
                    real_core = cell_features[:, core_indices_tensor]
                    recon_core = sc_recon[:, core_indices_tensor]
                    loss_corr = compute_gene_correlation_loss(real_core, recon_core)

                    total_loss = lambda_recon * loss_recon + beta * loss_kld + lambda_corr * loss_corr
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                    train_loss_epoch += total_loss.item()
            
        train_loss_epoch /= len(dataloader)
        scheduler.step(train_loss_epoch)
        curr_std = torch.exp(0.5 * sc_enc_logvar).mean().item()
        
        if train_loss_epoch < min_loss:
            min_loss = train_loss_epoch
            best_vae = copy.deepcopy(vae)
            epoch_final = epoch

    print(f"Best Loss: {min_loss:.4f} at epoch {epoch_final}")
    torch.save(best_vae.state_dict(), os.path.join(output_dir, 'scvae_best.pth'))
    
    # Compute Final Priors
    vae = best_vae
    vae.eval()
    final_dataset = TensorDataset(X_tensor, labels)
    final_loader = DataLoader(final_dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    all_mu_list, all_logvar_list, all_labels_list = [], [], []
    with torch.autocast(device_type='cuda', dtype=dtype_preference):
        with torch.no_grad():
            for batch_x, batch_y in tqdm(final_loader, desc="Computing Final Priors"):
                batch_x = batch_x.to(used_device, dtype=torch.float32)
                batch_y_hot = F.one_hot(batch_y, num_cell_types).float().to(used_device)
                mu, logvar = vae.encode(batch_x[:, core_indices_tensor], batch_y_hot)
                
                all_mu_list.append(mu.float().cpu().numpy())
                all_logvar_list.append(logvar.float().cpu().numpy())
                all_labels_list.append(batch_y.cpu().numpy())
                
    all_mu = np.concatenate(all_mu_list, axis=0)
    all_logvar = np.concatenate(all_logvar_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    
    final_cell_type_mu_logvar = {}
    for label_id in range(num_cell_types):
        mask = all_labels == label_id
        if mask.sum() == 0: continue
        final_cell_type_mu_logvar[label_id] = (
            torch.tensor(all_mu[mask].mean(axis=0, keepdims=True)),
            torch.tensor(all_logvar[mask].mean(axis=0, keepdims=True))
        )

    torch.save(final_cell_type_mu_logvar, os.path.join(output_dir, 'cell_type_mu_logvar_best.pt'))
    return best_vae, final_cell_type_mu_logvar
