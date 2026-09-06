import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import set_seed

# NOTE: the gene-context encoder uses the standard nn.TransformerEncoder (matching the
# deconv_20260610 / manuscript checkpoint). Route2/prior-anchored inference only uses the
# decoder + priors, but the encoder is kept identical so the published checkpoint loads
# strictly (state-dict keys transformer_encoder.layers.*).
class AttentionVAE(nn.Module):
    def __init__(self, input_size: int, output_size:int, hidden_size_list: list, 
                 mid_hidden_size: int, num_cell_types: int, embedding_dim: int, 
                 nhead: int, num_layers: int, ff_dim: int = 64, seed: int = 18):
        super(AttentionVAE, self).__init__()
        set_seed(seed)
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.gene_embedding = nn.Parameter(torch.randn(input_size, embedding_dim))
        nn.init.normal_(self.gene_embedding, mean=0, std=0.02)
        
        self.linear_proj = nn.Linear(1, embedding_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,
                                               dim_feedforward=ff_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, 1)
        
        self.label_emb_dim = mid_hidden_size * 4
        self.label_embedding = nn.Linear(num_cell_types, self.label_emb_dim)
        nn.init.xavier_uniform_(self.label_embedding.weight)
        
        self.enc_input_dim = self.input_size + self.label_emb_dim
        self.enc_feature_size_list = [self.enc_input_dim] + hidden_size_list + [self.label_emb_dim]
    
        self.encoder_layers = nn.ModuleList()
        in_dim = self.enc_input_dim
        for h_dim in self.enc_feature_size_list[1:]:
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ))
            in_dim = h_dim
        
        self.fc_mu = nn.Linear(in_dim, mid_hidden_size)
        self.fc_var = nn.Linear(in_dim, mid_hidden_size)

        self.dec_input_dim = mid_hidden_size + self.label_emb_dim
        self.dec_feature_size_list = [self.dec_input_dim] + hidden_size_list[::-1] + [self.output_size]
        
        self.decoder_layers = nn.ModuleList()
        in_dim = self.dec_input_dim
        for h_dim in self.dec_feature_size_list[1:-1]:
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ))
            in_dim = h_dim
        
        self.final_layer = nn.Linear(in_dim, self.output_size)
    
    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple:
        x_in = x.unsqueeze(-1)
        x_val = self.linear_proj(x_in)
        x_embed = x_val + self.gene_embedding
        x_feat = self.output_proj(self.transformer_encoder(x_embed)).squeeze(-1)
        label_embed = self.label_embedding(labels)
        h = torch.cat([x_feat, label_embed], dim=1)
        
        for layer in self.encoder_layers:
            h = layer(h)  
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_embed = self.label_embedding(labels)
        h = torch.cat([z, label_embed], dim=1)
        for layer in self.decoder_layers:
            h = layer(h)
        out = self.final_layer(h)
        return F.relu(out)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor, core_indices: torch.Tensor) -> tuple:
        x_core = x[:, core_indices]
        mu, logvar = self.encode(x_core, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar