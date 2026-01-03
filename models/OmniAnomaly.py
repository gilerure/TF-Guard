import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from layers.Embed import DataEmbedding

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_dim = configs.hidden_dim
        self.n_feats = configs.c_out 

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model)
        self.lstm = nn.GRU(configs.d_model, self.hidden_dim, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 2*self.hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        x = self.enc_embedding(x)
        B, L, D = x.shape
        
        hidden = torch.rand(2, B, self.hidden_dim).to(self.device) if hidden is not None else hidden
        out, hidden = self.lstm(x.contiguous().view(-1, B, D), hidden)

        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.hidden_dim, self.hidden_dim], dim=-1)

        ## Reparameterization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        x = mu + eps*std

        ## Decoder
        x = self.decoder(x) # (L, B, n_feats) 
        return x.reshape(B, L*self.n_feats)