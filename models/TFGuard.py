import torch
from torch import nn
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import CrossAttention
from layers.Scale import ScaleMLP, FourierScaleMLP
from layers.Autoformer_EncDec import MOSDecomp
       
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.P = configs.patch_size
        self.K = configs.K

        # Scaling MLP
        self.timeScaler = ScaleMLP(hidden_dim=configs.hidden_dim)
        self.freqScaler = FourierScaleMLP(input_channels=configs.enc_in, hidden_dim=configs.hidden_dim)

        # Embedding layer with positional encoding
        self.embedding = DataEmbedding(c_in=configs.enc_in, d_model=configs.d_model)

        # Mixture of Seasonal Decomposition layer
        self.decomp = MOSDecomp(in_channels=configs.d_model, kernel_sizes=configs.kernel_size)

        # Cross attention layer
        self.cross_attn = CrossAttention(in_channels=configs.d_model, 
                                        d_keys=configs.hidden_dim, 
                                        d_values=configs.hidden_dim,
                                        )


        # Output projection layer
        self.projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.hidden_dim),
            nn.ReLU(),
            nn.Linear(configs.hidden_dim, configs.hidden_dim),
            nn.ReLU(),
            nn.Linear(configs.hidden_dim, configs.c_out)
        )

    def forward(self, x_enc):
        # Two branches in time domain and frequency domain
        x_time = x_enc
        x_freq = x_enc

        # Scaling layer in both time domain and frequency domain
        scale = self.timeScaler(x_time)
        x_time = x_time * scale
        x_freq = self.freqScaler(x_freq)

        # Embedding layer
        x_time = self.embedding(x_time) # (128, 256, 32)
        x_freq = self.embedding(x_freq)
        seasonal_time = x_time
        seasonal_freq = x_freq

        # Mixture of Seasonal Decomposition
        seasonal_time, _ = self.decomp(seasonal_time)

        # Frequency Wave Modeling
        B, L, D = seasonal_freq.shape
        P = self.P
        N = L // P # patch count
        
        patches = seasonal_freq.reshape(B, N, P, D) # [B, N, P, D]
        freq_patches = torch.fft.rfft(patches, dim=2) 
        
        amp = torch.abs(freq_patches)
        wave = torch.zeros_like(amp)
        wave[:, 1:, :, :] = amp[:, 1:, :, :] - amp[:, :-1, :, :]
        wave[:, 0, :, :] = amp[:, 0, :, :]
        
        values, indices = torch.topk(wave, k=self.K, dim=2) 
        
        mask = torch.zeros_like(freq_patches, dtype=torch.bool)
        mask.scatter_(dim=2, index=indices, value=True)
        
        filtered_freq = freq_patches * mask
        restored_patches = torch.fft.irfft(filtered_freq, n=P, dim=2)
        seasonal_freq = restored_patches.reshape(B, L, D)

        # Cross attention 
        attn_out = self.cross_attn(seasonal_freq, seasonal_time, seasonal_time) 

        # Projection layer
        out = self.projection(attn_out)
        return out


