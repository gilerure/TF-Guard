from torch import nn
import torch

def min_max_normalization(X):
    min_val = X.min(dim=0, keepdim=True)[0]
    max_val = X.max(dim=0, keepdim=True)[0]

    denom = max_val - min_val
    denom[denom == 0] = 1 

    X_norm = (X - min_val) / denom
    return X_norm

def z_score_normalize(X):
    mean = torch.mean(X, axis=1, keepdims=True)
    std = torch.std(X, axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

class ScaleMLP(nn.Module):
    def __init__(self, hidden_dim=16):
        super(ScaleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = z_score_normalize(x) # shape: (B, 256, 1)

        mean = x.mean(dim=0).detach() # (256, 1)
        var = x.var(dim=0).detach() # (256, 1)
        mad = (x - mean).abs().mean(dim=0).detach()

        stats = min_max_normalization(torch.cat([mean, var, mad], dim=-1))
        scale = self.mlp(stats)  # (256, 1)
        return scale
    
    
class FourierScaleMLP(nn.Module):
    def __init__(self, input_channels=32, hidden_dim=16):
        super(FourierScaleMLP, self).__init__()
        input_dim = 6 * input_channels
        output_channels = input_channels
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_channels),  
            nn.Sigmoid()  
        )

    def forward(self, x):
        # Fourier transform
        x_ft = torch.fft.rfft(x, dim=1) # (B, F, D)

        # Calculate magnitude and angle 
        mag = torch.abs(x_ft) # (B, F, D)
        angle = torch.angle(x_ft)

        # Construct the input of MLP
        mag = z_score_normalize(mag) # (B, F, D)
        angle = z_score_normalize(angle)

        mag_mean = mag.mean(dim=0) # (F, D)
        mag_var = mag.var(dim=0) 
        mag_mad = (mag - mag_mean).abs().mean(dim=0)

        angle_mean = angle.mean(dim=0)
        angle_var = angle.var(dim=0)
        angle_mad = (angle - angle_mean).abs().mean(dim=0)

        stats = min_max_normalization(torch.cat([mag_mean, mag_var, mag_mad, angle_mean, angle_var, angle_mad], dim=-1))
        
        # Scaling in frequency domain
        scale = self.mlp(stats)  # (F, D)
        x_ft = x_ft * scale

        # Inverse Fourier transform
        x_out = torch.fft.irfft(x_ft, n=x.size(1), dim=1) 

        return x_out