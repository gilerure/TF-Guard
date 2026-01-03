import torch, math
from torch import nn

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        pad_front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        pad_end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([pad_front, x, pad_end], dim=1)
        x = self.avg(x.permute(0, 2, 1))  # [B, D, L]
        x = x.permute(0, 2, 1)            # [B, L, D]
        return x
    
    
class MOSDecomp(nn.Module):
    def __init__(self, in_channels=32, kernel_sizes=(8, 16, 32, 64)):
        super(MOSDecomp, self).__init__()
        self.experts = nn.ModuleList([MovingAvg(k) for k in kernel_sizes])
        self.gate = nn.Sequential(
            nn.Linear(in_channels * len(kernel_sizes), in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, len(kernel_sizes)),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        trends = []
        for expert in self.experts:
            t = expert(x) 
            trends.append(t.unsqueeze(-1))   # [B, L, D, 1]

        trend_stack = torch.cat(trends, dim=-1)
        B, L, D, U = trend_stack.shape
        gate_input = trend_stack.view(B, L, -1) # [B, L, D * U]
        weights = self.gate(gate_input).unsqueeze(dim=2) # [B, L, 1, U]

        trend = (trend_stack * weights).sum(-1)   # [B, L, D]
        seasonal = x - trend
        return seasonal, trend