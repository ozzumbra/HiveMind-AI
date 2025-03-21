import torch
import torch.nn as nn
from hex_utils import generate_hex_kernel

class HoneycombConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, radius=1, bias=True):
        super(HoneycombConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_points = generate_hex_kernel(radius)
        self.kernel_size = len(self.kernel_points)
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Простейшая заглушка
        # Реализация настоящей гекс-свёртки сложнее и зависит от специфики входных данных
        batch_size, in_channels, height, width = x.size()
        out = torch.zeros((batch_size, self.out_channels, height, width), device=x.device)
        
        for idx, (dq, dr) in enumerate(self.kernel_points):
            shifted = torch.roll(x, shifts=(dq, dr), dims=(2, 3))
            weight = self.weight[:, :, idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            out += shifted * weight
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out

