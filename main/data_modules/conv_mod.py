import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    
    def __init__(self, in_chans, out_chans, drop_prob=0.0, pool_size=2):
       
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.pool_size = pool_size

        layers = [nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
                  nn.InstanceNorm2d(out_chans),  
                  nn.ReLU(),
                  nn.Dropout2d(drop_prob)]

        if pool_size > 1:
            layers.append(nn.MaxPool2d(pool_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
       
        input = torch.clamp(input, min=-1e6, max=1e6)
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob}, max_pool_size={self.pool_size})'