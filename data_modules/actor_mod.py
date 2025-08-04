import torch.nn as nn
from data_modules.conv_mod import ConvBlock


class PolicyModel(nn.Module): 
    def __init__(self, resolution, in_chans, chans, num_pool_layers, drop_prob, fc_size):
        
        super().__init__()
        self.resolution = resolution
        self.in_chans = in_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.fc_size = fc_size
        
        self.pool_size = 2
        self.flattened_size = resolution * resolution * chans

        self.channel_layer = ConvBlock(in_chans, chans, drop_prob, pool_size=1)
        self.down_sample_layers = nn.ModuleList([])
        ch = chans
        
        self.flattened_size = (chans * (2 ** num_pool_layers)) * (resolution // (2 ** num_pool_layers)) * (resolution // (2 ** num_pool_layers))

        for _ in range(num_pool_layers):
            next_ch = min(ch * 2, 1024)
            self.down_sample_layers += [nn.Sequential(
                                                      ConvBlock(ch, next_ch, drop_prob, pool_size=self.pool_size),
                                                      nn.BatchNorm2d(next_ch)
                                                     )
                                        ]
            ch *= 2
        self.fc_out = nn.Sequential(
            nn.Linear(in_features=self.flattened_size, out_features=self.fc_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.fc_size, out_features=self.fc_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.fc_size, out_features=resolution)
        )

    def forward(self, image):
        image_emb = self.channel_layer(image) 
        for layer in self.down_sample_layers:
            image_emb = layer(image_emb) 
        flattened_emb = image_emb.flatten(start_dim=1)
        output = self.fc_out(flattened_emb) 
        return output