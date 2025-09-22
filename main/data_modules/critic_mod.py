import torch.nn as nn
import torch


class Critic(nn.Module): 
    def __init__(self, state_dim=320*320, action_dim=320, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.state_processor = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(state_dim, hidden_dim)  
        )
        
        self.action_processor = nn.Linear(action_dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        state_features = self.state_processor(state)   
        action_features = self.action_processor(action) 
        combined = torch.cat([state_features, action_features], dim=1)
        return self.net(combined)