import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class ResNet50Module(nn.Module):
    def __init__(
            self,
            num_classes: int = 2,  
            dropout_prob: float = 0.1,
    ):
        
        super().__init__()

        self.num_classes = num_classes
          
        self.resnet50 = models.resnet50(weights='DEFAULT')
        self.append_dropout(self.resnet50, rate=dropout_prob)

        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features // 4, num_classes)
        )
        
        print(self.resnet50)
        

    def append_dropout(self, module, rate):
        for name, child_module in module.named_children():
            if len(list(child_module.children())) > 0:
                self.append_dropout(child_module, rate)
            if isinstance(child_module, nn.ReLU) and not isinstance(child_module, nn.Dropout2d):
                new_module = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout2d(p=rate)  
                )
                setattr(module, name, new_module)

    def forward(self, image):
            logits = self.resnet50(image)
            probs = F.softmax(logits, dim=-1)
            return probs, logits