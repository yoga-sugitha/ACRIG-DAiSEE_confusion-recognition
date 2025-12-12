import torch.nn as nn
from torchvision import models

class InceptionV3(nn.Module):
    def __init__(self, num_classes=2, c_t=512, act_fn=nn.ReLU, dropout=0.2):
        super().__init__()
        # Load pretrained InceptionV3
        self.backbone = models.inception_v3(weights='DEFAULT')
        # Replace the fully connected layer with Identity to extract features
        self.backbone.fc = nn.Identity()
        
        # Custom classifier head
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, c_t),
            act_fn(),
            nn.Dropout(p=dropout),
            nn.Linear(c_t, c_t // 2),
            act_fn(),
            nn.Dropout(p=dropout),
            nn.Linear(c_t // 2, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)  # Shape: (B, 2048)
        x = self.fc(x)
        return x