import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionV3(nn.Module):
    def __init__(
        self, 
        num_classes=2, 
        pretrained=True,
        c_t=512,
        dropout=0.2,
        act_fn=nn.ReLU
    ):
        super().__init__()
        
        if pretrained:
            weights = Inception_V3_Weights.DEFAULT
            # Must set aux_logits=True to load pretrained weights
            self.backbone = inception_v3(weights=weights, aux_logits=True)
            # Now, disable the auxiliary classifier manually
            self.backbone.aux_logits = False
            self.backbone.AuxLogits = None  # Optional: remove to save memory
        else:
            # For non-pretrained, you can safely use aux_logits=False
            self.backbone = inception_v3(aux_logits=False)
        
        # Replace the main classifier (fc)
        self.backbone.fc = nn.Sequential(
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
        # Forward pass — now returns only main logits
        out = self.backbone(x)
        # If aux_logits were enabled, it would return InceptionOutputs,
        # but we disabled it, so it returns just a tensor
        return out