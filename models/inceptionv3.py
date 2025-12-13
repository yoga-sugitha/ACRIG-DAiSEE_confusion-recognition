import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.inception import InceptionOutputs

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
        
        # Load backbone
        if pretrained:
            weights = Inception_V3_Weights.DEFAULT
            self.backbone = inception_v3(weights=weights, aux_logits=True)
        else:
            self.backbone = inception_v3(aux_logits=False)
        
        # Freeze original fc (optional, but good practice)
        # We won't use it — we'll replace the whole head
        
        # Remove original fc — set to Identity so backbone returns features
        self.backbone.fc = nn.Identity()
        
        # Build your custom head
        self.classifier = nn.Sequential(
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
        features = self.backbone(x)
        
        # Handle InceptionOutputs (from aux_logits=True)
        if isinstance(features, InceptionOutputs):
            features = features.logits  # main output is in .logits
        
        # Now pass through your custom classifier
        return self.classifier(features)