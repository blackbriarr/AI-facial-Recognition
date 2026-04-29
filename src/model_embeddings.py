import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class FaceEmbeddingModel(nn.Module):
    def __init__(self, num_classes=250, embedding_dim=128):
        super().__init__()
        # Pretrained ResNet18 backbone (shows research, good accuracy)
        backbone = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.embedding_head = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x).flatten(1)  # [B, 512]
        embeddings = self.embedding_head(features)  # [B, 128]
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize!
        logits = self.classifier(features)
        return embeddings, logits
