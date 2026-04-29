import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class FaceEmbeddingModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=128, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_head = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = self.classifier(features)
        return embeddings, logits