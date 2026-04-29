import torch
import torch.nn as nn
from torchvision import models


class FaceEmbeddingModel(nn.Module):
    def __init__(self, num_classes=250, embedding_dim=256, pretrained=False):
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.embedding = nn.Linear(in_features, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        emb = self.embedding(x)
        emb = self.embedding_bn(emb)
        logits = self.classifier(emb)
        return emb, logits