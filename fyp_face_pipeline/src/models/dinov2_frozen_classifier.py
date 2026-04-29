import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=300, model_name="facebook/dinov2-base", dropout=0.4, unfreeze_last_n=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        for p in self.backbone.parameters():
            p.requires_grad = False

        blocks = self.backbone.encoder.layer
        for block in blocks[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        emb = outputs.last_hidden_state[:, 0]
        logits = self.classifier(emb)
        return logits, emb

    def backbone_params(self):
        for block in self.backbone.encoder.layer[-2:]:
            for p in block.parameters():
                if p.requires_grad:
                    yield p

    def head_params(self):
        return self.classifier.parameters()