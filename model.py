# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class FakeNewsModel(nn.Module):
    def __init__(self, model_name="distilbert-base-multilingual-cased", use_kg=True):
        super().__init__()
        self.use_kg = use_kg

        # Load transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)

        # 🔴 CRITICAL: Freeze transformer weights
        for param in self.transformer.parameters():
            param.requires_grad = False

        hidden_size = self.transformer.config.hidden_size  # safer than hardcoding
        kg_size = 3 if use_kg else 0

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + kg_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, kg_features=None):
        # 🔴 Disable gradient tracking for transformer
        with torch.no_grad():
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs.last_hidden_state[:, 0, :]

        if self.use_kg and kg_features is not None:
            combined = torch.cat((pooled_output, kg_features), dim=1)
        else:
            combined = pooled_output

        logits = self.classifier(combined)
        return logits
