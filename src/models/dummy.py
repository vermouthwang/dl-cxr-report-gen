"""
Dummy model for smoke-testing the training pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.iu_xray import BOS_ID, EOS_ID, PAD_ID


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        hidden_dim = int(config.get("hidden_dim", 128))
        image_feature_dim = int(config.get("image_feature_dim", 64))

        # Tiny CNN: (B, 3, 224, 224) -> (B, image_feature_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, image_feature_dim),
        )
        self.image_to_hidden = nn.Linear(image_feature_dim, hidden_dim)

        # Text decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (1, B, H) initial LSTM hidden state."""
        feat = self.encoder(images)              # (B, image_feature_dim)
        h = self.image_to_hidden(feat)           # (B, hidden_dim)
        return h.unsqueeze(0)                    # (1, B, hidden_dim)

    def forward(
        self,
        images: torch.Tensor,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        h0 = self._encode_image(images)
        c0 = torch.zeros_like(h0)

        emb = self.embedding(input_tokens)               # (B, T, H)
        out, _ = self.lstm(emb, (h0, c0))                # (B, T, H)
        logits = self.output_proj(out)                    # (B, T, V)

        # Cross-entropy with padding masked via ignore_index
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_tokens.reshape(-1),
            ignore_index=PAD_ID,
        )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_length: int, beam_size: int = 1) -> list[list[int]]:
        """
        Greedy autoregressive generation. beam_size is accepted for interface
        compatibility but ignored — this is a dummy model.
        """
        if beam_size != 1:
            pass 

        B = images.size(0)
        device = images.device

        h = self._encode_image(images)
        c = torch.zeros_like(h)

        # Start with BOS
        current = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results: list[list[int]] = [[] for _ in range(B)]

        for _ in range(max_length):
            emb = self.embedding(current[:, -1:])         # (B, 1, H)
            out, (h, c) = self.lstm(emb, (h, c))          # (B, 1, H)
            next_token = self.output_proj(out[:, -1]).argmax(dim=-1)  # (B,)

            for i in range(B):
                if finished[i]:
                    continue
                tid = int(next_token[i].item())
                if tid == EOS_ID:
                    finished[i] = True
                else:
                    results[i].append(tid)

            current = torch.cat([current, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return results