import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# Special token IDs (must match src/data/iu_xray.py)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



# Visual Encoder (DenseNet-121)

class VisualEncoder(nn.Module):
    """
    DenseNet-121 feature extractor.
    Returns both a global pooled vector and the full spatial feature map.
    The spatial map is used by co-attention; the global vector seeds the
    sentence LSTM at t=0.
    """

    def __init__(self, embed_size: int = 512, pretrained: bool = True):
        super().__init__()
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features          # (B, 1024, 7, 7) for 224x224
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed    = nn.Linear(1024, embed_size)
        self.bn       = nn.BatchNorm1d(embed_size, momentum=0.01)


    def forward(self, images: torch.Tensor):
        """
        Args: images (B, 3, 224, 224)
        Returns:
            avg_feats (B, embed_size) - global context vector
            spatial   (B, 1024, H, W) - spatial map for attention
        """
        spatial   = F.relu(self.features(images), inplace=True)   # (B,1024,7,7)
        avg       = self.avg_pool(spatial).view(images.size(0), -1)  # (B,1024)
        avg_feats = self.bn(self.embed(avg))                          # (B,embed)
        return avg_feats, spatial


class TransformerCaptioner(nn.Module):
    def __init__(self, vocab_size, config=None):
        super().__init__()
        config = config or {}
        embed_size         = int(config.get("embed_size", 512))
        n_heads            = int(config.get("n_heads", 8))
        n_layers           = int(config.get("n_layers", 6))
        d_ff               = int(config.get("d_ff", 2048))
        dropout            = float(config.get("dropout", 0.1))
        pretrained_encoder = bool(config.get("pretrained_encoder", True))
        freeze_encoder     = bool(config.get("freeze_encoder", True))

        self.vocab_size = vocab_size

        # Reused encoder from before, but we only care about the spatial features for the Transformer
        self.encoder = VisualEncoder(embed_size=embed_size, pretrained=pretrained_encoder)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Linear layer to map DenseNet's 1024 channels to Transformer's embed_size
        self.feature_proj = nn.Linear(1024, embed_size)

        self.decoder_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(embed_size) # As defined before

        # Decoder-only stack: visual tokens go straight to cross-attention.
        # (Earlier draft used nn.Transformer(num_encoder_layers=0); PyTorch 2.11
        #  errors on that path because nn.TransformerEncoder.forward indexes
        #  self.layers[0]. nn.TransformerDecoder is the cleaner expression of
        #  the spec anyway.)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, images, input_tokens, target_tokens, lengths):
        # 1. Get features from your reused DenseNet
        _, spatial = self.encoder(images)

        # 2. Prepare visual tokens: (B, 1024, 7, 7) -> (B, 49, 512)
        visual_tokens = spatial.flatten(2).permute(0, 2, 1) # (B, H*W, 1024), "sentence" of 49 tokens, each with 1024-dim features
        visual_tokens = self.feature_proj(visual_tokens)

        # 3. Prepare caption tokens (the BOS-prefixed input sequence)
        tgt = self.pos_encoder(self.decoder_embedding(input_tokens))

        # 4. Create mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(input_tokens.size(1)).to(images.device)
        padding_mask = (input_tokens == PAD_ID)

        # 5. Decoder cross-attends from caption tokens to the 49 visual tokens
        out = self.transformer(tgt, visual_tokens, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
        logits = self.fc_out(out)

        # 6. Cross-entropy against EOS-suffixed targets, ignoring PAD positions
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_tokens.reshape(-1),
            ignore_index=PAD_ID,
        )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, images, max_length, beam_size=1):
        """Greedy autoregressive decoding. beam_size accepted for interface parity, not implemented."""
        del beam_size

        device = images.device
        B = images.size(0)

        # Encode the image once; reuse as memory across all decoding steps.
        _, spatial = self.encoder(images)
        visual_tokens = self.feature_proj(spatial.flatten(2).permute(0, 2, 1))

        tokens = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            tgt = self.pos_encoder(self.decoder_embedding(tokens))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1)).to(device)
            out = self.transformer(tgt, visual_tokens, tgt_mask=tgt_mask)
            next_token = self.fc_out(out[:, -1, :]).argmax(dim=-1)  # (B,)
            # Once a sample has emitted EOS, freeze it by appending PAD.
            next_token = torch.where(finished, torch.full_like(next_token, PAD_ID), next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == EOS_ID)
            if finished.all():
                break

        out_lists = []
        for row in tokens.tolist():
            row = row[1:]  # drop BOS
            if EOS_ID in row:
                row = row[: row.index(EOS_ID)]
            out_lists.append(row)
        return out_lists
