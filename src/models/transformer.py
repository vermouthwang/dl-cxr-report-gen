import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

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
    def __init__(self, vocab_size, embed_size=512):
        super().__init__()
        # Reused encoder from before, but we only care about the spatial features for the Transformer
        self.encoder = VisualEncoder(embed_size=embed_size)
        
        # Linear layer to map DenseNet's 1024 channels to Transformer's embed_size
        self.feature_proj = nn.Linear(1024, embed_size)
        
        self.decoder_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size) # As defined before
        
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=8,
            num_encoder_layers=2, # You don't need many here since DenseNet did the work
            num_decoder_layers=6,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, images, captions):
        # 1. Get features from your reused DenseNet
        _, spatial = self.encoder(images) 
        
        # 2. Prepare visual tokens: (B, 1024, 7, 7) -> (B, 49, 512)
        visual_tokens = spatial.flatten(2).permute(0, 2, 1) # (B, H*W, 1024), "sentence" of 49 tokens, each with 1024-dim features
        visual_tokens = self.feature_proj(visual_tokens)
        
        # 3. Prepare caption tokens
        tgt = self.pos_encoder(self.decoder_embedding(captions))
        
        # 4. Create mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(captions.size(1)).to(images.device)
        padding_mask = (captions == 0)  # Assuming 0 is the padding index
        
        # 5. Transformer does the cross-attention between 49 visual tokens and captions
        out = self.transformer(visual_tokens, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
        return self.fc_out(out)