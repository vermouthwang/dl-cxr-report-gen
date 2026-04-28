"""Clinical-term-guided Transformer for chest X-ray report generation.

Architecture
------------
- Encoder: DenseNet-121 (ImageNet-pretrained), spatial feature map (B, 1024, 7, 7)
  → flatten to (B, 49, 1024) → linear projection to (B, 49, d_model). The 49
  spatial tokens are passed as memory to the decoder's cross-attention.
- Decoder: standard pre-norm Transformer decoder (causal self-attn + cross-attn
  + position-wise FFN), `n_layers` layers.

Clinical guidance (both toggleable via config; ablation table)
--------------------------------------------------------------
- `clinical.loss_weighting` (training-time): nn.CrossEntropyLoss is weighted by
  a per-token alpha vector built from data_meta/clinical_terms.json. Findings
  get `alpha_finding`, negations get `alpha_negation`, hedges get `alpha_hedge`.
  When OFF, falls back to plain CE with ignore_index=PAD_ID.
- `clinical.decoding_bias` (inference-time): at every greedy step, an additive
  bias `bias_strength * mask` is added to the next-token logits, where `mask`
  is 1.0 on clinical/negation positions and 0.0 elsewhere. When OFF, plain
  greedy decoding.

When BOTH flags are OFF, this model is functionally a vanilla DenseNet+Transformer
— the design point that makes the 4-cell A/B/C/D ablation a pure config change.

Interface contract (from src/data/iu_xray.py and src/training/trainer.py):
    forward(images, input_tokens, target_tokens, lengths) -> {"loss", "logits"}
    generate(images, max_length, beam_size=1) -> list[list[int]]   # excludes BOS and EOS
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

from src.data.clinical_vocab import ClinicalLexicon
from src.data.iu_xray import BOS_ID, EOS_ID, PAD_ID

logger = logging.getLogger(__name__)


_NUM_SPATIAL_TOKENS = 49      # 7x7 feature map from DenseNet-121 at 224x224 input
_DENSENET_FEATURE_DIM = 1024  # last DenseBlock channel count


class ClinicalTransformer(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        self.vocab_size = vocab_size

        # ---- decoder hyperparameters ----
        d_model = int(config.get("d_model", 512))
        n_heads = int(config.get("n_heads", 8))
        n_layers = int(config.get("n_layers", 3))
        d_ff = int(config.get("d_ff", 2048))
        dropout = float(config.get("dropout", 0.3))
        max_pos = int(config.get("max_position_embeddings", 128))

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.max_pos = max_pos

        # ---- encoder: DenseNet-121 features + projection to d_model ----
        pretrained = bool(config.get("pretrained_encoder", True))
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        self.encoder = backbone.features  # (B, 3, 224, 224) -> (B, 1024, 7, 7)
        self.image_proj = nn.Linear(_DENSENET_FEATURE_DIM, d_model)
        self.image_norm = nn.LayerNorm(d_model)

        if bool(config.get("use_spatial_pos_embed", True)):
            self.image_pos_embed = nn.Parameter(torch.zeros(1, _NUM_SPATIAL_TOKENS, d_model))
            nn.init.trunc_normal_(self.image_pos_embed, std=0.02)
        else:
            self.image_pos_embed = None

        self.freeze_encoder = bool(config.get("freeze_encoder", False))
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("ClinicalTransformer: DenseNet encoder is FROZEN (requires_grad=False).")

        # ---- token + positional embedding ----
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_pos, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # ---- Transformer decoder ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: more stable under AMP than post-norm
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Tie input/output embeddings (parity with most modern Transformers).
        if bool(config.get("tie_word_embeddings", True)):
            self.output_proj.weight = self.token_embed.weight

        self._init_decoder_weights()

        # ---- clinical guidance ----
        self.clinical_cfg = dict(config.get("clinical") or {})
        self.use_loss_weighting = bool(self.clinical_cfg.get("loss_weighting", False))
        self.use_decoding_bias = bool(self.clinical_cfg.get("decoding_bias", False))
        self.bias_strength = float(self.clinical_cfg.get("bias_strength", 1.0))

        # Always register buffers (size-fixed) so the same checkpoint loads
        # regardless of which flags were ON at save time. When a flag is OFF,
        # the corresponding buffer is just unused.
        weight_vec, bias_vec = self._build_clinical_vectors()
        # persistent=False: recompute from config on each __init__, never load
        # stale values from a checkpoint.
        self.register_buffer("clinical_weight", weight_vec, persistent=False)
        self.register_buffer("clinical_bias_mask", bias_vec, persistent=False)

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------
    def _init_decoder_weights(self) -> None:
        """Standard small-init for everything outside the (already-pretrained) DenseNet."""
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        # Re-zero PAD row to keep padding embedding fixed at zero.
        with torch.no_grad():
            self.token_embed.weight[PAD_ID].fill_(0.0)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.zeros_(self.image_proj.bias)
        # output_proj is tied to token_embed if tie_word_embeddings=True; otherwise init it
        if self.output_proj.weight is not self.token_embed.weight:
            nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    # ----------------------------------------------------------------------
    # Clinical vector construction
    # ----------------------------------------------------------------------
    def _build_clinical_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (weight_vec, bias_mask) from the lexicon + on-disk vocab snapshot.

        The lexicon resolves seed words against `data_meta/vocab.json` (a snapshot
        of the runtime vocabulary). We assert the snapshot's vocab_size matches
        the runtime vocab_size — a mismatch indicates the snapshot is stale,
        which would silently misalign clinical IDs with the actual embedding rows.

        If both clinical flags are OFF, returns trivial (ones, zeros) vectors.
        """
        if not (self.use_loss_weighting or self.use_decoding_bias):
            ones = torch.ones(self.vocab_size, dtype=torch.float32)
            zeros = torch.zeros(self.vocab_size, dtype=torch.float32)
            return ones, zeros

        lexicon_path = Path(self.clinical_cfg.get("lexicon_path", "data_meta/clinical_terms.json"))
        vocab_path = Path(self.clinical_cfg.get("vocab_path", "data_meta/vocab.json"))
        if not lexicon_path.is_file():
            raise FileNotFoundError(
                f"Clinical lexicon not found: {lexicon_path}. "
                f"Disable both clinical flags or fix model.config.clinical.lexicon_path."
            )
        if not vocab_path.is_file():
            raise FileNotFoundError(
                f"Vocab snapshot not found: {vocab_path}. "
                f"Run `python scripts/dump_vocab.py` to regenerate it."
            )

        with open(vocab_path) as f:
            vocab_payload = json.load(f)
        word_to_id: dict[str, int] = vocab_payload["word_to_id"]
        if len(word_to_id) != self.vocab_size:
            raise ValueError(
                f"Vocab snapshot at {vocab_path} has size {len(word_to_id)}, "
                f"but runtime vocab_size is {self.vocab_size}. The snapshot is stale; "
                f"regenerate it with `python scripts/dump_vocab.py`."
            )

        lexicon = ClinicalLexicon.from_files(lexicon_path, word_to_id)

        alpha_finding = float(self.clinical_cfg.get("alpha_finding", 3.0))
        alpha_negation = float(self.clinical_cfg.get("alpha_negation", 2.0))
        alpha_hedge = float(self.clinical_cfg.get("alpha_hedge", 1.0))
        include_negations_in_bias = bool(self.clinical_cfg.get("bias_include_negations", True))

        weight_vec = lexicon.weight_vector(
            alpha_finding=alpha_finding,
            alpha_negation=alpha_negation,
            alpha_hedge=alpha_hedge,
            dtype=torch.float32,
        )
        bias_vec = lexicon.bias_mask(
            include_negations=include_negations_in_bias,
            dtype=torch.float32,
        )

        # Diagnostics: empty categories warn loudly so users notice vocab drift.
        empty = [c for c, ids in lexicon.category_to_ids.items() if not ids]
        if empty:
            logger.warning(
                f"ClinicalTransformer: categories with zero matched tokens: {empty}. "
                f"These will not contribute to weighted CE or decoding bias."
            )
        if lexicon.missing:
            logger.info(
                f"ClinicalTransformer: lexicon seeds missing from vocab "
                f"(per tier): {lexicon.missing}"
            )
        loss_state = "ON" if self.use_loss_weighting else "OFF"
        bias_state = "ON" if self.use_decoding_bias else "OFF"
        logger.info(
            f"ClinicalTransformer: clinical guidance loss_weighting={loss_state} "
            f"(alphas: F={alpha_finding}, N={alpha_negation}, H={alpha_hedge}), "
            f"decoding_bias={bias_state} (strength={self.bias_strength}, "
            f"include_negations={include_negations_in_bias})."
        )
        return weight_vec, bias_vec

    # ----------------------------------------------------------------------
    # Image encoding
    # ----------------------------------------------------------------------
    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 49, d_model) memory tokens."""
        feat = self.encoder(images)              # (B, 1024, 7, 7)
        feat = F.relu(feat, inplace=False)       # standard DenseNet pre-classifier activation
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)   # (B, 49, 1024)
        feat = self.image_proj(feat)             # (B, 49, d_model)
        feat = self.image_norm(feat)
        if self.image_pos_embed is not None:
            feat = feat + self.image_pos_embed
        return feat

    # ----------------------------------------------------------------------
    # Decoder forward (shared by training and generation)
    # ----------------------------------------------------------------------
    def _decode(
        self,
        input_tokens: torch.Tensor,        # (B, T)
        memory: torch.Tensor,              # (B, 49, d_model)
    ) -> torch.Tensor:                     # (B, T, vocab_size)
        B, T = input_tokens.shape
        if T > self.max_pos:
            raise ValueError(
                f"Input length {T} exceeds max_position_embeddings={self.max_pos}. "
                f"Increase model.config.max_position_embeddings."
            )

        positions = torch.arange(T, device=input_tokens.device).unsqueeze(0).expand(B, T)
        x = self.token_embed(input_tokens) + self.pos_embed(positions)
        x = self.embed_dropout(x)

        # Both masks must be the SAME dtype (PyTorch >= 2.x deprecates mixed
        # bool/float masks). We use bool for both: True means "do not attend".
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=input_tokens.device),
            diagonal=1,
        )
        pad_mask = input_tokens.eq(PAD_ID)  # (B, T) bool — True at PAD positions

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=None,  # all 49 image tokens are valid
        )
        out = self.decoder_norm(out)
        return self.output_proj(out)       # (B, T, V)

    # ----------------------------------------------------------------------
    # Training forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        lengths: torch.Tensor,             # unused; kept for interface compatibility
    ) -> dict:
        memory = self._encode_image(images)
        logits = self._decode(input_tokens, memory)        # (B, T, V)

        weight = self.clinical_weight if self.use_loss_weighting else None
        # cross_entropy auto-skips PAD via ignore_index. weight is broadcast over
        # the class dim and is independent of ignore_index.
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_tokens.reshape(-1),
            weight=weight,
            ignore_index=PAD_ID,
        )
        return {"loss": loss, "logits": logits}

    # ----------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int,
        beam_size: int = 1,
    ) -> list[list[int]]:
        """Greedy autoregressive decoding with optional clinical logit bias.

        Returns one list[int] per image. Lists exclude BOS, and stop at (excluding)
        EOS — matching the convention used by dummy.py and hierarchical_lstm_adapter.
        """
        if beam_size != 1:
            raise NotImplementedError(
                f"ClinicalTransformer.generate: beam_size={beam_size} not supported yet. "
                f"Only greedy (beam_size=1) is implemented."
            )

        was_training = self.training
        self.eval()
        try:
            B = images.size(0)
            device = images.device

            memory = self._encode_image(images)            # (B, 49, d_model)
            current = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            results: list[list[int]] = [[] for _ in range(B)]

            bias_vec = None
            if self.use_decoding_bias and self.bias_strength != 0.0:
                bias_vec = self.bias_strength * self.clinical_bias_mask  # (V,)

            for _ in range(max_length):
                logits = self._decode(current, memory)     # (B, T, V)
                next_logits = logits[:, -1, :]             # (B, V)
                if bias_vec is not None:
                    next_logits = next_logits + bias_vec   # broadcast over batch

                next_token = next_logits.argmax(dim=-1)    # (B,)

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
        finally:
            if was_training:
                self.train()

        return results
