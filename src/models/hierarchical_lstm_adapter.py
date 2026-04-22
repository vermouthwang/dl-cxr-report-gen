"""
Adapter: presents the shared model interface over Yousuf's HierarchicalLSTM.

Why an adapter (not edit his file):
  Yousuf's HierarchicalLSTM is authored on a separate branch and is the
  canonical expression of his hierarchical architecture. Modifying it
  in-place would force him to re-integrate every time he iterates. Instead,
  we wrap it. His file stays untouched; the adapter does the translation
  between interfaces.

What this adapter does:
  1. Accepts our constructor signature: __init__(vocab_size, config)
     and unpacks config into HierarchicalLSTM kwargs.
  2. Accepts our forward signature:
        forward(images, input_tokens, target_tokens, lengths)
     and reconstructs the captions+masks tensors his model expects:
        captions  = [BOS, w1, ..., wN, EOS]  (reconstructed from input + target)
        masks     = 1s for real tokens, 0s for padding, from `lengths`
  3. Returns the dict {"loss": ..., "logits": ...} our trainer expects.
     Note: HierarchicalLSTM does not naturally produce flat (B, T, V) logits.
     We return a placeholder zero tensor with the right shape. The trainer
     never reads logits during training (only output["loss"]), so this is safe.
  4. Adapts generate() signature from (images, max_len=None) to
     (images, max_length, beam_size=1). beam_size is accepted but ignored
     (Yousuf's model is greedy).
  5. Configures sentence boundaries to use <SEP> (id 4) instead of <EOS>,
     matching the updated tokenizer (see src/data/iu_xray.py).

Important setup notes for Yousuf's model:
  - pretrained DenseNet-121 weights are downloaded on first instantiation.
    On PACE, run `scripts/cache_densenet_weights.py` on the login node once
    so the weights are cached in $TORCH_HOME before compute-node runs.
  - encoder submodule is named `self.encoder` inside HierarchicalLSTM, so
    the optimizer's param_groups "hlstm.encoder" prefix will match correctly
    for the lower-LR encoder fine-tuning (see configs/lstm.yaml).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.iu_xray import BOS_ID, EOS_ID, PAD_ID, SEP_ID
from src.models.hierarchical_lstm import HierarchicalLSTM


class HierarchicalLSTMAdapter(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()

        # Unpack config with defaults matching his YAML.
        hl_kwargs = dict(
            vocab_size=vocab_size,
            embed_size=int(config.get("embed_size", 512)),
            hidden_size=int(config.get("hidden_size", 512)),
            word_num_layers=int(config.get("word_num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            s_max=int(config.get("s_max", 6)),
            n_max=int(config.get("n_max", 30)),
            bos_id=BOS_ID,
            # Use <SEP> as the sentence boundary token. HierarchicalLSTM's
            # `eos_id` param is what its reshape splits on - we repurpose that
            # parameter to mean "sentence separator" without renaming it in
            # his code. The real end-of-report EOS still exists at id 2;
            # his reshape just won't see it as a split point, which is
            # correct (the whole report is one unit of sentences, not one
            # sentence itself).
            eos_id=SEP_ID,
            pad_id=PAD_ID,
            pretrained_encoder=bool(config.get("pretrained_encoder", True)),
            reshape_captions=True,   # we always feed flat sequences
        )
        self.hlstm = HierarchicalLSTM(**hl_kwargs)
        self.vocab_size = vocab_size

        # For generate() - actual end of report, not sentence boundary
        self.true_eos_id = EOS_ID

    def forward(
        self,
        images: torch.Tensor,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        """
        Our trainer passes pre-shifted tensors:
            input_tokens  = [BOS, w1, ..., wN]
            target_tokens = [w1, ..., wN, EOS]
        Yousuf's model expects the full un-shifted sequence:
            captions      = [BOS, w1, ..., wN, EOS]
        We reconstruct it by concatenating input_tokens[:, 0:1] (BOS) with
        target_tokens (which ends in EOS).
        """
        B, T = input_tokens.shape
        device = input_tokens.device

        # Reconstruct full sequence: [BOS, w1, ..., wN, EOS]
        bos_col = input_tokens[:, :1]                                # (B, 1)
        captions = torch.cat([bos_col, target_tokens], dim=1)        # (B, T+1)

        # Masks: 1 for real tokens, 0 for pad.
        # Full sequence has lengths[b] + 1 real tokens (BOS + all non-pad targets).
        full_lengths = lengths + 1
        arange = torch.arange(T + 1, device=device).unsqueeze(0)     # (1, T+1)
        masks = (arange < full_lengths.unsqueeze(1)).long()          # (B, T+1)

        loss = self.hlstm(images, captions, masks)

        # Placeholder logits - HierarchicalLSTM does not produce flat logits
        # per time step. Trainer does not use logits during training.
        # Return a zero tensor with the right shape to satisfy the interface.
        dummy_logits = torch.zeros(B, T, self.vocab_size, device=device)

        return {"loss": loss, "logits": dummy_logits}

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_length: int, beam_size: int = 1) -> list[list[int]]:
        """
        Adapt the generate() signature.
        Yousuf's model is greedy-only; beam_size is accepted but ignored.
        max_length is interpreted as the per-sentence limit (passed as max_len);
        total generated tokens can still exceed this across multiple sentences.
        """
        reports = self.hlstm.generate(images, max_len=max_length)
        return reports
