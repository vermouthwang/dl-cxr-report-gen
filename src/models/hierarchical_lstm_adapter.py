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
     and bundles (input_tokens, target_tokens, lengths) into the tuple
     Yousuf's forward expects.
  3. Returns the dict {"loss": ..., "logits": ...} our trainer expects.
     Note: HierarchicalLSTM does not naturally produce flat (B, T, V) logits.
     We return a placeholder zero tensor with the right shape. The trainer
     never reads logits during training (only output["loss"]), so this is safe.
  4. Adapts generate(): Yousuf's returns a (B, T_out) LongTensor, our trainer
     expects list[list[int]]. The adapter converts.
  5. Configures sentence boundaries to use <SEP> (id 4) instead of <EOS>,
     matching the updated tokenizer (see src/data/iu_xray.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.iu_xray import BOS_ID, EOS_ID, PAD_ID, SEP_ID
from src.models.hierarchical_lstm import HierarchicalLSTM


class HierarchicalLSTMAdapter(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()

        hl_kwargs = dict(
            vocab_size=vocab_size,
            embed_size=int(config.get("embed_size", 512)),
            hidden_size=int(config.get("hidden_size", 512)),
            word_num_layers=int(config.get("word_num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            s_max=int(config.get("s_max", 6)),
            n_max=int(config.get("n_max", 30)),
            bos_id=BOS_ID,
            eos_id=SEP_ID,        # use <SEP> as the sentence-boundary splitter
            pad_id=PAD_ID,
            pretrained_encoder=bool(config.get("pretrained_encoder", True)),
        )
        self.hlstm = HierarchicalLSTM(**hl_kwargs)
        self.vocab_size = vocab_size

    def forward(self, images, input_tokens, target_tokens, lengths):
        B, T = input_tokens.shape
        device = input_tokens.device

        # Drop the final EOS from each sample's sequence before handing to
        # HierarchicalLSTM. Reason: our dataloader appends EOS (id 2) at
        # position lengths[b]-1 of target_tokens. His _reshape_flat_captions
        # splits on SEP (id 4); the trailing EOS, arriving after the final
        # SEP, forms a lone-token "sentence" whose teacher-forcing target
        # after shift is all-PAD, which makes CrossEntropyLoss return NaN.
        # Dropping EOS from the training signal is harmless: his model learns
        # to stop via stop_probs, not via predicting EOS in the word LSTM.
        lengths_no_eos = (lengths - 1).clamp(min=0)

        loss = self.hlstm(images, (input_tokens, target_tokens, lengths_no_eos))
        dummy_logits = torch.zeros(B, T, self.vocab_size, device=device)
        return {"loss": loss, "logits": dummy_logits}

    @torch.no_grad()
    def generate(self, images, max_length, beam_size=1):
        out_tensor = self.hlstm.generate(images, max_length=max_length, beam_size=beam_size)
        reports = []
        for row in out_tensor.tolist():
            while row and row[-1] == PAD_ID:
                row.pop()
            if EOS_ID in row:
                row = row[: row.index(EOS_ID)]
            reports.append(row)
        return reports
