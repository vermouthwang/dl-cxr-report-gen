"""
CPU sanity check for the vanilla Transformer integration. Runs in ~10s.

Exercises:
  1. Model factory dispatch ('transformer' is registered)
  2. forward(): output dict shape, loss has grad_fn
  3. loss.backward() works (gradients flow into decoder)
  4. Encoder really is frozen (its params have requires_grad=False)
  5. generate(): returns list[list[int]] with length <= max_length
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import get_model

VOCAB_SIZE = 975
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2

config = {
    "embed_size": 512,
    "n_heads": 8,
    "n_layers": 2,                  # smaller for fast CPU test
    "dropout": 0.1,
    "pretrained_encoder": False,    # skip downloading weights
    "freeze_encoder": True,
}

print("[1] Instantiating model via factory...")
model = get_model("transformer", vocab_size=VOCAB_SIZE, config=config)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    OK. Total params: {n_total:,}  Trainable: {n_train:,}")

print("[2] Encoder freeze check...")
enc_params = list(model.encoder.parameters())
frozen = sum(1 for p in enc_params if not p.requires_grad)
print(f"    {frozen}/{len(enc_params)} encoder params frozen.")
assert frozen == len(enc_params), "Encoder is not fully frozen!"

# Dummy batch
B, T = 2, 10
images = torch.randn(B, 3, 224, 224)
input_tokens = torch.tensor([
    [BOS_ID, 50, 51, 52, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
    [BOS_ID, 60, 61, 62, 63, 64, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
], dtype=torch.long)
target_tokens = torch.tensor([
    [50, 51, 52, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
    [60, 61, 62, 63, 64, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
], dtype=torch.long)
lengths = torch.tensor([4, 6], dtype=torch.long)

print("[3] Forward...")
out = model(images, input_tokens, target_tokens, lengths)
assert isinstance(out, dict) and {"loss", "logits"} <= out.keys(), f"Bad output: {out}"
assert out["loss"].dim() == 0, f"loss not scalar: {out['loss'].shape}"
assert out["logits"].shape == (B, T, VOCAB_SIZE), f"logits shape: {out['logits'].shape}"
assert out["loss"].grad_fn is not None, "loss has no grad_fn"
print(f"    loss={out['loss'].item():.4f}  logits.shape={tuple(out['logits'].shape)}")

print("[4] Backward...")
out["loss"].backward()
dec_total = sum(1 for _ in model.transformer.parameters())
dec_grad = sum(1 for p in model.transformer.parameters() if p.grad is not None)
print(f"    Decoder params with grad: {dec_grad}/{dec_total}")
assert dec_grad == dec_total, "decoder params missing grad"
enc_grad = sum(1 for p in model.encoder.parameters() if p.grad is not None)
print(f"    Encoder params with grad: {enc_grad} (should be 0 — frozen)")
assert enc_grad == 0, "encoder is frozen but got grads"

print("[5] Generate (greedy)...")
model.eval()
with torch.no_grad():
    reports = model.generate(images, max_length=20)
assert isinstance(reports, list) and len(reports) == B
for i, r in enumerate(reports):
    assert isinstance(r, list) and all(isinstance(t, int) for t in r)
    assert len(r) <= 20
    print(f"    Report {i}: {r}")

print("\nAll sanity checks passed.")
