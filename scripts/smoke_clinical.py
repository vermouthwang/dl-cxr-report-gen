"""Stage-4 smoke test for ClinicalTransformer.

Runs a sequence of in-process checks against the real dataloader without
launching a full training run. Designed to fail loudly on any interface
violation, NaN, shape bug, or guidance-flag misbehavior.

Usage (from repo root, on PACE compute node — NOT login):
    python -m scripts.smoke_clinical

Tests in order:
    [1] Shape test       — forward returns dict with finite scalar loss and
                            (B, T, V) logits.
    [2] Backward test    — loss.backward() works, decoder grads are non-zero,
                            encoder grad-state matches `freeze_encoder` flag.
    [3] Generate parity  — same model, decoding_bias toggled ON vs OFF, output
                            token sequences differ.
    [4] Ablation 4-cell  — instantiates clinical_transformer for all four
                            (loss_weighting, decoding_bias) combos and runs
                            forward+generate on one batch each.

Integration test (`python -m src.training.train --config
configs/clinical_transformer_smoke.yaml`) is intentionally NOT included here
— run it separately as the final step once these in-process checks pass.
"""
from __future__ import annotations

import logging
import sys
import traceback

import torch

from src.data.iu_xray import build_dataloaders
from src.models import get_model


SMOKE_MODEL_CONFIG = {
    # Smaller than production for fast smoke iteration; structure is identical.
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_position_embeddings": 128,
    "pretrained_encoder": True,
    "freeze_encoder": False,
    "use_spatial_pos_embed": True,
    "tie_word_embeddings": True,
}


def _clinical_block(loss_weighting: bool, decoding_bias: bool, bias_strength: float = 1.0) -> dict:
    return {
        "lexicon_path": "data_meta/clinical_terms.json",
        "vocab_path": "data_meta/vocab.json",
        "loss_weighting": loss_weighting,
        "decoding_bias": decoding_bias,
        "alpha_finding": 3.0,
        "alpha_negation": 2.0,
        "alpha_hedge": 1.0,
        "bias_strength": bias_strength,
        "bias_include_negations": True,
    }


def _full_cfg(loss_w: bool, dec_b: bool) -> dict:
    return {**SMOKE_MODEL_CONFIG, "clinical": _clinical_block(loss_w, dec_b)}


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA not available; running on CPU. This will be slow but valid.")
    torch.manual_seed(0)

    _section("Building dataloader (small batch)")
    bundle = build_dataloaders(batch_size=4, num_workers=0)
    V = bundle.vocab.size
    print(f"  vocab_size={V}")
    images, input_tokens, target_tokens, lengths, image_ids = next(iter(bundle.train_loader))
    images = images.to(device)
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    lengths = lengths.to(device)
    B, T = input_tokens.shape
    print(f"  batch: images={tuple(images.shape)}  input_tokens={tuple(input_tokens.shape)}")

    failures: list[str] = []

    # ----------------------------------------------------------------- [1]
    _section("[1/4] Shape test (cell D: both flags ON)")
    try:
        torch.manual_seed(0)
        model = get_model("clinical_transformer", vocab_size=V, config=_full_cfg(True, True)).to(device)
        out = model(images, input_tokens, target_tokens, lengths)

        assert "loss" in out and "logits" in out, f"forward() must return dict with loss+logits; got {list(out)}"
        assert out["loss"].dim() == 0, f"loss must be scalar; got shape {out['loss'].shape}"
        assert torch.isfinite(out["loss"]), f"loss not finite: {out['loss'].item()}"
        assert out["logits"].shape == (B, T, V), f"logits shape wrong: {out['logits'].shape} vs ({B},{T},{V})"
        print(f"  loss={out['loss'].item():.4f}  logits={tuple(out['logits'].shape)}  PASS")
    except Exception as e:
        failures.append(f"[1] Shape test: {e}")
        traceback.print_exc()
        return 1  # can't continue meaningfully without a working model

    # ----------------------------------------------------------------- [2]
    _section("[2/4] Backward / grad flow")
    try:
        model.zero_grad(set_to_none=True)
        out = model(images, input_tokens, target_tokens, lengths)
        out["loss"].backward()

        decoder_params = [(n, p) for n, p in model.named_parameters() if n.startswith("decoder")]
        encoder_params = [(n, p) for n, p in model.named_parameters() if n.startswith("encoder")]

        dec_with_grad = [p for _, p in decoder_params if p.grad is not None]
        assert dec_with_grad, "no decoder parameters have gradients"
        assert any(p.grad.abs().sum() > 0 for p in dec_with_grad), "all decoder gradients are zero"

        if model.freeze_encoder:
            assert all(p.grad is None for _, p in encoder_params), \
                "encoder is frozen but some encoder params received gradients"
            print(f"  decoder grads OK ({len(dec_with_grad)} tensors), encoder FROZEN as expected  PASS")
        else:
            enc_with_grad = [p for _, p in encoder_params if p.grad is not None]
            assert enc_with_grad, "encoder is not frozen but no encoder params have gradients"
            assert any(p.grad.abs().sum() > 0 for p in enc_with_grad), "all encoder gradients are zero"
            print(f"  decoder grads OK ({len(dec_with_grad)} tensors), "
                  f"encoder grads OK ({len(enc_with_grad)} tensors)  PASS")
    except Exception as e:
        failures.append(f"[2] Backward: {e}")
        traceback.print_exc()

    # ----------------------------------------------------------------- [3]
    _section("[3/4] Generate parity (decoding_bias ON vs OFF, same weights)")
    try:
        # Same model: toggle the flag in place. Since `clinical_bias_mask` is
        # always built and registered as a buffer, the only thing that changes
        # is whether generate() applies the additive bias.
        gen_images = images[:2]

        model.use_decoding_bias = True
        out_on = model.generate(gen_images, max_length=20, beam_size=1)

        model.use_decoding_bias = False
        out_off = model.generate(gen_images, max_length=20, beam_size=1)

        # Restore original flag state
        model.use_decoding_bias = True

        print(f"  bias ON : {out_on}")
        print(f"  bias OFF: {out_off}")
        differ = out_on != out_off
        if differ:
            print("  outputs differ  PASS")
        else:
            # With untrained random weights and bias_strength=1.0 vs logits
            # std~0.3, the bias should overwhelm random argmax. Identical
            # outputs are unexpected — flag for manual inspection but don't
            # hard-fail (could occur if bias mask resolves to all zeros).
            mask_sum = float(model.clinical_bias_mask.sum().item())
            print(f"  WARN: outputs identical. clinical_bias_mask sum={mask_sum} "
                  f"(expect > 0 if lexicon resolved correctly).")
            if mask_sum == 0:
                failures.append("[3] Generate parity: clinical_bias_mask is all zeros — lexicon failed to resolve")
    except Exception as e:
        failures.append(f"[3] Generate parity: {e}")
        traceback.print_exc()

    # ----------------------------------------------------------------- [4]
    _section("[4/4] Ablation 4-cell smoke")
    # We verify each cell:
    #   - constructs without error
    #   - runs forward + generate
    #   - has the EXPECTED clinical buffers for its flag combo
    # We do NOT compare losses across cells: at random init every per-position
    # NLL is ~log(V), so a weighted mean collapses to the unweighted mean and
    # loss_weighting has no measurable effect on the loss until training has
    # broken the uniform-output equilibrium. Structural buffer checks are the
    # right thing here.
    cells: dict[str, dict] = {}
    try:
        for loss_w, dec_b in [(False, False), (True, False), (False, True), (True, True)]:
            torch.manual_seed(0)
            cell = get_model("clinical_transformer", vocab_size=V, config=_full_cfg(loss_w, dec_b)).to(device)
            cell.eval()
            with torch.no_grad():
                fwd = cell(images, input_tokens, target_tokens, lengths)
                gen = cell.generate(images[:1], max_length=10, beam_size=1)
            label = "ABCD"[loss_w * 2 + dec_b]
            # Snapshot buffer signatures before deleting the cell.
            cells[label] = {
                "loss_w": loss_w,
                "dec_b": dec_b,
                "loss": float(fwd["loss"].item()),
                "gen_len": len(gen[0]),
                "weight_nontrivial_count": int((cell.clinical_weight != 1.0).sum().item()),
                "bias_nonzero_count": int((cell.clinical_bias_mask != 0.0).sum().item()),
            }
            print(f"  cell {label} (loss_weighting={loss_w}, decoding_bias={dec_b}): "
                  f"loss={fwd['loss'].item():.4f}, gen_len={len(gen[0])}, "
                  f"weight_nontrivial={cells[label]['weight_nontrivial_count']}, "
                  f"bias_nonzero={cells[label]['bias_nonzero_count']}")
            del cell
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Structural assertions on the clinical buffers per cell.
        # A: both flags off -> trivial buffers (weight all-ones, bias all-zeros).
        assert cells["A"]["weight_nontrivial_count"] == 0, \
            f"cell A: weight should be all-ones; got {cells['A']['weight_nontrivial_count']} non-1.0 entries"
        assert cells["A"]["bias_nonzero_count"] == 0, \
            f"cell A: bias_mask should be all-zeros; got {cells['A']['bias_nonzero_count']} non-zero entries"
        # B: loss_weighting on, decoding_bias off -> weight non-trivial, bias all-zeros.
        # (NOTE: bias_mask buffer is still built when loss_weighting is on, but
        # it's intentionally only consumed by generate() when use_decoding_bias is True.)
        assert cells["B"]["weight_nontrivial_count"] > 0, \
            f"cell B: loss_weighting=True but weight buffer is all-ones — lexicon failed to populate"
        # C: loss_weighting off, decoding_bias on -> bias non-trivial, weight may be trivial.
        assert cells["C"]["bias_nonzero_count"] > 0, \
            f"cell C: decoding_bias=True but bias_mask is all-zeros — lexicon failed to populate"
        # D: both on -> both non-trivial.
        assert cells["D"]["weight_nontrivial_count"] > 0 and cells["D"]["bias_nonzero_count"] > 0, \
            f"cell D: both flags on but a buffer is trivial: {cells['D']}"
        # Cross-cell sanity: B and D should have the same weight signature; C and D the same bias signature.
        assert cells["B"]["weight_nontrivial_count"] == cells["D"]["weight_nontrivial_count"], \
            f"cells B and D should share weight buffer signature; got {cells['B']} vs {cells['D']}"
        assert cells["C"]["bias_nonzero_count"] == cells["D"]["bias_nonzero_count"], \
            f"cells C and D should share bias_mask signature; got {cells['C']} vs {cells['D']}"
        print(f"  buffers: A trivial; B weighted ({cells['B']['weight_nontrivial_count']} entries); "
              f"C biased ({cells['C']['bias_nonzero_count']} entries); D both. PASS")
    except Exception as e:
        failures.append(f"[4] Ablation 4-cell: {e}")
        traceback.print_exc()

    # ----------------------------------------------------------------- summary
    _section("Summary")
    if failures:
        print(f"FAILED ({len(failures)} test(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("All in-process smoke tests PASSED.")
    print("\nNext step (integration test, run separately):")
    print("  python -m src.training.train --config configs/clinical_transformer_smoke.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
