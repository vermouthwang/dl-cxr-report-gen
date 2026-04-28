"""
Unified interface for all models. To add a new model, define it in a new file (e.g., src/models/LSTM.py) and add it to the _MODEL_REGISTRY below.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.clinical_transformer import ClinicalTransformer
from src.models.dummy import DummyModel
from src.models.hierarchical_lstm_adapter import HierarchicalLSTMAdapter

# Registry of model_name (as used in configs) -> class
_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "dummy": DummyModel,
    "hierarchical_lstm": HierarchicalLSTMAdapter,   # Yousuf
    # "transformer": VanillaTransformer,            # Murad
    "clinical_transformer": ClinicalTransformer,    # Yinghou
}


def get_model(name: str, vocab_size: int, config: dict) -> nn.Module:
    """
    Args:
        name: one of the keys in _MODEL_REGISTRY
        vocab_size: number of tokens in the vocabulary (including specials)
        config: model-specific config dict (e.g., hidden_dim, embed_size, ...)

    Raises:
        KeyError: if `name` isn't registered. Lists available models in the message.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model '{name}'. Available models: [{available}]. "
            f"To register a new model, edit src/models/__init__.py."
        )
    cls = _MODEL_REGISTRY[name]
    return cls(vocab_size=vocab_size, config=config)


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())