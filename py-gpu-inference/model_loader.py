#!/usr/bin/env python3
"""
Model loading and mean pooling utilities.

Handles loading transformer models (including contrastive checkpoints with
"encoder." prefix in state dict keys) and provides mean pooling with
control-token exclusion.
"""

import logging
from typing import List, Optional

import torch
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


def _get_attn_implementation() -> str:
    """Prefer Flash Attention 2 over SDPA when available (requires Ampere+ GPU)."""
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:
                logger.info(f"GPU SM {capability[0]}.{capability[1]} < 8.0, using SDPA")
                return "sdpa"
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_model(model_path: str, force_dtype=None) -> AutoModel:
    """
    Load a transformer model, automatically handling contrastive checkpoints
    that store weights with an "encoder." prefix in state dict keys.

    Args:
        model_path: Local path to checkpoint directory.
        force_dtype: Optional torch dtype to force (e.g. torch.bfloat16).

    Returns:
        Loaded AutoModel instance.
    """
    from pathlib import Path

    attn_impl = _get_attn_implementation()
    logger.info(f"Using {attn_impl} attention implementation")

    model_path_obj = Path(model_path)
    safetensors_path = model_path_obj / "model.safetensors"

    has_encoder_prefix = False
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            has_encoder_prefix = any(k.startswith("encoder.") for k in state_dict.keys())
        except Exception as e:
            logger.warning(f"Could not check checkpoint format: {e}, loading normally")

    if has_encoder_prefix:
        logger.info("Detected contrastive checkpoint (encoder. prefix in keys)")
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))

        new_state_dict = {
            (k[len("encoder."):] if k.startswith("encoder.") else k): v
            for k, v in state_dict.items()
        }

        config = AutoConfig.from_pretrained(model_path)
        torch_dtype = force_dtype or getattr(config, "torch_dtype", None)

        model = AutoModel.from_config(
            config,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            **({"torch_dtype": torch_dtype} if torch_dtype else {}),
        )

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if not missing and not unexpected:
            logger.info("All weights loaded successfully from contrastive checkpoint")

        return model
    else:
        logger.info("Standard checkpoint format (no encoder. prefix)")
        model = AutoModel.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        return model


def mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    exclude_tokens: Optional[List[str]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Mean pooling over last hidden state, excluding CLS/SEP/PAD and optional
    control tokens.

    Args:
        last_hidden_state: (batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, seq_len)
        input_ids: (batch_size, seq_len)
        tokenizer: HuggingFace tokenizer instance.
        exclude_tokens: Token strings to exclude (e.g. ['[product]', '[query]']).
        normalize: L2-normalize output embeddings.

    Returns:
        Pooled embeddings (batch_size, hidden_size).
    """
    pooling_mask = attention_mask.clone()

    # Exclude CLS (position 0)
    pooling_mask[:, 0] = 0

    # Exclude SEP
    sep_token_id = tokenizer.sep_token_id
    if sep_token_id is not None:
        pooling_mask = pooling_mask * (input_ids != sep_token_id).long()

    # Exclude control tokens
    if exclude_tokens:
        for token in exclude_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                pooling_mask = pooling_mask * (input_ids != token_id).long()

    input_mask_expanded = pooling_mask.unsqueeze(-1).type_as(last_hidden_state)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    mean_pooled = sum_embeddings / sum_mask

    if normalize:
        if mean_pooled.dtype == torch.bfloat16:
            mean_pooled = mean_pooled.float()
        mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

    return mean_pooled
