# LoRA patching utilities for unsloth
# Applies PEFT LoRA adapters to a loaded model

from unsloth.models.config import UnslothConfig


def apply_lora(model, config: UnslothConfig):
    """
    Apply LoRA adapters to the model using PEFT.

    Args:
        model: A loaded HuggingFace model.
        config: UnslothConfig with LoRA settings.

    Returns:
        PEFT model with LoRA adapters applied.
    """
    config.validate()

    try:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError("peft is required. Install with: pip install peft") from e

    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    return model


def print_trainable_parameters(model) -> str:
    """
    Print and return a summary of trainable vs total parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0
    summary = (
        f"Trainable params: {trainable:,} | "
        f"Total params: {total:,} | "
        f"Trainable %: {pct:.4f}"
    )
    print(summary)
    return summary
