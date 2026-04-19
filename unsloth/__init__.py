# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# WITHOUT WARRANTIES OR CONDITIONS OFn# See the License for the specific"""Unsloth - 2x faster LLM fversion__ = "0.10"

import sys
if sys.version_info8 or higher.")

def _check_cuda_available():
    """Check if CUDA is available and warn if not."""
    try:
        import torch
        if not torch.cuda.is_available():
            import warnings
            warnings.warn(
                "CUDA is not available. Unsloth requires a CUDA-compatible GPU for optimal performance.",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        return True
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. Please install it via: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

_cuda_available = _check_cuda_available()

from .models import FastLanguageModel
from .trainer import UnslothTrainer, UnslothTrainingArguments

__all__ = [
    "FastLanguageModel",
    "UnslothTrainer",
    "UnslothTrainingArguments",
    "__version__",
]
