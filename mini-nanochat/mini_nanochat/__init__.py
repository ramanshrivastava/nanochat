"""
Mini-Nanochat: Learning-focused LLM implementation

A reasoning-based, historically-grounded implementation of a minimal
LLM training system, built to deeply understand transformer training.
"""

__version__ = "0.2.0"
__author__ = "Learning Project"

from .tokenizer import Tokenizer
from .bpe import BPETrainer, train_bpe_from_file

__all__ = ["Tokenizer", "BPETrainer", "train_bpe_from_file"]
