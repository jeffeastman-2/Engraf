"""Layer-6 LLM integration package.

This package contains utilities to extract training/eval pairs from
LATN Layer-6 hypotheses and a small adapter skeleton for encoding the
structural tokens + LATN vectors into embeddings for an LLM.

Modules:
- dataset_extractor: helpers to build JSONL training examples
- adapter: encoder/adapter skeleton to project LATN semantic vectors
- dataset: PyTorch datasets for training (file-based and on-the-fly)
- synthetic_generator: comprehensive synthetic data generation

"""

__all__ = ["dataset_extractor", "adapter", "dataset", "synthetic_generator"]
