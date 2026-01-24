# Layer-6 LLM Integration (engraf/llm_layer6)

This package provides Layer-6 LLM integration for the LATN system,
enabling training and inference of small language models for spatial reasoning.

## Files

### Core Modules
- `dataset_extractor.py` — convert Layer-6 hypotheses into JSONL training pairs
- `adapter.py` — skeleton encoder that projects LATN semantic vectors and structural tokens
- `dataset.py` — PyTorch Dataset classes (file-based and on-the-fly generation)
- `synthetic_generator.py` — comprehensive synthetic training data generator

### Models
- `model.py` — full encoder-decoder Layer-6 LLM
- `model_encoder_only.py` — encoder-only with fixed output positions  
- `model_simple.py` — simplified encoder for smaller datasets

### Training Scripts
- `train.py` — train encoder-decoder model
- `train_encoder_only.py` — train encoder-only model
- `train_simple.py` — train simplified model (supports on-the-fly generation)

## Usage

### On-the-fly Training (Recommended)
```bash
# Train with 100k examples generated on-the-fly per epoch
python train_simple.py --num_train_examples 100000 --num_epochs 10
```

### File-based Training
```bash
# Generate dataset file first (if needed)
python synthetic_generator.py

# Train from file
python train_simple.py --dataset layer6_training_data.jsonl
```

## Vector Dimensions

Semantic vectors use `VECTOR_LENGTH` from `engraf.lexer.vector_space`, 
which is dynamically computed from `VECTOR_DIMENSIONS`. This ensures that
if new dimensions are added, the models automatically adjust.
