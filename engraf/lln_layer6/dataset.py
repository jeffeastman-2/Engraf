#!/usr/bin/env python3
"""
Layer-6 Dataset for PyTorch training.

Loads JSONL examples and creates batches with proper tokenization.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple


# Structural vocabulary for Layer-6 tokens
STRUCTURAL_TOKENS = [
    '[NP',   # 0
    ']NP',   # 1
    '[PP',   # 2
    ']PP',   # 3
    '[VP',   # 4
    ']VP',   # 5
    '[SP',   # 6
    ']SP',   # 7
    '<SEP>', # 8
    '<BOS>', # 9
    '<EOS>', # 10
    '<PAD>', # 11
]

STRUCTURAL_TOKEN_TO_ID = {tok: i for i, tok in enumerate(STRUCTURAL_TOKENS)}

# Special tokens for text vocabulary
SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']


class Layer6TextTokenizer:
    """Simple word-level tokenizer for natural language targets."""
    
    def __init__(self, vocab=None, max_vocab_size=5000):
        self.vocab = vocab or {}
        self.max_vocab_size = max_vocab_size
        
        # Initialize with special tokens
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[tok] = i
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_freq = {}
        
        for text in texts:
            for word in text.lower().split():
                # Remove punctuation
                word = ''.join(c for c in word if c.isalnum() or c in "'-")
                if word:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        next_idx = len(SPECIAL_TOKENS)
        
        for word, freq in sorted_words:
            if len(self.vocab) >= self.max_vocab_size:
                break
            if word not in self.vocab:
                self.vocab[word] = next_idx
                next_idx += 1
        
        print(f"Text vocabulary built: {len(self.vocab)} tokens")
    
    def encode(self, text: str, max_length=None) -> List[int]:
        """Convert text to token IDs."""
        tokens = []
        
        # Add BOS
        tokens.append(self.vocab.get('<BOS>', 0))
        
        # Tokenize and encode words
        for word in text.lower().split():
            word = ''.join(c for c in word if c.isalnum() or c in "'-")
            if word:
                token_id = self.vocab.get(word, self.vocab.get('<UNK>', 3))
                tokens.append(token_id)
        
        # Add EOS
        tokens.append(self.vocab.get('<EOS>', 0))
        
        # Pad or truncate
        if max_length:
            if len(tokens) < max_length:
                tokens.extend([self.vocab.get('<PAD>', 0)] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        id_to_vocab = {v: k for k, v in self.vocab.items()}
        words = []
        
        for token_id in token_ids:
            word = id_to_vocab.get(token_id, '<UNK>')
            if word not in ['<BOS>', '<EOS>', '<PAD>']:
                words.append(word)
        
        return ' '.join(words)


class Layer6Dataset(Dataset):
    """Dataset for Layer-6 LLM training."""
    
    def __init__(self, 
                 jsonl_path: str,
                 text_tokenizer: Layer6TextTokenizer = None,
                 max_structural_length: int = 100,
                 max_target_length: int = 100):
        """
        Args:
            jsonl_path: Path to JSONL dataset file
            text_tokenizer: Tokenizer for target text (will fit if not provided)
            max_structural_length: Max length for structural token sequences
            max_target_length: Max length for target text
        """
        self.jsonl_path = jsonl_path
        self.max_structural_length = max_structural_length
        self.max_target_length = max_target_length
        
        # Load examples
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        # Initialize or fit text tokenizer
        if text_tokenizer:
            self.text_tokenizer = text_tokenizer
        else:
            self.text_tokenizer = Layer6TextTokenizer()
            target_texts = [ex['target_string'] for ex in self.examples]
            self.text_tokenizer.fit(target_texts)
        
        # Build object ID vocabulary
        self.object_ids = set()
        for ex in self.examples:
            for obj_id in ex['scene_grounding']:
                if obj_id:
                    self.object_ids.add(obj_id)
        
        self.object_id_to_idx = {obj_id: i for i, obj_id in enumerate(sorted(self.object_ids))}
        self.object_id_to_idx[None] = len(self.object_id_to_idx)  # Padding for None
        
        print(f"Dataset loaded: {len(self.examples)} examples")
        print(f"Object vocabulary: {len(self.object_ids)} unique objects")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example.
        
        Returns:
            Dict with keys:
            - structural_tokens: (seq_len,) token IDs
            - semantic_vectors: (seq_len, 76) float32 tensors
            - grounding_ids: (seq_len,) object IDs
            - target_ids: (tgt_len,) token IDs
            - question: original question string
        """
        ex = self.examples[idx]
        
        # Structural tokens
        struct_tokens = ex['structural_tokens']
        struct_ids = [STRUCTURAL_TOKEN_TO_ID.get(tok, STRUCTURAL_TOKEN_TO_ID['<PAD>']) 
                      for tok in struct_tokens]
        
        # Semantic vectors - ensure all are 76-dim
        semantic_vecs = []
        for vec in ex['semantic_vectors']:
            if isinstance(vec, (list, tuple)):
                vec_array = np.array(vec, dtype=np.float32)
                # Pad to 76 dims if necessary
                if len(vec_array) < 76:
                    vec_array = np.pad(vec_array, (0, 76 - len(vec_array)), mode='constant', constant_values=0)
                semantic_vecs.append(vec_array[:76])  # Also truncate if longer
            else:
                # Single value - pad to 76 dims
                semantic_vecs.append(np.array([vec] + [0] * 75, dtype=np.float32))
        
        semantic_vecs = np.array(semantic_vecs, dtype=np.float32)
        
        # Grounding object IDs
        grounding = ex['scene_grounding']
        grounding_ids = [self.object_id_to_idx.get(obj_id, self.object_id_to_idx[None]) 
                         for obj_id in grounding]
        
        # Target text
        target_text = ex['target_string']
        target_ids = self.text_tokenizer.encode(target_text, self.max_target_length)
        
        # Pad/truncate structural sequence
        seq_len = len(struct_ids)
        
        if seq_len < self.max_structural_length:
            pad_len = self.max_structural_length - seq_len
            struct_ids.extend([STRUCTURAL_TOKEN_TO_ID['<PAD>']] * pad_len)
            semantic_vecs = np.vstack([semantic_vecs, np.zeros((pad_len, 76), dtype=np.float32)])
            grounding_ids.extend([self.object_id_to_idx[None]] * pad_len)
        else:
            struct_ids = struct_ids[:self.max_structural_length]
            semantic_vecs = semantic_vecs[:self.max_structural_length]
            grounding_ids = grounding_ids[:self.max_structural_length]
        
        return {
            'structural_tokens': torch.tensor(struct_ids, dtype=torch.long),
            'semantic_vectors': torch.tensor(semantic_vecs, dtype=torch.float32),
            'grounding_ids': torch.tensor(grounding_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'question': ex['question'],
        }


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        'structural_tokens': torch.stack([ex['structural_tokens'] for ex in batch]),
        'semantic_vectors': torch.stack([ex['semantic_vectors'] for ex in batch]),
        'grounding_ids': torch.stack([ex['grounding_ids'] for ex in batch]),
        'target_ids': torch.stack([ex['target_ids'] for ex in batch]),
        'questions': [ex['question'] for ex in batch],
    }


def create_dataloaders(jsonl_path: str,
                      batch_size: int = 4,
                      train_split: float = 0.8,
                      num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        jsonl_path: Path to JSONL dataset
        batch_size: Batch size for training
        train_split: Fraction for training (rest for validation)
        num_workers: Number of workers for data loading
    
    Returns:
        (train_loader, val_loader)
    """
    # Load full dataset once to build tokenizer
    dataset = Layer6Dataset(jsonl_path)
    
    # Split into train/val
    num_train = int(len(dataset) * train_split)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    from torch.utils.data import Subset
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers
    )
    
    print(f"Train: {len(train_set)} examples, Val: {len(val_set)} examples")
    
    return train_loader, val_loader, dataset.text_tokenizer
