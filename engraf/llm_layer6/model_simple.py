#!/usr/bin/env python3
"""
Simplified Encoder-Only Model for Layer-6 LLM.

Key insight: With only 43 training examples, we need a smaller model
that can actually fit the data. Use structural tokens as primary input,
semantic vectors as auxiliary features.

Simplified architecture:
- Structural token embedding (12 vocab, 64-dim)
- Direct LSTM encoding of tokens
- Simple position-based output heads (no semantic fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from engraf.lexer.vector_space import VECTOR_LENGTH

# Semantic vector dimension (from VECTOR_DIMENSIONS)
SEMANTIC_VECTOR_DIM = VECTOR_LENGTH


class Layer6EncoderSimple(nn.Module):
    """Simplified encoder for small datasets."""
    
    def __init__(self, 
                 structural_vocab_size=12,
                 embedding_dim=128,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.2):
        super().__init__()
        
        self.structural_vocab_size = structural_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token embedding only (lightweight)
        self.token_embedding = nn.Embedding(structural_vocab_size, embedding_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, token_ids):
        """
        Args:
            token_ids: (batch_size, seq_len) structural token indices
        
        Returns:
            context: (batch_size, seq_len, 2*hidden_dim) encoded representations
        """
        # Embed structural tokens
        token_emb = self.token_embedding(token_ids)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(token_emb)
        
        return lstm_out


class Layer6EncoderOnlySimple(nn.Module):
    """Simplified encoder-only model for small datasets.
    
    Much fewer parameters than the full model:
    - No semantic vector fusion
    - Smaller embedding dimensions
    - Simpler architecture overall
    
    Trade-off: Less semantic understanding, but can actually learn from 43 examples.
    """
    
    def __init__(self,
                 text_vocab_size,
                 max_output_length=15,
                 structural_vocab_size=12,
                 embedding_dim=128,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.2):
        super().__init__()
        
        self.text_vocab_size = text_vocab_size
        self.max_output_length = max_output_length
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = Layer6EncoderSimple(
            structural_vocab_size=structural_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Global pooling -> dense layers
        encoder_output_dim = hidden_dim * 2  # bidirectional
        self.pooling_dense = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Position-specific output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, text_vocab_size)
            for _ in range(max_output_length)
        ])
        
    def forward(self, token_ids, semantic_vectors=None, grounding_ids=None):
        """Forward pass.
        
        Args:
            token_ids: (batch_size, seq_len)
            semantic_vectors: (batch_size, seq_len, sem_dim) - IGNORED for simplicity
            grounding_ids: (batch_size, seq_len) - IGNORED for simplicity
        
        Returns:
            logits: (batch_size, max_output_length, text_vocab_size)
        """
        # Encode
        context = self.encoder(token_ids)
        # context: (batch_size, seq_len, hidden_dim * 2)
        
        # Pool over sequence dimension
        pooled = context.mean(dim=1)
        # pooled: (batch_size, hidden_dim * 2)
        
        # Dense layer
        dense_output = self.pooling_dense(pooled)
        # dense_output: (batch_size, hidden_dim)
        
        # Apply position-specific heads
        logits_list = [head(dense_output) for head in self.output_heads]
        logits = torch.stack(logits_list, dim=1)
        # logits: (batch_size, max_output_length, text_vocab_size)
        
        return logits


class TemplateDecoder:
    """Decodes logits to text."""
    
    def __init__(self, id_to_token, token_to_id):
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id
        
    def decode(self, logits):
        """Decode logits to text.
        
        Args:
            logits: (max_output_length, vocab_size) tensor
        
        Returns:
            text: decoded string
        """
        # Greedy decoding
        token_ids = logits.argmax(dim=-1).cpu().numpy()
        
        tokens = []
        for tid in token_ids:
            tid = int(tid)
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                if token == '<EOS>':
                    break
                if token not in ['<BOS>', '<PAD>']:
                    tokens.append(token)
        
        return ' '.join(tokens)


if __name__ == '__main__':
    # Test
    model = Layer6EncoderOnlySimple(
        text_vocab_size=27,
        max_output_length=15
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    batch_token_ids = torch.randint(0, 12, (4, 20))
    batch_sem_vecs = torch.randn(4, 20, SEMANTIC_VECTOR_DIM)
    batch_ground_ids = torch.randint(0, 10, (4, 20))
    
    logits = model(batch_token_ids, batch_sem_vecs, batch_ground_ids)
    print(f"Output logits shape: {logits.shape}")
    print("Expected: (4, 15, 27)")
