#!/usr/bin/env python3
"""
Encoder-Only Layer-6 LLM Model

Template-based generation with fixed-position token prediction.
Suitable for deterministic spatial reasoning with fill-in-the-blanks patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from engraf.lexer.vector_space import VECTOR_LENGTH

# Semantic vector dimension (from VECTOR_DIMENSIONS)
SEMANTIC_VECTOR_DIM = VECTOR_LENGTH


class Layer6Encoder(nn.Module):
    """Encodes Layer-6 structural tokens and semantic vectors."""
    
    def __init__(self, 
                 structural_vocab_size=12,
                 semantic_dim=SEMANTIC_VECTOR_DIM,
                 embedding_dim=256,
                 hidden_dim=512,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()
        
        self.structural_vocab_size = structural_vocab_size
        self.semantic_dim = semantic_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token embedding for structural markers
        self.token_embedding = nn.Embedding(structural_vocab_size, embedding_dim)
        
        # Project semantic vectors to embedding space
        self.semantic_projection = nn.Linear(semantic_dim, embedding_dim)
        
        # Grounding embedding (for scene object references)
        self.grounding_embedding = nn.Embedding(256, 32)
        
        # Combine token + semantic + grounding into unified representation
        combined_dim = embedding_dim + embedding_dim + 32
        self.fusion_layer = nn.Linear(combined_dim, hidden_dim)
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, token_ids, semantic_vectors, grounding_ids=None):
        """Encode Layer-6 input.
        
        Args:
            token_ids: (batch_size, seq_len) structural token indices
            semantic_vectors: (batch_size, seq_len, 76) semantic vectors from LATN
            grounding_ids: (batch_size, seq_len) scene object indices or None
        
        Returns:
            context: (batch_size, seq_len, hidden_dim) encoded representations
        """
        batch_size, seq_len = token_ids.shape
        
        # Embed structural tokens
        token_emb = self.token_embedding(token_ids)
        
        # Project semantic vectors
        semantic_emb = self.semantic_projection(semantic_vectors)
        
        # Embed grounding if available
        if grounding_ids is not None:
            grounding_emb = self.grounding_embedding(grounding_ids)
        else:
            grounding_emb = torch.zeros(batch_size, seq_len, 32, device=token_ids.device)
        
        # Fuse all representations
        combined = torch.cat([token_emb, semantic_emb, grounding_emb], dim=-1)
        fused = self.fusion_layer(combined)
        fused = F.relu(fused)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(fused)
        
        # Project back to hidden_dim
        context = self.output_projection(lstm_out)
        
        return context


class Layer6EncoderOnly(nn.Module):
    """Encoder-only model for deterministic spatial reasoning.
    
    Uses template-based generation with fixed-position predictions.
    No autoregressive decoding - all output positions predicted in parallel.
    """
    
    def __init__(self,
                 text_vocab_size,
                 max_output_length=15,
                 structural_vocab_size=12,
                 semantic_dim=SEMANTIC_VECTOR_DIM,
                 embedding_dim=256,
                 hidden_dim=512,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()
        
        self.text_vocab_size = text_vocab_size
        self.max_output_length = max_output_length
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = Layer6Encoder(
            structural_vocab_size=structural_vocab_size,
            semantic_dim=semantic_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Pooling: take mean of all encoded positions
        self.pooling = 'mean'  # or 'last'
        
        # Output heads: one per position
        # Each head predicts token at that position
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, text_vocab_size)
            )
            for _ in range(max_output_length)
        ])
        
    def forward(self, structural_tokens, semantic_vectors, grounding_ids=None):
        """Forward pass: encode and predict all output positions.
        
        Args:
            structural_tokens: (batch_size, seq_len)
            semantic_vectors: (batch_size, seq_len, 76)
            grounding_ids: (batch_size, seq_len) or None
        
        Returns:
            logits: (batch_size, max_output_length, vocab_size)
        """
        # Encode
        context = self.encoder(structural_tokens, semantic_vectors, grounding_ids)
        # context: (batch_size, input_seq_len, hidden_dim)
        
        # Pool the context
        if self.pooling == 'mean':
            pooled = context.mean(dim=1)  # (batch_size, hidden_dim)
        else:  # 'last'
            pooled = context[:, -1, :]    # (batch_size, hidden_dim)
        
        # Predict all output positions in parallel
        batch_size = pooled.shape[0]
        all_logits = []
        
        for head in self.output_heads:
            logits = head(pooled)  # (batch_size, vocab_size)
            all_logits.append(logits)
        
        # Stack: (batch_size, max_output_length, vocab_size)
        output_logits = torch.stack(all_logits, dim=1)
        
        return output_logits


class TemplateDecoder:
    """Decodes logits to text using template patterns.
    
    Handles common spatial reasoning patterns like:
    - "Moving the {obj1} {prep} the {obj2}."
    - "The {obj1} is {prep} the {obj2}."
    - "Yes, the {obj1} is {prep} the {obj2}."
    """
    
    def __init__(self, id_to_token, token_to_id):
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id
    
    def decode(self, logits):
        """Decode logits to text.
        
        Args:
            logits: (batch_size, max_length, vocab_size) or (max_length, vocab_size)
        
        Returns:
            List of decoded sentences
        """
        if logits.dim() == 3:
            # Batch mode
            batch_size = logits.shape[0]
            results = []
            for i in range(batch_size):
                results.append(self._decode_single(logits[i]))
            return results
        else:
            # Single example
            return self._decode_single(logits)
    
    def _decode_single(self, logits):
        """Decode a single example.
        
        Args:
            logits: (max_length, vocab_size)
        
        Returns:
            str: Decoded text
        """
        tokens = []
        
        for step_logits in logits:
            # Get most likely token
            token_id = torch.argmax(step_logits).item()
            word = self.id_to_token.get(token_id, '<UNK>')
            tokens.append(word)
            
            # Stop at EOS or PAD
            if word in ['<EOS>', '<PAD>']:
                break
        
        # Remove special tokens and join
        text_tokens = [t for t in tokens if t not in ['<BOS>', '<EOS>', '<PAD>']]
        return ' '.join(text_tokens)
