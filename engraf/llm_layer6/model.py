#!/usr/bin/env python3
"""
PyTorch Layer-6 LLM Model

Encoder-decoder architecture for Layer-6 spatial reasoning:
- Encoder: Processes Layer-6 structural tokens + semantic vectors
- Decoder: Generates natural language responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from engraf.lexer.vector_space import VECTOR_LENGTH

# Semantic vector dimension (from VECTOR_DIMENSIONS)
SEMANTIC_VECTOR_DIM = VECTOR_LENGTH


class Layer6Encoder(nn.Module):
    """Encodes Layer-6 structural tokens and semantic vectors.
    
    Combines:
    - Structural token embeddings ([NP, ]NP, [VP, ]VP, etc.)
    - Semantic vectors (SEMANTIC_VECTOR_DIM from LATN)
    - Scene grounding references (object_id strings)
    """
    
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
        # We'll use a simple 32-dim embedding for object IDs
        self.grounding_embedding = nn.Embedding(256, 32)  # Max 256 unique objects
        
        # Combine token + semantic + grounding into unified representation
        combined_dim = embedding_dim + embedding_dim + 32  # token + semantic + grounding
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
            (h_n, c_n): LSTM final states for decoder initialization
        """
        batch_size, seq_len = token_ids.shape
        
        # Embed structural tokens
        token_emb = self.token_embedding(token_ids)  # (bs, seq_len, embed_dim)
        
        # Project semantic vectors
        semantic_emb = self.semantic_projection(semantic_vectors)  # (bs, seq_len, embed_dim)
        
        # Embed grounding if available
        if grounding_ids is not None:
            grounding_emb = self.grounding_embedding(grounding_ids)  # (bs, seq_len, 32)
        else:
            grounding_emb = torch.zeros(batch_size, seq_len, 32, device=token_ids.device)
        
        # Fuse all representations
        combined = torch.cat([token_emb, semantic_emb, grounding_emb], dim=-1)  # (bs, seq_len, combined_dim)
        fused = self.fusion_layer(combined)  # (bs, seq_len, hidden_dim)
        fused = F.relu(fused)
        
        # BiLSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(fused)  # lstm_out: (bs, seq_len, hidden_dim*2)
        
        # Project back to hidden_dim
        context = self.output_projection(lstm_out)  # (bs, seq_len, hidden_dim)
        
        return context, (h_n, c_n)


class Layer6Decoder(nn.Module):
    """Decoder for generating natural language from encoded Layer-6 input."""
    
    def __init__(self,
                 vocab_size,
                 embedding_dim=256,
                 hidden_dim=512,
                 num_layers=2,
                 dropout=0.1,
                 max_length=100):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Token embedding for text
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (Bahdanau-style)
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, encoder_context, target_ids, hidden_state=None):
        """Decode to text.
        
        Args:
            encoder_context: (batch_size, encoder_seq_len, hidden_dim) from encoder
            target_ids: (batch_size, target_seq_len) target token IDs (teacher forcing)
            hidden_state: Tuple of (h_n, c_n) from encoder or None
        
        Returns:
            logits: (batch_size, target_seq_len, vocab_size)
        """
        # Embed target tokens
        embedded = self.embedding(target_ids)  # (bs, tgt_len, embed_dim)
        
        # LSTM decode
        lstm_out, hidden_state = self.lstm(embedded, hidden_state)  # (bs, tgt_len, hidden_dim)
        
        # Apply attention over encoder context
        # Simplified: use last encoder state for context
        attn_weights = F.softmax(
            torch.bmm(lstm_out, encoder_context.transpose(1, 2)),
            dim=-1
        )  # (bs, tgt_len, enc_len)
        context_vector = torch.bmm(attn_weights, encoder_context)  # (bs, tgt_len, hidden_dim)
        
        # Combine LSTM output with attention context
        combined = lstm_out + context_vector  # (bs, tgt_len, hidden_dim) - simplified
        
        # Project to vocabulary
        logits = self.output_projection(combined)  # (bs, tgt_len, vocab_size)
        
        return logits


class Layer6LLM(nn.Module):
    """End-to-end Layer-6 LLM: Encoder + Decoder."""
    
    def __init__(self,
                 text_vocab_size,
                 structural_vocab_size=12,
                 semantic_dim=SEMANTIC_VECTOR_DIM,
                 embedding_dim=256,
                 hidden_dim=512,
                 num_layers=2,
                 dropout=0.1,
                 max_length=100):
        super().__init__()
        
        self.encoder = Layer6Encoder(
            structural_vocab_size=structural_vocab_size,
            semantic_dim=semantic_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = Layer6Decoder(
            vocab_size=text_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_length=max_length
        )
        
        self.hidden_dim = hidden_dim
        
    def forward(self, 
                structural_tokens, 
                semantic_vectors,
                target_ids,
                grounding_ids=None):
        """Forward pass: encode Layer-6 input, decode to text.
        
        Args:
            structural_tokens: (batch_size, seq_len)
            semantic_vectors: (batch_size, seq_len, 76)
            target_ids: (batch_size, tgt_len)
            grounding_ids: (batch_size, seq_len) or None
        
        Returns:
            logits: (batch_size, tgt_len, vocab_size)
        """
        # Encode
        encoder_context, (h_n, c_n) = self.encoder(
            structural_tokens, 
            semantic_vectors,
            grounding_ids
        )
        
        # Prepare hidden state for decoder
        # h_n shape: (num_layers * 2, batch_size, hidden_dim) for bidirectional
        # We need: (num_layers, batch_size, hidden_dim) for decoder
        batch_size = structural_tokens.shape[0]
        num_layers = h_n.shape[0] // 2
        
        # Take only forward direction from bidirectional LSTM
        h_n_fwd = h_n[:num_layers]  # (num_layers, batch_size, hidden_dim)
        c_n_fwd = c_n[:num_layers]
        
        # Decode
        logits = self.decoder(encoder_context, target_ids, (h_n_fwd, c_n_fwd))
        
        return logits
