"""Adapter / encoder skeleton for Layer-6 inputs.

This file provides a minimal `Layer6Encoder` skeleton that projects the
LATN semantic vectors and optional scene properties into a shared embedding
space and combines them with a small structural token embedding.

The implementation is intentionally lightweight and dependency-free by
default; a PyTorch implementation example is included but commented
behind an import guard so the module can be imported where torch is
unavailable (e.g., static analysis, tests).
"""
from typing import Optional, Sequence

from engraf.lexer.vector_space import VECTOR_LENGTH

# Semantic vector dimension (from VECTOR_DIMENSIONS)
SEMANTIC_VECTOR_DIM = VECTOR_LENGTH

STRUCTURAL_VOCAB = ["[NP", "]NP", "[PP", "]PP", "[VP", "]VP", "[SP", "]SP", "<SEP>", "<BOS>", "<EOS>", "<OBJ>"]


def token_to_id(token: str) -> int:
    try:
        return STRUCTURAL_VOCAB.index(token)
    except ValueError:
        return -1


try:
    import torch
    import torch.nn as nn

    class Layer6Encoder(nn.Module):
        """PyTorch encoder projecting LATN vectors + structural tokens.

        Args:
            vocab_size: number of structural tokens (default 12)
            latn_dim: LATN vector dimension (default SEMANTIC_VECTOR_DIM)
            embed_dim: resulting embedding dim (default 256)
        """
        def __init__(self, vocab_size: int = None, latn_dim: int = SEMANTIC_VECTOR_DIM, embed_dim: int = 256):
            super().__init__()
            self.vocab_size = vocab_size or len(STRUCTURAL_VOCAB)
            self.latn_dim = latn_dim
            self.embed_dim = embed_dim

            self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
            self.latn_projection = nn.Linear(latn_dim, embed_dim)
            self.scene_projection = nn.Linear(9, embed_dim)

        def forward(self, token_ids: torch.LongTensor, latn_vectors: torch.FloatTensor, scene_props: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
            """Return embeddings shaped [batch, seq_len, embed_dim]."""
            struct_emb = self.token_embedding(token_ids)
            semantic_emb = self.latn_projection(latn_vectors)
            emb = struct_emb + semantic_emb
            if scene_props is not None:
                emb = emb + self.scene_projection(scene_props)
            return emb

except Exception:
    # Torch not installed — provide a lightweight fallback
    Layer6Encoder = None
