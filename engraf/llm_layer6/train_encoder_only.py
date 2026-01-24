#!/usr/bin/env python3
"""
Training script for Encoder-Only Layer-6 LLM.

Simpler training than encoder-decoder:
- No teacher forcing needed
- Single forward pass per example
- Standard supervised learning
"""

import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
import json

from engraf.llm_layer6.model_encoder_only import Layer6EncoderOnly
from engraf.llm_layer6.dataset import create_dataloaders
from engraf.lexer.vector_space import VECTOR_LENGTH

# Semantic vector dimension (from VECTOR_DIMENSIONS)
SEMANTIC_VECTOR_DIM = VECTOR_LENGTH


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        structural_tokens = batch['structural_tokens'].to(device)
        semantic_vectors = batch['semantic_vectors'].to(device)
        grounding_ids = batch['grounding_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # Forward pass
        logits = model(structural_tokens, semantic_vectors, grounding_ids)
        # logits: (batch_size, max_output_length, vocab_size)
        # target_ids: (batch_size, max_target_length)
        
        # Pad target to match max_output_length if needed
        batch_size, max_output_length, vocab_size = logits.shape
        _, max_target_length = target_ids.shape
        
        if max_target_length < max_output_length:
            # Pad target with PAD tokens
            pad_idx = 11  # <PAD> token id
            padding = torch.full(
                (batch_size, max_output_length - max_target_length),
                pad_idx,
                dtype=target_ids.dtype,
                device=device
            )
            target_ids = torch.cat([target_ids, padding], dim=1).contiguous()
        elif max_target_length > max_output_length:
            # Truncate target
            target_ids = target_ids[:, :max_output_length].contiguous()
        
        # Compute loss
        loss = criterion(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 5 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss = {avg_loss:.4f}")
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """Validate on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            structural_tokens = batch['structural_tokens'].to(device)
            semantic_vectors = batch['semantic_vectors'].to(device)
            grounding_ids = batch['grounding_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            logits = model(structural_tokens, semantic_vectors, grounding_ids)
            
            # Pad/truncate target to match output length
            batch_size, max_output_length, vocab_size = logits.shape
            _, max_target_length = target_ids.shape
            
            if max_target_length < max_output_length:
                pad_idx = 11
                padding = torch.full(
                    (batch_size, max_output_length - max_target_length),
                    pad_idx,
                    dtype=target_ids.dtype,
                    device=device
                )
                target_ids = torch.cat([target_ids, padding], dim=1).contiguous()
            elif max_target_length > max_output_length:
                target_ids = target_ids[:, :max_output_length].contiguous()
            
            loss = criterion(
                logits.view(-1, vocab_size),
                target_ids.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train Encoder-Only Layer-6 LLM')
    parser.add_argument('--dataset', type=str, default='layer6_training_data_expanded.jsonl',
                        help='Path to JSONL dataset')
    parser.add_argument('--output_dir', type=str, default='./layer6_checkpoints_encoder_only',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_output_length', type=int, default=15,
                        help='Maximum output sequence length')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Encoder-Only Layer-6 LLM Training")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"Max output length: {args.max_output_length}")
    print()
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, text_tokenizer = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size
    )
    
    print(f"Train: {len(train_loader.dataset)} examples")
    print(f"Val: {len(val_loader.dataset)} examples")
    print()
    
    # Create model
    print("Creating model...")
    model = Layer6EncoderOnly(
        text_vocab_size=len(text_tokenizer.vocab),
        max_output_length=args.max_output_length,
        structural_vocab_size=12,
        semantic_dim=SEMANTIC_VECTOR_DIM,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=text_tokenizer.vocab.get('<PAD>', 0))
    
    # Training loop
    best_val_loss = float('inf')
    print("Starting training...")
    print("-" * 70)
    print()
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'text_tokenizer': text_tokenizer,
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        print()
    
    print("-" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
