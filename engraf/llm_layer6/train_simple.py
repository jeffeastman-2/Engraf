#!/usr/bin/env python3
"""
Training script for Simplified Encoder-Only Layer-6 LLM.

Key difference from previous version:
- Much smaller model (fewer parameters)
- Can actually learn from 43 examples
- Focuses on structural tokens, ignores semantic vectors
"""

import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
import json

from engraf.llm_layer6.model_simple import Layer6EncoderOnlySimple
from engraf.llm_layer6.dataset import create_dataloaders


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
                pad_idx = 11  # <PAD>
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
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main(args):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Load data
    print(f"Loading dataset from {args.dataset}...")
    train_loader, val_loader, text_tokenizer = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size
    )
    print()
    
    # Create model
    print("Creating simplified encoder-only model...")
    model = Layer6EncoderOnlySimple(
        text_vocab_size=len(text_tokenizer.vocab),
        max_output_length=args.max_output_length,
        structural_vocab_size=12,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    )
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting training...")
    print()
    
    best_val_loss = float('inf')
    best_epoch = 0
    checkpoint_dir = Path('layer6_checkpoints_simple')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
        
        status = "✓ Best" if is_best else ""
        print(f"Epoch {epoch}/{args.num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f} {status}")
        
        if is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'text_tokenizer': text_tokenizer,
                'model_config': {
                    'text_vocab_size': len(text_tokenizer.vocab),
                    'max_output_length': args.max_output_length,
                    'embedding_dim': 128,
                    'hidden_dim': 256,
                    'num_layers': 2,
                    'dropout': 0.2
                }
            }
            
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        print()
    
    print("=" * 70)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train simplified encoder-only Layer-6 LLM'
    )
    parser.add_argument(
        '--dataset',
        default='layer6_training_data_expanded.jsonl',
        help='Path to training dataset (JSONL)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--max_output_length',
        type=int,
        default=15,
        help='Maximum output sequence length'
    )
    
    args = parser.parse_args()
    main(args)
