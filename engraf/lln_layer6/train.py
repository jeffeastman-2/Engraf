#!/usr/bin/env python3
"""
Layer-6 LLM Training Script

Train the encoder-decoder model on Layer-6 spatial reasoning dataset.
"""

import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
import argparse
from pathlib import Path
import json

from engraf.llm_layer6.model import Layer6LLM
from engraf.llm_layer6.dataset import create_dataloaders


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        structural_tokens = batch['structural_tokens'].to(device)
        semantic_vectors = batch['semantic_vectors'].to(device)
        grounding_ids = batch['grounding_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(
            structural_tokens,
            semantic_vectors,
            target_ids,
            grounding_ids
        )
        
        # Compute loss (ignore padding tokens)
        loss = criterion(
            logits.view(-1, logits.shape[-1]),
            target_ids.view(-1)
        )
        
        # Backward pass
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
            # Move to device
            structural_tokens = batch['structural_tokens'].to(device)
            semantic_vectors = batch['semantic_vectors'].to(device)
            grounding_ids = batch['grounding_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            logits = model(
                structural_tokens,
                semantic_vectors,
                target_ids,
                grounding_ids
            )
            
            # Compute loss
            loss = criterion(
                logits.view(-1, logits.shape[-1]),
                target_ids.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train Layer-6 LLM')
    parser.add_argument('--dataset', type=str, default='layer6_training_data_l5_expanded.jsonl',
                        help='Path to JSONL dataset')
    parser.add_argument('--output_dir', type=str, default='./layer6_checkpoints',
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
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Layer-6 LLM Training")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print()
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, text_tokenizer = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        train_split=0.8
    )
    print()
    
    # Create model
    print("Creating model...")
    model = Layer6LLM(
        text_vocab_size=len(text_tokenizer.vocab),
        structural_vocab_size=12,
        semantic_dim=76,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=text_tokenizer.vocab.get('<PAD>', 0))
    
    if HAS_TENSORBOARD:
        writer = SummaryWriter(output_dir / 'logs')
    else:
        writer = None
        print("TensorBoard not available, skipping logging")
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("Starting training...")
    print("-" * 70)
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'text_tokenizer': text_tokenizer,
        }
        
        # Save every epoch
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch:02d}.pt')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping (patience={patience})")
                break
    
    # Final summary
    print()
    print("-" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print()
    
    # Save config
    config = {
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'text_vocab_size': len(text_tokenizer.vocab),
        'structural_vocab_size': 12,
        'semantic_dim': 76,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
