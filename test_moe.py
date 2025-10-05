#!/usr/bin/env python3
"""Test script to verify MoE implementation."""

import torch
import sys
import os

# Add moe to path
sys.path.insert(0, os.path.dirname(__file__))

from moe.data.dataset import CharDataset
from moe.models.transformer import MoETransformer

def test_dataset():
    """Test dataset loading."""
    print("Testing dataset...")
    dataset = CharDataset('moe/data/tiny_shakespeare.txt', seq_len=128, split='train')
    print(f"  Vocab size: {dataset.vocab_size}")
    print(f"  Dataset length: {len(dataset)}")
    
    # Test a sample
    x, y = dataset[0]
    assert x.shape == (128,)
    assert y.shape == (128,)
    print(f"  Sample shape: x={x.shape}, y={y.shape}")
    print("  ✓ Dataset test passed")
    return dataset.vocab_size

def test_model(vocab_size):
    """Test model creation and forward pass."""
    print("\nTesting model...")
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"  Using device: {device}")
    
    # Create small model
    model = MoETransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        n_experts=4,
        topk=2,
        d_hidden_mult=2.0,
        max_seq_len=256,
        dropout=0.0,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Forward with targets
    ce_loss, logits, aux_losses = model(x, y)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  CE loss: {ce_loss.item():.4f}")
    print(f"  LB loss: {aux_losses['lb_loss'].item():.4f}")
    print(f"  Z loss: {aux_losses['z_loss'].item():.4f}")
    print(f"  Entropy: {aux_losses['entropy'].item():.4f}")
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("  ✓ Model test passed")
    
    # Test generation
    print("\nTesting generation...")
    context = torch.randint(0, vocab_size, (1, 10), device=device)
    generated = model.generate(context, max_new_tokens=50, temperature=1.0)
    print(f"  Generated shape: {generated.shape}")
    assert generated.shape == (1, 60)
    print("  ✓ Generation test passed")
    
    return True

def main():
    print("="*60)
    print("MoE Transformer Implementation Test")
    print("="*60)
    
    vocab_size = test_dataset()
    test_model(vocab_size)
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)

if __name__ == '__main__':
    main()
