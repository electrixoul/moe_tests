import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import json
import math
import time
from pathlib import Path

from data.dataset import CharDataset
from models.transformer import MoETransformer


def get_lr(step, warmup_steps, max_lr, max_steps):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return 0.0
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * coeff


def train_epoch(model, dataloader, optimizer, device, args, epoch, total_steps):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_lb_loss = 0
    total_z_loss = 0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Update learning rate
        lr = get_lr(total_steps, args.warmup_steps, args.lr, args.max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        ce_loss, logits, aux_losses = model(x, y)
        
        # Total loss (式 3.8)
        loss = (ce_loss + 
                args.lb_weight * aux_losses['lb_loss'] + 
                args.zloss_weight * aux_losses['z_loss'])
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_lb_loss += aux_losses['lb_loss'].item()
        total_z_loss += aux_losses['z_loss'].item()
        num_batches += 1
        total_steps += 1
        
        # Print progress (without tqdm)
        if batch_idx % args.log_interval == 0:
            ppl = math.exp(ce_loss.item())
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Step {total_steps} | LR {lr:.6f} | "
                  f"Loss {loss.item():.4f} | CE {ce_loss.item():.4f} | "
                  f"LB {aux_losses['lb_loss'].item():.4f} | "
                  f"Z {aux_losses['z_loss'].item():.4f} | "
                  f"PPL {ppl:.2f}")
        
        if total_steps >= args.max_steps:
            break
    
    avg_loss = total_loss / num_batches
    avg_ce = total_ce_loss / num_batches
    avg_lb = total_lb_loss / num_batches
    avg_z = total_z_loss / num_batches
    
    return avg_loss, avg_ce, avg_lb, avg_z, total_steps


@torch.no_grad()
def evaluate(model, dataloader, device, args):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    num_batches = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        ce_loss, logits, aux_losses = model(x, y)
        loss = (ce_loss + 
                args.lb_weight * aux_losses['lb_loss'] + 
                args.zloss_weight * aux_losses['z_loss'])
        
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_ce = total_ce_loss / num_batches
    ppl = math.exp(avg_ce)
    
    return avg_loss, avg_ce, ppl


@torch.no_grad()
def generate_sample(model, dataset, device, max_new_tokens=200):
    """Generate a sample text."""
    model.eval()
    
    # Start with a simple prompt
    prompt = "\n"
    context = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
    
    # Generate
    generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    text = dataset.decode(generated[0].cpu().tolist())
    
    return text


def main():
    parser = argparse.ArgumentParser(description='Train MoE Transformer')
    
    # Data
    parser.add_argument('--dataset', type=str, default='tiny_shakespeare',
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='moe/data',
                        help='Data directory')
    parser.add_argument('--seq_len', type=int, default=256,
                        help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    
    # Model
    parser.add_argument('--model_dim', type=int, default=384,
                        help='Model dimension')
    parser.add_argument('--n_head', type=int, default=6,
                        help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--moe_experts', type=int, default=8,
                        help='Number of experts')
    parser.add_argument('--topk', type=int, default=2,
                        help='Top-k routing')
    parser.add_argument('--moe_hidden_mult', type=float, default=2.0,
                        help='MoE hidden dimension multiplier')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--use_noisy_gating', action='store_true',
                        help='Use noisy gating')
    
    # Training
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Max learning rate')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Warmup steps')
    parser.add_argument('--max_steps', type=int, default=20000,
                        help='Max training steps')
    parser.add_argument('--wd', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--lb_weight', type=float, default=0.05,
                        help='Load balancing loss weight')
    parser.add_argument('--zloss_weight', type=float, default=0.001,
                        help='Z-loss weight')
    
    # Logging
    parser.add_argument('--save_dir', type=str, default='runs/moe_test',
                        help='Save directory')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=2000,
                        help='Save checkpoint interval')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/mps/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load datasets
    data_path = os.path.join(args.data_dir, f'{args.dataset}.txt')
    train_dataset = CharDataset(data_path, seq_len=args.seq_len, split='train')
    val_dataset = CharDataset(data_path, seq_len=args.seq_len, split='val')
    
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create model
    model = MoETransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=args.model_dim,
        n_heads=args.n_head,
        n_layers=args.n_layer,
        n_experts=args.moe_experts,
        topk=args.topk,
        d_hidden_mult=args.moe_hidden_mult,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        use_noisy_gating=args.use_noisy_gating,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.wd,
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    total_steps = 0
    
    for epoch in range(1000):  # Large number, will break based on max_steps
        epoch_start = time.time()
        
        # Train
        train_loss, train_ce, train_lb, train_z, total_steps = train_epoch(
            model, train_loader, optimizer, device, args, epoch, total_steps
        )
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | CE: {train_ce:.4f} | "
              f"LB: {train_lb:.4f} | Z: {train_z:.4f} | PPL: {math.exp(train_ce):.2f}")
        
        # Evaluate
        val_loss, val_ce, val_ppl = evaluate(model, val_loader, device, args)
        print(f"Val Loss: {val_loss:.4f} | CE: {val_ce:.4f} | PPL: {val_ppl:.2f}")
        
        # Generate sample
        sample = generate_sample(model, train_dataset, device)
        print(f"\n--- Generated Sample ---\n{sample}\n--- End Sample ---\n")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'step': total_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'config': vars(args),
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if total_steps % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'step': total_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'config': vars(args),
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_step_{total_steps}.pt'))
        
        # Check if done
        if total_steps >= args.max_steps:
            print(f"\nReached max steps ({args.max_steps}). Training complete!")
            break
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
