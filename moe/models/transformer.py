import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .moe_layer import MoELayer


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d]
            mask: Optional causal mask
        Returns:
            [B, T, d]
        """
        B, T, d = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, T, 3*d]
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, T, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, n_heads, T, T]
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum of values
        out = attn @ v  # [B, n_heads, T, d_head]
        out = out.transpose(1, 2).contiguous().reshape(B, T, d)  # [B, T, d]
        
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with MoE-FFN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int = 8,
        topk: int = 2,
        d_hidden_mult: float = 2.0,
        dropout: float = 0.0,
        use_noisy_gating: bool = False,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(
            d_model=d_model,
            n_experts=n_experts,
            topk=topk,
            d_hidden_mult=d_hidden_mult,
            dropout=dropout,
            use_noisy_gating=use_noisy_gating,
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: [B, T, d]
            mask: Causal mask
        Returns:
            x: [B, T, d]
            moe_aux: MoE auxiliary losses
        """
        # Self-attention with residual
        x = x + self.attn(self.ln1(x), mask)
        
        # MoE-FFN with residual
        moe_out, moe_aux = self.moe(self.ln2(x))
        x = x + moe_out
        
        return x, moe_aux


class MoETransformer(nn.Module):
    """MoE Transformer for character-level language modeling."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 4,
        n_experts: int = 8,
        topk: int = 2,
        d_hidden_mult: float = 2.0,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        use_noisy_gating: bool = False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_experts=n_experts,
                topk=topk,
                d_hidden_mult=d_hidden_mult,
                dropout=dropout,
                use_noisy_gating=use_noisy_gating,
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (share embeddings with output projection)
        self.head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            x: Input token indices [B, T]
            targets: Target token indices [B, T] (optional)
        Returns:
            If targets provided: (loss, logits, aux_losses)
            Otherwise: logits
        """
        B, T = x.shape
        device = x.device
        
        # Token + positional embeddings
        tok_emb = self.token_emb(x)  # [B, T, d]
        pos_emb = self.pos_emb[:, :T, :]  # [1, T, d]
        x = tok_emb + pos_emb
        
        # Causal mask for attention
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        
        # Forward through transformer blocks
        all_moe_aux = []
        for block in self.blocks:
            x, moe_aux = block(x, mask)
            all_moe_aux.append(moe_aux)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]
        
        # Compute loss if targets provided
        if targets is not None:
            # Cross-entropy loss (式 3.2)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                reduction='mean'
            )
            
            # Aggregate MoE auxiliary losses from all layers
            total_lb_loss = sum(aux['lb_loss'] for aux in all_moe_aux) / len(all_moe_aux)
            total_z_loss = sum(aux['z_loss'] for aux in all_moe_aux) / len(all_moe_aux)
            total_entropy = sum(aux['entropy'] for aux in all_moe_aux) / len(all_moe_aux)
            
            aux_losses = {
                'ce_loss': ce_loss,
                'lb_loss': total_lb_loss,
                'z_loss': total_z_loss,
                'entropy': total_entropy,
                'all_layers_aux': all_moe_aux,
            }
            
            return ce_loss, logits, aux_losses
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Generate text autoregressively.
        
        Args:
            idx: Starting indices [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
        Returns:
            Generated indices [B, T + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.pos_emb.size(1) else idx[:, -self.pos_emb.size(1):]
            
            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
