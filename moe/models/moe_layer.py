import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer with Top-k routing.
    Implements the mathematical formulation from the design doc.
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        topk: int = 2,
        d_hidden_mult: float = 2.0,
        capacity_factor: float = 1.0,
        dropout: float = 0.0,
        use_noisy_gating: bool = False,
        noise_std: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            n_experts: Number of experts (E)
            topk: Number of top experts to route to (k)
            d_hidden_mult: Hidden dimension multiplier
            capacity_factor: Capacity factor for load balancing
            dropout: Dropout rate
            use_noisy_gating: Whether to use noisy gating during training
            noise_std: Standard deviation for noise
        """
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.topk = topk
        self.capacity_factor = capacity_factor
        self.use_noisy_gating = use_noisy_gating
        self.noise_std = noise_std
        
        d_hidden = int(d_model * d_hidden_mult)
        
        # Router: h -> logits for E experts (式 3.4)
        self.router = nn.Linear(d_model, n_experts)
        
        # Expert FFNs: W_1 (d -> d_hidden), W_2 (d_hidden -> d)
        self.experts_w1 = nn.Parameter(torch.randn(n_experts, d_model, d_hidden))
        self.experts_b1 = nn.Parameter(torch.zeros(n_experts, d_hidden))
        self.experts_w2 = nn.Parameter(torch.randn(n_experts, d_hidden, d_model))
        self.experts_b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.experts_w1)
        nn.init.xavier_uniform_(self.experts_w2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with MoE routing.
        
        Args:
            x: Input tensor [B, T, d]
            
        Returns:
            output: [B, T, d]
            aux_loss_dict: Dictionary with auxiliary losses and stats
        """
        B, T, d = x.shape
        N = B * T
        
        # Reshape to [N, d]
        h = x.view(N, d)
        
        # Router: compute logits s_t (式 3.4)
        logits = self.router(h)  # [N, E]
        
        # Add noise during training (optional noisy gating)
        if self.training and self.use_noisy_gating:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Compute router probabilities p_t = softmax(s_t)
        probs = F.softmax(logits, dim=-1)  # [N, E]
        
        # Top-k routing: get top-k expert indices and weights
        weights, indices = probs.topk(self.topk, dim=-1)  # both [N, k]
        
        # Renormalize weights g_{t,e} (式 3.4)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Create routing mask M and weight matrix G
        # M[t, e] = 1 if e in Top-k for token t
        M = torch.zeros(N, self.n_experts, device=x.device, dtype=torch.bool)
        M.scatter_(1, indices, True)
        
        # G[t, e] = g_{t,e} if e selected, 0 otherwise
        G = torch.zeros(N, self.n_experts, device=x.device, dtype=x.dtype)
        G.scatter_add_(1, indices, weights)
        
        # Dispatch and compute (式 3.4)
        y = torch.zeros_like(h)  # [N, d]
        
        for e in range(self.n_experts):
            # Get tokens assigned to expert e
            expert_mask = M[:, e]  # [N]
            if not expert_mask.any():
                continue
            
            expert_tokens = h[expert_mask]  # [N_e, d]
            
            # Expert FFN: Y_e = W_2 * GeLU(W_1 * X_e + b_1) + b_2
            hidden = F.gelu(expert_tokens @ self.experts_w1[e] + self.experts_b1[e])
            hidden = self.dropout(hidden)
            expert_out = hidden @ self.experts_w2[e] + self.experts_b2[e]  # [N_e, d]
            
            # Combine with routing weights
            y[expert_mask] += expert_out * G[expert_mask, e:e+1]
        
        # Reshape back
        y = y.view(B, T, d)
        
        # Compute auxiliary losses (式 3.5, 3.6)
        aux_loss_dict = self._compute_aux_losses(probs, M, logits)
        
        return y, aux_loss_dict
    
    def _compute_aux_losses(
        self, 
        probs: torch.Tensor,
        M: torch.Tensor, 
        logits: torch.Tensor
    ) -> dict:
        """
        Compute load balancing and z-loss (式 3.5, 3.6).
        
        Args:
            probs: Router probabilities [N, E]
            M: Routing mask [N, E]
            logits: Router logits [N, E]
            
        Returns:
            Dictionary with losses and statistics
        """
        N, E = probs.shape
        
        # f_e: fraction of tokens assigned to expert e (式 3.5)
        # For Top-k, we count how many tokens selected each expert
        f_e = M.float().mean(dim=0)  # [E]
        
        # P_e: average gate probability for expert e
        P_e = probs.mean(dim=0)  # [E]
        
        # Load balancing loss: minimize -(E * sum_e f_e * P_e) (式 3.5)
        lb_loss = -E * (f_e * P_e).sum()
        
        # Z-loss: minimize average of (logsumexp(logits))^2 (式 3.6)
        z = torch.logsumexp(logits, dim=-1)  # [N]
        z_loss = (z ** 2).mean()
        
        # Router entropy (optional, 式 3.7)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        
        return {
            'lb_loss': lb_loss,
            'z_loss': z_loss,
            'entropy': entropy,
            'f_e': f_e,  # for monitoring
            'P_e': P_e,  # for monitoring
        }
