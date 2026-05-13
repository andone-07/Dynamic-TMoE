import torch
from torch import nn
import torch.nn.functional as F

class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, n_vars):
        super().__init__()
        self.cycle_len = cycle_len
        self.n_vars = n_vars

        self.cycle_relations = nn.Parameter(
            torch.zeros(cycle_len, n_vars, n_vars),
            requires_grad=True
        )
        nn.init.xavier_uniform_(self.cycle_relations)

    def forward(self, cycle_index, length):
        B = cycle_index.size(0)
        gather_index = (cycle_index.view(-1, 1) + torch.arange(length, device=cycle_index.device).view(1, -1)) % self.cycle_len
        return self.cycle_relations[gather_index]

class CyclicRelationLayer(nn.Module):
    def __init__(self, n_vars, d_model, num_patches, cycle_length, dropout):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.num_patches = num_patches
        
        self.cycle_length = cycle_length

        self.cycle_queue = RecurrentCycle(
            cycle_len=self.cycle_length,
            n_vars=n_vars
        )

        self.residual_learner = nn.Sequential(
            nn.Linear(n_vars, n_vars * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_vars * 2, n_vars)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, patch_cycle_indices):
        """
        Args:
            x: [B, N, D, P]
            patch_cycle_indices: [B]
        Returns:
            output: [B, N, D, P]
        """
        B, N, D, P = x.shape

        # [B, P, N, N]
        R_cycle = self.cycle_queue(patch_cycle_indices, P)

        # [B, N, D, P] -> [B, P, N, D]
        x_permuted = x.permute(0, 3, 1, 2)
        x_flat = x_permuted.reshape(B * P, N, D)  # [B*P, N, D]

        x_normalized = F.normalize(x_flat, p=2, dim=-1)  # [B*P, N, D]

        # [B*P, N, D] @ [B*P, D, N] -> [B*P, N, N]
        R_current = torch.bmm(x_normalized, x_normalized.transpose(1, 2))  # [B*P, N, N]
        R_current = R_current.reshape(B, P, N, N)  # [B, P, N, N]

        # R_residual = R_current - R_cycle
        R_residual = R_current - R_cycle  # [B, P, N, N]
        
        R_residual_flat = R_residual.reshape(B * P, N, N)  # [B*P, N, N]
        
        # [B*P, N, N] -> [B*P, N, N]
        R_residual_learned_flat = self.residual_learner(R_residual_flat)
        R_residual_learned = R_residual_learned_flat.reshape(B, P, N, N)

        # R_final = R_cycle + R_residual_learned
        R_final = R_cycle + R_residual_learned
        # output = R_final @ x
        R_final_flat = R_final.reshape(B * P, N, N)
        output_flat = torch.bmm(R_final_flat, x_flat)
        output = output_flat.reshape(B, P, N, D)

        output_flat = output.reshape(B * P, N, D)
        x_permuted_flat = x_permuted.reshape(B * P, N, D)
        
        output_proj_flat = self.output_proj(output_flat) + x_permuted_flat
        output_final = output_proj_flat.reshape(B, P, N, D)
        
        # [B, P, N, D] -> [B, N, D, P]
        output_final = output_final.permute(0, 2, 3, 1)
        
        return output_final
