import torch
from torch import nn
import torch.nn.functional as F

class AnomalyStateRepository(nn.Module):
    def __init__(self, hidden_dim, max_capacity=20):
        """
        Args:
            hidden_dim
            max_capacity
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_capacity = max_capacity

        self.anomaly_states = {}

        self.pending_drift_patch_indices = set()

        self.fusion_weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.register_buffer('total_stored', torch.tensor(0))
        self.register_buffer('total_accessed', torch.tensor(0))
    
    def set_pending_drift_patches(self, drift_patch_indices):
        """
        Args:
            drift_patch_indices: list[int]
        """
        self.pending_drift_patch_indices = set(drift_patch_indices)
    
    def is_drift_patch(self, patch_idx):
        """
        Args:
            patch_idx: int
        Returns:
            whether the patch is marked as drift: bool
        """
        return patch_idx in self.pending_drift_patch_indices
    
    def store_anomaly_state(self, patch_idx, hidden_state, timestamp):
        """
        Args:
            patch_idx: int
            hidden_state: [B*N, hidden_dim]
            timestamp: int
        """

        stored_state = hidden_state[0].detach().clone()

        if patch_idx not in self.anomaly_states:
            self.anomaly_states[patch_idx] = []

        self.anomaly_states[patch_idx].append({
            'hidden_state': stored_state,
            'timestamp': timestamp
        })
        
        self.total_stored += 1
        self.pending_drift_patch_indices.discard(patch_idx)

        if len(self.anomaly_states[patch_idx]) > self.max_capacity:
            self._prune_states_for_patch(patch_idx)
    
    def _prune_states_for_patch(self, patch_idx):
        if patch_idx not in self.anomaly_states:
            return

        self.anomaly_states[patch_idx].sort(key=lambda x: x['timestamp'], reverse=True)
        self.anomaly_states[patch_idx] = self.anomaly_states[patch_idx][:self.max_capacity]
    
    def get_reference_state(self, patch_idx, current_hidden):
        if patch_idx not in self.anomaly_states or len(self.anomaly_states[patch_idx]) == 0:
            return None, False

        patch_anomaly_states = self.anomaly_states[patch_idx]
        self.total_accessed += 1

        return self._adaptive_fusion(current_hidden, patch_anomaly_states)
    
    def _weighted_average_states(self, current_hidden, patch_anomaly_states):
        """
        Args:
            current_hidden: [B*N, hidden_dim]
            patch_anomaly_states
        Returns:
            averaged_state: [B*N, hidden_dim]
            has_reference: True
        """
        batch_size = current_hidden.shape[0]
        current_h_single = current_hidden[0]

        similarities = []
        for item in patch_anomaly_states:
            stored_h = item['hidden_state']
            sim = F.cosine_similarity(current_h_single.unsqueeze(0), stored_h.unsqueeze(0), dim=-1)
            similarities.append(sim)

        similarities = torch.stack(similarities)
        weights = F.softmax(similarities, dim=0)

        weighted_h = torch.zeros_like(current_h_single)
        for weight, item in zip(weights, patch_anomaly_states):
            stored_h = item['hidden_state']
            weighted_h += weight * stored_h

        if weighted_h.dim() > 1:
            if weighted_h.shape[0] != batch_size:
                weighted_h = weighted_h[0].unsqueeze(0).expand(batch_size, -1)
        else:
            weighted_h = weighted_h.unsqueeze(0).expand(batch_size, -1)
        
        return weighted_h, True
    
    def _adaptive_fusion(self, current_hidden, patch_anomaly_states):
        """
        Args:
            current_hidden: [B*N, hidden_dim]
            patch_anomaly_states
        Returns:
            fused_state: [B*N, hidden_dim]
            has_reference: True
        """
        ref_state, _ = self._weighted_average_states(current_hidden, patch_anomaly_states)
        
        current_h = current_hidden
        ref_h = ref_state
        concat_h = torch.cat([current_h, ref_h], dim=-1)
        alpha = self.fusion_weight_net(concat_h)
        fused_h = alpha * current_h + (1 - alpha) * ref_h
        
        return fused_h, True
    
    def clear(self):
        self.anomaly_states.clear()
        self.pending_drift_patch_indices.clear()
