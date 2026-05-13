import torch
from torch import nn

from .experts import (
    IdentityExpert,
    TrendExpert,
    SeasonalityExpert,
    FluctuationExpert
)
from .expert_manager import EvolvableExpertManager
from .drift_detector import DriftPatternProfiler

class DynamicExpertPool(nn.Module):
    def __init__(self, d_model, num_patches, n_vars, cycle_length, num_experts=4, num_drift_experts=4,
                 dropout=0.1, drift_window_size=576, use_relation_layer=1):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.n_vars = n_vars
        self.cycle_length = cycle_length
        self.dropout = dropout
        self.num_drift_experts = num_drift_experts
        self.use_relation_layer = use_relation_layer

        if num_experts < 4:
            num_experts = 4
        self.num_base_experts = num_experts

        self.num_fourier_modes = num_patches // 2
        self.experts = nn.ModuleList()

        self.experts.append(IdentityExpert(d_model=d_model, use_relation_layer=use_relation_layer))
        self.experts.append(TrendExpert(d_model=d_model, use_relation_layer=use_relation_layer))
        self.experts.append(SeasonalityExpert(d_model=d_model, num_fourier_modes=self.num_fourier_modes, use_relation_layer=use_relation_layer))
        self.experts.append(FluctuationExpert(d_model=d_model, use_relation_layer=use_relation_layer))

        remaining = num_experts - 4
        if remaining > 0:
            for i in range(remaining):
                expert_type = ['trend', 'seasonality', 'fluctuation'][i % 3]
                self.experts.append(self._create_expert(expert_type))

        for i in range(num_drift_experts):
            expert_type = ['trend', 'seasonality', 'fluctuation'][i % 3]
            self.experts.append(self._create_expert(expert_type))
        
        self.expert_manager = EvolvableExpertManager(
            base_expert_count=num_experts,
            num_drift_experts=num_drift_experts,
            min_experts=4,
            tracking_window=drift_window_size,
            low_usage_threshold=0.01,
            removal_patience=30
        )

        self.pattern_profiler = DriftPatternProfiler()
    
    def _create_expert(self, expert_type):
        """
        Args:
            expert_type: str
        Returns:
            expert: TemporalExpert
        """
        if expert_type == 'identity':
            return IdentityExpert(
                d_model=self.d_model,
                n_vars=self.n_vars,
                num_patches=self.num_patches,
                cycle_length=self.cycle_length,
                dropout=self.dropout,
                use_relation_layer=self.use_relation_layer
            )
        elif expert_type == 'trend':
            return TrendExpert(
                d_model=self.d_model,
                n_vars=self.n_vars,
                num_patches=self.num_patches,
                cycle_length=self.cycle_length,
                dropout=self.dropout,
                use_relation_layer=self.use_relation_layer
            )
        elif expert_type == 'seasonality':
            return SeasonalityExpert(
                d_model=self.d_model, 
                num_fourier_modes=self.num_fourier_modes,
                n_vars=self.n_vars,
                num_patches=self.num_patches,
                cycle_length=self.cycle_length,
                dropout=self.dropout,
                use_relation_layer=self.use_relation_layer
            )
        elif expert_type == 'fluctuation':
            return FluctuationExpert(
                d_model=self.d_model,
                n_vars=self.n_vars,
                num_patches=self.num_patches,
                cycle_length=self.cycle_length,
                dropout=self.dropout,
                use_relation_layer=self.use_relation_layer
            )
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
    
    @property
    def current_num_experts(self):
        return self.expert_manager.get_num_experts()
    
    def get_enabled_expert_indices(self):
        return self.expert_manager.get_enabled_expert_indices()
    
    def enable_drift_expert(self, expert_type, ref_data, cur_data, gating_network, 
                           pretrain_epochs=20, pretrain_lr=0.001, pretrain_patience=5):
        result = self.expert_manager.enable_drift_expert(expert_type)
        
        if result[0] is None:
            return
        
        expert_id, is_new = result

        drift_expert = self.experts[expert_id].to(next(self.experts[0].parameters()).device)

        if is_new:
            other_experts = [self.experts[i] for i in self.get_enabled_expert_indices() if i != expert_id]
            loss_history = drift_expert.localized_pretrain(
                ref_data=ref_data,
                cur_data=cur_data,
                gating_network=gating_network,
                other_experts=other_experts,
                epochs=pretrain_epochs,
                lr=pretrain_lr,
                patience=pretrain_patience
            )
            
            return
        else:
            return
    
    def disable_expert_if_needed(self):
        expert_id = self.expert_manager.get_expert_to_disable()
        
        if expert_id is None:
            return None

        disabled_info = self.expert_manager.disable_expert(expert_id)

        return disabled_info
    
    def analyze_drift_pattern(self, ref_data, cur_data, gating_network):
        """
        Args:
            ref_data: [window_size, P, D]
            cur_data: [window_size, P, D]
            gating_network: TemporalMemoryRouter
        Returns:
            expert_type: str
            analysis_scores: dict
        """
        with torch.no_grad():
            window_size, P, D = cur_data.shape

            features = cur_data.mean(dim=1)
            hidden_state = gating_network.init_hidden(window_size, cur_data.device)

            routing_weights, _, hidden_state = gating_network(features, hidden_state)

            enabled_indices = self.get_enabled_expert_indices()
            K_enabled = len(enabled_indices)

            expert_outputs = []
            for idx in enabled_indices:
                expert_out = self.experts[idx](cur_data)
                expert_outputs.append(expert_out)

            expert_outputs = torch.stack(expert_outputs, dim=1)
            weights_expanded = routing_weights.unsqueeze(-1).unsqueeze(-1)

            weighted_outputs = expert_outputs * weights_expanded
            model_output = weighted_outputs.sum(dim=1)

        expert_type, analysis_scores = self.pattern_profiler.analyze_pattern(
            model_output, cur_data
        )
        
        return expert_type, analysis_scores
    
    def is_drift_pool_enabled(self):
        return self.expert_manager.is_drift_pool_enabled()
    
    def disable_drift_pool(self):
        self.expert_manager.disable_drift_pool()
    
    def forward(self, x, routing_weights, patch_cycle_indices):
        """
        Args:
            x: [B*N, P, D]
            routing_weights: [B*N, P, K_enabled]
            patch_cycle_indices: [B]   
        Returns:
            aggregated_output: [B*N, P, D]
        """
        BN, P, D = x.shape

        enabled_indices = self.get_enabled_expert_indices()
        K_enabled = len(enabled_indices)

        K_routing = routing_weights.shape[2]
        if K_routing != K_enabled:
            if K_routing < K_enabled:
                pad_size = K_enabled - K_routing
                padding = torch.zeros(BN, P, pad_size, device=routing_weights.device)
                routing_weights = torch.cat([routing_weights, padding], dim=2)
            else:
                routing_weights = routing_weights[:, :, :K_enabled]

        expert_temporal_outputs = []
        for idx in enabled_indices:
            expert = self.experts[idx]
            expert_out = expert(x)
            expert_temporal_outputs.append(expert_out)

        expert_outputs = []
        for idx, expert_temporal_out in zip(enabled_indices, expert_temporal_outputs):
            expert = self.experts[idx]
            expert_relation_out = expert.apply_relation(expert_temporal_out, patch_cycle_indices)
            expert_outputs.append(expert_relation_out)

        expert_outputs = torch.stack(expert_outputs, dim=1)

        weights_permuted = routing_weights.permute(0, 2, 1)
        weights_expanded = weights_permuted.unsqueeze(-1)
        weighted_outputs = expert_outputs * weights_expanded

        aggregated_output = weighted_outputs.sum(dim=1)
        
        return aggregated_output
