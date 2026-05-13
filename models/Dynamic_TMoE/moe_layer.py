import torch
from torch import nn

from .expert_pool import DynamicExpertPool
from .memory_router import TemporalMemoryRouter
from .anomaly_repository import AnomalyStateRepository
from .drift_detector import MMDDriftDetector

class TemporalDynamicMoELayer(nn.Module):
    def __init__(self, d_model, num_patches, n_vars, cycle_length, num_experts=4, num_drift_experts=4,
                 dropout=0.1, num_rnn_layers=1,
                 drift_window_size=576, use_relation_layer=1, enable_drift_detection=True, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.cycle_length = cycle_length
        self.use_relation_layer = use_relation_layer
        self.num_drift_experts = num_drift_experts
        self.drift_detection_enabled = enable_drift_detection

        self.expert_pool = DynamicExpertPool(
            d_model=d_model,
            num_patches=num_patches,
            n_vars=n_vars,
            cycle_length=cycle_length,
            num_experts=num_experts,
            num_drift_experts=num_drift_experts,
            dropout=dropout,
            drift_window_size=drift_window_size,
            use_relation_layer=use_relation_layer
        )

        self.anomaly_state_repo = AnomalyStateRepository(
            hidden_dim=d_model * 2,
            max_capacity=10
        )

        self.gating_network = TemporalMemoryRouter(
            d_model=d_model,
            num_base_experts=num_experts,
            num_drift_experts=self.num_drift_experts,
            hidden_dim=d_model * 2,
            dropout=dropout,
            expert_pool=self.expert_pool,
            num_layers=num_rnn_layers,
            anomaly_repo=self.anomaly_state_repo,
            top_k=top_k
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.register_buffer('forward_call_count', torch.tensor(0))
        self.drift_detector = None
        self.register_buffer('drift_handled_count', torch.tensor(0))

        if self.drift_detection_enabled and not self.expert_pool.is_drift_pool_enabled():
            self.expert_pool.expert_manager.enable_drift_pool()
    
    def forward(self, x, hidden_state=None, batch_start_patch_idx=0, patch_cycle_indices=None):
        """
        Args:
            x: [B*N, P, D]
            hidden_state: [B*N, hidden_dim]
            batch_start_patch_idx: int
            patch_cycle_indices: [B]
        Returns:
            out: [B*N, P, D]
            new_hidden_state
        """
        BN, P, D = x.shape

        routing_weights_list = []
        routing_decisions_list = []

        use_anomaly_ref = (self.training and 
                          len(self.anomaly_state_repo.anomaly_states) > 0)

        for p in range(P):
            current_patch_idx = batch_start_patch_idx + p
            
            x_p = x[:, p, :]
            
            routing_weights_p, routing_decisions_p, hidden_state = self.gating_network(
                x_p,
                hidden_state,
                use_anomaly_reference=use_anomaly_ref,
                patch_idx=current_patch_idx
            )

            if self.anomaly_state_repo.is_drift_patch(current_patch_idx):
                self.anomaly_state_repo.store_anomaly_state(
                    patch_idx=current_patch_idx,
                    hidden_state=hidden_state,
                    timestamp=self.forward_call_count.item()
                )
            
            routing_weights_list.append(routing_weights_p)
            routing_decisions_list.append(routing_decisions_p)

        new_hidden_state = hidden_state

        max_K = max(w.shape[1] for w in routing_weights_list)
        aligned_weights_list = []
        aligned_decisions_list = []
        
        for weights_p, decisions_p in zip(routing_weights_list, routing_decisions_list):
            current_K = weights_p.shape[1]
            if current_K < max_K:
                pad_size = max_K - current_K
                pad_weights = torch.zeros(weights_p.shape[0], pad_size, device=weights_p.device)
                pad_decisions = torch.zeros(decisions_p.shape[0], pad_size, dtype=torch.bool, device=decisions_p.device)
                weights_p = torch.cat([weights_p, pad_weights], dim=1)
                decisions_p = torch.cat([decisions_p, pad_decisions], dim=1)
            
            aligned_weights_list.append(weights_p)
            aligned_decisions_list.append(decisions_p)
        
        all_routing_weights = torch.stack(aligned_weights_list, dim=1)
        all_routing_decisions = torch.stack(aligned_decisions_list, dim=1)

        if patch_cycle_indices is None:
            B = x.shape[0] // self.n_vars
            patch_cycle_indices = torch.zeros(B, dtype=torch.long, device=x.device)
        
        expert_output = self.expert_pool(x, all_routing_weights, patch_cycle_indices)

        out = self.output_proj(expert_output) + x

        self.forward_call_count += 1

        if self.training:
            self.expert_pool.expert_manager.update_usage(all_routing_weights.detach())

        if self.drift_detector is not None and self.training:
            sample_size = min(max(1, BN // 10), 50)
            step = BN // sample_size
            indices = torch.arange(0, BN, step, device=x.device)[:sample_size]
            start_patch_indices = indices * P + batch_start_patch_idx
            self.drift_detector.add_current_samples_batch(x[indices].detach(), start_patch_indices)
        
        return out, new_hidden_state
    
    def set_pending_drift_patches(self, drift_patch_indices, mmd_score=0.0):
        if self.anomaly_state_repo is not None:
            self.anomaly_state_repo.set_pending_drift_patches(drift_patch_indices)
    
    def enable_drift_detection(self, window_size=576, history_size=50, k_sigma=3.0):
        self.drift_detector = MMDDriftDetector(
            d_model=self.d_model,
            window_size=window_size,
            history_size=history_size,
            k_sigma=k_sigma
        )

        if hasattr(self, 'forward_call_count'):
            self.drift_detector = self.drift_detector.to(self.forward_call_count.device)
    
    def calibrate_drift_detector(self, calibration_data):
        if self.drift_detector is None:
            return
        
        self.drift_detector.add_reference_samples(calibration_data)
    
    def check_and_handle_drift(self, finetune_epochs=20, finetune_lr=0.0005, finetune_patience=5, train_mode=True):
        if self.drift_detector is None:
            return

        if not self.drift_detector.should_check_drift():
            return

        drift_result = self.drift_detector.check_drift()
        if drift_result is None:
            return
        
        drift_detected, mmd_score, drift_patch_indices = drift_result
        
        if drift_detected:
            self.drift_handled_count += 1

            if drift_patch_indices:
                self.set_pending_drift_patches(drift_patch_indices, mmd_score)

            ref_data, cur_data = self.drift_detector.get_drift_data()

            if not train_mode and self.drift_detection_enabled:
                self.drift_detector.slide_window()
                return
            
            if ref_data is not None and cur_data is not None and train_mode:
                expert_type, scores = self.expert_pool.analyze_drift_pattern(
                    ref_data, cur_data, self.gating_network
                )

                self.expert_pool.enable_drift_expert(
                    expert_type=expert_type,
                    ref_data=ref_data,
                    cur_data=cur_data,
                    gating_network=self.gating_network,
                    pretrain_epochs=finetune_epochs,
                    pretrain_lr=finetune_lr,
                    pretrain_patience=finetune_patience
                )

                disabled_info = self.expert_pool.disable_expert_if_needed()

                self.drift_detector.slide_window()
            
            if train_mode:
                self.drift_detector.slide_window()
        else:
            if train_mode and self.drift_detection_enabled:
                pass
            elif not train_mode and self.drift_detection_enabled:
                pass
