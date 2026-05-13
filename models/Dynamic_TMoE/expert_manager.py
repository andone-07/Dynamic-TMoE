import torch
from torch import nn

class EvolvableExpertManager(nn.Module):
    """
    Evolvable Expert Manager: Utilizing Masks to Manage the Activation and Deactivation of Experts
    """
    def __init__(self, base_expert_count, num_drift_experts=4, min_experts=4, 
                 tracking_window=1000, low_usage_threshold=0.01, removal_patience=3):
        """
        Args:
            base_expert_count: Initial number of base experts
            num_drift_experts: Number of drift experts
            min_experts: Minimum number of active experts to retain
            tracking_window: Size of the tracking window, usually set to drift_window_size
            low_usage_threshold: Low usage rate threshold
            removal_patience: Patience value for disabling experts
        """
        super().__init__()
        self.base_expert_count = base_expert_count
        self.num_drift_experts = num_drift_experts
        self.min_experts = min_experts
        self.tracking_window = tracking_window
        self.low_usage_threshold = low_usage_threshold
        self.removal_patience = removal_patience
        
        self.total_expert_count = base_expert_count + num_drift_experts
        
        expert_mask = torch.ones(self.total_expert_count, dtype=torch.bool)
        expert_mask[base_expert_count:] = False
        self.register_buffer('expert_mask', expert_mask)

        self.register_buffer('drift_pool_enabled', torch.tensor(False))
        self.expert_info = []

        for i in range(base_expert_count):
            if i == 0:
                expert_type = 'identity'
            elif i == 1:
                expert_type = 'trend'
            elif i == 2:
                expert_type = 'seasonality'
            elif i == 3:
                expert_type = 'fluctuation'
            else:
                expert_type = ['trend', 'seasonality', 'fluctuation'][(i - 4) % 3]
            self.expert_info.append({
                'expert_id': i,
                'expert_type': expert_type,
                'is_base': True,
                'is_drift': False,
                'added_at_step': 0,
                'is_enabled': True
            })

        for i in range(num_drift_experts):
            idx = base_expert_count + i
            expert_type = ['trend', 'seasonality', 'fluctuation'][i % 3]
            self.expert_info.append({
                'expert_id': idx,
                'expert_type': expert_type,
                'is_base': False,
                'is_drift': True,
                'added_at_step': 0,
                'is_enabled': False
            })

        self.usage_history = {i: [] for i in range(self.total_expert_count)}
        self.low_usage_count = {i: 0 for i in range(self.total_expert_count)}
        self.accumulated_weights = {i: 0.0 for i in range(self.total_expert_count)}
        self.sample_count = 0
        self.step_count = 0
    
    def get_num_experts(self):
        if not self.drift_pool_enabled:
            return self.base_expert_count
        return self.expert_mask.sum().item()
    
    def get_enabled_expert_indices(self):
        if not self.drift_pool_enabled:
            return list(range(self.base_expert_count))
        return torch.where(self.expert_mask)[0].tolist()
    
    def get_expert_types(self):
        return [self.expert_info[i]['expert_type'] for i in range(len(self.expert_info)) 
                if self.expert_info[i]['is_enabled']]
    
    def enable_drift_pool(self):
        if not self.drift_pool_enabled:
            self.drift_pool_enabled = torch.tensor(True, device=self.drift_pool_enabled.device)

    def disable_drift_pool(self):
        if self.drift_pool_enabled:
            for i in range(self.base_expert_count, self.total_expert_count):
                if self.expert_info[i]['is_enabled']:
                    self.expert_mask[i] = False
                    self.expert_info[i]['is_enabled'] = False
            
            self.drift_pool_enabled = torch.tensor(False, device=self.drift_pool_enabled.device)
    
    def is_drift_pool_enabled(self):
        return self.drift_pool_enabled.item()
    
    def enable_drift_expert(self, expert_type):
        """
        Args:
            expert_type: str    
        Returns:
            (expert_id, is_new): tuple
            expert_id: int or None
            is_new: bool
        """
        for i in range(self.base_expert_count, self.total_expert_count):
            info = self.expert_info[i]
            if not info['is_enabled'] and info['expert_type'] == expert_type:
                self.expert_mask[i] = True
                info['is_enabled'] = True
                info['added_at_step'] = self.step_count
                
                return i, True

        for i in range(self.base_expert_count, self.total_expert_count):
            info = self.expert_info[i]
            if info['is_enabled'] and info['expert_type'] == expert_type:
                return i, False

        return None, False
    
    def update_usage(self, routing_weights):
        """
        Args:
            routing_weights: [B*N, P, K_enabled]
        """
        BN, P, K = routing_weights.shape
        routing_weights = routing_weights.reshape(BN * P, K)

        BN, K = routing_weights.shape
        
        batch_weights = routing_weights.sum(dim=0)

        enabled_indices = self.get_enabled_expert_indices()
        
        for idx, expert_id in enumerate(enabled_indices):
            if idx < K:
                self.accumulated_weights[expert_id] += batch_weights[idx].item()
        
        self.sample_count += BN

        if self.sample_count >= self.tracking_window:
            self._update_history()
    
    def _update_history(self):
        enabled_indices = self.get_enabled_expert_indices()
        
        for expert_id in enabled_indices:
            avg_usage = self.accumulated_weights[expert_id] / self.sample_count
            self.usage_history[expert_id].append(avg_usage)

            if avg_usage < self.low_usage_threshold:
                self.low_usage_count[expert_id] += 1
            else:
                self.low_usage_count[expert_id] = 0

            self.accumulated_weights[expert_id] = 0.0
        
        self.sample_count = 0
        self.step_count += 1
    
    def get_expert_to_disable(self):
        """
        Returns:
            expert_id: int or None
        """
        if self.get_num_experts() <= self.min_experts:
            return None

        candidates = []
        for expert_id in range(self.base_expert_count, self.total_expert_count):
            info = self.expert_info[expert_id]
            if not info['is_enabled']:
                continue
            
            if self.low_usage_count[expert_id] >= self.removal_patience:
                if len(self.usage_history[expert_id]) > 0:
                    avg_usage = sum(self.usage_history[expert_id]) / len(self.usage_history[expert_id])
                    candidates.append((expert_id, avg_usage))
        
        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1])
        expert_id, _ = candidates[0]
        
        return expert_id
    
    def disable_expert(self, expert_id):
        """
        Args:
            expert_id: int
        Returns:
            disabled_info: dict
        """
        if expert_id >= self.total_expert_count:
            return None
        
        info = self.expert_info[expert_id]

        if info['is_base']:
            return None

        self.expert_mask[expert_id] = False
        info['is_enabled'] = False
        self.low_usage_count[expert_id] = 0
        
        return info
