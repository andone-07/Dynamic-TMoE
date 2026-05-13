import torch
from torch import nn
import torch.nn.functional as F

class TemporalExpert(nn.Module):
    def __init__(self, d_model, n_vars=None, num_patches=None, cycle_length=None, dropout=0.1, use_relation_layer=1):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars
        self.num_patches = num_patches
        self.use_relation_layer = use_relation_layer

        if use_relation_layer == 1 and n_vars is not None and num_patches is not None and cycle_length is not None:
            from .cyclic_relation import CyclicRelationLayer
            self.relation_layer = CyclicRelationLayer(
                n_vars=n_vars,
                d_model=d_model,
                num_patches=num_patches,
                cycle_length=cycle_length,
                dropout=dropout
            )
        else:
            self.relation_layer = None
    
    def forward(self, x):
        raise NotImplementedError
    
    def apply_relation(self, x, patch_cycle_indices):
        """
        Args:
            x: [B*N, P, D]
            patch_cycle_indices: [B]
        Returns:
            output: [B*N, P, D]
        """
        if self.use_relation_layer == 0 or self.relation_layer is None:
            return x

        BN, P, D = x.shape
        B = patch_cycle_indices.shape[0]
        N = BN // B
        x_reshaped = x.view(B, N, P, D).permute(0, 1, 3, 2)

        x_relation = self.relation_layer(x_reshaped, patch_cycle_indices)
        output = x_relation.permute(0, 1, 3, 2).reshape(BN, P, D)
        
        return output
    
    def finetune(self, ref_data, cur_data, epochs=50, lr=0.001, patience=5):
        """
        Args:
            ref_data: [window_size, P, D]
            cur_data: [window_size, P, D]
        Returns:
            loss_history
        """
        self.train()

        train_data = torch.cat([ref_data, cur_data], dim=0).detach()

        for param in self.parameters():
            param.requires_grad = True

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        loss_history = []
        best_loss = float('inf')
        best_state_dict = None
        patience_counter = 0

        with torch.enable_grad():
            for epoch in range(epochs):
                optimizer.zero_grad()

                output = self.forward(train_data)
                loss = criterion(output, train_data)

                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                loss_history.append(current_loss)

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        if best_state_dict is not None:
            device = next(self.parameters()).device
            self.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        
        self.eval()
        return loss_history
    
    def localized_pretrain(self, ref_data, cur_data, gating_network, other_experts, 
                          epochs=20, lr=0.001, patience=5):
        """
        Localized pre-training of new experts
        """
        self.train()

        for param in self.parameters():
            param.requires_grad = True

        original_gating_training_mode = gating_network.training
        gating_network.train()

        frozen_params = []
        for expert in other_experts:
            for param in expert.parameters():
                frozen_params.append((param, param.requires_grad))
                param.requires_grad = False

        device = next(self.parameters()).device
        ref_data = ref_data.detach().clone().to(device)
        cur_data = cur_data.detach().clone().to(device)

        train_data = torch.cat([ref_data, cur_data], dim=0)
        window_size = train_data.shape[0]

        trainable_params = list(self.parameters())

        if hasattr(gating_network, 'base_output_layer'):
            for param in gating_network.base_output_layer.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        if hasattr(gating_network, 'drift_output_layer'):
            for param in gating_network.drift_output_layer.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        trainable_params = [p for p in trainable_params if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        loss_history = {
            'total_loss': [],
            'recon_loss': [],
            'gate_loss': []
        }
        
        best_loss = float('inf')
        best_state_dict = None
        patience_counter = 0
        
        with torch.enable_grad():
            for epoch in range(epochs):
                optimizer.zero_grad()

                expert_output = self.forward(train_data)  # [2*window_size, P, D]
                recon_loss = F.mse_loss(expert_output, train_data)

                total_loss = recon_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()

                current_total_loss = total_loss.item()
                current_recon_loss = recon_loss.item()

                loss_history['total_loss'].append(current_total_loss)
                loss_history['recon_loss'].append(current_recon_loss)
                loss_history['gate_loss'].append(0.0)

                if current_total_loss < best_loss:
                    best_loss = current_total_loss
                    best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        if best_state_dict is not None:
            device = next(self.parameters()).device
            best_state_dict = {k: v.to(device) for k, v in best_state_dict.items()}
            self.load_state_dict(best_state_dict)

        for param, requires_grad in frozen_params:
            param.requires_grad = requires_grad

        if not original_gating_training_mode:
            gating_network.eval()
        
        self.eval()

        return loss_history


class IdentityExpert(TemporalExpert):
    def __init__(self, d_model, n_vars=None, num_patches=None, cycle_length=None, dropout=0.1, use_relation_layer=1):
        super().__init__(d_model, n_vars, num_patches, cycle_length, dropout, use_relation_layer)
        self.transform = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        Args:
            x: [B*N, P, D]
        Returns:
            out: [B*N, P, D]
        """
        BN, P, D = x.shape
        out = self.transform(x)
        return out


class TrendExpert(TemporalExpert):
    def __init__(self, d_model, n_vars=None, num_patches=None, cycle_length=None, dropout=0.1, use_relation_layer=1):
        super().__init__(d_model, n_vars, num_patches, cycle_length, dropout, use_relation_layer)
        kernel_size = 13
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.trend_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x):
        BN, P, D = x.shape
        x_t = x.transpose(1, 2)
        trend = self.avg_pool(x_t).transpose(1, 2)
        out = self.trend_proj(trend)
        return out


class PeriodicActivation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        out = torch.cat([torch.sin(x1), torch.cos(x2)], dim=-1)
        
        return out


class SeasonalityExpert(TemporalExpert):
    def __init__(self, d_model, num_fourier_modes=None, n_vars=None, num_patches=None, cycle_length=None, dropout=0.1, use_relation_layer=1):
        super().__init__(d_model, n_vars, num_patches, cycle_length, dropout, use_relation_layer)
        self.d_model = d_model
        self.num_fourier_modes = num_fourier_modes

        self.freq_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.Tanh(),
            nn.Linear(d_model * 2, d_model * 2)
        )

        self.periodic_activation = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            PeriodicActivation(),
            nn.Linear(d_model * 2, d_model)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B*N, P, D]
        Returns:
            out: [B*N, P, D]
        """
        BN, P, D = x.shape

        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')

        x_freq_real = x_freq.real
        x_freq_imag = x_freq.imag

        x_freq_combined = torch.cat([x_freq_real, x_freq_imag], dim=-1)

        if self.num_fourier_modes is not None:
            modes = min(self.num_fourier_modes, x_freq_combined.shape[1])
            x_freq_selected = x_freq_combined[:, :modes, :]
            x_freq_processed = self.freq_mlp(x_freq_selected)

            x_freq_full = torch.zeros_like(x_freq_combined)
            x_freq_full[:, :modes, :] = x_freq_processed
        else:
            x_freq_full = self.freq_mlp(x_freq_combined)

        x_freq_real_processed = x_freq_full[..., :D]
        x_freq_imag_processed = x_freq_full[..., D:]
        x_freq_processed = torch.complex(x_freq_real_processed, x_freq_imag_processed)

        x_seasonal = torch.fft.irfft(x_freq_processed, n=P, dim=1, norm='ortho')
        x_seasonal = self.periodic_activation(x_seasonal)

        out = self.output_proj(x_seasonal)
        
        return out


class FluctuationExpert(TemporalExpert):
    def __init__(self, d_model, n_vars=None, num_patches=None, cycle_length=None, dropout=0.1, use_relation_layer=1):
        super().__init__(d_model, n_vars, num_patches, cycle_length, dropout, use_relation_layer)
        self.conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1, dilation=1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        shortcut = x
        x = x.transpose(1, 2)

        out1 = self.conv1(x)
        out1, gate1 = out1.chunk(2, dim=1)
        x = out1 * torch.sigmoid(gate1)

        out2 = self.conv2(x)
        out2, gate2 = out2.chunk(2, dim=1)
        x = out2 * torch.sigmoid(gate2)
        
        x = x.transpose(1, 2)
        return self.norm(x + shortcut)
