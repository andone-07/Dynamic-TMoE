import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.RevIN import RevIN
from .moe_layer import TemporalDynamicMoELayer

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        padding = self.stride
        self.patch_embedding = PatchEmbedding(
            configs.d_model, 
            self.patch_len, 
            self.stride, 
            padding, 
            configs.dropout
        )

        self.num_patches = int((configs.seq_len - self.patch_len) / self.stride + 2)

        # Temporal MoE
        self.num_temporal_moe_layers = getattr(configs, 'num_temporal_moe_layers', 2)
        num_experts = getattr(configs, 'num_experts', 4)
        num_drift_experts = getattr(configs, 'num_drift_experts', 3)
        num_rnn_layers = getattr(configs, 'num_rnn_layers', 2)
        top_k = getattr(configs, 'top_k', 2)

        self.drift_detection_enabled = getattr(configs, 'enable_drift_detection', True)
        self.drift_window_size = getattr(configs, 'drift_window_size', 576)
        self.drift_history_size = getattr(configs, 'drift_history_size', 20)
        self.drift_k_sigma = getattr(configs, 'drift_k_sigma', 2.0)

        self.train_epochs = getattr(configs, 'train_epochs', 10)
        self.learning_rate = getattr(configs, 'learning_rate', 0.0001)
        self.finetune_patience = getattr(configs, 'finetune_patience', 5)

        self.cycle_length = configs.cycle_length

        self.channel_independence = getattr(configs, 'channel_independence', 1)
        self.use_relation_layer = getattr(configs, 'use_relation_layer', 1)
        
        self.temporal_moe_layers = nn.ModuleList([
            TemporalDynamicMoELayer(
                d_model=configs.d_model,
                num_patches=self.num_patches,
                n_vars=configs.enc_in,
                cycle_length=self.cycle_length,
                num_experts=num_experts,
                num_drift_experts=num_drift_experts,
                dropout=configs.dropout,
                num_rnn_layers=num_rnn_layers,
                drift_window_size=self.drift_window_size,
                use_relation_layer=self.use_relation_layer,
                enable_drift_detection=bool(self.drift_detection_enabled),
                top_k=top_k
            ) for _ in range(self.num_temporal_moe_layers)
        ])

        revin_affine = getattr(configs, 'revin_affine', True)
        self.revin_layer = RevIN(configs.enc_in, eps=1e-5, affine=revin_affine)

        self.register_buffer('global_patch_counter', torch.tensor(0, dtype=torch.long))

        self.head_input_dim = configs.d_model * int((configs.seq_len - self.patch_len) / self.stride + 2)
        if self.channel_independence:
            self.projection_layers = nn.ModuleList([
                nn.Linear(self.head_input_dim, configs.pred_len)
                for _ in range(configs.enc_in)
            ])
        else:
            self.projection_layer = nn.Linear(self.head_input_dim, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        Args:
            x_enc: [B, S, N]
            x_mark_enc: [B, S, mark_dim]
        Returns:
            predictions: [B, pred_len, N]
        """
        B, S, N = x_enc.shape
        x_norm, means, stdev = self.revin_layer.normalize(x_enc)  # [B, S, N]

        x_patches = x_norm.permute(0, 2, 1)  # [B, N, S]
        enc_out, n_vars = self.patch_embedding(x_patches)  # [B*N, P, D]

        if x_mark_enc is not None:
            first_timestamp = x_mark_enc[:, 0, :]  # [B, mark_dim]
            patch_cycle_indices = (first_timestamp[:, 0] % self.cycle_length).long()
        else:
            patch_cycle_indices = torch.zeros(B, dtype=torch.long, device=x_enc.device)

        temporal_hidden_states = [None] * self.num_temporal_moe_layers

        final_temporal_hidden_states = []

        batch_start_patch_idx = self.global_patch_counter.item()
        BN = enc_out.shape[0]
        P = enc_out.shape[1]
        
        for layer_idx, moe_layer in enumerate(self.temporal_moe_layers):
            enc_out, new_hidden = moe_layer(
                enc_out,
                hidden_state=temporal_hidden_states[layer_idx],
                batch_start_patch_idx=batch_start_patch_idx,
                patch_cycle_indices=patch_cycle_indices
            )
            final_temporal_hidden_states.append(new_hidden)
        
        if self.training:
            self.global_patch_counter += BN * P

        self._last_temporal_hidden_states = final_temporal_hidden_states

        P = enc_out.shape[1]
        D = enc_out.shape[2]
        enc_out = enc_out.view(B, N, P, D)  # [B, N, P, D]
        enc_out_flat = enc_out.reshape(B, N, P * D)  # [B, N, D*P]


        if self.channel_independence:
            predictions = torch.stack([
                self.projection_layers[i](enc_out_flat[:, i, :])
                for i in range(N)
            ], dim=2)  # [B, pred_len, N]
        else:
            predictions = self.projection_layer(enc_out_flat)
            predictions = predictions.permute(0, 2, 1)  # [B, pred_len, N]

        predictions = self.revin_layer.denormalize(predictions, means, stdev)
        
        return predictions
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise NotImplementedError(f"Task {self.task_name} not supported")
    
    def enable_drift_detection(self, window_size=576, history_size=50, k_sigma=3.0):
        for layer_idx, moe_layer in enumerate(self.temporal_moe_layers):
            moe_layer.enable_drift_detection(
                window_size=window_size,
                history_size=history_size,
                k_sigma=k_sigma
            )
    
    def check_and_handle_drift(self, train_mode=True):
        for moe_layer in self.temporal_moe_layers:
            moe_layer.check_and_handle_drift(
                finetune_epochs=self.train_epochs,
                finetune_lr=self.learning_rate,
                finetune_patience=self.finetune_patience,
                train_mode=train_mode
            )
