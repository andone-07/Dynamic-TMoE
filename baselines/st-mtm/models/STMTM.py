import torch
import torch.nn as nn

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.FourierFC import SeasonalFrequencyMLP
from utils.losses import ContrastiveLoss
from utils.augmentations import SeasonalTrendMask


class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x
    
class TemporalProjection_Head(nn.Module):
    def __init__(self, seq_len, head_dropout=0):
        super().__init__()
        self.linear = nn.Linear(seq_len, 1)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x):
        x = self.linear(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.dropout(x)
        return x.squeeze()
    
class STMTMEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.trend_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        
        self.seasonal_encoder = SeasonalFrequencyMLP(configs.d_model, configs.d_hidden, configs.d_model, configs.seq_len)
        self.component_gating_layer = nn.Sequential(
                                        nn.Linear(2*configs.d_model, 2), 
                                        nn.Softmax(dim=-1)
                                        ) 
        
    def aggregate_trend_season(self, zt, zs):
        gate_input = torch.cat([zt, zs], dim=-1) # b, l, 2*d_model
        weights = self.component_gating_layer(gate_input) # b, l, 2 
        a = weights[:, :, 0].unsqueeze(-1)
        b = weights[:, :, 1].unsqueeze(-1)
        z = a * zt + b * zs
        return z
    
    def aggregate_seasonal_repr(self, xs_masked_list, ac):
        zs_masked_list = [self.seasonal_encoder(xs_masked) for xs_masked in xs_masked_list]
        zs_masked = torch.stack(zs_masked_list, dim=0).permute(1, 0, 2, 3)
        w_ac = torch.nn.functional.softmax(ac, dim=-1).view(ac.size(0), ac.size(1), 1, 1)
        agg_zs_masked = (w_ac * zs_masked).sum(axis=1)
        return agg_zs_masked
                
    def forward(self, xt, xs):
        zt, _ = self.trend_encoder(xt) # b, l, d_model
        zs = self.seasonal_encoder(xs) # b, l, d_model
        z = self.aggregate_trend_season(zt, zs) # b, l, d_model
        return z
    
    

class Model(nn.Module):
    """
    STMTM
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.configs = configs

        # Embedding
        self.trend_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.season_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # mask and ts decomposition
        self.mask = SeasonalTrendMask(configs.seq_len, configs.kernel_size, configs.seg_len, configs.p_tmask, configs.topk)
        # Encoder
        self.encoder = STMTMEncoder(configs)

        # Decoder
        if self.task_name == 'pretrain':
            # for reconstruction
            self.decoder = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)
            self.rec_criterion = torch.nn.MSELoss()
            self.cont_criterion = ContrastiveLoss(configs.batch_size, configs.topk, configs.tau)
            self.series_projector = TemporalProjection_Head(configs.seq_len, configs.head_dropout)
            self.alpha = configs.alpha
                
        elif self.task_name == 'finetune':
            self.pred_head = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)

    def forecast(self, x_enc, x_mark_enc):

        # data shape
        bs, seq_len, n_vars = x_enc.shape

        # mean subtraction
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means

        # channel independent
        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]
        
        # seasonal trend decomposition
        xt_enc, xs_enc = self.mask.extract_season_trend(x_enc)
        
        # embedding
        xt_enc = self.trend_embedding(xt_enc)
        xs_enc = self.season_embedding(xs_enc)
        
        # encoder
        enc_out = self.encoder(xt_enc, xs_enc)
        enc_out = torch.reshape(enc_out, (bs, n_vars, seq_len, -1))
        
        # decoder
        dec_out = self.pred_head(enc_out).permute(0, 2, 1) # dec_out: [bs x pred_len x n_vars]
        # mean addition
        dec_out = dec_out + means
        return dec_out
    
    def pretrain(self, x_enc, x_mark_enc):

        # data shape
        bs, seq_len, n_vars = x_enc.shape
        x_true = torch.clone(x_enc)
        
        # mean subtraction
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        
        # channel independent
        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.unsqueeze(-1) # x_enc: [bs x n_vars x seq_len x 1]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]
        
        # seasonal trend mask
        xt_m, (ac, xs_m_list) = self.mask(x_enc)
        xt_m_enc = self.trend_embedding(xt_m) 
        xs_m_enc_list = [self.season_embedding(xs_m) for xs_m in xs_m_list] 
        
        # trend encoder 
        enc_t_out, _ = self.encoder.trend_encoder(xt_m_enc)
        # seasonal encoder & adaptive aggregation
        enc_s_out = self.encoder.aggregate_seasonal_repr(xs_m_enc_list, ac)
        
        # series projector
        xs_m_proj_list = [self.series_projector(xs_m_enc) for xs_m_enc in xs_m_enc_list]
        proj_s_out = self.series_projector(enc_s_out)
        
        # contextual contrastive loss 
        cont_loss = self.cont_criterion(proj_s_out, xs_m_proj_list)

        # aggregation of trend and seasonality through component-wise gating layer
        enc_out = self.encoder.aggregate_trend_season(enc_t_out, enc_s_out)
        enc_out = torch.reshape(enc_out, (bs, n_vars, seq_len, -1))
        
        # decoder
        dec_out = self.decoder(enc_out)
        dec_out = dec_out.reshape(bs, n_vars, seq_len).permute(0, 2, 1)
        # mean addition
        dec_out = dec_out + means
        
        # reconstruction loss
        mse_loss = self.rec_criterion(dec_out, x_true)
        loss = mse_loss + self.alpha * cont_loss

        return loss, mse_loss, cont_loss


    def forward(self, x_enc, x_mark_enc):

        if self.task_name == 'pretrain':
            return self.pretrain(x_enc, x_mark_enc)
        if self.task_name == 'finetune':
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None
