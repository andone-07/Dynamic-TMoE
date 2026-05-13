import numpy as np
import torch
import torch.nn as nn
import math

#%%

def _arange_1d(start, end, step, sign):
    return torch.arange(start, end+step, step).int()*sign

def arange_1d(starts, ends, steps, reverse=False):
    if isinstance(ends, int):
        ends = torch.full(starts.size(), ends)
    if isinstance(steps, int):
        steps = torch.full(starts.size(), steps)
        
    sign = -1 if reverse else 1
    out = [_arange_1d(start, end, step, sign) for start, end, step in zip(starts.tolist(), ends.tolist(), steps.tolist())]
    return out

def cat(lst1, lst2):
    lst = [torch.unique(torch.cat([x, y], dim=-1), sorted=True) for x, y in zip(lst1, lst2)]
    return lst

def get_window_indices(anc_indices, window_size, l):
    window_indices = (anc_indices[:, None] + torch.arange(-window_size // 2, window_size // 2 + 1)).view(-1)
    window_indices = torch.unique(window_indices)
    window_indices = window_indices[(window_indices>=0)&(window_indices<l)]
    return torch.unique(window_indices)

class SeasonalTrendMask(nn.Module):
    def __init__(self, length, kernel_size, segment_len, p_tmask=0.2, topk=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.l = length
        self.segment_len = segment_len
        self.topk = topk
        self.equal_last_seg = True if length%segment_len==0 else False
        self.n_tseg = length // segment_len if self.equal_last_seg else length//segment_len + 1
        
        self.trend_extractor = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.padding, count_include_pad=False)
        self.tmask_sampler = torch.distributions.bernoulli.Bernoulli(p_tmask)
        
    def extract_season_trend(self, x):
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        
        xt = self.trend_extractor(x.transpose(-1, -2)).transpose(-1, -2)
        if self.kernel_size%2==0:
            xt = xt[:, :-1]
        xs = x - xt
        return xt, xs

    def mask_trend(self, xt):
        t_mask = self.tmask_sampler.sample((xt.size(0), self.n_tseg, xt.size(-1))) # 1 : mask, 0 : no mask
        t_mask = t_mask.unsqueeze(2).repeat(1, 1, self.segment_len, 1).view(xt.size(0), -1, xt.size(-1)).to(xt.device)
        if self.equal_last_seg is False:
            last_len = self.n_tseg*self.segment_len - self.l
            t_mask = t_mask[:, :-last_len]
        xt_masked = xt * (1-t_mask)
        return xt_masked
    
    def autocorrelation_based_periods(self, xs):
        xs_fft = torch.fft.rfft(xs.permute(0, 2, 1).contiguous(), dim=-1)
        res = xs_fft * torch.conj(xs_fft)
        corr = torch.fft.irfft(res, n=self.l, dim=-1) # (b, d, l)
        corr = corr.mean(dim=1) # (b, l)
        corr[:, :self.segment_len] = -1e5
        corr[:, -self.segment_len:] = -1e5
        topk_corr, periods = torch.topk(corr, self.topk, dim=-1) # (b, topk)
        return topk_corr, periods
     
    def mask(self, indices_list):
        b = len(indices_list)
        mask = torch.zeros((b, self.l, 1))
        for i, index in enumerate(indices_list):
            mask[i, index] = 1

        return mask

    def mask_season(self, xs):
        autocorr, periods = self.autocorrelation_based_periods(xs) # (b, topk)
        period_mask_list = []
        for k in range(self.topk):
            period =  periods[:, k] # (b, )
            anc_mask_indices = torch.randint(high=self.l-self.segment_len//2, size=(xs.size(0), )).to(xs.device) # (b, d, 1)
            left_mask_indices = arange_1d(-anc_mask_indices, 0, period, reverse=True)
            right_mask_indices = arange_1d(anc_mask_indices, self.l, period, reverse=False)
            mask_index_list = cat(left_mask_indices, right_mask_indices)
            mask_index_list = [get_window_indices(mask_index, self.segment_len, self.l) for mask_index in mask_index_list]
            period_mask_list.append(self.mask(mask_index_list))            
        xs_masked_list = [xs*(1-s_mask.to(xs.device)) for s_mask in period_mask_list]
        return autocorr, xs_masked_list
                    
    def forward(self, x):
        # x : b, l, d
        
        xt, xs = self.extract_season_trend(x)
        xt_masked = self.mask_trend(xt)
        autocorr, xs_masked_list = self.mask_season(xs)

        return xt_masked, (autocorr, xs_masked_list)
    
