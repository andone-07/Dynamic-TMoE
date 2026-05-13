#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#%%
class SeasonalFrequencyMLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(SeasonalFrequencyMLP, self).__init__()
        self.length = seq_len
        self.n_freq = seq_len//2 + 1
        self.hidden_dim = hidden_dim
        
        self.W = nn.Parameter(torch.empty((self.n_freq, input_dim, hidden_dim), dtype=torch.cfloat))
        self.B = nn.Parameter(torch.empty((self.n_freq, hidden_dim), dtype=torch.cfloat))
        self.W_rc = nn.Parameter(torch.empty((input_dim, hidden_dim)))
        
        self.fc = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), 
                        nn.LeakyReLU(),
                        nn.Linear(hidden_dim, output_dim)
                        )
        
        self._reset_parameters()
        
    def _reset_parameters(self) -> None:

        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.B, -bound, bound)
        
        nn.init.kaiming_uniform_(self.W_rc, a=math.sqrt(5))
        
    def _forward(self, x):
        out = torch.einsum('bti,tio->bto', x, self.W) + self.B
        out_r, out_c = F.relu(out.real), F.relu(out.imag)
        out = torch.stack([out_r, out_c], dim=-1)
        out = torch.view_as_complex(out)
        return out
          
    def forward(self, x):
        # inputs : (b, t, d)
        x_rc = torch.einsum('btd,dk->btk', x, self.W_rc)
        x_fft = torch.fft.rfft(x, dim=1)[:, :self.n_freq]
        out_fft = self._forward(x_fft)
        out = torch.fft.irfft(out_fft, n=x.size(1), dim=1)
        
        out = out + x_rc
        out = self.fc(out)
        return out 
