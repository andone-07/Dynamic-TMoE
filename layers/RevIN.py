import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
    
    def _get_statistics(self, x):
        if x.dim() == 3 and x.shape[-1] != self.num_features:
            x = x.transpose(1, 2)
        
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        
        return mean, std
    
    def _normalize(self, x, mean, std):
        normalized_x = (x - mean) / std

        if self.affine:
            gamma = self.gamma.view(1, 1, -1)
            beta = self.beta.view(1, 1, -1)
            normalized_x = gamma * normalized_x + beta
        
        return normalized_x
    
    def _denormalize(self, x, mean, std):
        if self.affine:
            gamma = self.gamma.view(1, 1, -1)
            beta = self.beta.view(1, 1, -1)
            x = (x - beta) / gamma

        denormalized_x = x * std + mean
        
        return denormalized_x
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            mean, std = self._get_statistics(x)
            normalized_x = self._normalize(x, mean, std)
            return normalized_x, mean, std
        
        elif mode == 'denorm':
            raise ValueError("For denorm mode, use denormalize() method directly with mean and std")
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'norm' or 'denorm'")
    
    def normalize(self, x):
        mean, std = self._get_statistics(x)
        normalized_x = self._normalize(x, mean, std)
        return normalized_x, mean, std
    
    def denormalize(self, x, mean, std):
        return self._denormalize(x, mean, std)
    
    def extra_repr(self):
        return f'num_features={self.num_features}, eps={self.eps}, affine={self.affine}'

class RevINLoss(nn.Module):
    def __init__(self, gamma_reg=0.01, beta_reg=0.01):
        super(RevINLoss, self).__init__()
        self.gamma_reg = gamma_reg
        self.beta_reg = beta_reg
    
    def forward(self, revin_layer):
        if not revin_layer.affine:
            return torch.tensor(0.0, device=next(revin_layer.parameters()).device)

        gamma_loss = self.gamma_reg * torch.mean((revin_layer.gamma - 1.0) ** 2)
        beta_loss = self.beta_reg * torch.mean(revin_layer.beta ** 2)
        
        return gamma_loss + beta_loss